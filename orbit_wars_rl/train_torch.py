"""Self-play PPO training on VecTorchEnv.

Pure self-play from scratch (no heuristic warm-start, no shaping). Both players
use the current policy. Trains on player 0's perspective; player 1's actions
are sampled from the same policy using player 1's view of state.

Usage:
    python train_torch.py --num-envs 512 --total-steps 100_000_000

Targets:
    M4 MPS:  ~6,000 SPS  (3M steps ≈ 8 min, 100M steps ≈ 4.5h)
    5090:   ~15,000+ SPS
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# wandb is optional — only imported when --wandb flag is passed
try:
    import wandb as _wandb
except ImportError:
    _wandb = None

from config import Config
from model import EntityTransformer, count_params
from opponent_pool import OpponentPool, PoolMember
from ppo import PPOLearner
from torch_env import (
    VecTorchEnv,
    MAX_OWNED,
    NUM_ANGLE_BINS,
    NUM_SHIP_BINS,
    SHIP_COUNTS,
    FRACTION_BIN_VALUES,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def atomic_torch_save(obj, path: str | os.PathLike) -> None:
    """Write a torch checkpoint via atomic rename to avoid partial .pt files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def sample_action_batched(outputs: dict, fire_mask: torch.Tensor,
                          target_mask: torch.Tensor | None = None):
    """Sample fire/ship/target actions for a batch of envs (target-decode only).

    Angle is not part of the executed policy — the env computes the aim direction
    from the sampled target planet index.  A zero tensor is returned for angle_a
    so the action tensor passed to env.step keeps its expected 4-column shape
    [fire, angle, ship, target].

    Returns: (fire_a, angle_a, ship_a, target_a, lp_fire, lp_ship, lp_target)
    """
    fire_logits   = outputs["fire_logits"].masked_fill(~fire_mask, -1e9)
    ship_logits   = outputs["ship_logits"]
    target_logits = outputs["target_logits"]
    if target_mask is not None:
        target_logits = target_logits.masked_fill(~target_mask, -1e9)

    fire_dist   = torch.distributions.Bernoulli(logits=fire_logits)
    ship_dist   = torch.distributions.Categorical(logits=ship_logits)
    target_dist = torch.distributions.Categorical(logits=target_logits)

    fire_a   = fire_dist.sample()    # (N, MAX_OWNED)
    ship_a   = ship_dist.sample()    # (N, MAX_OWNED)
    target_a = target_dist.sample()  # (N, MAX_OWNED)
    # Angle is unused in target-decode; zeros satisfy env.step's action shape.
    angle_a  = torch.zeros_like(fire_a)

    # Log probs only for valid slots / fired actions.
    slot_valid = fire_mask.float()
    fired      = (fire_a > 0.5).float() * slot_valid
    lp_fire   = fire_dist.log_prob(fire_a) * slot_valid
    lp_ship   = ship_dist.log_prob(ship_a)   * fired
    lp_target = target_dist.log_prob(target_a) * fired

    return fire_a.long(), angle_a, ship_a, target_a, lp_fire, lp_ship, lp_target


def decode_ship_bins(ship_bins: torch.Tensor, max_ships: torch.Tensor, ship_bin_mode: str) -> torch.Tensor:
    """Decode sampled ship bins to ship counts using the same semantics as VecTorchEnv."""
    if ship_bin_mode == "fraction":
        frac_t = torch.tensor(FRACTION_BIN_VALUES, dtype=torch.float32, device=ship_bins.device)
        idx = ship_bins.long().clamp(0, len(FRACTION_BIN_VALUES) - 1)
        max_sendable = (max_ships.to(ship_bins.device).float() - 1.0).clamp(min=1.0)
        return torch.round(frac_t[idx] * max_sendable).clamp(min=1.0)

    counts_t = torch.tensor(SHIP_COUNTS, dtype=torch.float32, device=ship_bins.device)
    idx = ship_bins.long().clamp(0, len(SHIP_COUNTS) - 1)
    return counts_t[idx]


def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                dones: torch.Tensor, next_value: torch.Tensor,
                gamma: float, lam: float):
    """Vectorized GAE across (T, N).

    Args:
        rewards: (T, N)
        values:  (T, N)
        dones:   (T, N) bool
        next_value: (N,) bootstrap value after the rollout
    Returns:
        advantages, returns: each (T, N)
    """
    T, N = rewards.shape
    advantages = torch.zeros(T, N, device=rewards.device)
    last_gae = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        if t == T - 1:
            next_v = next_value
            next_nonterm = (~dones[t]).float()
        else:
            next_v = values[t + 1]
            next_nonterm = (~dones[t]).float()
        delta = rewards[t] + gamma * next_v * next_nonterm - values[t]
        last_gae = delta + gamma * lam * next_nonterm * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# In-training eval was removed: vs-frozen-initial gave false positives
# (degenerate policies "improved" over the unchanged baseline). Source of
# truth is local eval (eval.py) on downloaded checkpoints against raw
# Suneet/Zach/Rahul. See docs/bugs.md.
#
# _heuristic_moves_to_action_tensor is kept — pool-mode=mixed uses it to
# convert external-heuristic Python moves into the action tensor format.


# ----------------------------------------------------------------------------
# Multiprocessing pool for external heuristic opponents.
#
# Without this, a pool rollout serializes N agent_fn(obs) calls on one CPU.
# Suneet (3239 LoC forward-search) at 51 envs × 64 steps drops SPS from
# ~1500 to ~260 on g5.2xlarge. Each call is independent and stateless across
# turns, so we fan them out to a worker pool that pre-loads the agent module
# once via fork — Suneet's per-call cost is unchanged but parallelism scales
# with vCPUs. Expected gain: ~5-7× on 8 vCPUs.
# ----------------------------------------------------------------------------

import multiprocessing as _mp

_WORKER_AGENT_FN = None  # populated per-worker in _heur_worker_init


def _heur_worker_init(agent_path: str):
    """Each worker fork loads the agent module once and stashes its agent fn."""
    import importlib.util, sys
    # Pin each worker to ONE torch thread. With N workers each defaulting to
    # multi-threaded intra-op parallelism on an 8-vCPU box, they oversubscribe
    # the cores and thrash — a planner call measured 280ms vs 40ms pinned (7x).
    # Only the workers are pinned; the main (GPU-feeding) process keeps its threads.
    import torch
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("worker_agent", agent_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # required for @dataclass __module__ resolution
    spec.loader.exec_module(mod)
    global _WORKER_AGENT_FN
    _WORKER_AGENT_FN = mod.agent


def _heur_worker_call(obs):
    """Run the worker's agent on one obs. Logs exceptions and returns no-op."""
    try:
        return _WORKER_AGENT_FN(obs) or []
    except Exception as exc:
        import sys, traceback
        print(f"WARNING heur_worker_call: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return []


class HeuristicWorkerPool:
    """Persistent process pool around one external heuristic agent."""
    def __init__(self, agent_path: str, num_workers: int):
        # Use 'spawn' to avoid inheriting parent's CUDA context (fork+CUDA = deadlock).
        ctx = _mp.get_context("spawn")
        self.pool = ctx.Pool(
            processes=num_workers,
            initializer=_heur_worker_init,
            initargs=(agent_path,),
        )
        self.num_workers = num_workers
        self.agent_path = agent_path

    def map(self, obs_list, timeout: float = 30.0):
        result = self.pool.map_async(_heur_worker_call, obs_list)
        try:
            return result.get(timeout=timeout)
        except Exception as exc:
            import sys
            print(f"WARNING HeuristicWorkerPool.map fallback ({type(exc).__name__}: {exc})"
                  f" — {len(obs_list)} envs will use no-op actions this rollout", file=sys.stderr)
            return [[] for _ in obs_list]

    def close(self):
        self.pool.close()
        self.pool.join()


def _heuristic_moves_to_action_tensor(moves_per_env, env, player, device):
    """Convert list-of-lists [[from_pid, angle_rad, ships], ...] per env into
    a (num_envs, MAX_OWNED, 3) action tensor."""
    from torch_env import MAX_OWNED, NUM_ANGLE_BINS, ANGLE_BIN_WIDTH, SHIP_COUNTS
    import math as _math

    N = env.num_envs
    owned_idx, slot_valid = env.owned_indices_for(player)   # (N, MAX_OWNED)
    fire = torch.zeros(N, MAX_OWNED, dtype=torch.long)
    angle_bin = torch.zeros(N, MAX_OWNED, dtype=torch.long)
    ship_bin = torch.zeros(N, MAX_OWNED, dtype=torch.long)

    gather_idx = owned_idx.unsqueeze(-1).expand(-1, -1, 7).cpu()
    planets_cpu = env.planets.cpu()
    src = planets_cpu.gather(1, gather_idx)
    sv_cpu = slot_valid.cpu()

    for e in range(N):
        moves = moves_per_env[e]
        if not moves:
            continue
        pid_to_slot = {}
        for k in range(MAX_OWNED):
            if sv_cpu[e, k].item():
                pid_to_slot[int(src[e, k, 0].item())] = k
        for mv in moves:
            if not isinstance(mv, (list, tuple)) or len(mv) < 3:
                continue
            slot = pid_to_slot.get(int(mv[0]))
            if slot is None:
                continue
            ang = float(mv[1]) % (2 * _math.pi)
            ab = int(ang / ANGLE_BIN_WIDTH) % NUM_ANGLE_BINS
            ships = int(mv[2])
            best, best_diff = 0, 10**9
            for b, c in enumerate(SHIP_COUNTS):
                if abs(c - ships) < best_diff:
                    best_diff, best = abs(c - ships), b
            fire[e, slot] = 1
            angle_bin[e, slot] = ab
            ship_bin[e, slot] = best
    return torch.stack([fire, angle_bin, ship_bin], dim=-1).to(device)


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    print(f"Training on device: {device}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Rollout steps: {args.rollout_steps}")
    print(f"Batch per update: {args.num_envs * args.rollout_steps * 2}  "
          f"(T={args.rollout_steps} x N={args.num_envs} x P=2 players)")

    torch.manual_seed(args.seed)

    cfg = Config()
    cfg.ppo.total_env_steps = args.total_steps
    cfg.device = args.device
    cfg.ppo.num_minibatches = args.num_minibatches
    if args.learning_rate is not None:
        cfg.ppo.learning_rate = args.learning_rate
    if args.ppo_epochs is not None:
        cfg.ppo.ppo_epochs = args.ppo_epochs
    if args.clip_eps is not None:
        cfg.ppo.clip_eps = args.clip_eps
    if args.entropy_coef_fire is not None:
        cfg.ppo.entropy_coef_fire = args.entropy_coef_fire
    # --entropy-coef-target is the honest name; --entropy-coef-angle is a deprecated
    # alias kept so old launch scripts still work (the coef has always weighted the
    # target head, not the now-vestigial angle head).
    if args.entropy_coef_target is not None:
        cfg.ppo.entropy_coef_target = args.entropy_coef_target
    elif args.entropy_coef_angle is not None:
        cfg.ppo.entropy_coef_target = args.entropy_coef_angle
    if args.entropy_coef_ships is not None:
        cfg.ppo.entropy_coef_ships = args.entropy_coef_ships
    if args.max_grad_norm is not None:
        cfg.ppo.max_grad_norm = args.max_grad_norm
    if args.gae_lambda is not None:
        cfg.ppo.gae_lambda = args.gae_lambda
    if args.il_lambda is not None:
        cfg.ppo.il_lambda = args.il_lambda
    if args.il_decay_frac is not None:
        cfg.ppo.il_decay_frac = args.il_decay_frac
    if args.bc_coef is not None:
        cfg.ppo.bc_coef = args.bc_coef
    print(f"PPO config: lr={cfg.ppo.learning_rate}, ppo_epochs={cfg.ppo.ppo_epochs}, "
          f"num_minibatches={cfg.ppo.num_minibatches}, clip_eps={cfg.ppo.clip_eps}, "
          f"entropy_coef_fire={cfg.ppo.entropy_coef_fire}, gae_lambda={cfg.ppo.gae_lambda}, "
          f"kl_target={cfg.ppo.kl_target}")
    print(f"Entropy coefs: fire={cfg.ppo.entropy_coef_fire}, target={cfg.ppo.entropy_coef_target}, "
          f"ships={cfg.ppo.entropy_coef_ships} | max_grad_norm={cfg.ppo.max_grad_norm}")
    print(f"Action decode: {args.action_decode}")
    print(f"Reinforcement (own planets as targets): {'ON' if args.allow_reinforce else 'off'}")
    if args.allow_reinforce and args.reinforce_anneal_frac > 0.0:
        print(f"Reinforcement CURRICULUM: own-target logit bias {args.reinforce_bias_init}→0 over "
              f"{args.reinforce_anneal_frac * args.total_steps:,.0f} steps "
              f"(frac {args.reinforce_anneal_frac}), then 0 — enemy/neutral targeting untouched")
    if args.allow_reinforce and (args.reinforce_garrison_floor > 0.0 or args.reinforce_cost > 0.0):
        print(f"Reinforcement DISCIPLINE: garrison_floor={args.reinforce_garrison_floor} "
              f"(veto reinforce that drains source below this), "
              f"cost={args.reinforce_cost}/ship (reward penalty on ships reinforced)")
    print(f"Win margin coeff: {args.win_margin_coeff}")
    print(f"Shaping coeff: {args.shaping_coef}")
    print(f"Expansion coeff: {args.expansion_coef}")
    print(f"Defense coeff: {args.defense_coef}")
    print(f"Early capture coeff: {args.early_capture_coef} (decay over {args.early_capture_steps} steps)")
    if args.early_capture_anneal_frac > 0.0:
        print(f"Early capture ANNEAL: cosine {args.early_capture_coef}→0 over "
              f"{args.early_capture_anneal_frac * args.total_steps:,.0f} steps "
              f"(frac {args.early_capture_anneal_frac}), then 0")
    print(f"First Strike: {args.first_strike_mult}x for t<{args.first_strike_steps} steps" if args.first_strike_steps > 0 else "First Strike: off")
    print(f"Speed coeff: {args.speed_coef}")
    if args.handicap_frac > 0:
        print(f"Handicap: {args.handicap_frac*100:.0f}% of games start with {args.handicap_ships} ships (vs normal 10)")
    if args.ssdr_frac > 0:
        print(f"SSDR: {args.ssdr_frac*100:.0f}% of resets grant opponent 1..{args.ssdr_max_steps} extra planets (asymmetric start)")
    if args.srcs_multi_penalty > 0.0:
        print(f"srcs_multi penalty: coef={args.srcs_multi_penalty}, threshold={args.srcs_multi_threshold}, "
              f"decay_frac={args.srcs_multi_penalty_decay_frac} "
              f"({'cosine decay to 0' if args.srcs_multi_penalty_decay_frac > 0 else 'constant'})")
    if args.fleet_activity_coef > 0.0:
        print(f"fleet_activity reward: coef={args.fleet_activity_coef} (per step any planet fires)")

    # Honor model-config fields saved in the checkpoint (num_ship_bins,
    # ship_bin_mode, min_ship_bin) BEFORE creating env or model.
    if args.resume:
        _ckpt_peek = torch.load(args.resume, map_location="cpu", weights_only=False)
        ckpt_cfg = _ckpt_peek.get("config", {}) if isinstance(_ckpt_peek, dict) else {}
        if "num_ship_bins" in ckpt_cfg:
            cfg.model.num_ship_bins = int(ckpt_cfg["num_ship_bins"])
            print(f"Checkpoint declares num_ship_bins={cfg.model.num_ship_bins}")
        else:
            sd_peek = _ckpt_peek.get("model", _ckpt_peek)
            if "ship_head.weight" in sd_peek:
                n = int(sd_peek["ship_head.weight"].shape[0])
                if n != cfg.model.num_ship_bins:
                    cfg.model.num_ship_bins = n
                    print(f"Checkpoint ship_head implies num_ship_bins={n}")
        if "ship_bin_mode" in ckpt_cfg:
            cfg.model.ship_bin_mode = str(ckpt_cfg["ship_bin_mode"])
            print(f"Checkpoint declares ship_bin_mode={cfg.model.ship_bin_mode}")
        if "min_ship_bin" in ckpt_cfg:
            cfg.model.min_ship_bin = int(ckpt_cfg["min_ship_bin"])
        del _ckpt_peek

    # CLI overrides checkpoint metadata. This lets a run deliberately mask bin
    # 0 when resuming from a BC checkpoint saved with min_ship_bin=0.
    if args.min_ship_bin is not None:
        cfg.model.min_ship_bin = args.min_ship_bin
    cfg.model.action_decode = args.action_decode
    cfg.model.allow_reinforce = args.allow_reinforce

    env = VecTorchEnv(num_envs=args.num_envs, num_players=2,
                      device=device, episode_steps=500,
                      ship_bin_mode=cfg.model.ship_bin_mode,
                      action_decode=args.action_decode,
                      allow_reinforce=args.allow_reinforce,
                      reinforce_garrison_floor=args.reinforce_garrison_floor,
                      reinforce_cost=args.reinforce_cost,
                      win_margin_coeff=args.win_margin_coeff,
                      shaping_coef=args.shaping_coef,
                      expansion_coef=args.expansion_coef,
                      defense_coef=args.defense_coef,
                      early_capture_coef=args.early_capture_coef,
                      early_capture_steps=args.early_capture_steps,
                      first_strike_steps=args.first_strike_steps,
                      first_strike_mult=args.first_strike_mult,
                      speed_coef=args.speed_coef,
                      handicap_frac=args.handicap_frac,
                      handicap_ships=args.handicap_ships,
                      ssdr_frac=args.ssdr_frac,
                      ssdr_max_steps=args.ssdr_max_steps)
    env.reset(seeds=[args.seed + i for i in range(args.num_envs)])

    model = EntityTransformer(cfg.model).to(device)
    print(f"Model params: {count_params(model):,}")
    if args.resume:
        sd = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model" in sd: sd = sd["model"]
        model.load_state_dict(sd)
        print(f"Resumed from {Path(args.resume).resolve()}")
        if getattr(args, "reinit_critic", False):
            # CONTROL: re-initialise the value head to a fresh state while keeping
            # the warm policy. Isolates the cold-critic shock — if a known-stable
            # warm-critic method (joint) collapses with a fresh critic, resume is
            # confounded for new-critic methods (VDN) and we should go from-scratch.
            for _m in (model.value_fc1, model.value_fc2, model.value_out):
                _m.reset_parameters()
            print("  CONTROL: scalar critic re-initialised fresh (warm policy kept).")

    # IL regularization: load frozen reference policy if requested
    frozen_il_model = None
    if args.il_ref:
        frozen_il_model = EntityTransformer(cfg.model).to(device)
        sd = torch.load(args.il_ref, map_location="cpu", weights_only=False)
        if "model" in sd: sd = sd["model"]
        frozen_il_model.load_state_dict(sd)
        frozen_il_model.eval()
        print(f"IL reference loaded from {args.il_ref}  (lambda={cfg.ppo.il_lambda}, "
              f"decay_frac={cfg.ppo.il_decay_frac})")
    elif cfg.ppo.il_lambda > 0:
        # If --resume is set and no separate ref, default to the resume checkpoint
        if args.resume:
            frozen_il_model = EntityTransformer(cfg.model).to(device)
            sd = torch.load(args.resume, map_location="cpu", weights_only=False)
            if "model" in sd: sd = sd["model"]
            frozen_il_model.load_state_dict(sd)
            frozen_il_model.eval()
            print(f"IL reference defaulted to --resume checkpoint  (lambda={cfg.ppo.il_lambda})")
        else:
            print("WARNING: il_lambda > 0 but no --il-ref and no --resume — IL disabled")

    roi_heads = None
    if args.aux_roi_coef > 0.0:
        roi_heads = {
            "kv": nn.Linear(2 * cfg.model.entity_dim, 3, bias=False),
            "ts": nn.Linear(cfg.model.entity_dim, 3, bias=False),
        }
        print(f"ROI aux loss: coef={args.aux_roi_coef} (keeps pair_kv/target_scorer new cols encoding roi_20/roi_50/enemy_contest)")

    learner = PPOLearner(model, cfg, device=device, frozen_il_model=frozen_il_model,
                        roi_heads=roi_heads, aux_roi_coef=args.aux_roi_coef)

    # BC auxiliary supervision: load teacher samples once, sample a minibatch
    # per PPO update. Cross-entropy on teacher's actions directly penalizes
    # argmax-drift away from teacher — fixes the failure mode where KL-on-
    # distributions (il_lambda) keeps distributions close but argmax flips.
    bc_samples_for_aux = None
    if args.bc_samples and cfg.ppo.bc_coef > 0:
        import pickle as _pkl
        with open(args.bc_samples, "rb") as f:
            bc_samples_for_aux = _pkl.load(f)
        print(f"BC auxiliary samples: {len(bc_samples_for_aux)} from {args.bc_samples} "
              f"(bc_coef={cfg.ppo.bc_coef})")

    # LR scheduler: warmup + cosine decay over total updates.
    # On --resume, skip warmup by default (the model is already trained — no
    # reason to ramp LR from 0). User can force warmup with --with-warmup.
    updates_per_batch = cfg.ppo.ppo_epochs * cfg.ppo.num_minibatches
    total_updates = (args.total_steps // (args.num_envs * args.rollout_steps)) * updates_per_batch
    # --lr-schedule-steps decouples the LR decay horizon from actual training
    # length. Set it larger than --total-steps for slow/partial decay, or equal
    # for the default full cosine decay to zero.
    lr_schedule_steps = args.lr_schedule_steps if args.lr_schedule_steps > 0 else args.total_steps
    lr_total_updates = (lr_schedule_steps // (args.num_envs * args.rollout_steps)) * updates_per_batch
    skip_warmup = (args.resume and not args.with_warmup) or args.skip_warmup
    warmup = 0 if skip_warmup else cfg.ppo.lr_warmup_steps
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(lr_total_updates - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(learner.optimizer, lr_lambda)
    print(f"LR schedule: warmup={warmup} updates (skip_warmup={skip_warmup}), "
          f"total_updates={total_updates}, lr_schedule_steps={lr_schedule_steps}, "
          f"peak_lr={cfg.ppo.learning_rate}")

    # ----------------------------------------------------------------------
    # Opponent pool setup (PFSP self-play with optional external heuristics)
    # ----------------------------------------------------------------------
    pool: OpponentPool | None = None
    pool_opp_model: EntityTransformer | None = None  # reusable frozen model for 'self' opponents
    rng = random.Random(args.seed)
    if args.pool_mode != "none":
        # If --resume points to a checkpoint with a sibling pool file (e.g.
        # checkpoints/pool_step_<N>.pt next to torch_step_<N>.pt), reload it.
        resumed_pool_path = None
        if args.resume:
            stem = Path(args.resume).stem  # torch_step_<N>
            candidate = Path(args.resume).parent / f"pool_{stem.replace('torch_', '')}.pt"
            if candidate.exists():
                resumed_pool_path = str(candidate)

        if resumed_pool_path:
            pool = OpponentPool.load(resumed_pool_path)
            print(f"Pool resumed from {resumed_pool_path}: {len(pool)} members")
        else:
            pool = OpponentPool(
                max_self_members=args.pool_max_size,
                pfsp_alpha=args.pool_pfsp_alpha,
                mastered_winrate=args.pool_mastered_threshold,
                mastered_min_games=args.pool_mastered_min_games,
                pfsp_min_games=args.pool_pfsp_min_games,
                external_fraction=args.pool_external_fraction,
            )
            # Seed pool with the starting weights so it isn't empty on iteration 1
            pool.add_self_checkpoint(0, model.state_dict())

        # Preseed pool from a directory of .pt checkpoints (appended as 'self'
        # members). Lets us dilute the heuristic share from iter 1 by populating
        # the pool with prior-run snapshots instead of waiting for organic
        # snapshots to accumulate.
        if args.preseed_pool:
            import re
            preseed_dir = Path(args.preseed_pool)
            existing_self_steps = {m.step_saved for m in pool.members if m.kind == "self"}
            added = 0
            for pt_file in sorted(preseed_dir.glob("*.pt")):
                m = re.search(r"step_(\d+)", pt_file.stem)
                if not m: continue
                step = int(m.group(1))
                if step in existing_self_steps: continue
                sd = torch.load(pt_file, map_location="cpu", weights_only=False)
                if "model" in sd: sd = sd["model"]
                pool.add_self_checkpoint(step, sd)
                added += 1
            print(f"  pool preseeded: {added} self-checkpoints from {preseed_dir}")

        # External opponents come from CLI flag — appended even on resume so
        # the user can add new externals without modifying the pool file.
        if args.pool_mode == "mixed" and args.external_opponents:
            existing_ext_names = {m.name for m in pool.members if m.kind == "external_heuristic"}
            for path in args.external_opponents.split(","):
                path = path.strip()
                if not path: continue
                name = Path(path).stem
                if name in existing_ext_names:
                    print(f"  pool external already present from resume: {name}")
                    continue
                pool.add_external_heuristic(name, path)
                print(f"  pool external loaded: {name} ({path})")
        # Create one dispatcher per external heuristic, keyed by member name so
        # compute_pool_actions can dispatch by opp.name. Planner-externals (heavy
        # orbit_lite agents) use the in-process GPU adapter; the rest use CPU pools.
        planner_names = {Path(p.strip()).stem for p in args.planner_externals.split(",") if p.strip()}
        heur_worker_pools: dict[str, HeuristicWorkerPool] = {}
        planner_adapters: dict = {}
        nw = args.heuristic_workers if args.heuristic_workers > 0 else max(1, (os.cpu_count() or 2) - 1)
        for m in pool.members:
            if m.kind == "external_heuristic":
                src = getattr(m, "_source_path", None) or args.external_opponents.split(",")[0].strip()
                if m.name in planner_names:
                    from batched_planner import BatchedPlannerOpponent
                    planner_adapters[m.name] = BatchedPlannerOpponent(src, args.num_envs, device=str(device))
                    print(f"  planner adapter (GPU in-process): {m.name} on {device}")
                else:
                    heur_worker_pools[m.name] = HeuristicWorkerPool(src, nw)
                    print(f"  heuristic worker pool: {m.name} × {nw} workers")
        # Pre-allocate a frozen model for 'self' opponents (loaded with state_dict per rollout)
        pool_opp_model = copy.deepcopy(model).to(device)
        pool_opp_model.eval()
        print(f"Pool initialised: mode={args.pool_mode}, members={len(pool)}, "
              f"fraction={args.pool_fraction}, snapshot_every={args.pool_checkpoint_interval:,} steps\n")
    last_pool_snapshot_step = 0

    total_env_steps = 0
    iter_count = 0
    start = time.perf_counter()
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Embed run name in checkpoint filenames so rev11_2M is unambiguous
    if args.run_name:
        run_ts = f"{args.run_name}_{run_ts}"
    print(f"Run timestamp: {run_ts}")

    # --- W&B init -----------------------------------------------------------
    wb = None
    if args.wandb:
        if _wandb is None:
            print("WARNING: --wandb passed but wandb package not installed. "
                  "Run: pip install wandb")
        else:
            wb = _wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or run_ts,
                config={
                    "total_steps": args.total_steps,
                    "num_envs": args.num_envs,
                    "rollout_steps": args.rollout_steps,
                    "learning_rate": args.learning_rate,
                    "lr_schedule_steps": args.lr_schedule_steps,
                    "ppo_epochs": cfg.ppo.ppo_epochs,
                    "num_minibatches": cfg.ppo.num_minibatches,
                    "pool_mode": args.pool_mode,
                    "pool_fraction": args.pool_fraction,
                    "pool_external_fraction": args.pool_external_fraction,
                    "srcs_multi_penalty": args.srcs_multi_penalty,
                    "srcs_multi_threshold": args.srcs_multi_threshold,
                    "fleet_activity_coef": args.fleet_activity_coef,
                    "il_lambda": cfg.ppo.il_lambda,
                    "win_margin_coeff": args.win_margin_coeff,
                    "speed_coef": args.speed_coef,
                    "action_decode": args.action_decode,
                    "resume": args.resume or "",
                    "ship_bin_mode": cfg.model.ship_bin_mode,
                },
                resume="allow",
            )
            print(f"W&B run: {wb.url}")

    reward_history = deque(maxlen=200)     # p0 episode rewards
    reward_history_p1 = deque(maxlen=200)  # p1 episode rewards (should mirror p0)
    clipfrac_history = deque(maxlen=50)
    best_avg_reward = float("-inf")
    last_log = start
    last_ckpt_step = 0

    rollout_T = args.rollout_steps
    N = args.num_envs
    P = 2  # num players — both players' transitions are collected and used for PPO

    # Pre-allocate rollout buffers on CPU to keep GPU memory free for the model.
    # Shape leading dims: (T, N, P) so player 0 and player 1 each contribute a
    # full trajectory per env step. Effective PPO batch size = T*N*P.
    storage_dev = torch.device("cpu")
    storage = {
        "planet_features": torch.zeros(rollout_T, N, P, 48, cfg.model.planet_feature_dim, device=storage_dev),
        "fleet_features":  torch.zeros(rollout_T, N, P, 128, cfg.model.fleet_feature_dim, device=storage_dev),
        "global_features": torch.zeros(rollout_T, N, P, cfg.model.global_feature_dim, device=storage_dev),
        "planet_mask":     torch.zeros(rollout_T, N, P, 48, dtype=torch.bool, device=storage_dev),
        "fleet_mask":      torch.zeros(rollout_T, N, P, 128, dtype=torch.bool, device=storage_dev),
        "fire_mask":       torch.zeros(rollout_T, N, P, MAX_OWNED, dtype=torch.bool, device=storage_dev),
        "slot_valid":      torch.zeros(rollout_T, N, P, MAX_OWNED, dtype=torch.bool, device=storage_dev),
        "target_mask":     torch.zeros(rollout_T, N, P, MAX_OWNED, cfg.env.max_planets, dtype=torch.bool, device=storage_dev),
        "owned_indices":   torch.zeros(rollout_T, N, P, MAX_OWNED, dtype=torch.long, device=storage_dev),
        # Store pairwise features whenever the model USES them: the target head
        # (per-(slot,target) scorer) needs them in the PPO-update forward, or it
        # falls back to a zeros/uniform target head that disagrees with the rollout
        # policy → persistent rollout-vs-update mismatch (KL/clip explode, never
        # stabilise). Must NOT be gated on aux_roi_coef (a separate aux loss); doing
        # so silently broke the target head whenever aux_roi_coef=0 (rev49+).
        **({"pairwise_features": torch.zeros(rollout_T, N, P, MAX_OWNED, cfg.env.max_planets, cfg.model.pairwise_feature_dim, device=storage_dev)} if cfg.model.pairwise_feature_dim > 0 else {}),
        "fire_a":     torch.zeros(rollout_T, N, P, MAX_OWNED, dtype=torch.long, device=storage_dev),
        "ship_a":     torch.zeros(rollout_T, N, P, MAX_OWNED, dtype=torch.long, device=storage_dev),
        "ship_count_a": torch.zeros(rollout_T, N, P, MAX_OWNED, device=storage_dev),
        "target_a":   torch.zeros(rollout_T, N, P, MAX_OWNED, dtype=torch.long, device=storage_dev),
        "lp_fire":    torch.zeros(rollout_T, N, P, MAX_OWNED, device=storage_dev),
        "lp_ship":    torch.zeros(rollout_T, N, P, MAX_OWNED, device=storage_dev),
        "lp_target":  torch.zeros(rollout_T, N, P, MAX_OWNED, device=storage_dev),
        "values":     torch.zeros(rollout_T, N, P, device=storage_dev),
        "rewards":    torch.zeros(rollout_T, N, P, device=storage_dev),
        "dones":      torch.zeros(rollout_T, N, P, dtype=torch.bool, device=storage_dev),
        # train_mask[t, e, p] = True if (env=e, player=p) is OUR current policy at
        # step t. Pool-opponent slots are False so PPO won't train on them.
        "train_mask": torch.ones(rollout_T, N, P, dtype=torch.bool, device=storage_dev),
        # Planet ship counts snapshot per step — used for avgfleet/p90fleet metrics.
        # Shape (T, N) — mean ships per planet across all owned planets per env.
        # Measuring planet inventories (not action sizes) is the correct passivity proxy.
        "planet_ships_snap": torch.zeros(rollout_T, N, device=storage_dev),
    }

    def compute_pool_actions(opp: PoolMember, player: int, env_slice: slice) -> torch.Tensor:
        """Return action tensor (N_pool, MAX_OWNED, 3) for the opponent playing
        `player` in envs `env_slice`. Supports 'self' (frozen RL model on GPU)
        and 'external_heuristic' (.py agent via legacy Python obs)."""
        if opp.kind == "self":
            # Load opp weights into the reusable frozen model
            pool_opp_model.load_state_dict(opp.state_dict)
            feats = env.get_features(player, max_planets=cfg.env.max_planets, max_fleets=128)
            with torch.no_grad():
                outs = pool_opp_model(
                    feats["planet_features"], feats["fleet_features"],
                    feats["global_features"], feats["planet_mask"],
                    feats["fleet_mask"],
                    fire_mask=feats["fire_mask"], angle_mask=feats["angle_mask"],
                    slot_valid=feats["slot_valid"], owned_indices=feats["owned_indices"],
                    owned_count=feats["owned_count"],
                    pairwise_features=feats.get("pairwise_features"),
                )
            fire_a, angle_a, ship_a, target_a, *_ = sample_action_batched(
                outs, feats["fire_mask"], feats.get("target_mask")
            )
            return torch.stack([fire_a, angle_a, ship_a, target_a], dim=-1)[env_slice]

        if opp.kind == "external_heuristic":
            from torch_env import to_legacy_obs
            start, stop = env_slice.start or 0, env_slice.stop or N
            pa = planner_adapters.get(opp.name)
            if pa is not None:
                # In-process GPU planner: no obs pickling / no CPU worker pool.
                moves_per_env = pa.moves(env, player, env_slice=slice(start, stop))
            else:
                obs_list = [to_legacy_obs(env, env_idx=e, player=player)
                            for e in range(start, stop)]
                wp = heur_worker_pools.get(opp.name)
                if wp is not None:
                    moves_per_env = wp.map(obs_list)
                else:
                    # Fallback: serial path (no worker pool registered)
                    moves_per_env = []
                    for obs in obs_list:
                        try:
                            moves_per_env.append(opp.agent_fn(obs) or [])
                        except Exception:
                            moves_per_env.append([])
            # Build action tensor with the same converter the cloud eval uses,
            # but only over the env slice. _heuristic_moves_to_action_tensor
            # expects N rows so we build over the slice and return.
            class _SliceView:
                """Minimal view-like adapter so _heuristic_moves_to_action_tensor
                sees just the slice it should write into."""
                def __init__(self, env, slc):
                    self.num_envs = stop - start
                    self.planets = env.planets[slc]
                    self._env = env; self._slc = slc
                def owned_indices_for(self, player):
                    oi, sv = env.owned_indices_for(player)
                    return oi[self._slc], sv[self._slc]
            view = _SliceView(env, env_slice)
            act = _heuristic_moves_to_action_tensor(moves_per_env, view, player, device)
            if args.action_decode == "target":
                # External heuristics already emit explicit angles. Use -1 as a
                # sentinel so VecTorchEnv keeps angle-bin decoding for these rows
                # even when the learned policy uses target decoding.
                pad_target = torch.full(
                    act.shape[:-1] + (1,), -1, dtype=act.dtype, device=act.device
                )
                act = torch.cat([act, pad_target], dim=-1)
            return act

        raise ValueError(f"unknown opponent kind: {opp.kind}")

    def forward_player(player: int):
        """Run model forward for given player, return outputs + features dict."""
        feats = env.get_features(player, max_planets=cfg.env.max_planets, max_fleets=128)
        owned_count = feats["owned_count"]
        with torch.no_grad():
            outs = model(
                feats["planet_features"], feats["fleet_features"],
                feats["global_features"], feats["planet_mask"],
                feats["fleet_mask"],
                fire_mask=feats["fire_mask"],
                angle_mask=feats["angle_mask"],
                slot_valid=feats["slot_valid"],
                owned_indices=feats["owned_indices"],
                owned_count=owned_count,
                pairwise_features=feats.get("pairwise_features"),
            )
        return feats, outs

    print(f"\nStarting self-play training (target {args.total_steps:,} env steps)")
    print("=" * 70)

    while total_env_steps < args.total_steps:
        # --- Per-rollout opponent assignment --------------------------------
        # If a pool is configured, dedicate `pool_fraction` of envs to play
        # current-vs-(sampled pool opponent). The remaining envs do P=2
        # symmetric self-play. The pool opponent is sampled ONCE per rollout
        # (not per env) — keeps pool-opponent forward cheap.
        pool_opp: PoolMember | None = None
        N_self, N_pool = N, 0
        current_seat = 0
        if pool is not None and len(pool) > 0 and args.pool_fraction > 0:
            pool_opp = pool.sample(rng)
            N_pool = int(N * args.pool_fraction)
            N_self = N - N_pool
            current_seat = rng.randint(0, 1)  # alternate so current sees both seats
        opp_seat = 1 - current_seat
        pool_slice = slice(N_self, N) if N_pool > 0 else slice(0, 0)
        # Inform env which envs are self-play (SSDR active) vs pool (symmetric).
        # Pool envs get clean symmetric starts so old checkpoints aren't poisoned
        # by asymmetric boards they were never trained on.
        if env.ssdr_frac > 0.0:
            self_mask = torch.zeros(N, dtype=torch.bool)
            self_mask[:N_self] = True
            env.set_ssdr_mask(self_mask)
        # Training-wide anneal of early_capture_coef → 0 (dense→sparse shaping).
        # Cosine from the base coef at step 0 to 0 at frac*total_steps, then stays 0.
        # env.step() reads self.early_capture_coef fresh each step, so mutating the
        # attribute here per-rollout is sufficient (no env-code change needed).
        if args.early_capture_anneal_frac > 0.0 and args.early_capture_coef > 0.0:
            ec_decay_steps = args.early_capture_anneal_frac * args.total_steps
            ec_frac = min(total_env_steps / max(ec_decay_steps, 1), 1.0)
            env.early_capture_coef = args.early_capture_coef * 0.5 * (1.0 + math.cos(math.pi * ec_frac))
        # Reinforcement curriculum: anneal the own-target logit bias init→0 over
        # reinforce_anneal_frac of training. model.forward reads reinforce_logit_bias
        # in BOTH the rollout below and this iter's PPO update → consistent PPO ratio.
        # Only the learning `model` is biased; pool/opponent models keep the 0.0 default.
        if args.allow_reinforce and args.reinforce_anneal_frac > 0.0:
            rb_decay_steps = args.reinforce_anneal_frac * args.total_steps
            rb_frac = min(total_env_steps / max(rb_decay_steps, 1), 1.0)
            model.reinforce_logit_bias = args.reinforce_bias_init * (1.0 - rb_frac)
        # Reset train_mask: all True by default, mark opp's slots False below
        storage["train_mask"].fill_(True)
        # Zero the reinforce_rate accumulators for this rollout (env counts realized
        # reinforce/fire launches per (env,player); combined with train_mask below).
        if args.allow_reinforce:
            env.reset_reinforce_stats()

        # --- Rollout collection (no grad) -----------------------------------
        model.eval()
        for t in range(rollout_T):
            actions_per_player = {}
            for p in range(P):
                feats_p, outs_p = forward_player(p)
                fire_p, angle_p, ship_p, target_p, lpf_p, lps_p, lpt_p = sample_action_batched(
                    outs_p, feats_p["fire_mask"], feats_p.get("target_mask")
                )
                ship_count_p = decode_ship_bins(ship_p, feats_p["max_ships"], cfg.model.ship_bin_mode)
                storage["planet_features"][t, :, p].copy_(feats_p["planet_features"], non_blocking=True)
                storage["fleet_features"][t, :, p].copy_(feats_p["fleet_features"], non_blocking=True)
                storage["global_features"][t, :, p].copy_(feats_p["global_features"], non_blocking=True)
                storage["planet_mask"][t, :, p].copy_(feats_p["planet_mask"], non_blocking=True)
                storage["fleet_mask"][t, :, p].copy_(feats_p["fleet_mask"], non_blocking=True)
                storage["fire_mask"][t, :, p].copy_(feats_p["fire_mask"], non_blocking=True)
                storage["slot_valid"][t, :, p].copy_(feats_p["slot_valid"], non_blocking=True)
                storage["target_mask"][t, :, p].copy_(feats_p["target_mask"], non_blocking=True)
                storage["owned_indices"][t, :, p].copy_(feats_p["owned_indices"], non_blocking=True)
                if "pairwise_features" in storage and "pairwise_features" in feats_p:
                    storage["pairwise_features"][t, :, p].copy_(feats_p["pairwise_features"], non_blocking=True)
                storage["fire_a"][t, :, p].copy_(fire_p, non_blocking=True)
                storage["ship_a"][t, :, p].copy_(ship_p, non_blocking=True)
                storage["ship_count_a"][t, :, p].copy_(ship_count_p, non_blocking=True)
                storage["target_a"][t, :, p].copy_(target_p, non_blocking=True)
                storage["lp_fire"][t, :, p].copy_(lpf_p, non_blocking=True)
                storage["lp_ship"][t, :, p].copy_(lps_p, non_blocking=True)
                storage["lp_target"][t, :, p].copy_(lpt_p, non_blocking=True)
                storage["values"][t, :, p].copy_(outs_p["value"].squeeze(-1), non_blocking=True)
                # angle_p is zeros (unused in target-decode); env needs 4-col action tensor.
                actions_per_player[p] = torch.stack([fire_p, angle_p, ship_p, target_p], dim=-1)

            # Pool opponent override: in pool envs, replace `opp_seat`'s action
            # with the pool member's action, and mark its storage slot as
            # not-trainable so PPO ignores it.
            if pool_opp is not None and N_pool > 0:
                opp_action = compute_pool_actions(pool_opp, opp_seat, pool_slice)
                actions_per_player[opp_seat][pool_slice] = opp_action
                storage["train_mask"][t, pool_slice, opp_seat] = False

            state, rewards, done = env.step(actions_per_player)
            # Snapshot mean planet ships per env — used for avgfleet/p90fleet metrics.
            # Measures actual planet inventories (passivity proxy), not action sizes.
            planet_ships = env.planets[:, :, 5]           # (N, max_planets)
            alive = env.planet_alive                       # (N, max_planets) bool
            n_alive = alive.float().sum(dim=1).clamp(min=1)
            storage["planet_ships_snap"][t].copy_(
                (planet_ships * alive.float()).sum(dim=1) / n_alive, non_blocking=True
            )
            # rewards: (N, P); done: (N,) shared across players.
            storage["rewards"][t].copy_(rewards[:, :P], non_blocking=True)

            # Per-step srcs_multi penalty: discourage firing from too many
            # sources simultaneously.  Applied symmetrically to both players.
            # penalty_t[n,p] = effective_coef * max(0, n_fires[n,p] - threshold)
            # If srcs_multi_penalty_decay_frac > 0, the coefficient cosine-decays
            # from srcs_multi_penalty to 0 over that fraction of total_steps.
            if args.srcs_multi_penalty > 0.0 or args.fleet_activity_coef > 0.0:
                if args.srcs_multi_penalty > 0.0 and args.srcs_multi_penalty_decay_frac > 0.0:
                    decay_steps = args.srcs_multi_penalty_decay_frac * args.total_steps
                    t_frac = min(total_env_steps / max(decay_steps, 1), 1.0)
                    _pen_coef = args.srcs_multi_penalty * 0.5 * (1.0 + math.cos(math.pi * t_frac))
                else:
                    _pen_coef = args.srcs_multi_penalty
                for p in range(P):
                    fires_p = storage["fire_a"][t, :, p].float()    # (N, MAX_OWNED)
                    sv_p    = storage["slot_valid"][t, :, p].float() # (N, MAX_OWNED)
                    n_fires = (fires_p * sv_p).sum(dim=-1)           # (N,)
                    if args.srcs_multi_penalty > 0.0:
                        excess = (n_fires - args.srcs_multi_threshold).clamp(min=0)
                        storage["rewards"][t, :, p] -= _pen_coef * excess
                    if args.fleet_activity_coef > 0.0:
                        # Proportional up to threshold: each source adds activity_coef
                        # until threshold, then the srcs_multi penalty takes over.
                        # Nash = fire from exactly threshold sources (not binary token-fire).
                        activity = n_fires.clamp(max=args.srcs_multi_threshold)
                        storage["rewards"][t, :, p] += args.fleet_activity_coef * activity
            storage["dones"][t, :, 0].copy_(done, non_blocking=True)
            storage["dones"][t, :, 1].copy_(done, non_blocking=True)

            # Log both seats so symmetry is visible at a glance — with P=2
            # training they should mirror (avg p0 ≈ -avg p1, both near 0).
            for r in rewards[:, 0][done].tolist():
                reward_history.append(r)
            for r in rewards[:, 1][done].tolist():
                reward_history_p1.append(r)

            # Track pool opponent win/loss for PFSP weighting — only count
            # finished pool envs (current is in `current_seat`).
            if pool_opp is not None and N_pool > 0:
                done_pool = done[pool_slice]
                if done_pool.any():
                    r_cur = rewards[pool_slice, current_seat]
                    r_opp = rewards[pool_slice, opp_seat]
                    for cur_r, opp_r, d in zip(r_cur.tolist(), r_opp.tolist(), done_pool.tolist()):
                        if not d: continue
                        if cur_r > opp_r:   pool.record_result(pool_opp, "win")
                        elif opp_r > cur_r: pool.record_result(pool_opp, "loss")
                        else:               pool.record_result(pool_opp, "draw")

        # Bootstrap value at end of rollout — for both players
        next_value_p = torch.zeros(N, P, device=storage_dev)
        with torch.no_grad():
            for p in range(P):
                feats_final = env.get_features(p, max_planets=cfg.env.max_planets, max_fleets=128)
                outs_final = model(
                    feats_final["planet_features"], feats_final["fleet_features"],
                    feats_final["global_features"], feats_final["planet_mask"],
                    feats_final["fleet_mask"],
                    fire_mask=feats_final["fire_mask"],
                    angle_mask=feats_final["angle_mask"],
                    slot_valid=feats_final["slot_valid"],
                    owned_indices=feats_final["owned_indices"],
                    owned_count=feats_final["owned_count"],
                    pairwise_features=feats_final.get("pairwise_features"),
                )
                next_value_p[:, p] = outs_final["value"].squeeze(-1).cpu()

        # --- GAE (run on CPU since storage is on CPU) -----------------------
        # Fold P into the env axis so each player-stream is an independent
        # trajectory: (T, N, P) -> (T, N*P).
        rewards_flat = storage["rewards"].reshape(rollout_T, N * P)
        values_flat  = storage["values"].reshape(rollout_T, N * P)
        dones_flat   = storage["dones"].reshape(rollout_T, N * P)
        next_v_flat  = next_value_p.reshape(N * P)
        advantages, returns = compute_gae(
            rewards_flat, values_flat, dones_flat,
            next_v_flat, gamma=cfg.ppo.gamma, lam=cfg.ppo.gae_lambda,
        )

        # --- Flatten (T, N, P, ...) → (T*N*P, ...) for PPO update -----------
        # Pool-opponent slots have train_mask=False — drop them so PPO only
        # learns from samples where current model picked the action.
        TN = rollout_T * N * P
        # Keys with shape (T, N) instead of (T, N, P, ...) — skip standard flatten
        _PER_ENV_KEYS = {"planet_ships_snap"}
        flat = {}
        for k, v in storage.items():
            if k in _PER_ENV_KEYS:
                continue  # handled separately (not per-player)
            flat[k] = v.reshape(TN, *v.shape[3:])
        flat_adv  = advantages.reshape(TN)
        flat_ret  = returns.reshape(TN)
        train_idx = torch.nonzero(flat["train_mask"], as_tuple=False).squeeze(-1)
        if train_idx.numel() < TN:
            for k, v in list(flat.items()):
                flat[k] = v[train_idx]
            flat_adv = flat_adv[train_idx]
            flat_ret = flat_ret[train_idx]
        TN = flat_adv.numel()

        fired_train_slots = flat["fire_a"].float() * flat["slot_valid"].float()
        fired_count = fired_train_slots.sum().clamp(min=1.0)
        # avgfleet/p90fleet: mean and p90 of actual planet ship inventories (passivity
        # proxy). Previously computed from action ship sizes — that was wrong: the
        # p90 pinned to a fixed SHIP_COUNTS bin value regardless of policy changes.
        planet_snaps = storage["planet_ships_snap"].reshape(-1)  # (T*N,)
        avg_fleet_size = planet_snaps.mean()
        fleet_size_p90 = torch.quantile(planet_snaps, 0.9)

        # Build PPOLearner-compatible batch (matches make_batch in self_play.py)
        batch = {
            "planet_features": flat["planet_features"],
            "fleet_features":  flat["fleet_features"],
            "global_features": flat["global_features"],
            "planet_mask":     flat["planet_mask"],
            "fleet_mask":      flat["fleet_mask"],
            "fire_mask":       flat["fire_mask"],
            "target_mask":     flat["target_mask"],
            "slot_valid":      flat["slot_valid"],
            "owned_indices":   flat["owned_indices"],
            **( {"pairwise_features": flat["pairwise_features"]} if "pairwise_features" in flat else {}),
            "owned_count":     flat["slot_valid"].sum(dim=1).tolist(),
            "actions": {
                "fire":   flat["fire_a"],
                "ship":   flat["ship_a"],
                "target": flat["target_a"],
            },
            "old_log_probs": {
                "fire":   flat["lp_fire"],
                "ships":  flat["lp_ship"],
                "target": flat["lp_target"],
            },
            "advantages": flat_adv,
            "returns":    flat_ret,
            "old_values": flat["values"],
        }

        # Minibatches: split TN into num_minibatches chunks. Build on CPU then
        # move each minibatch to GPU just-in-time inside PPOLearner.update.
        idx = torch.randperm(TN)
        mb_size = TN // cfg.ppo.num_minibatches
        minibatches = []
        for mb in range(cfg.ppo.num_minibatches):
            mi = idx[mb * mb_size : (mb + 1) * mb_size]
            sub = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    sub[k] = v[mi].to(device, non_blocking=True)
                elif isinstance(v, dict):
                    sub[k] = {kk: vv[mi].to(device, non_blocking=True) for kk, vv in v.items()}
                elif isinstance(v, list):
                    sub[k] = [v[i] for i in mi.tolist()]
                else:
                    sub[k] = v
            minibatches.append(sub)

        # IL coefficient schedule: linear decay from il_lambda → 0 over
        # il_decay_frac of total training. Held at 0 thereafter so PPO can
        # eventually exceed the teacher.
        if learner.frozen_il_model is not None and cfg.ppo.il_lambda > 0:
            progress = total_env_steps / max(args.total_steps, 1)
            decay = max(0.0, 1.0 - progress / max(cfg.ppo.il_decay_frac, 1e-6))
            learner.set_il_coef(cfg.ppo.il_lambda * decay)

        # Sample a BC minibatch for auxiliary supervision (if enabled)
        bc_batch = None
        if bc_samples_for_aux is not None:
            from bc import _collate as _bc_collate
            bc_idx = np.random.choice(len(bc_samples_for_aux),
                                       size=min(cfg.bc.batch_size, len(bc_samples_for_aux)),
                                       replace=False)
            bc_subset = [bc_samples_for_aux[i] for i in bc_idx]
            bc_batch = _bc_collate(bc_subset, device)

        # PPO update
        model.train()
        metrics = learner.update(minibatches, scheduler=scheduler,
                                 kl_target=cfg.ppo.kl_target,
                                 bc_batch=bc_batch)
        metrics["avg_fleet_size"] = float(avg_fleet_size.item())
        metrics["p90_fleet_size"] = float(fleet_size_p90.item())
        # reinforce_rate: of the current policy's realized launches (train_mask-filtered,
        # across both seats), the fraction sent to our own planets (Vadasz ~0.57; target
        # 0.4-0.6). train_mask[0] is (N,P) and constant over the rollout's t.
        if args.allow_reinforce and env._fire_launch_count is not None:
            tm = storage["train_mask"][0].to(env.device).float()   # (N, P)
            fires = (env._fire_launch_count * tm).sum()
            reinf = (env._reinforce_launch_count * tm).sum()
            metrics["reinforce_rate"] = float((reinf / fires.clamp(min=1.0)).item())
        with torch.no_grad():
            metrics["old_value_mean"] = float(flat["values"].mean().item())
            metrics["old_value_std"] = float(flat["values"].std(unbiased=False).item())
            metrics["return_mean"] = float(flat_ret.mean().item())
            metrics["return_std"] = float(flat_ret.std(unbiased=False).item())
            metrics["adv_std"] = float(flat_adv.std(unbiased=False).item())
            # Explained variance: how much of the return variance the value head
            # captures. EV = 1 - Var(returns - values)/Var(returns); since
            # returns = advantages + values, (returns - values) == advantages.
            # The master PPO-health signal (should climb >0.8 within ~100 iters;
            # if it never passes ~0.5, suspect obs representation / architecture).
            _ret_var = float(flat_ret.var(unbiased=False).item())
            metrics["explained_variance"] = (
                1.0 - float(flat_adv.var(unbiased=False).item()) / _ret_var
                if _ret_var > 1e-8 else 0.0
            )
            metrics["reward_mean"] = float(flat["rewards"].mean().item())
            metrics["reward_std"] = float(flat["rewards"].std(unbiased=False).item())
            metrics["reward_nonzero"] = float((flat["rewards"].abs() > 1e-8).float().mean().item())
            metrics["planet_feat_std"] = float(flat["planet_features"].std(unbiased=False).item())
            metrics["fleet_feat_std"] = float(flat["fleet_features"].std(unbiased=False).item())
            metrics["global_feat_std"] = float(flat["global_features"].std(unbiased=False).item())
            if "pairwise_features" in flat:
                metrics["pairwise_feat_std"] = float(flat["pairwise_features"].std(unbiased=False).item())
            # Per-feature input-weight norms on the pairwise cross-attention. The
            # last 3 pairwise cols are the newer features (roi_20, roi_50,
            # enemy_contest); track whether they actually get used (norm climbing
            # toward the ~1.1 mean of the original 12) vs stay inert (~0.05).
            if hasattr(model, "pair_kv"):
                fw = model.pair_kv.weight          # [2D, D + F_pair]
                D = fw.shape[0] // 2
                feat_cols = fw[:, D:]              # [2D, F_pair]
                if feat_cols.shape[1] >= 3:
                    new_norms = feat_cols[:, -3:].norm(dim=0)
                    metrics["wnorm_roi20"] = float(new_norms[0].item())
                    metrics["wnorm_roi50"] = float(new_norms[1].item())
                    metrics["wnorm_enemy_contest"] = float(new_norms[2].item())
                    metrics["wnorm_pw_orig"] = float(feat_cols[:, :-3].norm(dim=0).mean().item())

        total_env_steps += rollout_T * N
        iter_count += 1
        clipfrac_history.append(metrics.get("clip_frac", 0.0))

        # --- Logging --------------------------------------------------------
        now = time.perf_counter()
        elapsed = now - start
        sps = total_env_steps / elapsed if elapsed > 0 else 0
        avg_r = float(np.mean(reward_history)) if reward_history else 0.0
        avg_r1 = float(np.mean(reward_history_p1)) if reward_history_p1 else 0.0
        avg_cf = float(np.mean(clipfrac_history)) if clipfrac_history else 0.0
        if now - last_log >= 5.0 or iter_count == 1:
            last_log = now
            # Per-slot diagnostics — surface slot-0-only firing + ship-bin-0 collapse
            # that the aggregate H_fire metric was masking (see docs/bugs.md).
            psf = metrics.get("per_slot_fire_probs") or [0.0]
            slot0 = psf[0] if len(psf) > 0 else 0.0
            slot_rest_max = max(psf[1:]) if len(psf) > 1 else 0.0
            # Primary line: the PPO-health decision set. EV / KL / clip_frac are
            # the only three signals that tell you whether training will work;
            # H_fire because entropy is an active lever. Everything else is
            # diagnostic and lives on the periodic 'diag' line + W&B.
            print(
                f"iter {iter_count:5d} | steps {total_env_steps:>11,} | SPS {sps:>7,.0f} | "
                f"EV {metrics.get('explained_variance', 0):.3f} | KL {metrics.get('approx_kl', 0):.4f} | "
                f"clip {avg_cf:.3f}(fire {metrics.get('clip_frac_fire', 0):.3f}) | "
                f"H_fire {metrics.get('fire_entropy', 0):.3f} | "
                f"V_loss {metrics.get('value_loss', 0):.4f} | "
                f"r_p0 {avg_r:+.3f} r_p1 {avg_r1:+.3f} | "
                f"LR {metrics['learning_rate']:.6f} | estop {metrics.get('kl_early_stop', 0):.0f}"
                + (f" | il_kl {metrics.get('il_kl', 0):.3f} il_coef {metrics.get('il_coef', 0):.3f}"
                   if metrics.get('il_coef', 0) > 0 else "")
            )
            # Secondary behavioural diagnostics — occasionally useful, not decision
            # drivers (W&B keeps them every iter). Console-print every 10th log.
            if iter_count == 1 or iter_count % 10 == 0:
                pencoef = ""
                if args.srcs_multi_penalty > 0.0 and args.srcs_multi_penalty_decay_frac > 0.0:
                    _pc = args.srcs_multi_penalty * 0.5 * (1.0 + math.cos(math.pi * min(
                        total_env_steps / max(args.srcs_multi_penalty_decay_frac * args.total_steps, 1), 1.0)))
                    pencoef = f" pencoef {_pc:.5f}"
                actcoef = f" actcoef {args.fleet_activity_coef:.4f}" if args.fleet_activity_coef > 0.0 else ""
                reinfstr = f"reinf {metrics.get('reinforce_rate', 0):.2f} | " if args.allow_reinforce else ""
                print(
                    f"   diag | fire[0] {slot0:.2f} rest_max {slot_rest_max:.2f} | "
                    f"fire_frac {metrics.get('fire_fraction', 0):.2f} "
                    f"owned {metrics.get('owned_planets', 0):.1f} "
                    f"ship0 {metrics.get('ship_bin0_rate', 0):.2f} "
                    f"meanshipbin {metrics.get('mean_ship_bin', 0):.1f} | "
                    f"avgfleet {metrics.get('avg_fleet_size', 0):.1f} "
                    f"p90 {metrics.get('p90_fleet_size', 0):.1f} | "
                    f"{reinfstr}"
                    f"H_ship {metrics.get('ship_entropy', 0):.2f} | "
                    f"Vμ {metrics.get('old_value_mean', 0):+.2f} Rμ {metrics.get('return_mean', 0):+.2f} "
                    f"Rσ {metrics.get('return_std', 0):.2f} Aσ {metrics.get('adv_std', 0):.2f} | "
                    f"rewμ {metrics.get('reward_mean', 0):+.4f} rewNZ {metrics.get('reward_nonzero', 0):.3f} | "
                    f"featσ p/f/g/pw {metrics.get('planet_feat_std', 0):.2f}/"
                    f"{metrics.get('fleet_feat_std', 0):.2f}/"
                    f"{metrics.get('global_feat_std', 0):.2f}/"
                    f"{metrics.get('pairwise_feat_std', 0):.2f} | "
                    f"wnorm roi20/roi50/ec {metrics.get('wnorm_roi20', 0):.3f}/"
                    f"{metrics.get('wnorm_roi50', 0):.3f}/"
                    f"{metrics.get('wnorm_enemy_contest', 0):.3f} "
                    f"(orig~{metrics.get('wnorm_pw_orig', 0):.2f})"
                    + actcoef + pencoef
                )
            # W&B logging
            if wb is not None:
                wb.log({
                    # Core training
                    "train/steps": total_env_steps,
                    "train/sps": sps,
                    "train/reward_p0": avg_r,
                    "train/reward_p1": avg_r1,
                    "train/lr": metrics["learning_rate"],
                    # PPO health
                    "ppo/explained_variance": metrics.get("explained_variance", 0),
                    "feat/wnorm_roi20": metrics.get("wnorm_roi20", 0),
                    "feat/wnorm_roi50": metrics.get("wnorm_roi50", 0),
                    "feat/wnorm_enemy_contest": metrics.get("wnorm_enemy_contest", 0),
                    "feat/wnorm_pw_orig": metrics.get("wnorm_pw_orig", 0),
                    "ppo/clip_frac": avg_cf,
                    "ppo/clip_frac_fire": metrics.get("clip_frac_fire", 0),
                    "ppo/approx_kl": metrics.get("approx_kl", 0),
                    "ppo/value_loss": metrics.get("value_loss", 0),
                    "ppo/early_stop": metrics.get("kl_early_stop", 0),
                    # Policy behaviour — the key kill-signal metrics
                    "policy/fire_0": slot0,
                    "policy/fire_rest_max": slot_rest_max,
                    "policy/fire_fraction": metrics.get("fire_fraction", 0),
                    "policy/owned_planets": metrics.get("owned_planets", 0),
                    "policy/ship_bin0_rate": metrics.get("ship_bin0_rate", 0),
                    "policy/mean_ship_bin": metrics.get("mean_ship_bin", 0),
                    "policy/avg_fleet": metrics.get("avg_fleet_size", 0),
                    "policy/p90_fleet": metrics.get("p90_fleet_size", 0),
                    "policy/reinforce_rate": metrics.get("reinforce_rate", 0),
                    # Entropy
                    "entropy/fire": metrics.get("fire_entropy", 0),
                    "entropy/ship": metrics.get("ship_entropy", 0),
                    # Value / return stats
                    "value/mean": metrics.get("old_value_mean", 0),
                    "value/std": metrics.get("old_value_std", 0),
                    "value/return_mean": metrics.get("return_mean", 0),
                    "value/adv_std": metrics.get("adv_std", 0),
                    # Reward stats
                    "reward/mean": metrics.get("reward_mean", 0),
                    "reward/nonzero_frac": metrics.get("reward_nonzero", 0),
                    # IL (zero when not active)
                    "il/kl": metrics.get("il_kl", 0),
                    "il/coef": metrics.get("il_coef", 0),
                }, step=total_env_steps)

            # Collapse warnings — flag early instead of finding via replay at 10M
            if iter_count > 20:  # skip BC-resume noise
                if slot0 > 0.8 and slot_rest_max < 0.1:
                    print("  ⚠ slot-0-only firing: slot 0 fire_prob>0.8 while all "
                          "other slots <0.1 — fire-head is collapsing to one source")
                if metrics.get("ship_bin0_rate", 0) > 0.5:
                    print(f"  ⚠ ship-bin-0 collapse: {metrics['ship_bin0_rate']:.0%} of fires "
                          f"argmax to bin 0 (1 ship). 1-ship fleets can't capture neutrals.")

        # Checkpoint best by reward
        if len(reward_history) >= 100 and avg_r > best_avg_reward + 0.02:
            best_avg_reward = avg_r
            atomic_torch_save(learner.state_dict(), f"checkpoints/torch_best_{run_ts}.pt")
            print(f"  ★ best updated: reward={avg_r:+.3f}")

        # Periodic checkpoint by fixed step interval
        if total_env_steps - last_ckpt_step >= args.checkpoint_interval:
            last_ckpt_step = total_env_steps
            path = f"checkpoints/torch_step_{total_env_steps}_{run_ts}.pt"
            atomic_torch_save(learner.state_dict(), path)
            print(f"  saved {path}")
            # Checkpoint-aligned metric snapshot (latest PPO-update metrics), so the
            # tracker can read metrics that line up exactly with each checkpoint
            # rather than the sparse every-20-iter diag line.
            print("  CKPT_METRICS step={} EV={:.3f} KL={:.4f} clip={:.3f} fire_frac={:.3f} "
                  "owned={:.1f} avgfleet={:.1f} fire_rate={:.3f} Hfire={:.3f} reinf={:.3f}".format(
                      total_env_steps,
                      metrics.get("explained_variance", 0), metrics.get("approx_kl", 0),
                      metrics.get("clip_frac", 0), metrics.get("fire_fraction", 0),
                      metrics.get("owned_planets", 0),
                      metrics.get("avg_fleet_size", 0), metrics.get("fire_rate_overall", 0),
                      metrics.get("fire_entropy", 0), metrics.get("reinforce_rate", 0)))
            # Persist pool alongside the disk checkpoint so spot interrupts
            # don't lose pool diversity. Naming mirrors the checkpoint stem.
            if pool is not None:
                pool_path = f"checkpoints/pool_step_{total_env_steps}_{run_ts}.pt"
                pool.save(pool_path)
                print(f"  saved {pool_path} ({len(pool)} members)")

        # Pool snapshot (much more frequent than full checkpoint): adds the
        # current weights to the opponent pool for PFSP sampling. Also evict
        # any external opponent we've mastered, and print a pool summary.
        if pool is not None and total_env_steps - last_pool_snapshot_step >= args.pool_checkpoint_interval:
            last_pool_snapshot_step = total_env_steps
            pool.add_self_checkpoint(total_env_steps, model.state_dict())
            evicted = pool.maybe_evict_mastered()
            if evicted:
                print(f"  pool: mastered & evicted external opponents: {evicted}")
            print(f"  pool snapshot @ step {total_env_steps:,}")
            print(pool.summary())

    elapsed = time.perf_counter() - start
    sps = total_env_steps / elapsed if elapsed > 0 else 0
    print(f"\nTraining complete: {total_env_steps:,} env steps in {elapsed:.0f}s")
    print(f"Final SPS: {sps:,.0f}")

    # Final checkpoint
    atomic_torch_save(learner.state_dict(), f"checkpoints/torch_step_{total_env_steps}_{run_ts}_final.pt")

    # Shut down external heuristic worker pools
    if 'heur_worker_pools' in dir():
        for wp in heur_worker_pools.values():
            try:
                wp.close()
            except Exception:
                pass

    if wb is not None:
        wb.finish()

    # Auto-terminate the host (cost control). Set
    # InstanceInitiatedShutdownBehavior=terminate on the EC2 instance and this
    # command will tear down the VM, ending billing.
    if args.terminate_on_done:
        print("\n--terminate-on-done: powering off the instance...")
        os.system("sudo shutdown -h +5 'training complete, auto-terminating'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="Split PPO update batch into this many minibatches "
                             "(increase if hitting CUDA OOM in attention)")
    parser.add_argument("--total-steps", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--reinit-critic", action="store_true",
                        help="CONTROL: re-initialise the value head after resume "
                             "(warm policy + cold critic) to isolate the VDN "
                             "cold-critic-shock confound.")
    parser.add_argument("--checkpoint-interval", type=int, default=5_000_000,
                        help="Save a periodic checkpoint every N env steps")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override PPO learning rate (default: cfg.ppo.learning_rate=3e-4)")
    parser.add_argument("--ppo-epochs", type=int, default=None,
                        help="Override PPO epochs per rollout (default: 4)")
    parser.add_argument("--clip-eps", type=float, default=None,
                        help="Override PPO clip epsilon (default: cfg.ppo.clip_eps=0.2)")
    parser.add_argument("--entropy-coef-fire", type=float, default=None,
                        help="Override fire-head entropy coefficient "
                             "(default: cfg.ppo.entropy_coef_fire=0.01)")
    parser.add_argument("--entropy-coef-target", type=float, default=None,
                        help="Override target-head entropy coefficient "
                             "(default: cfg.ppo.entropy_coef_target=0.02)")
    parser.add_argument("--entropy-coef-angle", type=float, default=None,
                        help="DEPRECATED alias for --entropy-coef-target "
                             "(the angle head is vestigial; this coef weights the target head)")
    parser.add_argument("--entropy-coef-ships", type=float, default=None,
                        help="Override ships-head entropy coefficient "
                             "(default: cfg.ppo.entropy_coef_ships=0.01)")
    parser.add_argument("--max-grad-norm", type=float, default=None,
                        help="Override gradient-clipping max norm "
                             "(default: cfg.ppo.max_grad_norm=0.5)")
    parser.add_argument("--gae-lambda", type=float, default=None,
                        help="Override GAE lambda (default: cfg.ppo.gae_lambda=0.95)")
    # IL regularization (KL-to-frozen-BC penalty) ------------------------
    parser.add_argument("--il-lambda", type=float, default=None,
                        help="Peak coef for KL(current||frozen_BC) penalty. "
                             "0 = disabled. Typical: 0.1–1.0. Decays linearly "
                             "to 0 over --il-decay-frac of training.")
    parser.add_argument("--il-decay-frac", type=float, default=None,
                        help="Fraction of total training over which il_lambda "
                             "decays to 0 (default cfg: 0.8).")
    parser.add_argument("--il-ref", type=str, default="",
                        help="Path to a separate frozen reference .pt for IL. "
                             "Default behaviour: when il_lambda>0 and --resume is "
                             "set, the resume checkpoint is used as the reference.")
    # BC auxiliary supervision (cross-entropy on teacher actions during PPO).
    # Unlike il_lambda (KL on distributions), this directly penalizes argmax
    # drift via supervised loss on teacher's labels.
    parser.add_argument("--bc-coef", type=float, default=None,
                        help="Coefficient on auxiliary BC loss during PPO. "
                             "Requires --bc-samples. Typical: 0.5–2.0.")
    parser.add_argument("--bc-samples", type=str, default="",
                        help="Path to .pkl of teacher samples (produced by "
                             "extract_teacher_samples.py or bc_frac.py cache).")
    parser.add_argument("--min-ship-bin", type=int, default=None,
                        help="Mask ship bins < this index to -inf (never sampled). "
                             "For fraction-head 10-bin model, set 1 to remove the "
                             "10%%-of-source bin that PPO collapses to in cold-start.")
    parser.add_argument("--action-decode", choices=["angle", "target"], default="angle",
                        help="Direction component executed during PPO rollouts. "
                             "angle keeps the legacy free angle-bin action; target "
                             "samples target_logits and converts the target planet "
                             "to an intercept angle in VecTorchEnv.")
    parser.add_argument("--allow-reinforce", action="store_true",
                        help="Make OWN planets legal targets (reinforcement) — ships "
                             "arriving at a friendly planet add to its garrison. Top "
                             "players reinforce ~57%% of launches; default agents 0%%. "
                             "Saved in the checkpoint so eval/export mask the same way. "
                             "Off by default (attack-only, backward-compatible).")
    parser.add_argument("--reinforce-bias-init", type=float, default=-8.0,
                        help="Reinforcement CURRICULUM: initial additive bias on OWN-target "
                             "logits (negative suppresses reinforcement early, ≈ a soft mask). "
                             "Annealed linearly → 0 over --reinforce-anneal-frac. Only active "
                             "with --allow-reinforce AND --reinforce-anneal-frac > 0.")
    parser.add_argument("--reinforce-anneal-frac", type=float, default=0.0,
                        help="Fraction of --total-steps over which the own-target bias anneals "
                             "from --reinforce-bias-init → 0 (then stays 0). 0 = no curriculum "
                             "(hard unmask at t=0 — caused the rev55 over-fire collapse). "
                             "Suggested: 0.3.")
    parser.add_argument("--reinforce-garrison-floor", type=float, default=0.0,
                        help="Reinforcement discipline #1: a reinforce launch may not drain its "
                             "source planet below this many ships (training-time mask/veto, NOT a "
                             "penalty → no Nash risk). Kills the 'drain a planet, then lose it' "
                             "regression. Inference is unconstrained (real env has no floor). "
                             "0 = off. Only active with --allow-reinforce.")
    parser.add_argument("--reinforce-cost", type=float, default=0.0,
                        help="Reinforcement discipline #2: per-ship transit cost — subtract this × "
                             "ships_reinforced from the launching player's reward each step. The "
                             "actual flood fix (rev56: costless reinforcement floods ~30×). Scales "
                             "with waste; the calibration knob. Watch reinforce_rate → target "
                             "~0.4-0.6. 0 = off. Only active with --allow-reinforce.")
    parser.add_argument("--win-margin-coeff", type=float, default=0.0,
                        help="Terminal bonus coefficient α: winner gets +1 + α*(my_score/total_score). "
                             "0 = pure ±1 reward (default). Suggested start: 0.5.")
    parser.add_argument("--shaping-coef", type=float, default=0.0,
                        help="Per-step material-delta shaping coefficient. "
                             "0 = off. Suggested diagnostic start: 0.03. "
                             "NOTE: rewards passive ship accumulation — failed in rev8.")
    parser.add_argument("--expansion-coef", type=float, default=0.0,
                        help="Potential-based shaping on owned-production lead "
                             "(planet/economy race). Unlike --shaping-coef, passive "
                             "play nets ~0 (production only changes on capture). "
                             "0 = off. rev14 expansion fix; suggested start: 0.01.")
    parser.add_argument("--defense-coef", type=float, default=0.0,
                        help="Per-step penalty for losing owned production (a planet "
                             "captured from us). Consolidation/defense incentive — rewards "
                             "HOLDING planets, complements --expansion-coef's GRAB. "
                             "0 = off. rev15 defense lever; suggested start: 0.02.")
    parser.add_argument("--early-capture-coef", type=float, default=0.0,
                        help="Spike reward for CAPTURING a planet (delta in owned count), decayed "
                             "linearly to 0 over --early-capture-steps (default 400). Active through "
                             "mid-game so GAE 18-step horizon can see captures throughout, not just "
                             "the opening. Coeff math: one capture event ≈ coeff × decay_at_t; "
                             "keep ≤10%% of terminal win → range 0.05-0.10. 0 = off.")
    parser.add_argument("--early-capture-steps", type=int, default=400,
                        help="Step at which the delta-capture decay reaches zero. Default 400.")
    parser.add_argument("--early-capture-anneal-frac", type=float, default=0.0,
                        help="Training-wide (not within-episode) anneal of --early-capture-coef to 0. "
                             "Cosine decay from full coef at step 0 to 0 at frac*total_steps, then "
                             "stays 0 (mirrors --srcs-multi-penalty-decay-frac). Cosine holds near full "
                             "early (bootstrap) and fades fastest mid-run. Removes the capture-shaping "
                             "crutch once the pool can sustain aggression. 0 = off (constant coef).")
    parser.add_argument("--first-strike-steps", type=int, default=0,
                        help="Apply first_strike_mult to capture reward for t < N steps. "
                             "Breaks opening paralysis by making early captures more lucrative. "
                             "Suggested: 50. 0 = off.")
    parser.add_argument("--first-strike-mult", type=float, default=2.0,
                        help="Multiplier applied to capture reward for t < --first-strike-steps. "
                             "Default 2.0 (doubles the capture reward in the opening).")
    parser.add_argument("--speed-coef", type=float, default=0.0,
                        help="Terminal time-to-victory bonus coefficient. Winners get "
                             "+((episode_steps - termination_step) / episode_steps) * coef, "
                             "so early eliminations are worth more than timeout/grind wins. "
                             "0 = off. Suggested start: 0.3-0.5.")
    parser.add_argument("--handicap-frac", type=float, default=0.0,
                        help="Fraction of games where player 0 starts with --handicap-ships "
                             "instead of the normal 10. Forces exposure to losing positions "
                             "during self-play so the agent learns to fight back from behind. "
                             "0 = off (symmetric starts). Suggested: 0.3.")
    parser.add_argument("--handicap-ships", type=int, default=5,
                        help="Starting ship count for player 0 in handicap games "
                             "(default 5 = half normal). Only used when --handicap-frac > 0.")
    parser.add_argument("--ssdr-frac", type=float, default=0.0,
                        help="Start-State Domain Randomisation: fraction of env resets that "
                             "fast-forward the game by 1..--ssdr-max-steps random steps before "
                             "handing control to the learner. Both players act randomly during "
                             "warmup, creating asymmetric mid-game starts that shatter the "
                             "symmetric-start passive Nash. 0 = off. Suggested: 0.3.")
    parser.add_argument("--ssdr-max-steps", type=int, default=20,
                        help="Max warmup steps for SSDR. Actual steps ~ U(1, ssdr_max_steps). "
                             "20 = up to ~4%% of a 500-step game pre-played. (default: 20)")
    parser.add_argument("--srcs-multi-penalty", type=float, default=0.0,
                        help="Per-step reward penalty per source over --srcs-multi-threshold. "
                             "Applied symmetrically to both players each rollout step. "
                             "Discourages carpet-bombing (firing from many sources at once). "
                             "Typical: 0.001–0.005. 0 = off.")
    parser.add_argument("--srcs-multi-threshold", type=float, default=2.0,
                        help="Number of simultaneous fire sources at or below which no "
                             "penalty is applied (default: 2.0). Fires > threshold incur "
                             "--srcs-multi-penalty per excess source per step.")
    parser.add_argument("--srcs-multi-penalty-decay-frac", type=float, default=0.0,
                        help="If > 0, the srcs_multi penalty cosine-decays from "
                             "--srcs-multi-penalty to 0 over this fraction of --total-steps. "
                             "E.g. 0.5 = penalty is full strength at step 0, decays to 0 "
                             "by step total_steps*0.5, stays 0 after. 0 = constant penalty.")
    parser.add_argument("--aux-roi-coef", type=float, default=0.0,
                        help="Coefficient for ROI auxiliary regression loss. Keeps "
                             "pair_kv.weight[:, 108:111] and target_scorer new columns "
                             "anchored to encoding roi_20/roi_50/enemy_contest throughout "
                             "PPO. Reg heads are transient (not saved). Typical: 0.01–0.05.")
    parser.add_argument("--fleet-activity-coef", type=float, default=0.0,
                        help="Per-step reward added when any planet fires (n_fires > 0). "
                             "Breaks the fire=0 Nash created by srcs_multi penalty alone — "
                             "fire=0 forgoes this reward, making passivity costly. "
                             "Pair with --srcs-multi-penalty: activity reward makes firing "
                             "attractive, penalty caps how many sources are used. "
                             "Typical: 0.001–0.003. 0 = off.")
    # Opponent pool (PFSP self-play with optional external heuristics) ----
    parser.add_argument("--pool-mode", choices=["none", "self", "mixed"], default="none",
                        help="none: pure current-vs-current self-play (default). "
                             "self: pool of past-self checkpoints only. "
                             "mixed: self-checkpoints + external heuristics from --external-opponents.")
    parser.add_argument("--pool-fraction", type=float, default=0.5,
                        help="Fraction of envs that play current-vs-pool-opponent. "
                             "The rest do P=2 symmetric self-play.")
    parser.add_argument("--pool-checkpoint-interval", type=int, default=1_000_000,
                        help="Snapshot current weights into the pool every N env steps. "
                             "Should be much smaller than --checkpoint-interval.")
    parser.add_argument("--pool-max-size", type=int, default=20,
                        help="Max past-self checkpoints in the pool (FIFO eviction).")
    parser.add_argument("--heuristic-workers", type=int, default=0,
                        help="Number of worker processes per external heuristic. "
                             "0 = auto (cpu_count - 1). Used by pool-mode=mixed.")
    parser.add_argument("--pool-pfsp-alpha", type=float, default=2.0,
                        help="PFSP sampling exponent: weight = (1 - win_rate)^alpha.")
    parser.add_argument("--pool-mastered-threshold", type=float, default=0.9,
                        help="Win-rate above this triggers eviction of an external opponent.")
    parser.add_argument("--pool-mastered-min-games", type=int, default=50,
                        help="Minimum games against an external before mastery-eviction is considered.")
    parser.add_argument("--pool-pfsp-min-games", type=int, default=30,
                        help="Minimum games before trusting win-rate for PFSP weight. "
                             "Below this threshold wr=0.5 is assumed, preventing early "
                             "lucky streaks from sand-bagging an opponent (e.g. Hellburner "
                             "getting 0.003 PFSP weight after only 17 rollout games).")
    parser.add_argument("--pool-external-fraction", type=float, default=0.0,
                        help="Fixed fraction of pool samples reserved for external heuristic "
                             "opponents, bypassing PFSP. e.g. 0.4 = 40%% of pool games always "
                             "go to external opponents (split uniformly among them); the "
                             "remaining 60%% is governed by PFSP over self-checkpoints. "
                             "Use for targeted runs where you need sustained pressure from "
                             "a specific opponent regardless of rollout win-rate.")
    parser.add_argument("--preseed-pool", type=str, default="",
                        help="Directory of .pt checkpoints to preseed the pool "
                             "with as 'self' members. Step is parsed from filename "
                             "(e.g. torch_step_5013504.pt -> step=5013504). Useful "
                             "for diluting the heuristic-share early in training "
                             "and for resuming pool diversity across runs.")
    parser.add_argument("--external-opponents", type=str, default="",
                        help="Comma-separated paths to .py heuristic agents (e.g. "
                             "'opponents/candidate_suneet_lb1200.py,opponents/candidate_zach_public.py'). "
                             "Only used when --pool-mode=mixed.")
    parser.add_argument("--planner-externals", type=str, default="",
                        help="Comma list of external-opponent member names (file stems, e.g. "
                             "'candidate_flowdiff') that run via the in-process GPU "
                             "BatchedPlannerOpponent instead of CPU worker pools. For heavy "
                             "orbit_lite planners whose per-step cost saturates the CPU.")
    parser.add_argument("--lr-schedule-steps", type=int, default=0,
                        help="Decouple the LR cosine decay horizon from --total-steps. "
                             "Set larger than --total-steps for slow/partial decay "
                             "(e.g. 2x total-steps → LR only decays halfway). "
                             "Default 0 = use --total-steps (full decay to zero).")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip the LR warmup phase. Auto-enabled when "
                             "--resume is set (use --with-warmup to override).")
    parser.add_argument("--with-warmup", action="store_true",
                        help="Force LR warmup even on --resume.")
    parser.add_argument("--terminate-on-done", action="store_true",
                        help="Run 'sudo shutdown -h +1' after training. Combined "
                             "with EC2 InstanceInitiatedShutdownBehavior=terminate "
                             "this stops the instance to end billing.")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="orbit-wars",
                        help="W&B project name (default: orbit-wars).")
    parser.add_argument("--run-name", type=str, default="",
                        help="Short revision label embedded in checkpoint filenames. "
                             "e.g. 'rev12' → torch_step_1015808_rev12_20260601_120000.pt. "
                             "Makes it impossible to confuse checkpoints from different runs.")
    parser.add_argument("--wandb-run-name", type=str, default="",
                        help="W&B run name. Defaults to run timestamp if not set.")
    args = parser.parse_args()

    if not args.device:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    train(args)
