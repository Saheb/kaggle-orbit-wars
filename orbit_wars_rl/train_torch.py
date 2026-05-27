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
import os
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

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
    import importlib.util
    spec = importlib.util.spec_from_file_location("worker_agent", agent_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    global _WORKER_AGENT_FN
    _WORKER_AGENT_FN = mod.agent


def _heur_worker_call(obs):
    """Run the worker's agent on one obs. Swallows exceptions to a no-op."""
    try:
        return _WORKER_AGENT_FN(obs) or []
    except Exception:
        return []


class HeuristicWorkerPool:
    """Persistent process pool around one external heuristic agent."""
    def __init__(self, agent_path: str, num_workers: int):
        # Force 'fork' so child inherits already-imported torch_env etc. and
        # the initializer can re-load the agent module cheaply.
        ctx = _mp.get_context("fork")
        self.pool = ctx.Pool(
            processes=num_workers,
            initializer=_heur_worker_init,
            initargs=(agent_path,),
        )
        self.num_workers = num_workers
        self.agent_path = agent_path

    def map(self, obs_list):
        return self.pool.map(_heur_worker_call, obs_list)

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
    if args.il_lambda is not None:
        cfg.ppo.il_lambda = args.il_lambda
    if args.il_decay_frac is not None:
        cfg.ppo.il_decay_frac = args.il_decay_frac
    if args.bc_coef is not None:
        cfg.ppo.bc_coef = args.bc_coef
    print(f"PPO config: lr={cfg.ppo.learning_rate}, ppo_epochs={cfg.ppo.ppo_epochs}, "
          f"num_minibatches={cfg.ppo.num_minibatches}, kl_target={cfg.ppo.kl_target}")
    print(f"Action decode: {args.action_decode}")
    print(f"Win margin coeff: {args.win_margin_coeff}")

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

    env = VecTorchEnv(num_envs=args.num_envs, num_players=2,
                      device=device, episode_steps=500,
                      ship_bin_mode=cfg.model.ship_bin_mode,
                      action_decode=args.action_decode,
                      win_margin_coeff=args.win_margin_coeff)
    env.reset(seeds=[args.seed + i for i in range(args.num_envs)])

    model = EntityTransformer(cfg.model).to(device)
    print(f"Model params: {count_params(model):,}")
    if args.resume:
        sd = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model" in sd: sd = sd["model"]
        model.load_state_dict(sd)
        print(f"Resumed from {args.resume}")

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

    learner = PPOLearner(model, cfg, device=device, frozen_il_model=frozen_il_model)

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
        # Create one worker pool per external heuristic. Keyed by member name
        # so compute_pool_actions can dispatch by opp.name.
        heur_worker_pools: dict[str, HeuristicWorkerPool] = {}
        nw = args.heuristic_workers if args.heuristic_workers > 0 else max(1, (os.cpu_count() or 2) - 1)
        for m in pool.members:
            if m.kind == "external_heuristic":
                src = getattr(m, "_source_path", None) or args.external_opponents.split(",")[0].strip()
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
    print(f"Run timestamp: {run_ts}")
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
        "angle_mask":      torch.zeros(rollout_T, N, P, MAX_OWNED, NUM_ANGLE_BINS, dtype=torch.bool, device=storage_dev),
        "slot_valid":      torch.zeros(rollout_T, N, P, MAX_OWNED, dtype=torch.bool, device=storage_dev),
        "target_mask":     torch.zeros(rollout_T, N, P, MAX_OWNED, cfg.env.max_planets, dtype=torch.bool, device=storage_dev),
        "owned_indices":   torch.zeros(rollout_T, N, P, MAX_OWNED, dtype=torch.long, device=storage_dev),
        "pairwise_features": torch.zeros(rollout_T, N, P, MAX_OWNED, cfg.env.max_planets, cfg.model.pairwise_feature_dim, device=storage_dev),
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
    }

    def compute_pool_actions(opp: PoolMember, player: int, env_slice: slice) -> torch.Tensor:
        """Return action tensor (N_pool, MAX_OWNED, 3) for the opponent playing
        `player` in envs `env_slice`. Supports 'self' (frozen RL model on GPU)
        and 'external_heuristic' (.py agent via legacy Python obs)."""
        if opp.kind == "self":
            # Load opp weights into the reusable frozen model
            pool_opp_model.load_state_dict(opp.state_dict)
            feats = env.get_features(player, max_planets=cfg.env.max_planets, max_fleets=cfg.env.max_fleets)
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
        feats = env.get_features(player, max_planets=cfg.env.max_planets, max_fleets=cfg.env.max_fleets)
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
        # Reset train_mask: all True by default, mark opp's slots False below
        storage["train_mask"].fill_(True)

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
                storage["angle_mask"][t, :, p].copy_(feats_p["angle_mask"], non_blocking=True)
                storage["slot_valid"][t, :, p].copy_(feats_p["slot_valid"], non_blocking=True)
                storage["target_mask"][t, :, p].copy_(feats_p["target_mask"], non_blocking=True)
                storage["owned_indices"][t, :, p].copy_(feats_p["owned_indices"], non_blocking=True)
                if "pairwise_features" in feats_p:
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

            _, rewards, done = env.step(actions_per_player)
            # rewards: (N, P); done: (N,) shared across players.
            storage["rewards"][t].copy_(rewards[:, :P], non_blocking=True)
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
                feats_final = env.get_features(p)
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
        flat = {}
        for k, v in storage.items():
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
        avg_fleet_size = (flat["ship_count_a"] * fired_train_slots).sum() / fired_count
        fleet_size_p90 = torch.quantile(
            flat["ship_count_a"][fired_train_slots.bool()],
            0.9,
        ) if fired_train_slots.bool().any() else torch.tensor(0.0)

        # Build PPOLearner-compatible batch (matches make_batch in self_play.py)
        batch = {
            "planet_features": flat["planet_features"],
            "fleet_features":  flat["fleet_features"],
            "global_features": flat["global_features"],
            "planet_mask":     flat["planet_mask"],
            "fleet_mask":      flat["fleet_mask"],
            "fire_mask":       flat["fire_mask"],
            "angle_mask":      flat["angle_mask"],
            "target_mask":     flat["target_mask"],
            "slot_valid":      flat["slot_valid"],
            "owned_indices":   flat["owned_indices"],
            "pairwise_features": flat["pairwise_features"],
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
            print(
                f"iter {iter_count:5d} | steps {total_env_steps:>11,} | "
                f"SPS {sps:>7,.0f} | r_p0 {avg_r:+.3f} r_p1 {avg_r1:+.3f} | "
                f"clip_frac {avg_cf:.3f} | KL {metrics.get('approx_kl', 0):.4f} | "
                f"H_fire {metrics.get('fire_entropy', 0):.3f} "
                f"H_ang {metrics.get('angle_entropy', 0):.2f} "
                f"H_ship {metrics.get('ship_entropy', 0):.2f} | "
                f"V_loss {metrics.get('value_loss', 0):.4f} | "
                f"LR {metrics['learning_rate']:.6f} | "
                f"early_stop={metrics.get('kl_early_stop', 0):.0f} | "
                f"fire[0]={slot0:.2f} fire[rest_max]={slot_rest_max:.2f} "
                f"srcs_multi={metrics.get('avg_sources_multi', 0):.2f} "
                f"ship0={metrics.get('ship_bin0_rate', 0):.2f} "
                f"meanshipbin={metrics.get('mean_ship_bin', 0):.1f} "
                f"avgfleet={metrics.get('avg_fleet_size', 0):.1f} "
                f"p90fleet={metrics.get('p90_fleet_size', 0):.1f}"
                + (f" | il_kl={metrics.get('il_kl', 0):.3f} "
                   f"il_coef={metrics.get('il_coef', 0):.3f}"
                   if metrics.get('il_coef', 0) > 0 else "")
            )
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
    parser.add_argument("--checkpoint-interval", type=int, default=5_000_000,
                        help="Save a periodic checkpoint every N env steps")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override PPO learning rate (default: cfg.ppo.learning_rate=3e-4)")
    parser.add_argument("--ppo-epochs", type=int, default=None,
                        help="Override PPO epochs per rollout (default: 4)")
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
    parser.add_argument("--win-margin-coeff", type=float, default=0.0,
                        help="Terminal bonus coefficient α: winner gets +1 + α*(my_score/total_score). "
                             "0 = pure ±1 reward (default). Suggested start: 0.5.")
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
                             "'../candidate_suneet_lb1200.py,../candidate_zach_public.py'). "
                             "Only used when --pool-mode=mixed.")
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
    args = parser.parse_args()

    if not args.device:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    train(args)
