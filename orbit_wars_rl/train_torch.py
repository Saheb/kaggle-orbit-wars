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
from model import EntityTransformer, count_params, PHASE4_COMPAT_MISSING_KEYS
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


def _load_phase4_compatible(model: EntityTransformer, state_dict: dict, label: str) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in missing if k not in PHASE4_COMPAT_MISSING_KEYS]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"{label} checkpoint/model mismatch: missing={bad_missing}, unexpected={list(unexpected)}"
        )


def sample_action_batched(outputs: dict, fire_mask: torch.Tensor,
                          target_mask: torch.Tensor | None = None):
    """Sample fire/ship/target actions for a batch of envs (target-decode only).

    Angle is not part of the executed policy — the env computes the aim direction
    from the sampled target planet index.  A zero tensor is returned for angle_a
    so the action tensor passed to env.step keeps its expected 4-column shape
    [fire, angle, ship, target].

    Returns: (fire_a, angle_a, ship_a, target_a, lp_fire, lp_ship, lp_target)
    """
    fire_logits_target = outputs["fire_logits"]
    ship_logits_target = outputs["ship_logits"]
    target_logits = outputs["target_logits"]
    if target_mask is not None:
        target_logits = target_logits.masked_fill(~target_mask, -1e9)
    target_dist = torch.distributions.Categorical(logits=target_logits)

    target_a = target_dist.sample()  # (N, MAX_OWNED)
    gather_idx = target_a.unsqueeze(-1)
    fire_logits = torch.gather(fire_logits_target, -1, gather_idx).squeeze(-1)
    fire_logits = fire_logits.masked_fill(~fire_mask, -1e9)
    ship_logits = torch.gather(
        ship_logits_target,
        2,
        gather_idx.unsqueeze(-1).expand(-1, -1, 1, ship_logits_target.shape[-1]),
    ).squeeze(2)
    fire_dist = torch.distributions.Bernoulli(logits=fire_logits)
    ship_dist = torch.distributions.Categorical(logits=ship_logits)
    fire_a   = fire_dist.sample()    # (N, MAX_OWNED)
    ship_a   = ship_dist.sample()    # (N, MAX_OWNED)
    # Angle is unused in target-decode; zeros satisfy env.step's action shape.
    angle_a  = torch.zeros_like(fire_a)

    # Target is part of the sampled joint action even when fire=0.
    slot_valid = fire_mask.float()
    fired      = (fire_a > 0.5).float() * slot_valid
    lp_fire   = fire_dist.log_prob(fire_a) * slot_valid
    lp_ship   = ship_dist.log_prob(ship_a)   * fired
    lp_target = target_dist.log_prob(target_a) * slot_valid

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


def index_features(feats: dict, ids: torch.Tensor) -> dict:
    """Index an env-batched `get_features()` dict down to the envs in `ids` (1-D Long).

    Lets a frozen pool-self opponent forward ONLY its assigned envs — reusing the features
    already computed for the learning model this step — instead of forwarding all N envs and
    discarding all but a few rows. The model has no cross-env coupling (LayerNorm, not
    BatchNorm) so per-row outputs are identical to forwarding the full batch then indexing.
    All feature tensors are env-batched on dim 0; `owned_count` is a length-N python list."""
    ids_list = ids.tolist()
    out = {}
    for k, v in feats.items():
        if torch.is_tensor(v):
            out[k] = v[ids]
        elif isinstance(v, list):
            out[k] = [v[i] for i in ids_list]
        else:
            out[k] = v
    return out


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


def _heuristic_moves_to_action_tensor(moves_per_env, env, player, device, *,
                                      owned_idx=None, slot_valid=None, src_pids=None):
    """Convert list-of-lists [[from_pid, angle_rad, ships], ...] per env into
    a (num_envs, MAX_OWNED, 3) action tensor plus a (num_envs, MAX_OWNED) float
    tensor of the raw continuous angles (NaN where no launch). The continuous
    angles let the env bypass 144-bin quantization for these aiming-heavy
    opponents (see torch_env._apply_actions angle_override).

    Fast path (SPS patch C): the rollout already holds this seat's `owned_idx`/`slot_valid`
    in its feature dict (get_features calls owned_indices_for, so they are IDENTICAL), and the
    source planet ids per owned slot can be gathered once. Pass them as `owned_idx`/`slot_valid`/
    `src_pids` to skip the redundant `env.owned_indices_for()` recompute AND the full
    `env.planets.cpu()` 7-column copy. When omitted, they are derived from `env` exactly as
    before (the path frozen_vs_deb_torch.py relies on)."""
    from torch_env import MAX_OWNED, NUM_ANGLE_BINS, ANGLE_BIN_WIDTH, SHIP_COUNTS
    import math as _math

    if owned_idx is None:
        owned_idx, slot_valid = env.owned_indices_for(player)   # (N, MAX_OWNED)
        gather_idx = owned_idx.unsqueeze(-1).expand(-1, -1, 7).cpu()
        # planet id per owned slot (column 0) — same value the precomputed fast path supplies.
        src_pids = env.planets.cpu().gather(1, gather_idx)[:, :, 0]
    N = owned_idx.shape[0]
    sv_cpu = slot_valid.cpu() if torch.is_tensor(slot_valid) else slot_valid
    src_pids_cpu = src_pids.cpu() if torch.is_tensor(src_pids) else src_pids
    fire = torch.zeros(N, MAX_OWNED, dtype=torch.long)
    angle_bin = torch.zeros(N, MAX_OWNED, dtype=torch.long)
    ship_bin = torch.zeros(N, MAX_OWNED, dtype=torch.long)
    cont_angle = torch.full((N, MAX_OWNED), float("nan"), dtype=torch.float32)

    for e in range(N):
        moves = moves_per_env[e]
        if not moves:
            continue
        pid_to_slot = {}
        for k in range(MAX_OWNED):
            if bool(sv_cpu[e, k]):
                pid_to_slot[int(src_pids_cpu[e, k])] = k
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
            cont_angle[e, slot] = ang
    return torch.stack([fire, angle_bin, ship_bin], dim=-1).to(device), cont_angle.to(device)


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
    cfg.ppo.phase4_residual_lr_mult = args.phase4_residual_lr_mult
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
    if args.critic_warmup_ev is not None:
        cfg.ppo.critic_warmup_ev = args.critic_warmup_ev
    if args.critic_warmup_max_updates is not None:
        cfg.ppo.critic_warmup_max_updates = args.critic_warmup_max_updates
    if args.bc_coef is not None:
        cfg.ppo.bc_coef = args.bc_coef
    print(f"PPO config: lr={cfg.ppo.learning_rate}, ppo_epochs={cfg.ppo.ppo_epochs}, "
          f"num_minibatches={cfg.ppo.num_minibatches}, clip_eps={cfg.ppo.clip_eps}, "
          f"entropy_coef_fire={cfg.ppo.entropy_coef_fire}, gae_lambda={cfg.ppo.gae_lambda}, "
          f"kl_target={cfg.ppo.kl_target}")
    if cfg.ppo.phase4_residual_lr_mult != 1.0:
        print(f"Phase4 residual LR multiplier: x{cfg.ppo.phase4_residual_lr_mult:.3g}")
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
    if args.allow_reinforce and args.reinforce_gate_min_planets > 0:
        print(f"Reinforcement EMPIRE GATE: own targets legal only at >= "
              f"{args.reinforce_gate_min_planets} planets (attack-only below; mask, no Nash risk)")
    if args.allow_reinforce and args.reinforce_forward_only:
        print("Reinforcement FORWARD-STAGING GATE: own targets legal only if closer to the "
              "nearest enemy than the source (rear→front staging; mask, no Nash risk)")
    if args.sufficient_commit_factor > 0.0:
        print(f"SUFFICIENT-COMMIT MASK: veto attack launches with ships <= target_defense × "
              f"{args.sufficient_commit_factor} (force concentration; mask, no Nash risk)")
    print(f"Win margin coeff: {args.win_margin_coeff}")
    print(f"Shaping coeff: {args.shaping_coef}")
    print(f"Expansion coeff: {args.expansion_coef}")
    print(f"Defense coeff: {args.defense_coef}")
    print(f"Early capture coeff: {args.early_capture_coef} (decay over {args.early_capture_steps} steps)")
    print(f"Production-share coeff: {args.prod_share_coef}")
    if args.prod_share_coef != 0.0:
        other_reward_knobs = [
            ("win_margin_coeff", args.win_margin_coeff),
            ("shaping_coef", args.shaping_coef),
            ("expansion_coef", args.expansion_coef),
            ("defense_coef", args.defense_coef),
            ("early_capture_coef", args.early_capture_coef),
            ("speed_coef", args.speed_coef),
            ("consolidation_coef", args.consolidation_coef),
            ("capture_utility_coef", args.capture_utility_coef),
            ("capture_idle_penalty", args.capture_idle_penalty),
            ("decisive_mass_coef", args.decisive_mass_coef),
        ]
        active_reward_knobs = [name for name, value in other_reward_knobs if value != 0.0]
        if active_reward_knobs:
            print("WARNING: --prod-share-coef is active together with other reward knobs: "
                  + ", ".join(active_reward_knobs))
    if args.early_capture_anneal_frac > 0.0:
        print(f"Early capture ANNEAL: cosine {args.early_capture_coef}→0 over "
              f"{args.early_capture_anneal_frac * args.total_steps:,.0f} steps "
              f"(frac {args.early_capture_anneal_frac}), then 0")
    print(f"First Strike: {args.first_strike_mult}x for t<{args.first_strike_steps} steps" if args.first_strike_steps > 0 else "First Strike: off")
    print(f"Speed coeff: {args.speed_coef}")
    if args.consolidation_coef != 0:
        print(f"Consolidation bonus: {args.consolidation_coef} per net-new capture HELD {args.consolidation_steps} steps (force-concentration lever)")
    if args.capture_utility_coef != 0 or args.capture_idle_penalty != 0:
        print(f"Capture-utility reward: +{args.capture_utility_coef} when a net-new capture "
              f"attacks or remains frontline within {args.capture_utility_window} steps; "
              f"idle penalty={args.capture_idle_penalty}")
    if args.decisive_mass_coef != 0:
        print(f"Decisive-mass bonus (Lever A): {args.decisive_mass_coef} per inflight strike reaching producer_v2's ENEMY capture floor (reactive-margin beta={args.decisive_mass_beta}) — force-concentration signal")
    if args.handicap_frac > 0:
        print(f"Handicap: {args.handicap_frac*100:.0f}% of games start with {args.handicap_ships} ships (vs normal 10)")
    if args.ssdr_frac > 0:
        print(f"SSDR: {args.ssdr_frac*100:.0f}% of resets grant opponent 1..{args.ssdr_max_steps} extra planets (asymmetric start)")
    if args.neutral_garrison_scale > 1.0:
        print(f"Neutral garrison scale: {args.neutral_garrison_scale:.1f}x (board-curriculum: "
              f"expensive neutrals force multi-source concentration; symmetric, both players)")
    if args.scenario_curriculum != "off" and args.scenario_fraction > 0.0:
        print(f"Scenario curriculum: {args.scenario_curriculum} "
              f"on {args.scenario_fraction*100:.1f}% of resets, "
              f"deadline={args.scenario_deadline}")
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
        if "phase4_residual_init_std" in ckpt_cfg:
            cfg.model.phase4_residual_init_std = float(ckpt_cfg["phase4_residual_init_std"])
        if bool(ckpt_cfg.get("game_phase_features", False)):
            cfg.model.game_phase_features = True  # resumed weights are 15-global
        # Resume-path discipline parity: if the CLI left these at defaults, inherit
        # the checkpoint values so env/model wiring matches the source policy.
        if not args.allow_reinforce and bool(ckpt_cfg.get("allow_reinforce", False)):
            args.allow_reinforce = True
        if args.reinforce_gate_min_planets == 0 and "reinforce_gate_min_planets" in ckpt_cfg:
            args.reinforce_gate_min_planets = int(ckpt_cfg["reinforce_gate_min_planets"])
        if not args.reinforce_forward_only and bool(ckpt_cfg.get("reinforce_forward_only", False)):
            args.reinforce_forward_only = True
        if args.reverse_edge_cooldown == 0 and "reverse_edge_cooldown" in ckpt_cfg:
            args.reverse_edge_cooldown = int(ckpt_cfg["reverse_edge_cooldown"])
        if args.reinforce_garrison_floor == 0.0 and "reinforce_garrison_floor" in ckpt_cfg:
            args.reinforce_garrison_floor = float(ckpt_cfg["reinforce_garrison_floor"])
        if args.sufficient_commit_factor == 0.0 and "sufficient_commit_factor" in ckpt_cfg:
            args.sufficient_commit_factor = float(ckpt_cfg["sufficient_commit_factor"])
        del _ckpt_peek

    # CLI overrides checkpoint metadata. This lets a run deliberately mask bin
    # 0 when resuming from a BC checkpoint saved with min_ship_bin=0.
    if args.min_ship_bin is not None:
        cfg.model.min_ship_bin = args.min_ship_bin
    if args.phase4_residual_init_std is not None:
        cfg.model.phase4_residual_init_std = args.phase4_residual_init_std
    cfg.model.action_decode = args.action_decode
    cfg.model.allow_reinforce = args.allow_reinforce
    # Persist the reinforce/sufficient-commit DISCIPLINE on cfg.model so the checkpoint records
    # them (ppo.state_dict) and eval/export auto-load them — they must match training or the
    # policy self-sabotages. Previously eval relied on CLI flags being remembered.
    cfg.model.reinforce_gate_min_planets = args.reinforce_gate_min_planets
    cfg.model.reinforce_forward_only = args.reinforce_forward_only
    cfg.model.reverse_edge_cooldown = args.reverse_edge_cooldown
    cfg.model.reinforce_garrison_floor = args.reinforce_garrison_floor
    cfg.model.sufficient_commit_factor = args.sufficient_commit_factor
    cfg.model.capture_utility_coef = args.capture_utility_coef
    cfg.model.capture_utility_window = args.capture_utility_window
    cfg.model.capture_idle_penalty = args.capture_idle_penalty
    # PROVENANCE only (eval always clamps via _ship_bin_to_count, so this doesn't change the eval
    # contract) — but record how the ckpt was trained (drop vs clamp) so it's never ambiguous.
    cfg.model.ship_overflow_mode = args.ship_overflow_mode
    # Game-phase features: on if the CLI asks OR a resumed ckpt had them (15-global weights).
    cfg.model.game_phase_features = cfg.model.game_phase_features or args.game_phase_features
    if cfg.model.game_phase_features:
        cfg.model.global_feature_dim = 15

    env = VecTorchEnv(num_envs=args.num_envs, num_players=2,
                      device=device, episode_steps=500,
                      ship_bin_mode=cfg.model.ship_bin_mode,
                      ship_overflow_mode=args.ship_overflow_mode,
                      action_decode=args.action_decode,
                      allow_reinforce=args.allow_reinforce,
                      game_phase_features=cfg.model.game_phase_features,
                      reinforce_garrison_floor=args.reinforce_garrison_floor,
                      reinforce_cost=args.reinforce_cost,
                      reinforce_gate_min_planets=args.reinforce_gate_min_planets,
                      reinforce_forward_only=args.reinforce_forward_only,
                      reverse_edge_cooldown=args.reverse_edge_cooldown,
                      sufficient_commit_factor=args.sufficient_commit_factor,
                      win_margin_coeff=args.win_margin_coeff,
                      shaping_coef=args.shaping_coef,
                      expansion_coef=args.expansion_coef,
                      defense_coef=args.defense_coef,
                      early_capture_coef=args.early_capture_coef,
                      prod_share_coef=args.prod_share_coef,
                      early_capture_steps=args.early_capture_steps,
                      first_strike_steps=args.first_strike_steps,
                      first_strike_mult=args.first_strike_mult,
                      speed_coef=args.speed_coef,
                      consolidation_coef=args.consolidation_coef,
                      consolidation_steps=args.consolidation_steps,
                      capture_utility_coef=args.capture_utility_coef,
                      capture_utility_window=args.capture_utility_window,
                      capture_idle_penalty=args.capture_idle_penalty,
                      decisive_mass_coef=args.decisive_mass_coef,
                      decisive_mass_beta=args.decisive_mass_beta,
                      decisive_diag=args.decisive_diag,
                      handicap_frac=args.handicap_frac,
                      handicap_ships=args.handicap_ships,
                      ssdr_frac=args.ssdr_frac,
                      ssdr_max_steps=args.ssdr_max_steps,
                      neutral_garrison_scale=args.neutral_garrison_scale,
                      scenario_curriculum=args.scenario_curriculum,
                      scenario_fraction=args.scenario_fraction,
                      scenario_deadline=args.scenario_deadline)
    env.reset(seeds=[args.seed + i for i in range(args.num_envs)])

    model = EntityTransformer(cfg.model).to(device)
    print(f"Model params: {count_params(model):,}")
    if args.resume:
        sd = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model" in sd: sd = sd["model"]
        _load_phase4_compatible(model, sd, "--resume")
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
        _load_phase4_compatible(frozen_il_model, sd, "--il-ref")
        frozen_il_model.eval()
        print(f"IL reference loaded from {args.il_ref}  (lambda={cfg.ppo.il_lambda}, "
              f"decay_frac={cfg.ppo.il_decay_frac})")
    elif cfg.ppo.il_lambda > 0:
        # If --resume is set and no separate ref, default to the resume checkpoint
        if args.resume:
            frozen_il_model = EntityTransformer(cfg.model).to(device)
            sd = torch.load(args.resume, map_location="cpu", weights_only=False)
            if "model" in sd: sd = sd["model"]
            _load_phase4_compatible(frozen_il_model, sd, "--resume IL default")
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
        print(f"ROI aux loss: coef={args.aux_roi_coef} (keeps pair_kv/target_scorer columns encoding roi_20/roi_50/enemy_contest)")

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
            # OpponentPool.load() restores the SAVED external_fraction — a CONFIG knob, not
            # state (unlike the member list / PFSP stats, which we DO keep). Without this the
            # current run's --pool-external-fraction is silently ignored on resume: a pool
            # saved at 0.6 kept training at 0.6 despite --pool-external-fraction 0.5. The
            # current run's flag must govern; override it (members/stats untouched).
            if pool.external_fraction != args.pool_external_fraction:
                print(f"  override external_fraction {pool.external_fraction:.2f} -> "
                      f"{args.pool_external_fraction:.2f} (CLI --pool-external-fraction wins on resume)")
                pool.external_fraction = args.pool_external_fraction
            # Same resume footgun for pfsp_externals (a CONFIG knob): CLI flag must govern.
            if pool.pfsp_externals != args.pfsp_externals:
                print(f"  override pfsp_externals {pool.pfsp_externals} -> "
                      f"{args.pfsp_externals} (CLI --pfsp-externals wins on resume)")
                pool.pfsp_externals = args.pfsp_externals
        else:
            pool = OpponentPool(
                max_self_members=args.pool_max_size,
                pfsp_alpha=args.pool_pfsp_alpha,
                mastered_winrate=args.pool_mastered_threshold,
                mastered_min_games=args.pool_mastered_min_games,
                pfsp_min_games=args.pool_pfsp_min_games,
                external_fraction=args.pool_external_fraction,
                pfsp_externals=args.pfsp_externals,
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
            # Only preseed from model checkpoints. The same directory can also
            # contain serialized opponent-pool payloads (`pool_step_*.pt`),
            # which are not valid model state_dicts.
            for pt_file in sorted(preseed_dir.glob("torch_step_*.pt")):
                m = re.search(r"step_(\d+)", pt_file.stem)
                if not m: continue
                step = int(m.group(1))
                if step in existing_self_steps: continue
                sd = torch.load(pt_file, map_location="cpu", weights_only=False)
                if isinstance(sd, dict) and "model" in sd:
                    sd = sd["model"]
                if not isinstance(sd, dict) or not sd or any(not hasattr(v, "detach") for v in sd.values()):
                    print(f"  WARN: skipping non-model preseed payload {pt_file.name}")
                    continue
                pool.add_self_checkpoint(step, sd)
                added += 1
            print(f"  pool preseeded: {added} self-checkpoints from {preseed_dir}")

        # Pinned RL champions: fixed strong opponents (e.g. rev38, rev53b), never
        # FIFO-evicted. Appended even on resume; skip if a same-named pin already
        # exists (e.g. restored from the saved pool).
        if args.pool_seed_rl:
            existing_pins = {m.name for m in pool.members if getattr(m, "pinned", False)}
            for path in args.pool_seed_rl.split(","):
                path = path.strip()
                if not path:
                    continue
                name = Path(path).stem
                if f"seed_{name}" in existing_pins:
                    print(f"  pool pinned RL already present from resume: seed_{name}")
                    continue
                sd = torch.load(path, map_location="cpu", weights_only=False)
                if "model" in sd:
                    sd = sd["model"]
                # Drop keys the current model doesn't have (e.g. rev38's deleted
                # angle_head); fail fast if it's missing any the model needs.
                model_keys = set(model.state_dict().keys())
                sd = {k: v for k, v in sd.items() if k in model_keys}
                missing = model_keys - set(sd.keys())
                if missing:
                    print(f"  WARN: pinned RL {name} incompatible — missing {len(missing)} keys "
                          f"(e.g. {sorted(missing)[:3]}); skipping")
                    continue
                pool.add_pinned_rl(name, sd)
                print(f"  pool pinned RL champion: seed_{name} ({path})")

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
        # compute_pool_actions can dispatch by opp.name. All externals run via CPU
        # worker pools (the in-process GPU planner adapter deadlocked rollout 1).
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
        print(f"Pool initialised: mode={args.pool_mode}, members={len(pool)} (at init), "
              f"pool-fraction={args.pool_fraction}, snapshot_every={args.pool_checkpoint_interval:,} steps")
        _pool_ext = max(0.0, min(1.0, args.pool_external_fraction))
        _pool_pin = max(0.0, min(1.0, args.pool_pinned_fraction))
        if args.pool_pinned_fraction > 0.0:
            _pool_self = max(0.0, 1.0 - _pool_ext - _pool_pin)
            _pin_total = args.pool_fraction * _pool_pin
        else:
            _pool_self = max(0.0, 1.0 - _pool_ext)
            _pin_total = 0.0
        _ext_total = args.pool_fraction * _pool_ext
        _pool_self_total = args.pool_fraction * _pool_self
        _current_self_total = max(0.0, 1.0 - args.pool_fraction)
        print(f"  pool exposure target: external≈{_ext_total:.2f} total, "
              f"pool-self≈{_pool_self_total:.2f} total, current-self≈{_current_self_total:.2f} total"
              + (f", pinned≈{_pin_total:.2f} total" if args.pool_pinned_fraction > 0.0 else ""))
        if (args.pool_mode == "mixed" and args.external_opponents and args.pool_external_fraction >= 1.0
                and any(m.kind == "self" for m in pool.members)):
            print("  WARN: --pool-external-fraction >= 1.0 means self-checkpoints in the pool "
                  "will not be sampled (their pool wr stays n=0). If you intended 80% external "
                  "+ 20% pool-self, use --pool-fraction 1.0 --pool-external-fraction 0.8.")
        if args.pool_pinned_fraction > 0.0:
            print(f"  Opponent-difficulty RAMP active: pinned-RL + external ease in 0 -> "
                  f"{args.pool_pinned_fraction:.3f}/{args.pool_external_fraction:.3f} of the pool slice "
                  f"over {args.pool_hard_ramp_steps:,} steps (true-zero start = self-play early).")
        print()
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
                    "capture_utility_coef": args.capture_utility_coef,
                    "capture_utility_window": args.capture_utility_window,
                    "capture_idle_penalty": args.capture_idle_penalty,
                    "prod_share_coef": args.prod_share_coef,
                    "il_lambda": cfg.ppo.il_lambda,
                    "win_margin_coeff": args.win_margin_coeff,
                    "speed_coef": args.speed_coef,
                    "action_decode": args.action_decode,
                    "resume": args.resume or "",
                    "ship_bin_mode": cfg.model.ship_bin_mode,
                    "game_phase_features": cfg.model.game_phase_features,
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
    }

    def _get_pool_self_model(member: PoolMember):
        """Frozen, eval-mode model holding `member`'s weights, cached ON the member so we
        load_state_dict ONCE per member instead of once per (member, seat) group per step.
        The model is a tiny deepcopy that lives and dies with the member — evicted members
        are GC'd together with their cached model, so there is no id()-reuse hazard a plain
        dict cache keyed on id(member) would have. `pool.save()` serializes only named
        dataclass fields, so the extra attribute is never persisted."""
        m = getattr(member, "_pool_model", None)
        if m is None:
            m = copy.deepcopy(pool_opp_model)
            m.load_state_dict(member.state_dict)
            m.to(device)
            m.eval()
            member._pool_model = m
        return m

    def compute_pool_actions(opp: PoolMember, player: int, env_ids: torch.Tensor,
                             cached_feats: dict | None = None) -> torch.Tensor:
        """Return (action tensor, cont_angle|None) for the opponent playing `player`
        in the envs listed in `env_ids` (1-D LongTensor). Supports 'self' (frozen RL
        model on GPU) and 'external_heuristic' (.py agent via CPU worker pool).

        Keyed by an arbitrary index set, NOT a contiguous slice: per-episode pool
        assignment lets different envs hold different members at the same step, so the
        step loop groups envs by (member, seat) and calls this once per group.

        `cached_feats` (if given) is the env-batched `get_features(player)` already computed
        for the learning model this step — reused for 'self' members instead of recomputing,
        and indexed to `env_ids` so the frozen model forwards only its assigned envs."""
        if opp.kind == "self":
            opp_model = _get_pool_self_model(opp)   # cached weights (no per-group load_state_dict)
            feats = cached_feats if cached_feats is not None else \
                env.get_features(player, max_planets=cfg.env.max_planets, max_fleets=128)
            sub = index_features(feats, env_ids)    # select our envs BEFORE the forward
            with torch.no_grad():
                outs = opp_model(
                    sub["planet_features"], sub["fleet_features"],
                    sub["global_features"], sub["planet_mask"],
                    sub["fleet_mask"],
                    fire_mask=sub["fire_mask"], angle_mask=sub["angle_mask"],
                    slot_valid=sub["slot_valid"], owned_indices=sub["owned_indices"],
                    owned_count=sub["owned_count"],
                    pairwise_features=sub.get("pairwise_features"),
                )
            fire_a, angle_a, ship_a, target_a, *_ = sample_action_batched(
                outs, sub["fire_mask"], sub.get("target_mask")
            )
            # Rows already correspond 1:1 to env_ids (we indexed before the forward).
            return torch.stack([fire_a, angle_a, ship_a, target_a], dim=-1), None

        if opp.kind == "external_heuristic":
            from torch_env import to_legacy_obs_batch
            # Patch B: ONE batched GPU->CPU copy per field over the group, instead of
            # ~8 tiny per-env syncs × |ids| (the dominant per-step transport tax).
            _t0 = time.perf_counter()
            obs_list = to_legacy_obs_batch(env, env_ids, player)
            _t_acc["ext_obs"] += time.perf_counter() - _t0
            wp = heur_worker_pools.get(opp.name)
            if wp is not None:
                _t0 = time.perf_counter()
                moves_per_env = wp.map(obs_list)
                _t_acc["ext_wait"] += time.perf_counter() - _t0
            else:
                # Fallback: serial path (no worker pool registered)
                moves_per_env = []
                for obs in obs_list:
                    try:
                        moves_per_env.append(opp.agent_fn(obs) or [])
                    except Exception:
                        moves_per_env.append([])
            # Patch C: reuse this seat's already-computed owned_idx/slot_valid (get_features
            # calls owned_indices_for, so cached_feats holds the IDENTICAL tensors) + one
            # subset gather for source planet ids, replacing the per-group owned_indices_for
            # recompute and the full env.planets.cpu() 7-column copy the old _IndexView did.
            _t0 = time.perf_counter()
            if cached_feats is not None:
                oi_full, sv_full = cached_feats["owned_indices"], cached_feats["slot_valid"]
            else:
                oi_full, sv_full = env.owned_indices_for(player)
            oi_sub, sv_sub = oi_full[env_ids], sv_full[env_ids]
            src_pids = env.planets[env_ids, :, 0].gather(1, oi_sub)   # (n, MAX_OWNED) planet ids
            act, cont_angle = _heuristic_moves_to_action_tensor(
                moves_per_env, env, player, device,
                owned_idx=oi_sub, slot_valid=sv_sub, src_pids=src_pids)
            _t_acc["ext_conv"] += time.perf_counter() - _t0
            if args.action_decode == "target":
                # Target_idx = -1 sentinel so VecTorchEnv keeps angle decoding for these
                # rows (not target decoding). The continuous angle (cont_angle) is applied
                # separately via angle_overrides so it bypasses 144-bin quantization.
                pad_target = torch.full(
                    act.shape[:-1] + (1,), -1, dtype=act.dtype, device=act.device
                )
                act = torch.cat([act, pad_target], dim=-1)
            return act, cont_angle

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
    # Critic-only warmup (BC warmstart): fit the value head with the policy frozen
    # until EV reaches the threshold, before PPO trusts any advantage. Self-skips on
    # a warm-critic resume (first rollout's EV already >= threshold).
    critic_warmup_active = cfg.ppo.critic_warmup_ev > 0.0
    critic_warmup_count = 0
    if critic_warmup_active:
        print(f"Critic warmup ENABLED: value-head-only until EV>={cfg.ppo.critic_warmup_ev} "
              f"(max {cfg.ppo.critic_warmup_max_updates} rollouts), policy frozen.")
    print("=" * 70)

    # --- Per-EPISODE pool assignment (correctness fix) ----------------------------
    # The opponent identity + our seat are drawn ONCE PER EPISODE — when an env RESETS
    # — and held until that episode ends, instead of being re-rolled every rollout.
    # Previously `pool_opp`/`current_seat` were sampled per-rollout while episodes (~300-
    # 500 steps) span ~3-4 rollouts, so a game could be played by different controllers
    # across rollout boundaries and credited to whichever assignment was active at its
    # done step — contaminating PFSP attribution and injecting mid-game controller swaps.
    # Now each env's assignment is sticky for the life of its episode; result is credited
    # to THAT assignment with the raw (pre-shaping) winner.
    # NOTE: SSDR is supported per-env (self-play mask); self-boost (handicap, parked) is
    # NOT yet ported to per-episode seats and is guarded out below.
    pool_active = pool is not None and len(pool) > 0 and args.pool_fraction > 0

    def _draw_assignment(steps_now: int):
        """Draw a fresh (is_pool, member, our_seat) for one resetting env. Each env is a
        pool env with prob `pool_fraction` (preserves the old aggregate pool exposure);
        the member is sampled per-env (more in-rollout opponent diversity than the old
        single-member-per-rollout), with the same ramp logic as before."""
        if not pool_active or rng.random() >= args.pool_fraction:
            return False, None, 0
        if args.pool_pinned_fraction > 0.0:
            ramp = (min(steps_now / args.pool_hard_ramp_steps, 1.0)
                    if args.pool_hard_ramp_steps > 0 else 1.0)
            member = pool.sample(rng,
                                 external_fraction=args.pool_external_fraction * ramp,
                                 pinned_fraction=args.pool_pinned_fraction * ramp)
        else:
            member = pool.sample(rng)
        if member is None:
            return False, None, 0
        return True, member, rng.randint(0, 1)

    if args.self_boost_planets > 0 and pool_active:
        raise NotImplementedError(
            "self-boost (handicap) is not ported to per-episode pool assignment "
            "(needs per-env boosted seat); it is parked-negative — relaunch without it.")
    if env.ssdr_frac > 0.0 and pool_active:
        # SSDR asymmetry is applied at env.step()'s auto-reset using the SSDR mask set at
        # rollout start (= ~env_is_pool), but a per-episode reassignment is drawn AFTER that
        # reset — so a reassigned POOL env can inherit an SSDR-asymmetric board, violating the
        # "pool envs get clean symmetric starts" contract. Proper fix = pre-draw each env's
        # NEXT-episode pool-ness and set the SSDR mask from it BEFORE stepping (so the board
        # matches the new episode). Inert for the current lineage (ssdr_frac=0); error loudly
        # rather than silently feed asymmetric boards to frozen pool opponents.
        raise NotImplementedError(
            "SSDR (ssdr_frac>0) + per-episode pool assignment is not yet aligned: a reassigned "
            "pool env can inherit an SSDR-asymmetric board (the mask lags the reassignment). "
            "Use ssdr_frac=0, or implement pre-drawn next-episode pool-ness for the SSDR mask.")

    # Persistent per-env assignment (Python lists; N is small). Drawn at startup; an
    # env's slot is refreshed only when that env resets (handled in the step loop).
    env_is_pool = [False] * N
    env_member: list[PoolMember | None] = [None] * N
    env_seat = [0] * N
    for _e in range(N):
        env_is_pool[_e], env_member[_e], env_seat[_e] = _draw_assignment(total_env_steps)

    while total_env_steps < args.total_steps:
        # --- Per-EPISODE opponent assignment --------------------------------
        # env_is_pool / env_member / env_seat are sticky per episode (drawn on reset,
        # see the step loop). Here we just publish the per-env self-play mask for SSDR
        # (pool envs get clean symmetric starts; self-play envs get SSDR) from the
        # CURRENT assignment. N_pool is now just a diagnostic count, not a slice.
        N_pool = sum(env_is_pool)
        # Inform env which envs are self-play (SSDR active) vs pool (symmetric).
        if env.ssdr_frac > 0.0:
            self_mask = torch.tensor([not ip for ip in env_is_pool], dtype=torch.bool)
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
        # Zero the reinforce_rate + overask accumulators for this rollout (env counts realized
        # reinforce/fire launches per (env,player); combined with train_mask below). Unconditional
        # so overask_rate logs for non-reinforce runs too; reinforce-specific counts stay gated.
        env.reset_reinforce_stats()

        # Hoard milestones (player 0): snapshot garrison/in-flight/planets when an env
        # is AT episode-step 16/32/50/100. Controlled-time + scale-free → replaces the
        # end-skewed avgfleet/p90. Accumulated on-device, synced once after the rollout.
        _MS = (16, 32, 50, 100)
        ms_garr = {m: torch.zeros((), device=env.device) for m in _MS}
        ms_infl = {m: torch.zeros((), device=env.device) for m in _MS}
        ms_plan = {m: torch.zeros((), device=env.device) for m in _MS}
        ms_n    = {m: torch.zeros((), device=env.device) for m in _MS}
        scenario_done_count = 0.0
        scenario_success_count = 0.0

        # --- Rollout collection (no grad) -----------------------------------
        # Per-rollout wall-time breakdown (SPS patch A): attributes main-thread time to
        # policy_forward / external obs-build / worker-wait / move-convert / env_step /
        # ppo_update so the diag line shows where the per-step tax actually is. CPU
        # perf_counter (no cuda.synchronize) → GPU-async work lands at the next sync; that
        # is exactly the main-thread wall-time we are trying to reduce.
        _t_acc = {"pf": 0.0, "ext_obs": 0.0, "ext_wait": 0.0, "ext_conv": 0.0,
                  "estep": 0.0, "upd": 0.0}
        model.eval()
        for t in range(rollout_T):
            actions_per_player = {}
            feats_by_player = {}   # reused for self-pool opponents (avoid recomputing get_features)
            _t_pf = time.perf_counter()
            for p in range(P):
                feats_p, outs_p = forward_player(p)
                feats_by_player[p] = feats_p
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
            _t_acc["pf"] += time.perf_counter() - _t_pf

            # Pool opponent override: for each pool env, replace its opp-seat action
            # (opp_seat = 1 - the env's assigned seat) with that env's assigned member's
            # action, and mark the slot not-trainable so PPO ignores it. Per-episode
            # assignment means different envs may hold different members/seats this step,
            # so we group by (member, opp_seat) and call compute_pool_actions per group.
            angle_overrides = None
            if N_pool > 0:
                groups: dict = {}  # (id(member), opp_seat) -> [member, opp_seat, [env ids]]
                for e in range(N):
                    if not env_is_pool[e]:
                        continue
                    os_ = 1 - env_seat[e]
                    groups.setdefault((id(env_member[e]), os_),
                                      [env_member[e], os_, []])[2].append(e)
                for member, os_, ids in groups.values():
                    ids_t = torch.tensor(ids, device=env.device, dtype=torch.long)
                    opp_action, opp_cont = compute_pool_actions(
                        member, os_, ids_t, cached_feats=feats_by_player.get(os_))
                    # Advanced-index assignment (index_put_) requires matching dtypes — it
                    # does NOT auto-cast the way the old contiguous-slice copy_ did; cast to
                    # the destination so behaviour matches the prior slice path exactly.
                    dst = actions_per_player[os_]
                    dst[ids_t] = opp_action.to(dst.dtype)
                    storage["train_mask"][t, ids_t, os_] = False
                    if opp_cont is not None:
                        # Continuous-angle override for an external member's envs (NaN
                        # elsewhere) → bypasses 144-bin quantization for its aiming.
                        if angle_overrides is None:
                            angle_overrides = {}
                        ovr = angle_overrides.get(os_)
                        if ovr is None:
                            ovr = torch.full(actions_per_player[os_].shape[:2],
                                             float("nan"), device=env.device)
                            angle_overrides[os_] = ovr
                        ovr[ids_t] = opp_cont.to(ovr.dtype)

            _t_es = time.perf_counter()
            state, rewards, done = env.step(actions_per_player, angle_overrides=angle_overrides)
            _t_acc["estep"] += time.perf_counter() - _t_es
            if getattr(env, "_last_scenario_id", None) is not None:
                scen_done = done & (env._last_scenario_id != 0)
                if scen_done.any():
                    scenario_done_count += float(scen_done.sum().item())
                    scenario_success_count += float(env._last_scenario_success[scen_done].float().sum().item())
            # Hoard milestones: when an env is at episode-step 16/32/50/100, accumulate
            # player-0 garrison (parked), in-flight, and owned-planet counts for that env.
            ownp = env.planets[:, :, 1].long()                            # (N, P) owner
            mine_p = (ownp == 0) & env.planet_alive                       # player-0 planets
            garr_p0 = (env.planets[:, :, 5] * mine_p.float()).sum(dim=1)  # (N,) parked ships
            plan_p0 = mine_p.float().sum(dim=1)                           # (N,) planets owned
            mine_f = (env.fleets[:, :, 1].long() == 0) & env.fleet_alive
            infl_p0 = (env.fleets[:, :, 6] * mine_f.float()).sum(dim=1)   # (N,) in-flight ships
            for _m in _MS:
                sel = (env.step_count == _m).float()                     # (N,) at-milestone
                ms_garr[_m] += (garr_p0 * sel).sum()
                ms_infl[_m] += (infl_p0 * sel).sum()
                ms_plan[_m] += (plan_p0 * sel).sum()
                ms_n[_m]    += sel.sum()
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

            # PFSP attribution + per-episode reassignment. Credit each finished POOL
            # env to ITS OWN assigned member using the RAW (pre-shaping) winner — NOT
            # the shaped `rewards` tensor, which by now carries material/expansion/
            # early-capture shaping — then draw a fresh assignment for the new episode
            # that env.step() already auto-reset. Crediting BEFORE reassign so the slot
            # still holds the just-finished episode's controller.
            done_list = torch.nonzero(done, as_tuple=False).flatten().tolist()
            if done_list:
                last_wins = env._last_wins  # (N, P) bool, raw winner at the done step
                for e in done_list:
                    if env_is_pool[e] and env_member[e] is not None:
                        cur_seat = env_seat[e]; o_seat = 1 - cur_seat
                        cur_w = bool(last_wins[e, cur_seat]); opp_w = bool(last_wins[e, o_seat])
                        if cur_w and not opp_w:   pool.record_result(env_member[e], "win")
                        elif opp_w and not cur_w: pool.record_result(env_member[e], "loss")
                        else:                     pool.record_result(env_member[e], "draw")
                    env_is_pool[e], env_member[e], env_seat[e] = _draw_assignment(total_env_steps)

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
        _PER_ENV_KEYS = set()
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
        # Hoard milestones (player 0, controlled episode-step → no end-skew): at 16/32/50/100,
        #   garr_frac   = parked / (parked + in-flight)   — scale-free deployment ratio
        #   ships/planet = parked / owned planets         — pile-up per planet
        #   planets     = owned planets                   — expansion trajectory
        # Replaces the end-skewed avgfleet/p90. Reference (Isaiah): garr_frac ~0.5 mid-game,
        # ~11-22 ships/planet, planets 2/6/9/10.
        ms_metrics = {}
        for _m in _MS:
            g, fl, pl, nn = (ms_garr[_m].item(), ms_infl[_m].item(),
                             ms_plan[_m].item(), ms_n[_m].item())
            ms_metrics[f"garr_frac@{_m}"] = g / (g + fl) if (g + fl) > 0 else 0.0
            ms_metrics[f"ships_per_planet@{_m}"] = g / pl if pl > 0 else 0.0
            ms_metrics[f"planets@{_m}"] = pl / nn if nn > 0 else 0.0

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

        # PPO update — OR a critic-only warmup step while the BC-warmstart critic is
        # still cold (policy frozen, value head only). The EV-based exit is checked
        # below, after this rollout's EV is computed.
        model.train()
        _t_upd = time.perf_counter()
        if critic_warmup_active:
            metrics = learner.value_warmup_update(minibatches)
            critic_warmup_count += 1
        else:
            metrics = learner.update(minibatches, scheduler=scheduler,
                                     kl_target=cfg.ppo.kl_target,
                                     bc_batch=bc_batch)
        _t_acc["upd"] += time.perf_counter() - _t_upd
        metrics.update(ms_metrics)
        if scenario_done_count > 0:
            metrics["scenario_count"] = scenario_done_count
            metrics["scenario_success_rate"] = scenario_success_count / scenario_done_count
        # train_mask time-fraction per (env,player): under per-episode assignment a slot's
        # learning-ness can flip mid-rollout (env resets pool<->self / seat), so the env's
        # rollout-SUMMED diagnostic accumulators below are weighted by the FRACTION of steps
        # each slot was the learning policy — ≈ correct vs the old t=0 snapshot (which counted
        # a flipped slot's whole-rollout total). Still approximate: assumes a slot's counts are
        # roughly uniform over the rollout (the env totals aren't split by when they happened).
        train_frac = storage["train_mask"].float().mean(0).to(env.device)   # (N, P) in [0,1]
        # reinforce_rate: of the current policy's realized launches (train_frac-weighted,
        # across both seats), the fraction sent to our own planets (Vadasz ~0.57; target 0.4-0.6).
        if args.allow_reinforce and env._fire_launch_count is not None:
            tm = train_frac                                        # (N, P)
            fires = (env._fire_launch_count * tm).sum()
            reinf = (env._reinforce_launch_count * tm).sum()
            neut = (env._neutral_launch_count * tm).sum()
            denom = fires.clamp(min=1.0)
            metrics["reinforce_rate"] = float((reinf / denom).item())   # own-target share
            # target-owner share among the current policy's launches (own/neutral/enemy);
            # Phase-2 target-head health — a selective head ≠ uniform across owners.
            metrics["target_share_neutral"] = float((neut / denom).item())
            metrics["target_share_enemy"] = float(((fires - reinf - neut) / denom).item())
            # reinf-by-step: own-target share by episode-window [<50, 50-100, >100]. Winners
            # peak MID-game (0.29/0.41/0.31); ours was back-loaded (0.05/0.19/0.42). The deb-in-pool
            # run should shift reinforcement EARLIER/MID — watch these climb at <50 and 50-100.
            if env._reinf_step is not None:
                tmu = tm.unsqueeze(-1)                                  # (N, players, 1)
                rs = (env._reinf_step * tmu).sum(dim=(0, 1))           # (3,)
                fs = (env._fire_step * tmu).sum(dim=(0, 1)).clamp(min=1.0)
                metrics["reinf_step_e"] = float((rs[0] / fs[0]).item())
                metrics["reinf_step_m"] = float((rs[1] / fs[1]).item())
                metrics["reinf_step_l"] = float((rs[2] / fs[2]).item())
        # overask_rate: fraction of the current policy's INTENDED launches whose ship_count >
        # source garrison (→ DROPPED in "drop" mode, clamped-to-full in "clamp"/eval). Always
        # logged (not gated on allow_reinforce); split by episode window [<50/50-100/>100].
        if getattr(env, "_attempt_step", None) is not None:
            tmu = train_frac.unsqueeze(-1)                                        # (N, players, 1)
            oa = (env._overask_step * tmu).sum(dim=(0, 1))                        # (3,)
            at = (env._attempt_step * tmu).sum(dim=(0, 1))                        # (3,)
            metrics["overask_rate"] = float((oa.sum() / at.sum().clamp(min=1.0)).item())
            metrics["overask_e"] = float((oa[0] / at[0].clamp(min=1.0)).item())
            metrics["overask_m"] = float((oa[1] / at[1].clamp(min=1.0)).item())
            metrics["overask_l"] = float((oa[2] / at[2].clamp(min=1.0)).item())
            # requested→emitted gap + fleet saturation + obs truncation (current policy via train_mask)
            req = float((env._attempt_step * tmu).sum().item())
            emi = float((env._emitted_step * tmu).sum().item())
            ssv = float((env._slotstarve_step * tmu).sum().item())
            metrics["moves_emit_req"] = emi / max(req, 1.0)            # emitted / requested
            metrics["fleet_saturation"] = ssv / max(emi + ssv, 1.0)   # dropped-for-no-slot / can_fire
            if getattr(env, "_obs_calls", None) is not None:
                metrics["obs_trunc_rate"] = float(env._obs_trunc.sum().item()) / max(float(env._obs_calls.sum().item()), 1.0)
                # severity: how much of the live fleet count / ship mass is hidden past the obs cap
                metrics["obs_trunc_fleet_frac"] = float(env._obs_trunc_fleets.sum().item()) / max(float(env._obs_total_fleets.sum().item()), 1.0)
                metrics["obs_trunc_ship_frac"] = float(env._obs_trunc_ships.sum().item()) / max(float(env._obs_total_ships.sum().item()), 1.0)
                # ENEMY omitted ship mass ÷ all enemy ship mass — the obs256 decider (is the
                # hidden mass the OPPONENT's, e.g. an inbound strike we can't see?).
                metrics["obs_trunc_enemy_ship_frac"] = float(env._obs_trunc_enemy_ships.sum().item()) / max(float(env._obs_total_enemy_ships.sum().item()), 1.0)
            if getattr(env, "_decisive_credit", None) is not None:
                # Lever A: avg decisive-mass crossings per (env,seat) this rollout, current policy.
                tm2 = tmu.squeeze(-1)                                  # (N, players) to match _decisive_credit
                metrics["decisive_strikes"] = float((env._decisive_credit * tm2).sum().item()) / max(float(tm2.sum().item()), 1.0)
            # dm_* GAP diagnostic: of the current policy's attacked enemy targets, how close does
            # the assembled inflight mass get to producer_v2's capture floor (the decmass target)?
            # All quantities are the EXACT reward floor (torch_env._decisive_mass_fields). gap DOWN +
            # cross UP = the policy is concentrating force; flat while WR climbs = adjacent competence.
            if args.decisive_diag and getattr(env, "_dm_targets", None) is not None:
                tg = (env._dm_targets * tmu).sum(dim=(0, 1))          # (3,) phase: e/m/l
                cr = (env._dm_cross * tmu).sum(dim=(0, 1))
                rs = (env._dm_ratio_sum * tmu).sum(dim=(0, 1))
                gp = (env._dm_gap_sum * tmu).sum(dim=(0, 1))
                ok = (env._dm_overkill_sum * tmu).sum(dim=(0, 1))
                nm = (env._dm_nearmiss * tmu).sum(dim=(0, 1))
                tg_tot = tg.sum().clamp(min=1.0)
                cr_tot = cr.sum().clamp(min=1.0)
                for i, ph in enumerate(("e", "m", "l")):
                    d = tg[i].clamp(min=1.0)
                    metrics[f"dm_gap_{ph}"] = float((gp[i] / d).item())
                    metrics[f"dm_cross_{ph}"] = float((cr[i] / d).item())
                metrics["dm_gap"] = float((gp.sum() / tg_tot).item())
                metrics["dm_cross"] = float((cr.sum() / tg_tot).item())
                metrics["dm_ratio"] = float((rs.sum() / tg_tot).item())
                metrics["dm_overkill"] = float((ok.sum() / cr_tot).item())   # mean ratio on crossed
                metrics["dm_nearmiss"] = float((nm.sum() / tg_tot).item())
                # mean attacked enemy targets per controlled env-step (assembly breadth)
                slotsteps = float(train_frac.sum().item()) * storage["train_mask"].shape[0]
                metrics["dm_tgt"] = float(tg.sum().item()) / max(slotsteps, 1.0)
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
            # Per-feature input-weight norms on the pairwise cross-attention. Track whether
            # newer target-value / keepability columns actually get used after a padded
            # checkpoint resume (norm climbing toward the original geometry/owner channels)
            # vs staying inert near zero.
            if hasattr(model, "pair_kv"):
                fw = model.pair_kv.weight          # [2D, D + F_pair]
                D = fw.shape[0] // 2
                feat_cols = fw[:, D:]              # [2D, F_pair]
                if feat_cols.shape[1] >= 15:
                    metrics["wnorm_roi20"] = float(feat_cols[:, 12].norm().item())
                    metrics["wnorm_roi50"] = float(feat_cols[:, 13].norm().item())
                    metrics["wnorm_enemy_contest"] = float(feat_cols[:, 14].norm().item())
                    metrics["wnorm_pw_orig"] = float(feat_cols[:, :12].norm(dim=0).mean().item())
                if feat_cols.shape[1] >= 20:
                    metrics["wnorm_reachable_enemy"] = float(feat_cols[:, 15].norm().item())
                    metrics["wnorm_capture_value"] = float(feat_cols[:, 16].norm().item())
                    metrics["wnorm_reactive_roi"] = float(feat_cols[:, 17].norm().item())
                    metrics["wnorm_friendly_reach"] = float(feat_cols[:, 18].norm().item())
                    metrics["wnorm_keepability"] = float(feat_cols[:, 19].norm().item())
            if hasattr(model, "fire_q"):
                metrics["phase4_fire_q_norm"] = float(model.fire_q.weight.norm().item())
                metrics["phase4_fire_k_norm"] = float(model.fire_k.weight.norm().item())
                metrics["phase4_ship_q_norm"] = float(model.ship_q.weight.norm().item())
                metrics["phase4_ship_k_norm"] = float(model.ship_k.weight.norm().item())
                metrics["phase4_fire_resid_out_norm"] = float(model.fire_scorer[2].weight.norm().item())
                metrics["phase4_ship_resid_out_norm"] = float(model.ship_scorer[2].weight.norm().item())

        total_env_steps += rollout_T * N
        iter_count += 1
        clipfrac_history.append(metrics.get("clip_frac", 0.0))

        # Critic-warmup exit: once the (policy-frozen) value head explains enough
        # return variance — or we hit the cap — stop warming and switch to PPO.
        if critic_warmup_active:
            _ev = metrics.get("explained_variance", 0.0)
            if _ev >= cfg.ppo.critic_warmup_ev or critic_warmup_count >= cfg.ppo.critic_warmup_max_updates:
                critic_warmup_active = False
                print(f"  ★ critic warmup DONE: EV={_ev:.3f} after {critic_warmup_count} "
                      f"rollouts ({total_env_steps:,} steps) — starting PPO.", flush=True)

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
                f"LR {metrics['learning_rate']:.6f}"
                + (f"/{metrics.get('phase4_residual_learning_rate', metrics['learning_rate']):.6f}"
                   if args.phase4_residual_lr_mult != 1.0 else "")
                + f" | estop {metrics.get('kl_early_stop', 0):.0f}"
                + (f" | il_kl {metrics.get('il_kl', 0):.3f} il_coef {metrics.get('il_coef', 0):.3f}"
                   if metrics.get('il_coef', 0) > 0 else "")
            )
            # Secondary behavioural diagnostics — occasionally useful, not decision
            # drivers (W&B keeps them every iter). Console-print every 5th log.
            if iter_count == 1 or iter_count % 5 == 0:
                pencoef = ""
                if args.srcs_multi_penalty > 0.0 and args.srcs_multi_penalty_decay_frac > 0.0:
                    _pc = args.srcs_multi_penalty * 0.5 * (1.0 + math.cos(math.pi * min(
                        total_env_steps / max(args.srcs_multi_penalty_decay_frac * args.total_steps, 1), 1.0)))
                    pencoef = f" pencoef {_pc:.5f}"
                actcoef = f" actcoef {args.fleet_activity_coef:.4f}" if args.fleet_activity_coef > 0.0 else ""
                reinfstr = (
                    f"reinf {metrics.get('reinforce_rate', 0):.2f} "
                    f"step<50/50-100/>100 {metrics.get('reinf_step_e',0):.2f}/"
                    f"{metrics.get('reinf_step_m',0):.2f}/{metrics.get('reinf_step_l',0):.2f} "
                    f"[ref:win 0.29/0.41/0.31] "
                    f"tgt n/e {metrics.get('target_share_neutral', 0):.2f}/"
                    f"{metrics.get('target_share_enemy', 0):.2f} | "
                ) if args.allow_reinforce else ""
                print(
                    f"   diag | fire[0] {slot0:.2f} rest_max {slot_rest_max:.2f} | "
                    f"fire_frac {metrics.get('fire_fraction', 0):.2f} "
                    f"owned {metrics.get('owned_planets', 0):.1f} "
                    f"ship0 {metrics.get('ship_bin0_rate', 0):.2f} "
                    f"meanshipbin {metrics.get('mean_ship_bin', 0):.1f} | "
                    f"pl@16/32/50/100 {metrics.get('planets@16',0):.0f}/{metrics.get('planets@32',0):.0f}/"
                    f"{metrics.get('planets@50',0):.0f}/{metrics.get('planets@100',0):.0f} "
                    f"garrfrac@50 {metrics.get('garr_frac@50',0):.2f} "
                    f"shipspp@50 {metrics.get('ships_per_planet@50',0):.0f} | "
                    f"overask {metrics.get('overask_rate',0):.2f} "
                    f"(<50/50-100/>100 {metrics.get('overask_e',0):.2f}/"
                    f"{metrics.get('overask_m',0):.2f}/{metrics.get('overask_l',0):.2f}) "
                    f"emit/req {metrics.get('moves_emit_req',0):.2f} satur {metrics.get('fleet_saturation',0):.3f} "
                    f"obstrunc {metrics.get('obs_trunc_rate',0):.3f} "
                    f"(fleet {metrics.get('obs_trunc_fleet_frac',0):.3f} ship {metrics.get('obs_trunc_ship_frac',0):.3f} "
                    f"enemyship {metrics.get('obs_trunc_enemy_ship_frac',0):.3f}) | "
                    f"decis {metrics.get('decisive_strikes',0):.2f} | "
                    f"{reinfstr}"
                    f"H_ship {metrics.get('ship_entropy', 0):.2f} "
                    f"H_tgt {metrics.get('target_entropy', 0):.2f} | "
                    f"Vμ {metrics.get('old_value_mean', 0):+.2f} Rμ {metrics.get('return_mean', 0):+.2f} "
                    f"Rσ {metrics.get('return_std', 0):.2f} Aσ {metrics.get('adv_std', 0):.2f} | "
                    f"rewμ {metrics.get('reward_mean', 0):+.4f} rewNZ {metrics.get('reward_nonzero', 0):.3f} | "
                    f"scen {metrics.get('scenario_success_rate', 0):.2f}/"
                    f"{metrics.get('scenario_count', 0):.0f} | "
                    f"featσ p/f/g/pw {metrics.get('planet_feat_std', 0):.2f}/"
                    f"{metrics.get('fleet_feat_std', 0):.2f}/"
                    f"{metrics.get('global_feat_std', 0):.2f}/"
                    f"{metrics.get('pairwise_feat_std', 0):.2f} | "
                    f"wnorm roi20/roi50/ec {metrics.get('wnorm_roi20', 0):.3f}/"
                    f"{metrics.get('wnorm_roi50', 0):.3f}/"
                    f"{metrics.get('wnorm_enemy_contest', 0):.3f} "
                    f"val/keep {metrics.get('wnorm_capture_value', 0):.3f}/"
                    f"{metrics.get('wnorm_keepability', 0):.3f} "
                    f"(orig~{metrics.get('wnorm_pw_orig', 0):.2f})"
                    + actcoef + pencoef
                )
                print(
                    f"   p4   | fire/ship tgtσ {metrics.get('fire_target_std', 0):.3f}/"
                    f"{metrics.get('ship_target_std', 0):.3f} | "
                    f"prior_rms {metrics.get('phase4_fire_prior_rms', 0):.3f}/"
                    f"{metrics.get('phase4_ship_prior_rms', 0):.3f} | "
                    f"resid_rms {metrics.get('phase4_fire_resid_rms', 0):.3f}/"
                    f"{metrics.get('phase4_ship_resid_rms', 0):.3f} | "
                    f"ρ {metrics.get('phase4_fire_resid_ratio', 0):.3f}/"
                    f"{metrics.get('phase4_ship_resid_ratio', 0):.3f} | "
                    f"|resid|μ {metrics.get('phase4_fire_resid_abs_mean', 0):.3f}/"
                    f"{metrics.get('phase4_ship_resid_abs_mean', 0):.3f} | "
                    f"flip f/s {metrics.get('phase4_fire_decision_flip', 0):.3f}/"
                    f"{metrics.get('phase4_ship_decision_flip', 0):.3f} | "
                    f"qk f {metrics.get('phase4_fire_q_norm', 0):.2f}/"
                    f"{metrics.get('phase4_fire_k_norm', 0):.2f} "
                    f"s {metrics.get('phase4_ship_q_norm', 0):.2f}/"
                    f"{metrics.get('phase4_ship_k_norm', 0):.2f} | "
                    f"out {metrics.get('phase4_fire_resid_out_norm', 0):.3f}/"
                    f"{metrics.get('phase4_ship_resid_out_norm', 0):.3f}"
                )
                # dm | decisive-mass GAP: are our inflight attacks reaching producer_v2's capture
                # floor? gap DOWN + cross UP = concentrating force toward the decmass target.
                if args.decisive_diag:
                    print(
                        f"   dm   | gap <50/50-100/>100 {metrics.get('dm_gap_e',0):.2f}/"
                        f"{metrics.get('dm_gap_m',0):.2f}/{metrics.get('dm_gap_l',0):.2f} "
                        f"| cross {metrics.get('dm_cross_e',0):.2f}/{metrics.get('dm_cross_m',0):.2f}/"
                        f"{metrics.get('dm_cross_l',0):.2f} | ratio {metrics.get('dm_ratio',0):.2f} "
                        f"overkill {metrics.get('dm_overkill',0):.2f} "
                        f"nearmiss {metrics.get('dm_nearmiss',0):.2f} "
                        f"tgt/step {metrics.get('dm_tgt',0):.2f}"
                    )
                if args.log_timing:
                    # SPS patch A: per-rollout wall-time breakdown (where the per-step tax is).
                    _tt = sum(_t_acc.values()) or 1.0
                    _ext = _t_acc['ext_obs'] + _t_acc['ext_wait'] + _t_acc['ext_conv']
                    print(
                        f"   timing | pf {_t_acc['pf']:.1f} ext_obs {_t_acc['ext_obs']:.1f} "
                        f"ext_wait {_t_acc['ext_wait']:.1f} ext_conv {_t_acc['ext_conv']:.1f} "
                        f"estep {_t_acc['estep']:.1f} upd {_t_acc['upd']:.1f} "
                        f"(sum {_tt:.1f}s, ext {_ext/_tt*100:.0f}%)"
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
                    "train/phase4_residual_lr": metrics.get("phase4_residual_learning_rate", metrics["learning_rate"]),
                    # PPO health
                    "ppo/explained_variance": metrics.get("explained_variance", 0),
                    "feat/wnorm_roi20": metrics.get("wnorm_roi20", 0),
                    "feat/wnorm_roi50": metrics.get("wnorm_roi50", 0),
                    "feat/wnorm_enemy_contest": metrics.get("wnorm_enemy_contest", 0),
                    "feat/wnorm_reachable_enemy": metrics.get("wnorm_reachable_enemy", 0),
                    "feat/wnorm_capture_value": metrics.get("wnorm_capture_value", 0),
                    "feat/wnorm_reactive_roi": metrics.get("wnorm_reactive_roi", 0),
                    "feat/wnorm_friendly_reach": metrics.get("wnorm_friendly_reach", 0),
                    "feat/wnorm_keepability": metrics.get("wnorm_keepability", 0),
                    "feat/wnorm_pw_orig": metrics.get("wnorm_pw_orig", 0),
                    "phase4/fire_target_std": metrics.get("fire_target_std", 0),
                    "phase4/ship_target_std": metrics.get("ship_target_std", 0),
                    "phase4/fire_prior_rms": metrics.get("phase4_fire_prior_rms", 0),
                    "phase4/ship_prior_rms": metrics.get("phase4_ship_prior_rms", 0),
                    "phase4/fire_resid_rms": metrics.get("phase4_fire_resid_rms", 0),
                    "phase4/ship_resid_rms": metrics.get("phase4_ship_resid_rms", 0),
                    "phase4/fire_resid_ratio": metrics.get("phase4_fire_resid_ratio", 0),
                    "phase4/ship_resid_ratio": metrics.get("phase4_ship_resid_ratio", 0),
                    "phase4/fire_resid_abs_mean": metrics.get("phase4_fire_resid_abs_mean", 0),
                    "phase4/ship_resid_abs_mean": metrics.get("phase4_ship_resid_abs_mean", 0),
                    "phase4/fire_decision_flip": metrics.get("phase4_fire_decision_flip", 0),
                    "phase4/ship_decision_flip": metrics.get("phase4_ship_decision_flip", 0),
                    "phase4/fire_q_norm": metrics.get("phase4_fire_q_norm", 0),
                    "phase4/fire_k_norm": metrics.get("phase4_fire_k_norm", 0),
                    "phase4/ship_q_norm": metrics.get("phase4_ship_q_norm", 0),
                    "phase4/ship_k_norm": metrics.get("phase4_ship_k_norm", 0),
                    "phase4/fire_resid_out_norm": metrics.get("phase4_fire_resid_out_norm", 0),
                    "phase4/ship_resid_out_norm": metrics.get("phase4_ship_resid_out_norm", 0),
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
                    **{f"hoard/{k}": v for k, v in ms_metrics.items()},
                    "policy/reinforce_rate": metrics.get("reinforce_rate", 0),
                    "policy/target_share_neutral": metrics.get("target_share_neutral", 0),
                    "policy/target_share_enemy": metrics.get("target_share_enemy", 0),
                    # Entropy
                    "entropy/fire": metrics.get("fire_entropy", 0),
                    "entropy/ship": metrics.get("ship_entropy", 0),
                    "entropy/target": metrics.get("target_entropy", 0),
                    # Value / return stats
                    "value/mean": metrics.get("old_value_mean", 0),
                    "value/std": metrics.get("old_value_std", 0),
                    "value/return_mean": metrics.get("return_mean", 0),
                    "value/adv_std": metrics.get("adv_std", 0),
                    # Reward stats
                    "reward/mean": metrics.get("reward_mean", 0),
                    "reward/nonzero_frac": metrics.get("reward_nonzero", 0),
                    "scenario/success_rate": metrics.get("scenario_success_rate", 0),
                    "scenario/count": metrics.get("scenario_count", 0),
                    # IL (zero when not active)
                    "il/kl": metrics.get("il_kl", 0),
                    "il/coef": metrics.get("il_coef", 0),
                    # decisive-mass GAP diagnostic (force concentration toward producer_v2's floor)
                    "dm/gap": metrics.get("dm_gap", 0),
                    "dm/cross": metrics.get("dm_cross", 0),
                    "dm/ratio": metrics.get("dm_ratio", 0),
                    "dm/overkill": metrics.get("dm_overkill", 0),
                    "dm/nearmiss": metrics.get("dm_nearmiss", 0),
                    "dm/tgt_per_step": metrics.get("dm_tgt", 0),
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
                  "owned={:.1f} garrfrac@50={:.2f} shipspp@50={:.0f} fire_rate={:.3f} Hfire={:.3f} reinf={:.3f}".format(
                      total_env_steps,
                      metrics.get("explained_variance", 0), metrics.get("approx_kl", 0),
                      metrics.get("clip_frac", 0), metrics.get("fire_fraction", 0),
                      metrics.get("owned_planets", 0),
                      metrics.get("garr_frac@50", 0), metrics.get("ships_per_planet@50", 0),
                      metrics.get("fire_rate_overall", 0),
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
            if args.pool_pinned_fraction > 0.0:
                ramp = (min(total_env_steps / args.pool_hard_ramp_steps, 1.0)
                        if args.pool_hard_ramp_steps > 0 else 1.0)
                print(f"  pool hard-ramp {ramp:.2f} ({total_env_steps:,}/{args.pool_hard_ramp_steps:,}): "
                      f"LIVE pinned_frac={args.pool_pinned_fraction * ramp:.3f} "
                      f"external_frac={args.pool_external_fraction * ramp:.3f} of pool slice "
                      f"(target {args.pool_pinned_fraction:.3f}/{args.pool_external_fraction:.3f})")
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
    parser.add_argument("--phase4-residual-lr-mult", type=float, default=1.0,
                        help="Multiplier applied only to Phase 4 residual params "
                             "(fire_q/k/scorer, ship_q/k/scorer). 1.0 = legacy single-LR behavior.")
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
    parser.add_argument("--phase4-residual-init-std", type=float, default=None,
                        help="Stddev for Phase 4 residual output-layer init. "
                             "0.0 = exact parity; small nonzero values let the "
                             "per-target residual path affect decisions sooner.")
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
    parser.add_argument("--critic-warmup-ev", type=float, default=None,
                        help="Critic-only warmup: before PPO, freeze the trunk + policy "
                             "heads and train ONLY the value head until explained-variance "
                             "reaches this (e.g. 0.8), so PPO never trusts a random critic and "
                             "unlearns a BC warmstart. 0/unset = disabled. Self-skips on a "
                             "warm-critic resume (EV already high → 0 warmup steps).")
    parser.add_argument("--critic-warmup-max-updates", type=int, default=None,
                        help="Safety cap on critic-warmup rollouts if EV never reaches the threshold.")
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
    parser.add_argument("--reinforce-gate-min-planets", type=int, default=0,
                        help="Reinforcement discipline #3 (empire-size gate): own planets become "
                             "legal reinforce targets only once the player owns >= this many planets; "
                             "below it, attack-only (must expand first). Grounded in top-player "
                             "replays (reinforce_rate ≈0 at 1 planet, ramps with empire size). A pure "
                             "action mask → no Nash risk; makes the early flood impossible by "
                             "construction. 0 = off. Only active with --allow-reinforce.")
    parser.add_argument("--reinforce-forward-only", action="store_true",
                        help="Reinforcement discipline #4 (forward-staging gate): an own reinforce "
                             "target is legal only if it is closer to the nearest enemy planet than "
                             "the launch source, so reinforcement flows rear→front (staging) and a "
                             "safe rear hoard is impossible by construction. Matches the 66-70%% "
                             "forward-staging in top-player replays; removes the costless safe-fire "
                             "outlet that floods symmetric self-play. Enemy/neutral targets "
                             "unconstrained. Only active with --allow-reinforce.")
    parser.add_argument("--reverse-edge-cooldown", type=int, default=0,
                        help="Reinforce discipline (reverse-edge cooldown): after an own-target "
                             "reinforce A->B, the reverse B->A reinforce is illegal for this many "
                             "steps — blocks the A->B->A ping-pong (rank1 recip<=3st <0.01 vs our "
                             "0.06-0.10). Ownership-change & episode resets clear stale edges so a "
                             "recaptured planet is never mis-blocked. Pure mask, internalised at "
                             "inference. 0 = off. Try 3. Enemy/neutral untouched; needs --allow-reinforce.")
    parser.add_argument("--game-phase-features", action="store_true",
                        help="Append 4 game-phase global channels (phase one-hot early<50/mid/late + "
                             "normalized steps-to-next-comet-spawn); global feature dim 11->15. "
                             "Breaks 11-global checkpoint loading → from-scratch runs only (Stage B).")
    parser.add_argument("--ship-overflow-mode", choices=["drop", "clamp"], default="clamp",
                        help="What torch_env does when a launch asks for more ships than the source "
                             "garrison: 'clamp' (DEFAULT — send min(ask,src), the whole garrison, "
                             "MATCHING EVAL's _ship_bin_to_count) or 'drop' (legacy — void the whole "
                             "launch, the train/eval-mismatch behavior). ~35%% of attacks overask; "
                             "default flipped to clamp 2026-06-15 (clamp is correct; pass 'drop' only "
                             "to reproduce the legacy bug or as an A/B control).")
    parser.add_argument("--sufficient-commit-factor", type=float, default=0.0,
                        help="SUFFICIENT-COMMIT MASK: veto an ATTACK launch (enemy/neutral target) "
                             "whose ship count <= target's current defense × this factor → fragments "
                             "fired under a target's garrison are impossible by construction, forcing "
                             "concentration (the opening under-commitment fix). 1.0 = strict (need "
                             "strictly more than current defense); 0.6 = relaxed fallback; 0 = off. "
                             "Pure mask (no reward tax → no fire=0 Nash); internalised at inference. "
                             "Independent of --allow-reinforce (acts on attacks, not reinforces).")
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
    parser.add_argument("--prod-share-coef", type=float, default=0.0,
                        help="Unified production-share capture reward coefficient. Rewards ownership "
                             "deltas by planet production share, anchored to capture time: gain pays "
                             "+coef*decay(now)*prod/total, loss pays -coef*decay(capture_time)*prod/total. "
                             "Initial homes are pre-existing state, so holding them pays no dense reward. "
                             "Use as the cleanup replacement for early_capture/expansion/defense shaping.")
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
    parser.add_argument("--consolidation-coef", type=float, default=0.0,
                        help="Force-concentration lever: ONE-TIME bonus when a NET-NEW captured "
                             "planet survives --consolidation-steps (success-gated → prices "
                             "'commit enough to HOLD' without defense_coef's flood). 0 = off. "
                             "Calibrate ≤10-15%% of terminal win (~6-10 sticky caps/game → 0.015-0.02).")
    parser.add_argument("--consolidation-steps", type=int, default=40,
                        help="Steps a captured planet must be held to earn --consolidation-coef "
                             "(autopsy median churn-loss ~20 → 40 = 2x clears the churn band).")
    parser.add_argument("--capture-utility-coef", type=float, default=0.0,
                        help="Capture follow-through reward: ONE-TIME bonus when a net-new captured "
                             "planet proves useful within --capture-utility-window by launching an "
                             "attack from that planet, or by still being one of the holder's top-3 "
                             "frontline planets at window end. Targets the capture-born/idle-safe "
                             "diagnostic: capture -> use/convert tempo, not capture -> sit/peel. "
                             "0 = off. Start small, e.g. 0.03-0.06.")
    parser.add_argument("--capture-utility-window", type=int, default=30,
                        help="Steps after a net-new capture during which it can earn "
                             "--capture-utility-coef. Matches the eval utility<=30 diagnostic.")
    parser.add_argument("--capture-idle-penalty", type=float, default=0.0,
                        help="Optional ONE-TIME penalty at --capture-utility-window for a net-new "
                             "capture that neither launched an attack nor remained frontline. "
                             "0 = no penalty; prefer small values if enabled.")
    parser.add_argument("--decisive-mass-coef", type=float, default=0.0,
                        help="Lever A (force-concentration): ONE-TIME bonus the step our INFLIGHT "
                             "force converging on an ENEMY planet first reaches producer_v2's capture "
                             "floor = garrison + prod*eta + enemy_inbound + beta*rho(eta)*reachable_"
                             "enemy_mass + 1 (eta = MAX arrival ETA of our converging mass; see "
                             "--decisive-mass-beta). Board-grounded, not outcome-tied → injects the "
                             "concentration gradient self-play can't price (we get out-massed ~2.3x, "
                             "planets@50=6 invariant). 0 = off. Start ~0.2; read eval out-massed%% / "
                             "garr@loss-vs-inbound. project_force_concentration_wall.")
    parser.add_argument("--decisive-mass-beta", type=float, default=2.2,
                        help="Weight on producer_v2's reactive-reinforcement floor margin "
                             "(beta*rho(eta)*reachable_enemy_mass). 2.2 = v2-faithful (planner-"
                             "conservative). LOWER it (e.g. 0.5-1.0) if `decis` stays ~0 on the "
                             "resumed policy — a high beta makes the floor strict → sparse signal.")
    parser.add_argument("--decisive-diag", action=argparse.BooleanOptionalAction, default=True,
                        help="Compute the dm_* GAP diagnostic every step using the EXACT decisive-"
                             "mass reward floor, even when --decisive-mass-coef=0. The `dm` diag "
                             "line reports, split by phase: dm_gap (mean max(0,floor-mass)/floor — "
                             "DOWN if attacks concentrate), dm_cross (fraction reaching the floor — "
                             "UP), plus ratio/overkill/near-miss/targets-per-step. Tells whether PPO "
                             "is moving toward the decmass target vs only improving adjacent "
                             "competence. project_force_concentration_wall. NOTE: runs "
                             "_decisive_mass_fields() EVERY step (a P×P enemy-pressure pass + fleet-"
                             "target resolution) — benchmark the SPS delta on the GPU box; use "
                             "--no-decisive-diag for max-throughput production once the measurement "
                             "need is satisfied.")
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
    parser.add_argument("--neutral-garrison-scale", type=float, default=1.0,
                        help="Board-curriculum: multiply neutral planet ships by this factor at "
                             "reset, symmetrically (both players face the same board). >1.0 makes "
                             "captures expensive → single-source can't capture early → must "
                             "aggregate multiple sources (concentration). Training-only; eval/LB "
                             "use default boards (1.0). 1.0 = off. Suggested: 3.0.")
    parser.add_argument("--scenario-curriculum",
                        choices=["off", "mixed", "agg_attack", "stage_attack", "hold_under_peel"],
                        default="off",
                        help="Tiny reset-state curriculum where the focal tactic is required for "
                             "a short terminal win. agg_attack requires multi-source capture of a "
                             "neutral; stage_attack requires topping up an existing friendly inbound; "
                             "hold_under_peel requires reinforcing a thin owned planet under inbound "
                             "enemy peel. mixed samples all three. Off by default.")
    parser.add_argument("--scenario-fraction", type=float, default=0.0,
                        help="Fraction of resets replaced by --scenario-curriculum boards. Keep this "
                             "small (e.g. 0.05-0.20) when mixing into normal self-play; 0 = off.")
    parser.add_argument("--scenario-deadline", type=int, default=20,
                        help="Scenario terminal deadline in env steps. Attack/stage scenarios must "
                             "capture the focal target before this step; hold scenarios must retain "
                             "the focal planet through this step.")
    parser.add_argument("--self-boost-planets", type=int, default=0,
                        help="Handicapped-real-planner curriculum: grant OUR seat this many extra "
                             "starting planets in POOL envs at step 0, tapering to 0 over "
                             "--self-boost-ramp-steps. Makes a strong pool planner (deb) beatable so "
                             "RL gets a win-gradient for holding, then weans off the head-start. 0 = off.")
    parser.add_argument("--self-boost-ramp-steps", type=int, default=5000000,
                        help="Steps over which --self-boost-planets tapers linearly to 0.")
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
                             "pair_kv.weight[:, 108:111] and target_scorer ROI columns "
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
                             "go to external opponents (split uniformly among them by default — "
                             "see --pfsp-externals); the remaining 60%% is governed by PFSP over "
                             "self-checkpoints. Use for targeted runs where you need sustained "
                             "pressure from a specific opponent regardless of rollout win-rate.")
    parser.add_argument("--pfsp-externals", action=argparse.BooleanOptionalAction, default=False,
                        help="Sample the EXTERNAL slice PFSP-weighted (by (1-ema_wr)^alpha) instead "
                             "of UNIFORM, so a multi-rung league (e.g. h10/h12/h14) concentrates "
                             "games on the rungs we lose to most / the matched-difficulty band "
                             "instead of 1/N each. Default OFF = legacy uniform (the pool's pfsp_w "
                             "column is display-only for externals when off). Flag-overridden on "
                             "resume like --pool-external-fraction.")
    parser.add_argument("--pool-pinned-fraction", type=float, default=0.0,
                        help="TARGET fraction of pool samples reserved for PINNED RL champions "
                             "(--pool-seed-rl, e.g. rev38), pulling them OUT of PFSP into a fixed "
                             "slice. >0 engages 3-way ramp mode: external / pinned-RL / PFSP-over-"
                             "organic-selves. Necessary because PFSP up-samples opponents you lose "
                             "to, so a weak from-scratch policy would see MORE rev38 early "
                             "(backwards). 0 = legacy (pins compete inside PFSP). With "
                             "--pool-fraction 0.75 a target of 0.267 ≈ 0.20 of TOTAL games.")
    parser.add_argument("--pool-hard-ramp-steps", type=int, default=0,
                        help="Steps over which the hard opponents (pinned RL + external peeler) "
                             "ramp in 0→target, linearly. Both --pool-pinned-fraction and "
                             "--pool-external-fraction are scaled by min(step/ramp,1). Eases the "
                             "unbeatable opponents in so a weak from-scratch BC isn't win-starved "
                             "(true-zero start = pure self-play early). 0 = no ramp (jump to "
                             "target). Only active when --pool-pinned-fraction > 0.")
    parser.add_argument("--preseed-pool", type=str, default="",
                        help="Directory of .pt checkpoints to preseed the pool "
                             "with as 'self' members. Step is parsed from filename "
                             "(e.g. torch_step_5013504.pt -> step=5013504). Useful "
                             "for diluting the heuristic-share early in training "
                             "and for resuming pool diversity across runs.")
    parser.add_argument("--pool-seed-rl", type=str, default="",
                        help="Comma-separated .pt checkpoints to PIN into the pool as fixed RL "
                             "champion opponents (e.g. our rev38/rev53b). Run via the GPU 'self' "
                             "path (fast, sim-gap-immune) and NEVER FIFO-evicted, unlike --preseed-pool. "
                             "Must match the current model architecture (pairwise dim etc.).")
    parser.add_argument("--external-opponents", type=str, default="",
                        help="Comma-separated paths to .py heuristic agents (e.g. "
                             "'opponents/candidate_suneet_lb1200.py,opponents/candidate_zach_public.py'). "
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
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging.")
    parser.add_argument("--log-timing", action="store_true",
                        help="Print per-rollout timing breakdowns in the console log. "
                             "Off by default; enable only for profiling.")
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
