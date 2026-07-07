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


def compute_diagnostics(metrics, *, train_mask, env, model, args, flat, flat_adv, flat_ret):
    """Populate `metrics` in place with per-rollout behavioural + PPO-health diagnostics.

    Read-only w.r.t. training state — every value is derived from the just-finished
    rollout's env accumulators, the flattened batch (`flat`/`flat_adv`/`flat_ret`), and
    the current model weights. W&B logs all keys; the console prints a curated subset.

    `train_mask` is storage["train_mask"] (T, N, P): its per-slot time-fraction weights the
    env's rollout-SUMMED accumulators. Under per-episode assignment a slot's learning-ness can
    flip mid-rollout (env resets pool<->self / seat), so weighting by the FRACTION of steps
    each slot was the learning policy is ≈ correct vs the old t=0 snapshot (which counted a
    flipped slot's whole-rollout total). Still approximate: assumes a slot's counts are roughly
    uniform over the rollout (the env totals aren't split by when they happened).
    """
    train_frac = train_mask.float().mean(0).to(env.device)   # (N, P) in [0,1]
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
    if args.staging_shaping_coef != 0.0 and getattr(env, "_staging_phi_n", 0) > 0:
        metrics["staging_phi"] = env._staging_phi_acc / env._staging_phi_n
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
    print(f"Batch per update: {args.num_envs * args.rollout_steps * args.num_players}  "
          f"(T={args.rollout_steps} x N={args.num_envs} x P={args.num_players} players)")

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
    if args.entropy_coef_target is not None:
        cfg.ppo.entropy_coef_target = args.entropy_coef_target
    if args.entropy_coef_ships is not None:
        cfg.ppo.entropy_coef_ships = args.entropy_coef_ships
    if args.max_grad_norm is not None:
        cfg.ppo.max_grad_norm = args.max_grad_norm
    if args.gae_lambda is not None:
        cfg.ppo.gae_lambda = args.gae_lambda
    if args.critic_warmup_ev is not None:
        cfg.ppo.critic_warmup_ev = args.critic_warmup_ev
    if args.critic_warmup_max_updates is not None:
        cfg.ppo.critic_warmup_max_updates = args.critic_warmup_max_updates
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
    print(f"Expansion coeff: {args.expansion_coef}")
    print(f"Early capture coeff: {args.early_capture_coef} (decay over {args.early_capture_steps} steps)")
    if args.early_capture_anneal_frac > 0.0:
        print(f"Early capture ANNEAL: cosine {args.early_capture_coef}→0 over "
              f"{args.early_capture_anneal_frac * args.total_steps:,.0f} steps "
              f"(frac {args.early_capture_anneal_frac}), then 0")
    print(f"First Strike: {args.first_strike_mult}x for t<{args.first_strike_steps} steps" if args.first_strike_steps > 0 else "First Strike: off")

    # Honor model-config fields saved in the checkpoint (num_ship_bins,
    # ship_bin_mode) BEFORE creating env or model.
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
        if "phase4_residual_init_std" in ckpt_cfg:
            cfg.model.phase4_residual_init_std = float(ckpt_cfg["phase4_residual_init_std"])
        # Blessed feature config guard (2026-07 cleanup): feature semantics are hard-coded
        # (game-phase 15-global ON, precise pressure resolver ON, friendly roi-deflation ON,
        # enemy-deflate/zero-roi/surface-threat REMOVED). A checkpoint trained under different
        # semantics cannot be resumed here — use the pre-cleanup git tag for those.
        _blessed = {"game_phase_features": True, "pressure_precise_resolver": True,
                    "roi_enemy_deflate": False, "zero_roi_channels": False,
                    "threat_eta_surface": False}
        _mismatch = {k: ckpt_cfg.get(k, False) for k, want in _blessed.items()
                     if bool(ckpt_cfg.get(k, False)) != want}
        if _mismatch:
            raise RuntimeError(
                f"Checkpoint feature semantics {_mismatch} do not match the blessed config "
                f"{_blessed}. This checkpoint predates the 2026-07 cleanup — resume it from "
                f"the pre-cleanup git tag (pre-cleanup-2026-07) instead.")
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
    # PROVENANCE only (eval always clamps via _ship_bin_to_count, so this doesn't change the eval
    # contract) — but record how the ckpt was trained (drop vs clamp) so it's never ambiguous.
    cfg.model.ship_overflow_mode = args.ship_overflow_mode
    env = VecTorchEnv(num_envs=args.num_envs, num_players=args.num_players,
                      device=device, episode_steps=500,
                      ship_bin_mode=cfg.model.ship_bin_mode,
                      ship_overflow_mode=args.ship_overflow_mode,
                      action_decode=args.action_decode,
                      allow_reinforce=args.allow_reinforce,
                      reinforce_garrison_floor=args.reinforce_garrison_floor,
                      reinforce_cost=args.reinforce_cost,
                      reinforce_gate_min_planets=args.reinforce_gate_min_planets,
                      reinforce_forward_only=args.reinforce_forward_only,
                      reverse_edge_cooldown=args.reverse_edge_cooldown,
                      sufficient_commit_factor=args.sufficient_commit_factor,
                      win_margin_coeff=args.win_margin_coeff,
                      expansion_coef=args.expansion_coef,
                      early_capture_coef=args.early_capture_coef,
                      early_capture_steps=args.early_capture_steps,
                      first_strike_steps=args.first_strike_steps,
                      first_strike_mult=args.first_strike_mult,
                      staging_shaping_coef=args.staging_shaping_coef,
                      staging_topk=args.staging_topk,
                      staging_gamma=cfg.ppo.gamma,
                      enable_comets=not args.disable_comets,
                      fleet_target_refresh_every=args.fleet_target_refresh)
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

    learner = PPOLearner(model, cfg, device=device)

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
                    "win_margin_coeff": args.win_margin_coeff,
                    "action_decode": args.action_decode,
                    "resume": args.resume or "",
                    "ship_bin_mode": cfg.model.ship_bin_mode,
                    "feature_config": "blessed-2026-07",  # game-phase+resolver ON, deflate variants removed
                    "staging_shaping_coef": args.staging_shaping_coef,
                    "staging_topk": args.staging_topk,
                    "entropy_coef_fire": args.entropy_coef_fire,
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
    P = args.num_players  # num players — every seat's transitions are collected for PPO
    # 4p pool = HOMOGENEOUS self-snapshots: a pool env puts the learner in one seat and fills the
    # other P-1 seats with ONE sampled member (the per-seat override + train-mask machinery below
    # is generalized to all non-learner seats). External heuristics in 4p are NOT validated.
    if P > 2 and args.pool_external_fraction > 0:
        raise NotImplementedError(
            "4p with EXTERNAL pool members is not validated — use --pool-mode self "
            "(self-snapshots fill the 3 non-learner seats; that path IS wired/tested).")

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
        # stabilise).
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
            m.load_state_dict(member.state_dict, strict=False)
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
                    fire_mask=sub["fire_mask"],
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
        return True, member, rng.randint(0, P - 1)

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
        # see the step loop). N_pool is just a diagnostic count, not a slice.
        N_pool = sum(env_is_pool)
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

        # Hoard milestones (player 0): snapshot garrison/planets when an env is AT
        # episode-step 16/32/50/100. Controlled-time + scale-free → replaces the
        # end-skewed avgfleet/p90. Accumulated on-device, synced once after the rollout.
        _MS = (16, 32, 50, 100)
        ms_garr = {m: torch.zeros((), device=env.device) for m in _MS}
        ms_plan = {m: torch.zeros((), device=env.device) for m in _MS}
        ms_n    = {m: torch.zeros((), device=env.device) for m in _MS}

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
                    # Homogeneous fill: the sampled member plays EVERY non-learner seat
                    # (P-1 seats in 4p; the single opponent seat in 2p).
                    for os_ in range(P):
                        if os_ == env_seat[e]:
                            continue
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
            # Hoard milestones: when an env is at episode-step 16/32/50/100, accumulate
            # player-0 garrison (parked) and owned-planet counts for that env.
            ownp = env.planets[:, :, 1].long()                            # (N, P) owner
            mine_p = (ownp == 0) & env.planet_alive                       # player-0 planets
            garr_p0 = (env.planets[:, :, 5] * mine_p.float()).sum(dim=1)  # (N,) parked ships
            plan_p0 = mine_p.float().sum(dim=1)                           # (N,) planets owned
            for _m in _MS:
                sel = (env.step_count == _m).float()                     # (N,) at-milestone
                ms_garr[_m] += (garr_p0 * sel).sum()
                ms_plan[_m] += (plan_p0 * sel).sum()
                ms_n[_m]    += sel.sum()
            # rewards: (N, P); done: (N,) shared across players.
            storage["rewards"][t].copy_(rewards[:, :P], non_blocking=True)

            # done is shared across all seats of an env (game ends for everyone at once).
            storage["dones"][t].copy_(done.unsqueeze(1).expand(N, P), non_blocking=True)

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
                        # Learner vs the snapshot field (member fills the other P-1 seats).
                        # Result is from the LEARNER's perspective: learner won → "win";
                        # a snapshot seat won → "loss"; nobody won (timeout draw) → "draw".
                        cur_seat = env_seat[e]
                        learner_won = bool(last_wins[e, cur_seat])
                        field_won = bool(last_wins[e].any())
                        if learner_won:   pool.record_result(env_member[e], "win")
                        elif field_won:   pool.record_result(env_member[e], "loss")
                        else:             pool.record_result(env_member[e], "draw")
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
        #   ships/planet = parked / owned planets         — pile-up per planet
        #   planets     = owned planets                   — expansion trajectory
        # Replaces the end-skewed avgfleet/p90. Reference (Isaiah): ~11-22 ships/planet,
        # planets 2/6/9/10.
        ms_metrics = {}
        for _m in _MS:
            g, pl, nn = ms_garr[_m].item(), ms_plan[_m].item(), ms_n[_m].item()
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

        # Offline Q-head gate (docs/q-head.md step 1): dump ONE real torch_env+GAE
        # batch (returns carry the dense capture/prod-share/staging reward, not a
        # kaggle-env terminal-only MC) and exit before any optimizer step. The
        # q_head_offline_probe then trains a Q-only head on this and measures
        # action-sensitivity. Run self-play (no external pool) — same reward mechanics.
        if getattr(args, "dump_rollout_and_exit", None) and not critic_warmup_active:
            torch.save({k: (v if not isinstance(v, dict)
                            else {kk: vv for kk, vv in v.items()})
                        for k, v in batch.items()}, args.dump_rollout_and_exit)
            print(f"[dump-rollout] saved batch (TN={TN}) -> {args.dump_rollout_and_exit}", flush=True)
            return

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
                                     kl_target=cfg.ppo.kl_target)
        _t_acc["upd"] += time.perf_counter() - _t_upd
        metrics.update(ms_metrics)
        compute_diagnostics(metrics, train_mask=storage["train_mask"], env=env,
                            model=model, args=args, flat=flat,
                            flat_adv=flat_adv, flat_ret=flat_ret)

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
                    f"shipspp@50 {metrics.get('ships_per_planet@50',0):.0f} | "
                    f"overask {metrics.get('overask_rate',0):.2f} "
                    f"(<50/50-100/>100 {metrics.get('overask_e',0):.2f}/"
                    f"{metrics.get('overask_m',0):.2f}/{metrics.get('overask_l',0):.2f}) "
                    f"emit/req {metrics.get('moves_emit_req',0):.2f} satur {metrics.get('fleet_saturation',0):.3f} "
                    f"obstrunc {metrics.get('obs_trunc_rate',0):.3f} "
                    f"(fleet {metrics.get('obs_trunc_fleet_frac',0):.3f} ship {metrics.get('obs_trunc_ship_frac',0):.3f} "
                    f"enemyship {metrics.get('obs_trunc_enemy_ship_frac',0):.3f}) | "
                    f"{reinfstr}"
                    f"H_ship {metrics.get('ship_entropy', 0):.2f} "
                    f"H_tgt {metrics.get('target_entropy', 0):.2f} | "
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
                    f"val/keep {metrics.get('wnorm_capture_value', 0):.3f}/"
                    f"{metrics.get('wnorm_keepability', 0):.3f} "
                    f"(orig~{metrics.get('wnorm_pw_orig', 0):.2f})"
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
                    # IL (zero when not active)
                    # PBRS staging potential (is the agent staging toward neutrals?)
                    "staging/phi": metrics.get("staging_phi", 0),
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
                  "owned={:.1f} shipspp@50={:.0f} fire_rate={:.3f} Hfire={:.3f} reinf={:.3f}".format(
                      total_env_steps,
                      metrics.get("explained_variance", 0), metrics.get("approx_kl", 0),
                      metrics.get("clip_frac", 0), metrics.get("fire_fraction", 0),
                      metrics.get("owned_planets", 0),
                      metrics.get("ships_per_planet@50", 0),
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
    parser.add_argument("--dump-rollout-and-exit", type=str, default=None,
                        help="Save ONE real torch_env+GAE rollout batch to this path and exit "
                             "before any optimizer step (offline Q-head gate, docs/q-head.md).")
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
    parser.add_argument("--critic-warmup-ev", type=float, default=None,
                        help="Critic-only warmup: before PPO, freeze the trunk + policy "
                             "heads and train ONLY the value head until explained-variance "
                             "reaches this (e.g. 0.8), so PPO never trusts a random critic and "
                             "unlearns a BC warmstart. 0/unset = disabled. Self-skips on a "
                             "warm-critic resume (EV already high → 0 warmup steps).")
    parser.add_argument("--critic-warmup-max-updates", type=int, default=None,
                        help="Safety cap on critic-warmup rollouts if EV never reaches the threshold.")
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
    parser.add_argument("--num-players", type=int, choices=[2, 4], default=2,
                        help="Players per game. 4 = FFA self-play (every seat is the learning "
                             "policy; --pool-fraction must be 0 — external 4p pool not yet wired).")
    parser.add_argument("--win-margin-coeff", type=float, default=0.0,
                        help="Terminal bonus coefficient α: winner gets +1 + α*(my_score/total_score). "
                             "0 = pure ±1 reward (default). Suggested start: 0.5.")
    parser.add_argument("--expansion-coef", type=float, default=0.0,
                        help="Potential-based shaping on owned-production lead "
                             "(planet/economy race). Passive play nets ~0 (production "
                             "only changes on capture). 0 = off. rev14 expansion fix; "
                             "suggested start: 0.01.")
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
                             "stays 0. Cosine holds near full "
                             "early (bootstrap) and fades fastest mid-run. Removes the capture-shaping "
                             "crutch once the pool can sustain aggression. 0 = off (constant coef).")
    parser.add_argument("--first-strike-steps", type=int, default=0,
                        help="Apply first_strike_mult to capture reward for t < N steps. "
                             "Breaks opening paralysis by making early captures more lucrative. "
                             "Suggested: 50. 0 = off.")
    parser.add_argument("--first-strike-mult", type=float, default=2.0,
                        help="Multiplier applied to capture reward for t < --first-strike-steps. "
                             "Default 2.0 (doubles the capture reward in the opening).")
    parser.add_argument("--staging-shaping-coef", type=float, default=0.0,
                        help="PBRS staging shaping (project_undermass_by_choice): potential-based "
                             "reward r += coef*(gamma*Phi(s') - Phi(s)), Phi = top-k Σ min(1, our_"
                             "inflight/capture_floor) over NEUTRAL targets. Injects the DIRECTED "
                             "gradient the idle fire head lacks (A≈0 on spare-fire), telescoping so "
                             "spray-safe. Neutral-ONLY (enemy = decmass, which failed). 0 = off. "
                             "Start ~0.2 (pre-test calibrated). Tripwire: fire_frac/launch_rate up "
                             "without caps up = spray.")
    parser.add_argument("--staging-topk", type=int, default=2,
                        help="k for the staging potential (top-k neutral targets summed). k=2 keeps "
                             "serial expansion-breadth while bounding simultaneous spread.")
    parser.add_argument("--disable-comets", action="store_true",
                        help="Train WITHOUT comets (no spawns/schedule — SPS lever, 2026-07-05). "
                             "The real kaggle game has comets; eval/export always keep them, so "
                             "this trades a training-distribution gap for throughput.")
    parser.add_argument("--fleet-target-refresh", type=int, default=4,
                        help="Re-resolve ALL cached fleet targets every K ticks (staleness bound "
                             "for the launch-time target cache — SPS lever, 2026-07-05). Accuracy "
                             "vs true collision: 1 (~fresh) 95.6%%, 2 95.0%%, 4 93.7%%, 8 91.9%%, "
                             "0 (launch-only; comet staleness never fixed) 79.2%%.")
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
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Weights & Biases logging (default: on). Use --no-wandb to disable.")
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
