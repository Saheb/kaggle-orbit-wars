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
from pathlib import Path

import numpy as np
import torch

from config import Config
from model import EntityTransformer, count_params
from opponent_pool import OpponentPool, PoolMember
from ppo import PPOLearner
from torch_env import VecTorchEnv, MAX_OWNED, NUM_ANGLE_BINS, NUM_SHIP_BINS


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def sample_action_batched(outputs: dict, fire_mask: torch.Tensor,
                          angle_mask: torch.Tensor):
    """Sample fire/angle/ship for a batch of envs. Returns actions + log_probs."""
    fire_logits  = outputs["fire_logits"].masked_fill(~fire_mask, -1e9)
    angle_logits = outputs["angle_logits"].masked_fill(~angle_mask, -1e9)
    ship_logits  = outputs["ship_logits"]

    fire_dist  = torch.distributions.Bernoulli(logits=fire_logits)
    angle_dist = torch.distributions.Categorical(logits=angle_logits)
    ship_dist  = torch.distributions.Categorical(logits=ship_logits)

    fire_a  = fire_dist.sample()                 # (N, MAX_OWNED)
    angle_a = angle_dist.sample()                # (N, MAX_OWNED)
    ship_a  = ship_dist.sample()                 # (N, MAX_OWNED)

    # log_probs only for valid slots / fired actions
    slot_valid = fire_mask  # already equivalent for our usage
    lp_fire  = fire_dist.log_prob(fire_a) * slot_valid.float()
    fired = (fire_a > 0.5).float() * slot_valid.float()
    lp_angle = angle_dist.log_prob(angle_a) * fired
    lp_ship  = ship_dist.log_prob(ship_a)  * fired

    return fire_a.long(), angle_a, ship_a, lp_fire, lp_angle, lp_ship


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


# ----------------------------------------------------------------------------
# Periodic eval — current policy vs frozen baseline
# ----------------------------------------------------------------------------

def _act_deterministic(model, env, player):
    """Sample action from current policy (used during eval rollouts)."""
    feats = env.get_features(player)
    with torch.no_grad():
        out = model(
            feats["planet_features"], feats["fleet_features"], feats["global_features"],
            feats["planet_mask"], feats["fleet_mask"],
            fire_mask=feats["fire_mask"], angle_mask=feats["angle_mask"],
            slot_valid=feats["slot_valid"], owned_indices=feats["owned_indices"],
            owned_count=feats["owned_count"],
        )
    fl = out["fire_logits"].masked_fill(~feats["fire_mask"], -1e9)
    al = out["angle_logits"].masked_fill(~feats["angle_mask"], -1e9)
    f = torch.distributions.Bernoulli(logits=fl).sample().long()
    a = torch.distributions.Categorical(logits=al).sample()
    s = torch.distributions.Categorical(logits=out["ship_logits"]).sample()
    return torch.stack([f, a, s], dim=-1)


def eval_vs_baseline(current_model, baseline_model, device, n_games=64, episode_steps=500):
    """Play current vs baseline model. Returns (win_rate, n_completed)."""
    from torch_env import VecTorchEnv
    current_model.eval()
    baseline_model.eval()
    env = VecTorchEnv(num_envs=n_games, num_players=2, device=device, episode_steps=episode_steps)
    env.reset(seeds=list(range(10000, 10000 + n_games)))
    current_wins = 0
    baseline_wins = 0
    done_count = 0
    for _ in range(episode_steps + 50):
        a0 = _act_deterministic(current_model, env, 0)
        a1 = _act_deterministic(baseline_model, env, 1)
        _, rewards, done = env.step({0: a0, 1: a1})
        for i in torch.where(done)[0].tolist():
            r0, r1 = rewards[i, 0].item(), rewards[i, 1].item()
            if r0 > r1: current_wins += 1
            elif r1 > r0: baseline_wins += 1
            done_count += 1
        if done_count >= n_games:
            break
    win_rate = current_wins / max(done_count, 1)
    return win_rate, done_count


# ----------------------------------------------------------------------------
# Eval vs heuristic agent (e.g. ../main.py). The heuristic runs in Python
# per-env so this is slow — keep n_games small (16-32).
# ----------------------------------------------------------------------------

def _load_heuristic(agent_path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("heuristic_opp", agent_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


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

    # Gather planet ids per (env, slot)
    gather_idx = owned_idx.unsqueeze(-1).expand(-1, -1, 7).cpu()
    planets_cpu = env.planets.cpu()
    src = planets_cpu.gather(1, gather_idx)        # (N, MAX_OWNED, 7)
    sv_cpu = slot_valid.cpu()

    for e in range(N):
        moves = moves_per_env[e]
        if not moves:
            continue
        # Map planet_id → slot index for this env
        pid_to_slot = {}
        for k in range(MAX_OWNED):
            if sv_cpu[e, k].item():
                pid_to_slot[int(src[e, k, 0].item())] = k
        for mv in moves:
            if not isinstance(mv, (list, tuple)) or len(mv) < 3:
                continue
            from_pid = int(mv[0])
            slot = pid_to_slot.get(from_pid)
            if slot is None:
                continue
            ang = float(mv[1]) % (2 * _math.pi)
            ab = int(ang / ANGLE_BIN_WIDTH) % NUM_ANGLE_BINS
            ships = int(mv[2])
            # Find nearest ship-bin
            best, best_diff = 0, 10**9
            for b, c in enumerate(SHIP_COUNTS):
                if abs(c - ships) < best_diff:
                    best_diff, best = abs(c - ships), b
            fire[e, slot] = 1
            angle_bin[e, slot] = ab
            ship_bin[e, slot] = best
    return torch.stack([fire, angle_bin, ship_bin], dim=-1).to(device)


def eval_vs_heuristic(current_model, heuristic_path, device, n_games=16, episode_steps=500):
    """Play current model (player 0) vs Python heuristic (player 1).

    Slow (heuristic runs in Python per env per step) so n_games defaults to 16.
    Returns (win_rate, n_completed).
    """
    from torch_env import VecTorchEnv, to_legacy_obs
    current_model.eval()
    agent_fn = _load_heuristic(heuristic_path)

    env = VecTorchEnv(num_envs=n_games, num_players=2, device=device, episode_steps=episode_steps)
    env.reset(seeds=list(range(20000, 20000 + n_games)))
    current_wins = 0
    heuristic_wins = 0
    done_count = 0
    for _ in range(episode_steps + 50):
        # Player 0: our model
        a0 = _act_deterministic(current_model, env, 0)
        # Player 1: heuristic — call per env
        moves_per_env = []
        for e in range(n_games):
            obs = to_legacy_obs(env, env_idx=e, player=1)
            try:
                moves = agent_fn(obs) or []
            except Exception:
                moves = []
            moves_per_env.append(moves)
        a1 = _heuristic_moves_to_action_tensor(moves_per_env, env, 1, device)

        _, rewards, done = env.step({0: a0, 1: a1})
        for i in torch.where(done)[0].tolist():
            r0, r1 = rewards[i, 0].item(), rewards[i, 1].item()
            if r0 > r1: current_wins += 1
            elif r1 > r0: heuristic_wins += 1
            done_count += 1
        if done_count >= n_games:
            break
    win_rate = current_wins / max(done_count, 1)
    return win_rate, done_count


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
    print(f"PPO config: lr={cfg.ppo.learning_rate}, ppo_epochs={cfg.ppo.ppo_epochs}, "
          f"num_minibatches={cfg.ppo.num_minibatches}, kl_target={cfg.ppo.kl_target}")

    env = VecTorchEnv(num_envs=args.num_envs, num_players=2,
                      device=device, episode_steps=500)
    env.reset(seeds=[args.seed + i for i in range(args.num_envs)])

    model = EntityTransformer(cfg.model).to(device)
    print(f"Model params: {count_params(model):,}")
    if args.resume:
        sd = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model" in sd: sd = sd["model"]
        model.load_state_dict(sd)
        print(f"Resumed from {args.resume}")

    learner = PPOLearner(model, cfg, device=device)

    # LR scheduler: warmup + cosine decay over total updates.
    # On --resume, skip warmup by default (the model is already trained — no
    # reason to ramp LR from 0). User can force warmup with --with-warmup.
    updates_per_batch = cfg.ppo.ppo_epochs * cfg.ppo.num_minibatches
    total_updates = (args.total_steps // (args.num_envs * args.rollout_steps)) * updates_per_batch
    skip_warmup = (args.resume and not args.with_warmup) or args.skip_warmup
    warmup = 0 if skip_warmup else cfg.ppo.lr_warmup_steps
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(learner.optimizer, lr_lambda)
    print(f"LR schedule: warmup={warmup} updates (skip_warmup={skip_warmup}), "
          f"total_updates={total_updates}, peak_lr={cfg.ppo.learning_rate}")

    # Freeze a snapshot of the initial policy for periodic eval.
    # As training proceeds, current vs frozen-initial win rate should rise.
    baseline_model = copy.deepcopy(model).to(device)
    baseline_model.eval()
    print("Baseline policy snapshotted for periodic eval.\n")

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
            )
            # Seed pool with the starting weights so it isn't empty on iteration 1
            pool.add_self_checkpoint(0, model.state_dict())

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
        # Pre-allocate a frozen model for 'self' opponents (loaded with state_dict per rollout)
        pool_opp_model = copy.deepcopy(model).to(device)
        pool_opp_model.eval()
        print(f"Pool initialised: mode={args.pool_mode}, members={len(pool)}, "
              f"fraction={args.pool_fraction}, snapshot_every={args.pool_checkpoint_interval:,} steps\n")
    last_pool_snapshot_step = 0

    total_env_steps = 0
    iter_count = 0
    start = time.perf_counter()
    reward_history = deque(maxlen=200)     # p0 episode rewards
    reward_history_p1 = deque(maxlen=200)  # p1 episode rewards (should mirror p0)
    clipfrac_history = deque(maxlen=50)
    best_avg_reward = float("-inf")
    last_log = start
    last_eval_step = 0
    last_ckpt_step = 0
    eval_history: list[tuple[int, float]] = []   # (step, win_rate vs baseline)
    best_eval_winrate = 0.0
    no_improve_evals = 0

    rollout_T = args.rollout_steps
    N = args.num_envs
    P = 2  # num players — both players' transitions are collected and used for PPO

    # Pre-allocate rollout buffers on CPU to keep GPU memory free for the model.
    # Shape leading dims: (T, N, P) so player 0 and player 1 each contribute a
    # full trajectory per env step. Effective PPO batch size = T*N*P.
    storage_dev = torch.device("cpu")
    storage = {
        "planet_features": torch.zeros(rollout_T, N, P, 48, 18, device=storage_dev),
        "fleet_features":  torch.zeros(rollout_T, N, P, 128, 9, device=storage_dev),
        "global_features": torch.zeros(rollout_T, N, P, 10, device=storage_dev),
        "planet_mask":     torch.zeros(rollout_T, N, P, 48, dtype=torch.bool, device=storage_dev),
        "fleet_mask":      torch.zeros(rollout_T, N, P, 128, dtype=torch.bool, device=storage_dev),
        "fire_mask":       torch.zeros(rollout_T, N, P, MAX_OWNED, dtype=torch.bool, device=storage_dev),
        "angle_mask":      torch.zeros(rollout_T, N, P, MAX_OWNED, NUM_ANGLE_BINS, dtype=torch.bool, device=storage_dev),
        "slot_valid":      torch.zeros(rollout_T, N, P, MAX_OWNED, dtype=torch.bool, device=storage_dev),
        "owned_indices":   torch.zeros(rollout_T, N, P, MAX_OWNED, dtype=torch.long, device=storage_dev),
        "fire_a":     torch.zeros(rollout_T, N, P, MAX_OWNED, dtype=torch.long, device=storage_dev),
        "angle_a":    torch.zeros(rollout_T, N, P, MAX_OWNED, dtype=torch.long, device=storage_dev),
        "ship_a":     torch.zeros(rollout_T, N, P, MAX_OWNED, dtype=torch.long, device=storage_dev),
        "lp_fire":    torch.zeros(rollout_T, N, P, MAX_OWNED, device=storage_dev),
        "lp_angle":   torch.zeros(rollout_T, N, P, MAX_OWNED, device=storage_dev),
        "lp_ship":    torch.zeros(rollout_T, N, P, MAX_OWNED, device=storage_dev),
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
            feats = env.get_features(player, max_planets=48, max_fleets=128)
            with torch.no_grad():
                outs = pool_opp_model(
                    feats["planet_features"], feats["fleet_features"],
                    feats["global_features"], feats["planet_mask"],
                    feats["fleet_mask"],
                    fire_mask=feats["fire_mask"], angle_mask=feats["angle_mask"],
                    slot_valid=feats["slot_valid"], owned_indices=feats["owned_indices"],
                    owned_count=feats["owned_count"],
                )
            fire_a, angle_a, ship_a, *_ = sample_action_batched(
                outs, feats["fire_mask"], feats["angle_mask"]
            )
            return torch.stack([fire_a, angle_a, ship_a], dim=-1)[env_slice]

        if opp.kind == "external_heuristic":
            from torch_env import to_legacy_obs
            start, stop = env_slice.start or 0, env_slice.stop or N
            moves_per_env = []
            for e in range(start, stop):
                obs = to_legacy_obs(env, env_idx=e, player=player)
                try:
                    moves = opp.agent_fn(obs) or []
                except Exception:
                    moves = []
                moves_per_env.append(moves)
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
            return _heuristic_moves_to_action_tensor(moves_per_env, view, player, device)

        raise ValueError(f"unknown opponent kind: {opp.kind}")

    def forward_player(player: int):
        """Run model forward for given player, return outputs + features dict."""
        feats = env.get_features(player, max_planets=48, max_fleets=128)
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
                fire_p, angle_p, ship_p, lpf_p, lpa_p, lps_p = sample_action_batched(
                    outs_p, feats_p["fire_mask"], feats_p["angle_mask"]
                )
                storage["planet_features"][t, :, p].copy_(feats_p["planet_features"], non_blocking=True)
                storage["fleet_features"][t, :, p].copy_(feats_p["fleet_features"], non_blocking=True)
                storage["global_features"][t, :, p].copy_(feats_p["global_features"], non_blocking=True)
                storage["planet_mask"][t, :, p].copy_(feats_p["planet_mask"], non_blocking=True)
                storage["fleet_mask"][t, :, p].copy_(feats_p["fleet_mask"], non_blocking=True)
                storage["fire_mask"][t, :, p].copy_(feats_p["fire_mask"], non_blocking=True)
                storage["angle_mask"][t, :, p].copy_(feats_p["angle_mask"], non_blocking=True)
                storage["slot_valid"][t, :, p].copy_(feats_p["slot_valid"], non_blocking=True)
                storage["owned_indices"][t, :, p].copy_(feats_p["owned_indices"], non_blocking=True)
                storage["fire_a"][t, :, p].copy_(fire_p, non_blocking=True)
                storage["angle_a"][t, :, p].copy_(angle_p, non_blocking=True)
                storage["ship_a"][t, :, p].copy_(ship_p, non_blocking=True)
                storage["lp_fire"][t, :, p].copy_(lpf_p, non_blocking=True)
                storage["lp_angle"][t, :, p].copy_(lpa_p, non_blocking=True)
                storage["lp_ship"][t, :, p].copy_(lps_p, non_blocking=True)
                storage["values"][t, :, p].copy_(outs_p["value"].squeeze(-1), non_blocking=True)
                actions_per_player[p] = torch.stack([fire_p, angle_p, ship_p], dim=-1)

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

        # Build PPOLearner-compatible batch (matches make_batch in self_play.py)
        batch = {
            "planet_features": flat["planet_features"],
            "fleet_features":  flat["fleet_features"],
            "global_features": flat["global_features"],
            "planet_mask":     flat["planet_mask"],
            "fleet_mask":      flat["fleet_mask"],
            "fire_mask":       flat["fire_mask"],
            "angle_mask":      flat["angle_mask"],
            "slot_valid":      flat["slot_valid"],
            "owned_indices":   flat["owned_indices"],
            "owned_count":     flat["slot_valid"].sum(dim=1).tolist(),
            "actions": {
                "fire":  flat["fire_a"],
                "angle": flat["angle_a"],
                "ship":  flat["ship_a"],
            },
            "old_log_probs": {
                "fire":  flat["lp_fire"],
                "angle": flat["lp_angle"],
                "ships": flat["lp_ship"],
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
            minibatches.append(sub)

        # PPO update
        model.train()
        metrics = learner.update(minibatches, scheduler=scheduler,
                                 kl_target=cfg.ppo.kl_target)

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
            print(
                f"iter {iter_count:5d} | steps {total_env_steps:>11,} | "
                f"SPS {sps:>7,.0f} | r_p0 {avg_r:+.3f} r_p1 {avg_r1:+.3f} | "
                f"clip_frac {avg_cf:.3f} | KL {metrics.get('approx_kl', 0):.4f} | "
                f"H_fire {metrics.get('fire_entropy', 0):.3f} "
                f"H_ang {metrics.get('angle_entropy', 0):.2f} "
                f"H_ship {metrics.get('ship_entropy', 0):.2f} | "
                f"V_loss {metrics.get('value_loss', 0):.4f} | "
                f"LR {metrics['learning_rate']:.6f} | "
                f"early_stop={metrics.get('kl_early_stop', 0):.0f}"
            )

        # Checkpoint best by reward
        if len(reward_history) >= 100 and avg_r > best_avg_reward + 0.02:
            best_avg_reward = avg_r
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(learner.state_dict(), "checkpoints/torch_best.pt")
            print(f"  ★ best updated: reward={avg_r:+.3f}")

        # Periodic checkpoint by fixed step interval
        if total_env_steps - last_ckpt_step >= args.checkpoint_interval:
            last_ckpt_step = total_env_steps
            os.makedirs("checkpoints", exist_ok=True)
            path = f"checkpoints/torch_step_{total_env_steps}.pt"
            torch.save(learner.state_dict(), path)
            print(f"  saved {path}")
            # Persist pool alongside the disk checkpoint so spot interrupts
            # don't lose pool diversity. Naming mirrors the checkpoint stem.
            if pool is not None:
                pool_path = f"checkpoints/pool_step_{total_env_steps}.pt"
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

        # Periodic eval vs frozen baseline (and optionally heuristic)
        if args.eval_interval > 0 and total_env_steps - last_eval_step >= args.eval_interval:
            last_eval_step = total_env_steps
            win_rate, ng = eval_vs_baseline(model, baseline_model, device,
                                             n_games=args.eval_games)
            heur_str = ""
            if args.eval_heuristic:
                try:
                    heur_wr, heur_ng = eval_vs_heuristic(model, args.eval_heuristic,
                                                         device, n_games=args.eval_heuristic_games)
                    heur_str = f"  | vs heuristic: {heur_wr:.1%} ({heur_ng} games)"
                except Exception as e:
                    heur_str = f"  | vs heuristic: FAILED ({type(e).__name__}: {str(e)[:80]})"
            eval_history.append((total_env_steps, win_rate))
            improved = win_rate > best_eval_winrate + 0.01
            tag = "★" if improved else " "
            print(f"  {tag} EVAL @ step {total_env_steps:,}: "
                  f"vs initial = {win_rate:.1%} ({ng} games){heur_str} "
                  f"[best={best_eval_winrate:.1%}, no_improve={no_improve_evals}]")
            if improved:
                best_eval_winrate = win_rate
                no_improve_evals = 0
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(learner.state_dict(), "checkpoints/torch_eval_best.pt")
            else:
                no_improve_evals += 1
                if no_improve_evals >= args.early_stop_patience:
                    print(f"\nEarly stop: no eval improvement for "
                          f"{args.early_stop_patience} consecutive checks "
                          f"(best win_rate = {best_eval_winrate:.1%})")
                    break

    elapsed = time.perf_counter() - start
    sps = total_env_steps / elapsed if elapsed > 0 else 0
    print(f"\nTraining complete: {total_env_steps:,} env steps in {elapsed:.0f}s")
    print(f"Final SPS: {sps:,.0f}")
    print(f"Best eval win_rate vs initial: {best_eval_winrate:.1%}")
    if eval_history:
        print("Eval history (step, win_rate):")
        for s, w in eval_history:
            print(f"  {s:>12,}  {w:.1%}")

    # Final checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(learner.state_dict(), f"checkpoints/torch_step_{total_env_steps}_final.pt")

    # Auto-terminate the host (cost control). Set
    # InstanceInitiatedShutdownBehavior=terminate on the EC2 instance and this
    # command will tear down the VM, ending billing.
    if args.terminate_on_done:
        print("\n--terminate-on-done: powering off the instance...")
        os.system("sudo shutdown -h +1 'training complete, auto-terminating'")


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
    parser.add_argument("--eval-interval", type=int, default=5_000_000,
                        help="Run eval vs frozen initial policy every N env steps "
                             "(0 to disable)")
    parser.add_argument("--eval-games", type=int, default=64)
    parser.add_argument("--eval-heuristic", type=str, default="",
                        help="Path to heuristic .py file (e.g. ../main.py). "
                             "Eval against it at every checkpoint (slow, small N).")
    parser.add_argument("--eval-heuristic-games", type=int, default=16)
    parser.add_argument("--early-stop-patience", type=int, default=3,
                        help="Stop if N consecutive evals show no improvement "
                             "(set high to disable)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override PPO learning rate (default: cfg.ppo.learning_rate=3e-4)")
    parser.add_argument("--ppo-epochs", type=int, default=None,
                        help="Override PPO epochs per rollout (default: 4)")
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
    parser.add_argument("--pool-pfsp-alpha", type=float, default=2.0,
                        help="PFSP sampling exponent: weight = (1 - win_rate)^alpha.")
    parser.add_argument("--pool-mastered-threshold", type=float, default=0.9,
                        help="Win-rate above this triggers eviction of an external opponent.")
    parser.add_argument("--pool-mastered-min-games", type=int, default=50,
                        help="Minimum games against an external before mastery-eviction is considered.")
    parser.add_argument("--external-opponents", type=str, default="",
                        help="Comma-separated paths to .py heuristic agents (e.g. "
                             "'../candidate_suneet_lb1200.py,../candidate_zach_public.py'). "
                             "Only used when --pool-mode=mixed.")
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
