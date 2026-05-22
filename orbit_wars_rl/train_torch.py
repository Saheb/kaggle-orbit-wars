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
import os
import time
from collections import deque

import numpy as np
import torch

from config import Config
from model import EntityTransformer, count_params
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
# Training loop
# ----------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    print(f"Training on device: {device}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Rollout steps: {args.rollout_steps}")
    print(f"Batch per update: {args.num_envs * args.rollout_steps}")

    torch.manual_seed(args.seed)

    cfg = Config()
    cfg.ppo.total_env_steps = args.total_steps
    cfg.device = args.device
    cfg.ppo.num_minibatches = args.num_minibatches

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

    # LR scheduler: warmup + cosine decay over total updates
    updates_per_batch = cfg.ppo.ppo_epochs * cfg.ppo.num_minibatches
    total_updates = (args.total_steps // (args.num_envs * args.rollout_steps)) * updates_per_batch
    warmup = cfg.ppo.lr_warmup_steps
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(learner.optimizer, lr_lambda)

    total_env_steps = 0
    iter_count = 0
    start = time.perf_counter()
    reward_history = deque(maxlen=200)
    clipfrac_history = deque(maxlen=50)
    best_avg_reward = float("-inf")
    last_log = start

    rollout_T = args.rollout_steps
    N = args.num_envs
    P = 2  # num players

    # Pre-allocate rollout buffers on CPU to keep GPU memory free for the model.
    # Each minibatch is moved to GPU just-in-time inside the PPO update.
    storage_dev = torch.device("cpu")
    storage = {
        "planet_features": torch.zeros(rollout_T, N, 48, 18, device=storage_dev),
        "fleet_features":  torch.zeros(rollout_T, N, 128, 9, device=storage_dev),
        "global_features": torch.zeros(rollout_T, N, 10, device=storage_dev),
        "planet_mask":     torch.zeros(rollout_T, N, 48, dtype=torch.bool, device=storage_dev),
        "fleet_mask":      torch.zeros(rollout_T, N, 128, dtype=torch.bool, device=storage_dev),
        "fire_mask":       torch.zeros(rollout_T, N, MAX_OWNED, dtype=torch.bool, device=storage_dev),
        "angle_mask":      torch.zeros(rollout_T, N, MAX_OWNED, NUM_ANGLE_BINS, dtype=torch.bool, device=storage_dev),
        "slot_valid":      torch.zeros(rollout_T, N, MAX_OWNED, dtype=torch.bool, device=storage_dev),
        "owned_indices":   torch.zeros(rollout_T, N, MAX_OWNED, dtype=torch.long, device=storage_dev),
        "fire_a":     torch.zeros(rollout_T, N, MAX_OWNED, dtype=torch.long, device=storage_dev),
        "angle_a":    torch.zeros(rollout_T, N, MAX_OWNED, dtype=torch.long, device=storage_dev),
        "ship_a":     torch.zeros(rollout_T, N, MAX_OWNED, dtype=torch.long, device=storage_dev),
        "lp_fire":    torch.zeros(rollout_T, N, MAX_OWNED, device=storage_dev),
        "lp_angle":   torch.zeros(rollout_T, N, MAX_OWNED, device=storage_dev),
        "lp_ship":    torch.zeros(rollout_T, N, MAX_OWNED, device=storage_dev),
        "values":     torch.zeros(rollout_T, N, device=storage_dev),
        "rewards":    torch.zeros(rollout_T, N, device=storage_dev),
        "dones":      torch.zeros(rollout_T, N, dtype=torch.bool, device=storage_dev),
    }

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
        # --- Rollout collection (no grad) -----------------------------------
        model.eval()
        for t in range(rollout_T):
            # Player 0 — store features, sample, record log_probs + value
            feats_p0, outs_p0 = forward_player(0)
            fire0, angle0, ship0, lpf0, lpa0, lps0 = sample_action_batched(
                outs_p0, feats_p0["fire_mask"], feats_p0["angle_mask"]
            )
            # Copy to CPU storage (async, non-blocking)
            storage["planet_features"][t].copy_(feats_p0["planet_features"], non_blocking=True)
            storage["fleet_features"][t].copy_(feats_p0["fleet_features"], non_blocking=True)
            storage["global_features"][t].copy_(feats_p0["global_features"], non_blocking=True)
            storage["planet_mask"][t].copy_(feats_p0["planet_mask"], non_blocking=True)
            storage["fleet_mask"][t].copy_(feats_p0["fleet_mask"], non_blocking=True)
            storage["fire_mask"][t].copy_(feats_p0["fire_mask"], non_blocking=True)
            storage["angle_mask"][t].copy_(feats_p0["angle_mask"], non_blocking=True)
            storage["slot_valid"][t].copy_(feats_p0["slot_valid"], non_blocking=True)
            storage["owned_indices"][t].copy_(feats_p0["owned_indices"], non_blocking=True)
            storage["fire_a"][t].copy_(fire0, non_blocking=True)
            storage["angle_a"][t].copy_(angle0, non_blocking=True)
            storage["ship_a"][t].copy_(ship0, non_blocking=True)
            storage["lp_fire"][t].copy_(lpf0, non_blocking=True)
            storage["lp_angle"][t].copy_(lpa0, non_blocking=True)
            storage["lp_ship"][t].copy_(lps0, non_blocking=True)
            storage["values"][t].copy_(outs_p0["value"].squeeze(-1), non_blocking=True)

            # Player 1 — sample only (no storage). Reuse computed features.
            feats_p1, outs_p1 = forward_player(1)
            fire1, angle1, ship1, _, _, _ = sample_action_batched(
                outs_p1, feats_p1["fire_mask"], feats_p1["angle_mask"]
            )

            actions = {
                0: torch.stack([fire0, angle0, ship0], dim=-1),  # (N, MAX_OWNED, 3)
                1: torch.stack([fire1, angle1, ship1], dim=-1),
            }
            _, rewards, done = env.step(actions)
            storage["rewards"][t].copy_(rewards[:, 0], non_blocking=True)
            storage["dones"][t].copy_(done, non_blocking=True)

            for r in rewards[:, 0][done].tolist():
                reward_history.append(r)

        # Bootstrap value at end of rollout
        with torch.no_grad():
            feats_final = env.get_features(0)
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
            next_value = outs_final["value"].squeeze(-1)

        # --- GAE (run on CPU since storage is on CPU) -----------------------
        advantages, returns = compute_gae(
            storage["rewards"], storage["values"], storage["dones"],
            next_value.cpu(), gamma=cfg.ppo.gamma, lam=cfg.ppo.gae_lambda,
        )

        # --- Flatten (T, N, ...) → (T*N, ...) for PPO update ----------------
        TN = rollout_T * N
        flat = {}
        for k, v in storage.items():
            flat[k] = v.reshape(TN, *v.shape[2:])
        flat_adv  = advantages.reshape(TN)
        flat_ret  = returns.reshape(TN)

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
        avg_cf = float(np.mean(clipfrac_history)) if clipfrac_history else 0.0
        if now - last_log >= 5.0 or iter_count == 1:
            last_log = now
            print(
                f"iter {iter_count:5d} | steps {total_env_steps:>11,} | "
                f"SPS {sps:>7,.0f} | reward {avg_r:+.3f} | "
                f"clip_frac {avg_cf:.3f} | KL {metrics.get('approx_kl', 0):.4f} | "
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

        # Periodic checkpoint
        if total_env_steps % args.checkpoint_interval < (rollout_T * N):
            os.makedirs("checkpoints", exist_ok=True)
            path = f"checkpoints/torch_step_{total_env_steps}.pt"
            torch.save(learner.state_dict(), path)
            print(f"  saved {path}")

    print(f"\nTraining complete: {total_env_steps:,} env steps in {elapsed:.0f}s")
    print(f"Final SPS: {sps:,.0f}")


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
    parser.add_argument("--checkpoint-interval", type=int, default=2_000_000)
    args = parser.parse_args()

    if not args.device:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    train(args)
