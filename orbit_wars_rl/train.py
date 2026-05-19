"""Main training loop for Orbit Wars Entity Transformer PPO.

Usage:
    python train.py --seed 42 --total-steps 100000 --wandb
    python train.py --device cpu --num-envs 2
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from collections import deque

import torch
import numpy as np

from config import Config
from model import EntityTransformer, count_params
from ppo import PPOLearner
from self_play import (
    OpponentPool, collect_rollout, compute_gae, make_batch, make_minibatches
)
from env import OrbitWarsEnv


def train(cfg: Config, use_wandb: bool = False, resume_from: str = ""):
    device = torch.device(cfg.device)
    print(f"Training on device: {device}")

    torch.manual_seed(cfg.seed)
    model = EntityTransformer(cfg.model)
    print(f"Model params: {count_params(model):,}")

    if resume_from:
        sd = torch.load(resume_from, map_location="cpu", weights_only=True)
        if "model" in sd:
            sd = sd["model"]
        model.load_state_dict(sd)
        print(f"Resumed model weights from {resume_from}")

    learner = PPOLearner(model, cfg, device=device)
    opponent_pool = OpponentPool(max_size=cfg.self_play.opponent_pool_size)

    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, config=cfg.__dict__)
        wandb.watch(model, log="gradients", log_freq=100)

    total_env_steps = 0
    episode_count = 0
    start_time = time.time()
    reward_history = deque(maxlen=100)
    clip_frac_history = deque(maxlen=100)
    avg_clip_frac = 0.0  # keep in scope for checkpoint warning

    # LR scheduler: warmup (per optimizer step) + cosine decay
    warmup_steps = cfg.ppo.lr_warmup_steps
    total_opt_steps = cfg.ppo.total_env_steps  # proxy; actual opt steps ~ total/500 * 16

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_opt_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(learner.optimizer, lr_lambda)

    print(f"\nStarting training: {cfg.ppo.total_env_steps:,} env steps")
    print(f"Batch size: {cfg.ppo.batch_size}, Minibatches: {cfg.ppo.num_minibatches}")
    print(f"Self-play pool: {cfg.self_play.opponent_pool_size} snapshots")
    print(f"Shaping coef: {cfg.ppo.shaping_coef}")
    print()

    env = OrbitWarsEnv(num_players=cfg.env.num_players, seed=cfg.seed)

    while total_env_steps < cfg.ppo.total_env_steps:
        all_transitions = []
        episode_rewards = []

        while len(all_transitions) < cfg.ppo.batch_size:
            # Epsilon schedule: 0.1 → 0.02
            progress = total_env_steps / cfg.ppo.total_env_steps
            epsilon = max(0.02, 0.1 * (1.0 - progress))

            # Sample opponent from pool (or None → random)
            opponent_model = None
            if len(opponent_pool) > 0 and random.random() < cfg.self_play.opponent_sample_prob_old:
                params = opponent_pool.sample()
                opponent_model = EntityTransformer(cfg.model).to(device)
                opponent_model.load_state_dict(params)
                opponent_model.eval()

            transitions = collect_rollout(
                model, env, device,
                epsilon=epsilon,
                shaping_coef=cfg.ppo.shaping_coef,
                opponent_model=opponent_model,
            )

            if transitions:
                advantages, returns = compute_gae(
                    transitions,
                    gamma=cfg.ppo.gamma,
                    lam=cfg.ppo.gae_lambda,
                )
                all_transitions.extend(zip(transitions, advantages, returns))
                episode_rewards.append(transitions[-1].reward)

            total_env_steps += len(transitions)
            episode_count += 1

        # Subsample to batch_size
        if len(all_transitions) > cfg.ppo.batch_size:
            indices = np.random.choice(len(all_transitions), cfg.ppo.batch_size, replace=False)
            batch_transitions = [all_transitions[i] for i in indices]
        else:
            batch_transitions = all_transitions

        trans_list = [t[0] for t in batch_transitions]
        adv_list = np.array([t[1] for t in batch_transitions])
        ret_list = np.array([t[2] for t in batch_transitions])

        batch = make_batch(trans_list, adv_list, ret_list, device)
        minibatches = make_minibatches(batch, cfg.ppo.num_minibatches, device)

        # PPO update — scheduler steps per optimizer step inside update()
        metrics = learner.update(minibatches, scheduler=scheduler)
        metrics["learning_rate"] = learner.get_lr()

        reward_history.extend(episode_rewards)
        clip_frac_history.append(metrics.get("clip_frac", 0))
        avg_clip_frac = float(np.mean(clip_frac_history)) if clip_frac_history else 0.0

        if episode_count % 10 == 0:
            elapsed = time.time() - start_time
            sps = total_env_steps / elapsed if elapsed > 0 else 0
            avg_reward = float(np.mean(reward_history)) if reward_history else 0.0

            print(
                f"Episode {episode_count:5d} | Steps {total_env_steps:>10,} | "
                f"SPS {sps:7.0f} | Reward {avg_reward:+6.2f} | "
                f"Clip {avg_clip_frac:.3f} | "
                f"KL {metrics.get('approx_kl', 0):.4f} | "
                f"V_loss {metrics.get('value_loss', 0):.4f} | "
                f"LR {metrics['learning_rate']:.6f}"
            )

            if use_wandb:
                import wandb
                wandb.log({
                    "episode": episode_count,
                    "total_steps": total_env_steps,
                    "sps": sps,
                    "avg_reward": avg_reward,
                    "pool_size": len(opponent_pool),
                    **{f"train/{k}": v for k, v in metrics.items()},
                })

        # Add current policy to opponent pool periodically
        if episode_count % 50 == 0:
            opponent_pool.add(learner.model.state_dict(), total_env_steps)

        # Checkpoint
        if total_env_steps % cfg.self_play.checkpoint_interval_steps < cfg.ppo.batch_size:
            ckpt_dir = f"checkpoints/step_{total_env_steps}"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(learner.state_dict(), os.path.join(ckpt_dir, "checkpoint.pt"))
            print(f"Checkpoint saved at {ckpt_dir}")

            if avg_clip_frac > 0.3:
                print(f"WARNING: clip_frac = {avg_clip_frac:.3f} — consider reducing LR")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(learner.state_dict(), "checkpoints/final.pt")
    print(f"\nTraining complete! Total steps: {total_env_steps:,}")

    if use_wandb:
        import wandb
        wandb.finish()

    return learner


# Need random for opponent pool sampling
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=10_000_000)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num-players", type=int, choices=[2, 4], default=2)
    parser.add_argument("--shaping-coef", type=float, default=None,
                        help="Material-delta shaping coefficient (default: cfg value)")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint or BC model to warm-start from")
    args = parser.parse_args()

    cfg = Config()
    cfg.seed = args.seed
    cfg.ppo.total_env_steps = args.total_steps
    cfg.env.num_players = args.num_players
    if args.device:
        cfg.device = args.device
    if args.shaping_coef is not None:
        cfg.ppo.shaping_coef = args.shaping_coef

    train(cfg, use_wandb=args.wandb, resume_from=args.resume)
