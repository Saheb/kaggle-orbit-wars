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
    OpponentPool, collect_rollout, collect_rollouts_vec,
    compute_gae, make_batch, make_minibatches
)
from env import OrbitWarsEnv
from fast_env import FastOrbitWarsEnv


def train(
    cfg: Config,
    use_wandb: bool = False,
    resume_from: str = "",
    num_envs: int = 1,
    opponent_policy: str = "random",
    env_backend: str = "kaggle",
    epsilon_start: float = 0.1,
    epsilon_final: float = 0.02,
    eval_interval_steps: int = 0,
    eval_games: int = 0,
    eval_opponent: str = "random",
    eval_fire_threshold: float = 0.5,
    bc_agent_path: str = "",
    bc_games: int = 0,
    bc_batch_size: int = 128,
    bc_replay_data: str = "",
    opponent_checkpoint: str = "",
    self_play_prob: float = 0.5,
):
    device = torch.device(cfg.device)
    print(f"Training on device: {device}")

    torch.manual_seed(cfg.seed)
    model = EntityTransformer(cfg.model)
    print(f"Model params: {count_params(model):,}")

    if resume_from:
        try:
            sd = torch.load(resume_from, map_location="cpu", weights_only=True)
        except Exception:
            sd = torch.load(resume_from, map_location="cpu", weights_only=False)
        if "model" in sd:
            sd = sd["model"]
        model.load_state_dict(sd)
        print(f"Resumed model weights from {resume_from}")

    learner = PPOLearner(model, cfg, device=device)
    opponent_pool = OpponentPool(max_size=cfg.self_play.opponent_pool_size)
    bc_samples = []
    if bc_agent_path and cfg.ppo.bc_coef > 0:
        if bc_games <= 0:
            raise ValueError("--bc-games must be > 0 when --bc-agent and --bc-coef are used")
        from bc import collect_heuristic_trajectories, trajectory_to_training_sample

        print(f"Collecting {bc_games} teacher games for PPO BC regularization...")
        trajectories = collect_heuristic_trajectories(
            bc_agent_path,
            num_games=bc_games,
            opponent=opponent_policy,
            verbose=True,
        )
        bc_samples = [
            sample
            for traj in trajectories
            if (sample := trajectory_to_training_sample(traj)) is not None
        ]
        if not bc_samples:
            raise ValueError("No BC samples collected for PPO regularization")
        print(f"BC regularization samples: {len(bc_samples)}")

    # Load replay BC data if provided
    if args.bc_replay_data:
        import pickle
        from bc import _collate, bc_loss
        with open(args.bc_replay_data, "rb") as f:
            replay_samples = pickle.load(f)
        # Convert replay samples to torch tensors if needed
        converted = []
        for s in replay_samples:
            sample = {}
            for k in ["planet_features", "fleet_features", "global_features",
                      "planet_mask", "fleet_mask", "fire_mask", "angle_mask",
                      "slot_valid", "owned_indices", "fire_target", "angle_target", "ship_target"]:
                v = s.get(k)
                if v is not None:
                    if isinstance(v, np.ndarray):
                        sample[k] = torch.from_numpy(v)
                    elif isinstance(v, torch.Tensor):
                        sample[k] = v
                    else:
                        sample[k] = torch.tensor(v)
            if "owned_count" in s:
                sample["owned_count"] = s["owned_count"]
            converted.append(sample)
        bc_samples.extend(converted)
        print(f"Loaded {len(converted)} replay BC samples from {args.bc_replay_data}")
        print(f"Total BC samples: {len(bc_samples)}")

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
    last_log_steps = 0

    # Best-checkpoint tracking: save whenever rolling reward improves
    best_avg_reward = float("-inf")
    best_ckpt_dir = "checkpoints/best"

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
    print(f"Parallel envs: {num_envs}")
    print(f"Opponent policy: {opponent_policy}")
    print(f"Env backend: {env_backend}")
    rollout_device = torch.device("cpu") if device.type == "mps" else device
    print(f"Rollout inference device: {rollout_device}")
    print(f"Uniform exploration epsilon: {epsilon_start} -> {epsilon_final}")
    print(
        f"BC regularization: coef={cfg.ppo.bc_coef} "
        f"samples={len(bc_samples)} agent={bc_agent_path or 'none'}"
    )

    # Load fixed self-play opponent checkpoint if provided
    fixed_opponent = None
    if opponent_checkpoint:
        fixed_opponent = EntityTransformer(cfg.model).to(torch.device("cpu"))
        try:
            sd = torch.load(opponent_checkpoint, map_location="cpu", weights_only=True)
        except Exception:
            sd = torch.load(opponent_checkpoint, map_location="cpu", weights_only=False)
        if "model" in sd:
            sd = sd["model"]
        fixed_opponent.load_state_dict(sd)
        fixed_opponent.eval()
        print(f"Loaded self-play opponent from {opponent_checkpoint}")
        print(f"Self-play prob: {self_play_prob}")

    print()

    use_vec = num_envs > 1
    if use_vec:
        from vec_env import VecEnvPool
        vec_pool = VecEnvPool(
            num_envs,
            num_players=cfg.env.num_players,
            base_seed=cfg.seed,
            opponent_policy=opponent_policy,
            env_backend=env_backend,
        )
        print(f"Spawning {num_envs} environment workers...")
    else:
        env_cls = FastOrbitWarsEnv if env_backend == "fast" else OrbitWarsEnv
        env = env_cls(
            num_players=cfg.env.num_players,
            seed=cfg.seed,
            opponent_policy=opponent_policy,
        )
    last_eval_steps = 0

    try:
        while total_env_steps < cfg.ppo.total_env_steps:
            progress = total_env_steps / cfg.ppo.total_env_steps
            epsilon = max(epsilon_final, epsilon_start * (1.0 - progress))
            episode_rewards = []

            if use_vec:
                # Vectorized: workers parallelize env steps; keep rollout
                # inference on CPU for MPS because tiny per-step transfers
                # dominate. CUDA can keep batched inference on GPU.
                model.to(rollout_device)
                all_transitions_with_gae, episode_rewards = collect_rollouts_vec(
                    model, vec_pool, rollout_device,
                    batch_size=cfg.ppo.batch_size,
                    gamma=cfg.ppo.gamma,
                    lam=cfg.ppo.gae_lambda,
                    epsilon=epsilon,
                    shaping_coef=cfg.ppo.shaping_coef,
                )
                all_transitions = all_transitions_with_gae
            else:
                # Single env: CPU inference (13ms < 21ms MPS for BS=1)
                model.cpu()
                all_transitions = []

                while len(all_transitions) < cfg.ppo.batch_size:
                    # Select opponent: fixed self-play checkpoint, pool snapshot, or random
                    opponent_model = None
                    if fixed_opponent is not None and random.random() < self_play_prob:
                        opponent_model = fixed_opponent
                    elif len(opponent_pool) > 0 and random.random() < cfg.self_play.opponent_sample_prob_old:
                        params = opponent_pool.sample()
                        opponent_model = EntityTransformer(cfg.model).to(torch.device("cpu"))
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
                            transitions, gamma=cfg.ppo.gamma, lam=cfg.ppo.gae_lambda,
                        )
                        all_transitions.extend(zip(transitions, advantages, returns))
                        episode_rewards.append(transitions[-1].reward)

                    total_env_steps += len(transitions)
                    episode_count += 1

            # Update step counters (vec mode counts steps inside collect_rollouts_vec)
            if use_vec:
                total_env_steps += len(all_transitions)
                episode_count += len(episode_rewards)

            # Move model to training device for PPO gradient updates
            model.to(device)

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

            # Assemble BC mini-batch once per rollout (if BC regularization on).
            # Passed to update() so PPO and BC gradients are combined in a single
            # backward per minibatch step — no separate competing optimizer call.
            ppo_bc_batch = None
            if bc_samples and cfg.ppo.bc_coef > 0:
                from bc import _collate

                n = min(bc_batch_size, len(bc_samples))
                idx = np.random.choice(len(bc_samples), n, replace=len(bc_samples) < n)
                ppo_bc_batch = _collate([bc_samples[i] for i in idx], torch.device("cpu"))

            metrics = learner.update(minibatches, scheduler=scheduler,
                                     kl_target=cfg.ppo.kl_target,
                                     bc_batch=ppo_bc_batch)
            metrics["learning_rate"] = learner.get_lr()

            reward_history.extend(episode_rewards)
            clip_frac_history.append(metrics.get("clip_frac", 0))
            avg_clip_frac = float(np.mean(clip_frac_history)) if clip_frac_history else 0.0

            if total_env_steps - last_log_steps >= cfg.ppo.batch_size:
                last_log_steps = total_env_steps
                elapsed = time.time() - start_time
                sps = total_env_steps / elapsed if elapsed > 0 else 0
                avg_reward = float(np.mean(reward_history)) if reward_history else 0.0

                bc_tag = (
                    f" | BC {metrics.get('bc_loss', 0):.4f}"
                    if cfg.ppo.bc_coef > 0 and bc_samples else ""
                )
                print(
                    f"Episode {episode_count:5d} | Steps {total_env_steps:>10,} | "
                    f"SPS {sps:7.0f} | Reward {avg_reward:+6.2f} | "
                    f"Clip {avg_clip_frac:.3f} | "
                    f"KL {metrics.get('approx_kl', 0):.4f} | "
                    f"V_loss {metrics.get('value_loss', 0):.4f} | "
                    f"LR {metrics['learning_rate']:.6f}"
                    f"{bc_tag}"
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

                # Best-checkpoint: save whenever rolling reward improves by ≥0.02
                if len(reward_history) >= 50 and avg_reward > best_avg_reward + 0.02:
                    best_avg_reward = avg_reward
                    os.makedirs(best_ckpt_dir, exist_ok=True)
                    torch.save(learner.state_dict(), os.path.join(best_ckpt_dir, "checkpoint.pt"))
                    print(f"  ★ Best checkpoint updated: reward={avg_reward:+.4f} @ {total_env_steps:,} steps")

            # Add current policy to opponent pool periodically
            if episode_count % 50 == 0:
                opponent_pool.add(learner.model.state_dict(), total_env_steps)

            # Periodic checkpoint
            if total_env_steps % cfg.self_play.checkpoint_interval_steps < cfg.ppo.batch_size:
                ckpt_dir = f"checkpoints/step_{total_env_steps}"
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(learner.state_dict(), os.path.join(ckpt_dir, "checkpoint.pt"))
                print(f"Checkpoint saved at {ckpt_dir}")

                if avg_clip_frac > 0.3:
                    print(f"WARNING: clip_frac = {avg_clip_frac:.3f} — consider reducing LR")

            if (
                eval_interval_steps > 0
                and eval_games > 0
                and total_env_steps - last_eval_steps >= eval_interval_steps
            ):
                from eval import evaluate_against_baseline

                prev_training = learner.model.training
                learner.model.to(device)
                results = evaluate_against_baseline(
                    learner.model,
                    device,
                    num_games=eval_games,
                    seed_start=cfg.seed + total_env_steps,
                    opponent=eval_opponent,
                    num_players=cfg.env.num_players,
                    fire_threshold=eval_fire_threshold,
                )
                last_eval_steps = total_env_steps
                print(
                    f"Eval vs {eval_opponent}: "
                    f"fire_threshold={eval_fire_threshold:.3f} | "
                    f"win_rate={results['win_rate']:.2%} "
                    f"({results['wins']}/{results['total_games']}) | "
                    f"avg_material={results['avg_material']:.1f}"
                )
                if prev_training:
                    learner.model.train()
                if use_wandb:
                    import wandb

                    wandb.log({
                        "eval/total_steps": total_env_steps,
                        "eval/win_rate": results["win_rate"],
                        "eval/avg_material": results["avg_material"],
                    })
    finally:
        if use_vec:
            vec_pool.close()

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
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Parallel environment workers (>1 enables VecEnvPool + batched MPS)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override PPO learning rate")
    parser.add_argument("--lr-warmup-steps", type=int, default=None,
                        help="Override PPO LR warmup optimizer steps")
    parser.add_argument("--ppo-epochs", type=int, default=None,
                        help="Override PPO epochs per rollout batch")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override PPO rollout batch size")
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                        help="Override checkpoint interval in env steps")
    parser.add_argument("--bc-coef", type=float, default=None,
                        help="Auxiliary behavior-cloning loss coefficient during PPO")
    parser.add_argument("--kl-target", type=float, default=None,
                        help="KL early-stop threshold per PPO epoch (default: 0.05, inf=disabled)")
    parser.add_argument("--bc-agent", type=str, default="",
                        help="Teacher agent .py file for PPO BC regularization")
    parser.add_argument("--bc-games", type=int, default=0,
                        help="Teacher games to collect once for PPO BC regularization")
    parser.add_argument("--bc-replay-data", type=str, default="",
                        help="Path to pickle file with BC samples from replays (from replay_to_bc.py)")
    parser.add_argument("--bc-batch-size", type=int, default=128,
                        help="Auxiliary BC batch size per PPO update")
    parser.add_argument("--opponent-policy", choices=["random", "starter", "none"],
                        default="random",
                        help="Fast-wrapper opponent policy for PPO rollouts")
    parser.add_argument("--env-backend", choices=["kaggle", "fast"], default="kaggle",
                        help="Rollout env backend. Eval still uses official Kaggle env.")
    parser.add_argument("--epsilon-start", type=float, default=0.1,
                        help="Initial probability of uniform legal-action exploration")
    parser.add_argument("--epsilon-final", type=float, default=0.02,
                        help="Final/min probability of uniform legal-action exploration")
    parser.add_argument("--eval-interval-steps", type=int, default=0,
                        help="Run deterministic Kaggle-style eval every N env steps")
    parser.add_argument("--eval-games", type=int, default=0,
                        help="Number of games per deterministic eval")
    parser.add_argument("--eval-opponent", default="random",
                        help="Eval opponent: 'random' or path to agent .py")
    parser.add_argument("--eval-fire-threshold", type=float, default=0.5,
                        help="Deterministic fire threshold used by eval.py agent wrapper")
    parser.add_argument("--opponent-checkpoint", type=str, default="",
                        help="Path to checkpoint for fixed self-play opponent")
    parser.add_argument("--self-play-prob", type=float, default=0.5,
                        help="Probability of using fixed opponent vs random (0=random only, 1=opponent only)")
    args = parser.parse_args()

    cfg = Config()
    cfg.seed = args.seed
    cfg.ppo.total_env_steps = args.total_steps
    cfg.env.num_players = args.num_players
    if args.device:
        cfg.device = args.device
    if args.shaping_coef is not None:
        cfg.ppo.shaping_coef = args.shaping_coef
    if args.learning_rate is not None:
        cfg.ppo.learning_rate = args.learning_rate
    if args.lr_warmup_steps is not None:
        cfg.ppo.lr_warmup_steps = args.lr_warmup_steps
    if args.ppo_epochs is not None:
        cfg.ppo.ppo_epochs = args.ppo_epochs
    if args.batch_size is not None:
        cfg.ppo.batch_size = args.batch_size
    if args.checkpoint_interval is not None:
        cfg.self_play.checkpoint_interval_steps = args.checkpoint_interval
    if args.bc_coef is not None:
        cfg.ppo.bc_coef = args.bc_coef
    if args.kl_target is not None:
        cfg.ppo.kl_target = args.kl_target

    train(
        cfg,
        use_wandb=args.wandb,
        resume_from=args.resume,
        num_envs=args.num_envs,
        opponent_policy=args.opponent_policy,
        env_backend=args.env_backend,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        eval_interval_steps=args.eval_interval_steps,
        eval_games=args.eval_games,
        eval_opponent=args.eval_opponent,
        eval_fire_threshold=args.eval_fire_threshold,
        bc_agent_path=args.bc_agent,
        bc_games=args.bc_games,
        bc_batch_size=args.bc_batch_size,
        bc_replay_data=args.bc_replay_data,
        opponent_checkpoint=args.opponent_checkpoint,
        self_play_prob=args.self_play_prob,
    )
