#!/usr/bin/env python3
"""Quick training validation: collect a few rollouts and run PPO update."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np

from orbit_wars_rl.config import Config
from orbit_wars_rl.model import EntityTransformer, count_params
from orbit_wars_rl.ppo import PPOLearner
from self_play import OpponentPool, collect_rollout, compute_gae, make_batch, make_minibatches
from orbit_wars_rl.env import OrbitWarsEnv


def main():
    print("=== Quick Training Validation ===\n")

    cfg = Config()
    device = torch.device(cfg.device)
    print(f"Device: {device}")

    model = EntityTransformer(cfg.model).to(device)
    print(f"Model params: {count_params(model):,}")

    learner = PPOLearner(model, cfg, device=device)

    env = OrbitWarsEnv(num_players=2, seed=42)

    # Collect 3 episodes
    print("\nCollecting rollouts...")
    all_transitions = []
    for ep in range(3):
        transitions = collect_rollout(model, env, device, epsilon=0.3)
        advantages, returns = compute_gae(transitions, gamma=0.995, lam=0.95)
        all_transitions.extend(list(zip(transitions, advantages, returns)))
        print(f"  Episode {ep}: {len(transitions)} steps, final reward={transitions[-1].reward:.1f}")

    print(f"\nTotal transitions: {len(all_transitions)}")

    # Subsample
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

    print(f"\nBatch size: {batch['planet_features'].shape[0]}")
    print(f"Minibatches: {len(minibatches)}")

    # Run PPO update
    print("\nRunning PPO update...")
    metrics = learner.update(minibatches)

    print("\nUpdate metrics:")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print(f"\nClip fraction: {metrics.get('clip_frac', 0):.4f}")
    print(f"Approx KL: {metrics.get('approx_kl', 0):.6f}")

    if abs(metrics.get('approx_kl', 0)) > 0.1:
        print("\nWARNING: High KL divergence — consider reducing learning rate")
    elif metrics.get('clip_frac', 0) > 0.3:
        print("\nWARNING: High clip fraction — consider reducing learning rate")
    else:
        print("\nTraining loop validated successfully!")


if __name__ == "__main__":
    main()