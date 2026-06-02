from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_env import VecTorchEnv


def test_speed_coef_rewards_fast_terminal_wins_more_than_slow_wins():
    env = VecTorchEnv(
        num_envs=2,
        num_players=2,
        device="cpu",
        episode_steps=500,
        speed_coef=0.5,
    )
    env.reset(seeds=[1, 2])

    # Force both envs into a p0-only terminal state; differ only by termination step.
    env.planet_alive[:] = False
    env.fleet_alive[:] = False
    env.planet_alive[:, 0] = True
    env.planets[:, 0, 1] = 0
    env.planets[:, 0, 5] = 10
    env.step_count[:] = torch.tensor([150, 490])

    rewards, done = env._check_done()

    assert done.tolist() == [True, True]
    assert torch.allclose(rewards[:, 1], torch.tensor([-1.0, -1.0]))
    assert torch.allclose(
        rewards[:, 0],
        torch.tensor([
            1.0 + ((500 - 150) / 500) * 0.5,
            1.0 + ((500 - 490) / 500) * 0.5,
        ]),
    )
