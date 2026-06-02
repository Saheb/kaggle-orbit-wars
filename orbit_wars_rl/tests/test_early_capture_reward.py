from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_env import MAX_OWNED, VecTorchEnv


def _zero_actions(num_envs: int) -> dict[int, torch.Tensor]:
    return {
        0: torch.zeros(num_envs, MAX_OWNED, 3, dtype=torch.long),
        1: torch.zeros(num_envs, MAX_OWNED, 3, dtype=torch.long),
    }


def test_early_capture_reward_decays_and_cuts_off():
    coef = 0.07
    decay_steps = 400
    env = VecTorchEnv(
        num_envs=4,
        num_players=2,
        device="cpu",
        episode_steps=500,
        early_capture_coef=coef,
        early_capture_steps=decay_steps,
    )
    env.reset(seeds=[10, 11, 12, 13])

    # step() increments step_count before reward shaping, so these captures are
    # scored at post-step 150, 200, 400, and 450.
    env.step_count[:] = torch.tensor([149, 199, 399, 449])
    env.prev_owned[:] = torch.tensor([[1.0, 1.0]] * 4)
    env.fleet_alive[:] = False
    env.planet_alive[:] = False
    env.planet_alive[:, :3] = True
    env.planets[:, 0, 1] = 0
    env.planets[:, 1, 1] = 1
    env.planets[:, 2, 1] = 0
    env.planets[:, :3, 5] = 10

    _, rewards, done = env.step(_zero_actions(4))

    assert not done.any()
    expected_p0 = torch.tensor(
        [
            coef * (1.0 - 150 / decay_steps),
            coef * (1.0 - 200 / decay_steps),
            0.0,
            0.0,
        ]
    )
    assert torch.allclose(rewards[:, 0], expected_p0)
    assert torch.allclose(rewards[:, 1], -expected_p0)


def test_early_capture_reward_is_capped_zero_sum_and_one_shot():
    coef = 0.07
    decay_steps = 400
    env = VecTorchEnv(
        num_envs=3,
        num_players=2,
        device="cpu",
        episode_steps=500,
        early_capture_coef=coef,
        early_capture_steps=decay_steps,
    )
    env.reset(seeds=[20, 21, 22])

    env.step_count[:] = 99
    env.prev_owned[:] = torch.tensor([[1.0, 1.0]] * 3)
    env.fleet_alive[:] = False
    env.planet_alive[:] = False
    env.planet_alive[:, :6] = True
    env.planets[:, :, 5] = 10
    env.planets[:, 0, 1] = 0
    env.planets[:, 1, 1] = 1

    # Env 0: p0 gains four planets but reward is capped to one capture unit.
    env.planets[0, 2:6, 1] = torch.tensor([0, 0, 0, 0])
    # Env 1: both players gain planets, so relative zero-sum reward is zero.
    env.planets[1, 2:6, 1] = torch.tensor([0, 1, 0, 1])
    # Env 2: p1 gains four planets, symmetric with env 0.
    env.planets[2, 2:6, 1] = torch.tensor([1, 1, 1, 1])

    _, first_rewards, _ = env.step(_zero_actions(3))
    _, second_rewards, _ = env.step(_zero_actions(3))

    capped_reward = coef * (1.0 - 100 / decay_steps)
    expected = torch.tensor(
        [
            [capped_reward, -capped_reward],
            [0.0, 0.0],
            [-capped_reward, capped_reward],
        ]
    )
    assert torch.allclose(first_rewards, expected)
    assert torch.allclose(second_rewards, torch.zeros_like(second_rewards))
