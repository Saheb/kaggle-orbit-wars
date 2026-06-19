from __future__ import annotations

import math
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


def _decay(coef: float, t: float, episode_steps: int) -> float:
    # Rev30 capture-reward decay: exponential with a permanent 10% floor over the
    # full episode (early_capture_steps is unused — decay runs on episode_steps).
    return coef * (math.exp(-2.5 * t / episode_steps) + 0.10)


def test_early_capture_reward_decays_with_floor():
    coef = 0.07
    episode_steps = 500
    env = VecTorchEnv(
        num_envs=4,
        num_players=2,
        device="cpu",
        episode_steps=episode_steps,
        early_capture_coef=coef,
        early_capture_steps=400,
    )
    env.reset(seeds=[10, 11, 12, 13])

    # step() increments step_count before reward shaping, so these captures are
    # scored at post-step 150, 200, 400, and 450. The exponential decay keeps a
    # permanent 10% floor, so late captures still earn a (small) reward.
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
        [_decay(coef, t, episode_steps) for t in (150, 200, 400, 450)]
    )
    assert torch.allclose(rewards[:, 0], expected_p0)
    # Symmetric delta credits each player's OWN net ownership change. p1's owned
    # count is unchanged here (p0 captures a neutral planet, not p1's), so p1 is
    # not penalised — only true opponent captures are zero-sum (defender delta -1).
    assert torch.allclose(rewards[:, 1], torch.zeros_like(expected_p0))


def test_early_capture_reward_is_capped_and_one_shot():
    coef = 0.07
    episode_steps = 500
    env = VecTorchEnv(
        num_envs=3,
        num_players=2,
        device="cpu",
        episode_steps=episode_steps,
        early_capture_coef=coef,
        early_capture_steps=400,
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

    # Env 0: p0 gains four neutral planets but reward is capped to one capture unit.
    env.planets[0, 2:6, 1] = torch.tensor([0, 0, 0, 0])
    # Env 1: both players gain (neutral) planets — symmetric delta credits each
    # player's own net gain independently (no cross-player netting), so both earn
    # one capped unit rather than cancelling out.
    env.planets[1, 2:6, 1] = torch.tensor([0, 1, 0, 1])
    # Env 2: p1 gains four planets, symmetric with env 0.
    env.planets[2, 2:6, 1] = torch.tensor([1, 1, 1, 1])

    _, first_rewards, _ = env.step(_zero_actions(3))
    _, second_rewards, _ = env.step(_zero_actions(3))

    capped_reward = _decay(coef, 100, episode_steps)
    expected = torch.tensor(
        [
            [capped_reward, 0.0],
            [capped_reward, capped_reward],
            [0.0, capped_reward],
        ]
    )
    assert torch.allclose(first_rewards, expected)
    assert torch.allclose(second_rewards, torch.zeros_like(second_rewards))
