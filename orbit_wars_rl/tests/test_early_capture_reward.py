"""Early-capture (delta-capture) reward — Rev30 semantics, used by the blessed runs.

Semantics under test (torch_env.step, early_capture_coef != 0):
- decay = exp(-2.5 * t / episode_steps) + 0.10 (permanent 10% floor, NO cutoff —
  early_capture_steps is legacy/unused; decay runs to episode end).
- reward[pl] = coef * decay(t) * clamp(owned[pl] - prev_owned[pl], -1, 1): symmetric
  per-player delta, capped at one capture unit per step, NO opponent netting (a
  neutral capture pays the capturer only; capturing FROM the opponent pays +1/-1
  automatically via the two deltas).
- one-shot: prev_owned refreshes each step, so an unchanged board pays 0.
"""
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


def _decay(t: float, episode_steps: int = 500) -> float:
    return math.exp(-2.5 * t / episode_steps) + 0.10


def test_early_capture_reward_decays_and_cuts_off():
    """Exponential decay with the 10% floor: a capture at t=450 still pays (no cliff)."""
    coef = 0.07
    env = VecTorchEnv(
        num_envs=4,
        num_players=2,
        device="cpu",
        episode_steps=500,
        early_capture_coef=coef,
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
    env.planets[:, 2, 1] = 0     # p0's net-new planet (prev_owned said 1)
    env.planets[:, :3, 5] = 10

    _, rewards, done = env.step(_zero_actions(4))

    assert not done.any()
    expected_p0 = torch.tensor([coef * _decay(t) for t in (150, 200, 400, 450)])
    assert torch.allclose(rewards[:, 0], expected_p0, atol=1e-5)
    # p1's owned count is unchanged (delta 0) — a neutral gain is NOT netted onto the opponent.
    assert torch.allclose(rewards[:, 1], torch.zeros(4), atol=1e-6)


def test_early_capture_reward_is_capped_zero_sum_and_one_shot():
    """Delta is clamped to ±1/step (no multi-capture farming) and pays only once."""
    coef = 0.07
    env = VecTorchEnv(
        num_envs=3,
        num_players=2,
        device="cpu",
        episode_steps=500,
        early_capture_coef=coef,
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
    # Env 1: both players gain planets — each is paid for their own (capped) delta.
    env.planets[1, 2:6, 1] = torch.tensor([0, 1, 0, 1])
    # Env 2: p1 gains four planets, symmetric with env 0.
    env.planets[2, 2:6, 1] = torch.tensor([1, 1, 1, 1])

    _, first_rewards, _ = env.step(_zero_actions(3))
    _, second_rewards, _ = env.step(_zero_actions(3))

    capped = coef * _decay(100)
    expected = torch.tensor(
        [
            [capped, 0.0],
            [capped, capped],
            [0.0, capped],
        ]
    )
    assert torch.allclose(first_rewards, expected, atol=1e-5)
    # One-shot: the board is unchanged on the second step → no further payment.
    assert torch.allclose(second_rewards, torch.zeros_like(second_rewards), atol=1e-6)
