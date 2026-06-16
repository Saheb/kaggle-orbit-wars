"""Unit tests for capture follow-through reward shaping.

The reward is intentionally consequence-based: a net-new capture must either become an
attack source within K steps or still be a frontline planet at K. These tests drive the
state machine directly to avoid orbital/physics noise.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_env import MAX_OWNED, VecTorchEnv

COEF, PENALTY, K = 0.5, 0.2, 3


def _make():
    env = VecTorchEnv(
        num_envs=1,
        num_players=2,
        device="cpu",
        capture_utility_coef=COEF,
        capture_utility_window=K,
        capture_idle_penalty=PENALTY,
    )
    env.reset(seeds=[0])
    P = env.planets.shape[1]
    env.planet_alive[:] = True
    env.planets[0, :, 1] = -1
    env.cu_owner[:] = -1
    env.cu_age[:] = 0
    env.cu_credited[:] = False
    env.cu_is_capture[:] = False
    env.cu_used_attack[:] = False
    # Deterministic geometry: p1 enemy at index 4; p0's baseline frontline planets 1/2/3
    # are closer to it than idle test planet 0.
    coords = {
        0: (90.0, 90.0),
        1: (11.0, 10.0),
        2: (12.0, 10.0),
        3: (13.0, 10.0),
        4: (10.0, 10.0),
    }
    for idx, (x, y) in coords.items():
        env.planets[0, idx, 2] = x
        env.planets[0, idx, 3] = y
    return env


def _set_current(env, owner_map):
    P = env.planets.shape[1]
    row = torch.full((P,), -1.0)
    for idx, owner in owner_map.items():
        row[idx] = float(owner)
    env.planets[0, :, 1] = row


def _baseline(env, owner_map):
    _set_current(env, owner_map)
    env.cu_owner[0] = env.planets[0, :, 1].long()


def _drive(env, owner_map):
    _set_current(env, owner_map)
    return env._capture_utility_bonus(torch.zeros(1, 2))


def test_attack_from_captured_planet_rewards_once():
    env = _make()
    _drive(env, {0: 0})                       # net-new capture, age 0
    env.cu_used_attack[0, 0] = True            # emitted attack from captured planet
    r = _drive(env, {0: 0})
    assert r[0, 0].item() == COEF, r
    r2 = _drive(env, {0: 0})
    assert r2[0, 0].item() == 0.0, r2


def test_frontline_at_window_rewards_without_attack():
    env = _make()
    _baseline(env, {1: 0, 2: 0, 4: 1})
    rewards = []
    for _ in range(K + 1):
        # Planet 0 is close to enemy 4, so it is one of player 0's top-3 frontline planets.
        env.planets[0, 0, 2] = 9.0
        env.planets[0, 0, 3] = 10.0
        rewards.append(_drive(env, {0: 0, 1: 0, 2: 0, 4: 1})[0, 0].item())
    assert rewards == [0.0, 0.0, 0.0, COEF], rewards


def test_idle_capture_penalized_at_window_and_closed():
    env = _make()
    _baseline(env, {1: 0, 2: 0, 3: 0, 4: 1})
    rewards = [
        _drive(env, {0: 0, 1: 0, 2: 0, 3: 0, 4: 1})[0, 0].item()
        for _ in range(K + 2)
    ]
    expected = torch.tensor([0.0, 0.0, 0.0, -PENALTY, 0.0])
    assert torch.allclose(torch.tensor(rewards), expected), rewards
    env.cu_used_attack[0, 0] = True
    assert _drive(env, {0: 0, 1: 0, 2: 0, 3: 0, 4: 1})[0, 0].item() == 0.0


def test_initial_owner_never_rewards():
    env = _make()
    _baseline(env, {0: 0, 4: 1})
    seq = [_drive(env, {0: 0, 4: 1})[0, 0].item() for _ in range(K + 2)]
    assert seq == [0.0] * (K + 2), seq


def test_no_enemy_planets_does_not_create_frontline_reward():
    env = _make()
    rewards = [_drive(env, {0: 0})[0, 0].item() for _ in range(K + 1)]
    expected = torch.tensor([0.0, 0.0, 0.0, -PENALTY])
    assert torch.allclose(torch.tensor(rewards), expected), rewards


def test_emitted_attack_marks_captured_planet_used():
    env = VecTorchEnv(
        num_envs=1,
        num_players=2,
        device="cpu",
        action_decode="target",
        capture_utility_coef=COEF,
        capture_utility_window=K,
    )
    env.reset(seeds=[0])
    env.planet_alive[:] = False
    env.planet_alive[0, 0:2] = True
    env.planets[0, :, 1] = -1
    env.planets[0, 0, 1] = 0
    env.planets[0, 0, 5] = 10
    env.planets[0, 1, 1] = 1
    env.planets[0, 1, 5] = 1
    env.cu_owner[0, 0] = 0
    env.cu_is_capture[0, 0] = True
    actions = torch.full((1, MAX_OWNED, 4), -1, dtype=torch.long)
    actions[0, 0, 0] = 1      # fire from the only owned source
    actions[0, 0, 1] = 0
    actions[0, 0, 2] = 0      # 1 ship
    actions[0, 0, 3] = 1      # enemy target
    env._apply_actions(actions, owner_id=0)
    assert bool(env.cu_used_attack[0, 0])


def test_slot_starved_attack_does_not_mark_captured_planet_used():
    env = VecTorchEnv(
        num_envs=1,
        num_players=2,
        device="cpu",
        action_decode="target",
        capture_utility_coef=COEF,
        capture_utility_window=K,
    )
    env.reset(seeds=[0])
    env.planet_alive[:] = False
    env.planet_alive[0, 0:2] = True
    env.planets[0, :, 1] = -1
    env.planets[0, 0, 1] = 0
    env.planets[0, 0, 5] = 10
    env.planets[0, 1, 1] = 1
    env.planets[0, 1, 5] = 1
    env.cu_owner[0, 0] = 0
    env.cu_is_capture[0, 0] = True
    env.fleet_alive[:] = True
    actions = torch.full((1, MAX_OWNED, 4), -1, dtype=torch.long)
    actions[0, 0, 0] = 1
    actions[0, 0, 1] = 0
    actions[0, 0, 2] = 0
    actions[0, 0, 3] = 1
    env._apply_actions(actions, owner_id=0)
    assert not bool(env.cu_used_attack[0, 0])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} capture-utility tests passed.")
