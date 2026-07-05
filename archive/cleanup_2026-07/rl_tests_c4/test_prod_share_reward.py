"""Unit tests for the unified production-share capture reward (_prod_share_bonus).

Mirrors the offline red-team (reward_redteam.py) in the vectorized path: symmetric +
capture-time-anchored ⇒ capture-then-lose cancels exactly (no tennis farm), holding pays
nothing extra, and regular-board captures are bounded by 1.1·coef (so loss stays negative under ±1).
Drives ownership transitions directly to avoid orbital/physics noise (same approach as
test_capture_utility_reward)."""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_env import COMET_PRODUCTION, COMET_SLOT_START, VecTorchEnv

C = 0.5
TOTAL = 80.0


def dec(t):
    return math.exp(-2.5 * t / 500.0) + 0.10


def _make():
    env = VecTorchEnv(num_envs=1, num_players=2, device="cpu", prod_share_coef=C)
    env.reset(seeds=[0])
    env.planet_alive[:] = True
    env.planets[0, :, 1] = -1.0          # all neutral
    env.planets[0, :, 6] = 0.0           # zero production unless set per test
    env.prev_planet_owner[:] = -1        # so first capture credits via the gain path
    env.capture_time[:] = 0
    env.total_board_prod[:] = TOTAL
    return env


def _drive(env, owner_map, prod_map, t):
    env.step_count = torch.tensor([t], dtype=torch.long, device=env.device)
    for idx, o in owner_map.items():
        env.planets[0, idx, 1] = float(o)
    for idx, p in prod_map.items():
        env.planets[0, idx, 6] = float(p)
    return env._prod_share_bonus(torch.zeros(1, 2))


def test_reset_initial_ownership_pays_no_dense_reward():
    env = VecTorchEnv(num_envs=1, num_players=2, device="cpu", prod_share_coef=C)
    env.reset(seeds=[0])
    env.step_count = torch.tensor([1], dtype=torch.long, device=env.device)
    r = env._prod_share_bonus(torch.zeros(1, 2))
    assert torch.allclose(r, torch.zeros_like(r)), r


def test_losing_initial_planet_is_negative_delta():
    env = VecTorchEnv(num_envs=1, num_players=2, device="cpu", prod_share_coef=C)
    env.reset(seeds=[0])
    owner = env.planets[0, :, 1].long()
    regular = torch.arange(env.planets.shape[1], device=env.device) < COMET_SLOT_START
    candidates = torch.where((owner == 0) & env.planet_alive[0] & regular)[0]
    assert len(candidates) > 0
    pid = int(candidates[0].item())
    prod = float(env.planets[0, pid, 6].item())
    total = float(env.total_board_prod[0].item())
    assert prod > 0.0

    env.step_count = torch.tensor([10], dtype=torch.long, device=env.device)
    env.planets[0, pid, 1] = 1.0
    r = env._prod_share_bonus(torch.zeros(1, 2))
    assert math.isclose(r[0, 0].item(), -C * dec(0) * prod / total, abs_tol=1e-6), r
    assert math.isclose(r[0, 1].item(), C * dec(10) * prod / total, abs_tol=1e-6), r


def test_capture_credits_value_weighted():
    env = _make()
    r = _drive(env, {3: 0}, {3: 4}, 2)                       # neutral -> us, prod 4, t=2
    assert math.isclose(r[0, 0].item(), C * dec(2) * 4 / TOTAL, abs_tol=1e-6), r
    # a richer planet pays proportionally more
    r2 = _drive(env, {5: 0}, {5: 5}, 2)
    assert r2[0, 0].item() > r[0, 0].item()


def test_capture_then_lose_cancels_exactly():
    env = _make()
    g = C * dec(2) * 4 / TOTAL
    r1 = _drive(env, {3: 0}, {3: 4}, 2)                      # we capture at t=2
    assert math.isclose(r1[0, 0].item(), g, abs_tol=1e-6)
    r2 = _drive(env, {3: 1}, {3: 4}, 400)                    # we LOSE it at t=400
    # our loss is anchored to t=2 (not t=400) → exact cancel; opponent gains at t=400
    assert math.isclose(r2[0, 0].item(), -g, abs_tol=1e-6), r2
    assert math.isclose(r2[0, 1].item(), C * dec(400) * 4 / TOTAL, abs_tol=1e-6), r2
    assert abs((r1[0, 0] + r2[0, 0]).item()) < 1e-6           # round-trip = 0 (no farm)


def test_tennis_nets_zero():
    env = _make()
    tot = 0.0
    for (t, owner) in [(2, 0), (5, 1), (8, 0), (11, 1)]:     # cap/lose/cap/lose
        tot += _drive(env, {3: owner}, {3: 4}, t)[0, 0].item()
    assert abs(tot) < 1e-6, tot


def test_hold_pays_nothing_extra():
    env = _make()
    _drive(env, {3: 0}, {3: 4}, 2)                           # capture
    for t in range(3, 8):                                    # hold (no ownership change)
        r = _drive(env, {3: 0}, {3: 4}, t)
        assert r[0, 0].item() == 0.0, (t, r)


def test_full_board_hold_is_bounded():
    env = _make()
    # 4 planets, prod 20 each, total 80; capture ALL early → share sums to 1.0
    r = _drive(env, {0: 0, 1: 0, 2: 0, 3: 0}, {0: 20, 1: 20, 2: 20, 3: 20}, 1)
    assert r[0, 0].item() <= 1.1 * C + 1e-6, r                # bound → loss stays negative under ±1
    assert math.isclose(r[0, 0].item(), C * dec(1) * 1.0, abs_tol=1e-6), r


def test_terminal_is_pure_pm1_when_no_margin():
    # win_margin_coeff defaults to 0 → _check_done returns exactly ±1 (our chosen terminal).
    env = VecTorchEnv(num_envs=2, num_players=2, device="cpu", prod_share_coef=C)
    env.reset(seeds=[0, 1])
    rewards, done = env._check_done()
    # force a decided board: env0 player0 owns everything, env1 player1 owns everything
    env.planets[0, :, 1] = 0.0
    env.planets[1, :, 1] = 1.0
    env.fleets[:, :, :] = 0.0
    env.fleet_alive[:] = False
    rewards, done = env._check_done()
    assert rewards[0, 0].item() == 1.0 and rewards[0, 1].item() == -1.0, rewards
    assert rewards[1, 1].item() == 1.0 and rewards[1, 0].item() == -1.0, rewards


def test_comet_slots_are_ignored_by_prod_share():
    env = _make()
    c = COMET_SLOT_START
    env.step_count = torch.tensor([50], dtype=torch.long, device=env.device)
    env.planet_alive[0, c] = True
    env.planets[0, c, 1] = 0.0
    env.planets[0, c, 6] = COMET_PRODUCTION
    r = env._prod_share_bonus(torch.zeros(1, 2))
    assert torch.allclose(r, torch.zeros_like(r)), r

    env.step_count = torch.tensor([60], dtype=torch.long, device=env.device)
    env.planet_alive[0, c] = False
    env.planets[0, c, 1] = -1.0
    r = env._prod_share_bonus(torch.zeros(1, 2))
    assert torch.allclose(r, torch.zeros_like(r)), r


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")
