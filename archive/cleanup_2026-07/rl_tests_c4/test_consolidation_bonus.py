"""Unit tests for the consolidation bonus (force-concentration lever, 2026-06-15).

Drives VecTorchEnv._consolidation_bonus directly with controlled owner sequences (avoids
physics confounds) to verify the state machine: a NET-NEW captured planet earns the bonus
ONCE after surviving K steps; fleeting/lost captures earn nothing; initial owners never earn;
recaptures re-arm; multiple same-step consolidations each count.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_env import VecTorchEnv

COEF, K = 1.0, 3


def _make():
    env = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                      consolidation_coef=COEF, consolidation_steps=K)
    env.reset(seeds=[0])
    # clean baseline: everything neutral, nothing captured/credited
    env.cap_owner[:] = -1
    env.cap_age[:] = 0
    env.cap_credited[:] = False
    env.cap_is_capture[:] = False
    env.planet_alive[:] = True
    return env


def _drive(env, owner_map):
    """Set planet owners (idx->owner; all others neutral -1), run one consolidation step."""
    P = env.planets.shape[1]
    row = torch.full((P,), -1.0)
    for idx, o in owner_map.items():
        row[idx] = float(o)
    env.planets[0, :, 1] = row
    return env._consolidation_bonus(torch.zeros(1, 2))


def test_capture_held_K_rewards_once():
    env = _make()
    seq = [_drive(env, {0: 0})[0, 0].item() for _ in range(6)]   # capture p0 by player0, hold
    # age 0,1,2 → 0; age==K(3) → +COEF; then credited → 0
    assert seq == [0.0, 0.0, 0.0, COEF, 0.0, 0.0], seq


def test_lost_before_K_earns_nothing_then_recapture_rearms():
    env = _make()
    r0 = _drive(env, {0: 0})        # p0 captures
    r1 = _drive(env, {0: 0})        # holds (age1)
    r2 = _drive(env, {0: 1})        # p1 takes it before K → p0 earns 0, p1 starts fresh
    assert r0[0, 0] == 0 and r1[0, 0] == 0 and r2[0, 0] == 0
    seq1 = [_drive(env, {0: 1}) for _ in range(3)]                 # p1 holds: age1,2,3
    assert [r[0, 1].item() for r in seq1] == [0.0, 0.0, COEF]      # p1 consolidates after K
    assert [r[0, 0].item() for r in seq1] == [0.0, 0.0, 0.0]       # p0 earns NOTHING from the planet it lost


def test_initial_owner_never_rewarded():
    env = _make()
    # planet 1 owned by player 0 from the start (a home planet): baseline tracks it, NOT a capture
    env.cap_owner[0, 1] = 0
    env.cap_is_capture[0, 1] = False
    seq = [_drive(env, {1: 0})[0, 0].item() for _ in range(K + 3)]
    assert seq == [0.0] * (K + 3), seq        # held forever, never a "capture" → never paid


def test_multiple_same_step_consolidations_each_count():
    env = _make()
    for _ in range(K):
        r = _drive(env, {0: 0, 2: 0})         # player0 captures p0 and p2, holds
        assert r[0, 0] == 0
    r = _drive(env, {0: 0, 2: 0})             # both hit age K same step → 2*COEF
    assert r[0, 0] == 2 * COEF, r[0, 0].item()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} consolidation tests passed.")
