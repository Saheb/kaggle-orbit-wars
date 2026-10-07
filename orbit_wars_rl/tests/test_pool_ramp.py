"""Opponent-difficulty RAMP: pinned RL champions (rev38, the anchor) are pulled OUT of PFSP
into a fixed ramped slice, so a weak from-scratch policy eases into the unbeatable opponents
instead of being win-starved (PFSP would up-sample them).

Covers `OpponentPool.sample(pinned_fraction=)`:
  - true-zero start (ramp=0): pure self-play (None when no organic snapshots).
  - full ramp: ~target split pinned / PFSP-over-organic.
  - legacy preservation: pinned_fraction=None keeps pins competing inside PFSP.
(The external-heuristic slice was removed in the 2026-10 cleanup.)
"""
from __future__ import annotations

import os
import random
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opponent_pool import OpponentPool


def _sd(v):
    return {"w": torch.tensor([float(v)])}


def _build(n_organic=4):
    """rev38 pinned + n_organic past-self snapshots."""
    pool = OpponentPool(max_self_members=40, pfsp_min_games=30)
    pool.add_pinned_rl("rev38", _sd(99))
    for step in range(1, n_organic + 1):
        pool.add_self_checkpoint(step * 1000, _sd(step))
    return pool


def _classify(m):
    if m is None:
        return "self_mirror"          # caller falls back to current-vs-current self-play
    if m.pinned:
        return "pinned"
    return "organic"


def test_true_zero_start_is_pure_self_play():
    """ramp=0 with no organic snapshots yet → every sample is None (self-play)."""
    pool = OpponentPool(max_self_members=40)
    pool.add_pinned_rl("rev38", _sd(99))
    rng = random.Random(0)
    counts = Counter(_classify(pool.sample(rng, pinned_fraction=0.0)) for _ in range(2000))
    assert counts["self_mirror"] == 2000


def test_full_ramp_hits_target_split():
    """At full ramp, pinned≈0.267 of samples; remainder = organic PFSP."""
    pool = _build(n_organic=4)
    rng = random.Random(1)
    target = 0.267
    counts = Counter(_classify(pool.sample(rng, pinned_fraction=target)) for _ in range(20000))
    assert abs(counts["pinned"] / 20000 - target) < 0.02, counts
    # remaining budget goes to PFSP over organic snapshots, never to the mirror
    assert abs(counts["organic"] / 20000 - (1.0 - target)) < 0.02, counts
    assert counts["self_mirror"] == 0


def test_pinned_never_in_pfsp_organic_slice():
    """In ramp mode the pinned champion only ever comes from its own slice, NEVER from
    the PFSP-over-organic budget (it's excluded from the organic candidate set)."""
    pool = _build(n_organic=3)
    rng = random.Random(2)
    organic = [m for m in pool.members if m.kind == "self" and not m.pinned]
    assert len(organic) == 3
    # with pinned_fraction 0 the pin is unreachable (pulled out, slice width 0)
    counts = Counter(_classify(pool.sample(rng, pinned_fraction=0.0)) for _ in range(3000))
    assert counts["pinned"] == 0
    assert counts["organic"] == 3000  # all remainder → organic PFSP


def test_partial_ramp_scales_linearly():
    """Half-ramp (target*0.5) roughly halves the pinned fraction."""
    pool = _build(n_organic=4)
    rng = random.Random(3)
    half = 0.267 * 0.5
    counts = Counter(_classify(pool.sample(rng, pinned_fraction=half)) for _ in range(20000))
    assert abs(counts["pinned"] / 20000 - half) < 0.02


def test_legacy_mode_pinned_competes_in_pfsp():
    """pinned_fraction=None: the pin is a 'self' member competing inside PFSP, so it CAN be
    drawn from the PFSP budget, and the sampler never returns None."""
    pool = _build(n_organic=2)
    rng = random.Random(4)
    counts = Counter(_classify(pool.sample(rng)) for _ in range(8000))
    assert counts["self_mirror"] == 0
    assert counts["pinned"] > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} ramp tests passed.")
