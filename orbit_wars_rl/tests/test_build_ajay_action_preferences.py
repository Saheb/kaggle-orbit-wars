"""Regression checks for Ajay action preference sampling helpers."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from producer_action_ranking import ActionCandidate
from orbit_wars_rl.scripts.build_ajay_action_preferences import _negative_candidates


def _cand(source_id: int, target_id: int, ships: int, score: float) -> ActionCandidate:
    return ActionCandidate(
        source_idx=0,
        source_id=source_id,
        target_idx=target_id,
        target_id=target_id,
        ships=ships,
        eta=5,
        score=score,
        valid=True,
        target_is_mine=False,
        target_is_neutral=True,
        source_ships=20,
        target_prod=2,
        floor_at_arrival=10,
    )


def test_negative_candidates_skip_exact_and_keep_top_order():
    candidates = [
        _cand(1, 7, 10, 9.0),
        _cand(1, 8, 10, 8.0),
        _cand(2, 9, 12, 7.0),
    ]
    negatives = _negative_candidates(
        candidates,
        source_id=1,
        target_id=7,
        ships=10,
        limit=2,
    )
    assert [(c.source_id, c.target_id, c.ships) for c in negatives] == [(1, 8, 10), (2, 9, 12)]
    print("test_negative_candidates_skip_exact_and_keep_top_order: PASS")


if __name__ == "__main__":
    print("Running build_ajay_action_preferences tests...\n")
    test_negative_candidates_skip_exact_and_keep_top_order()
    print("\nAll build_ajay_action_preferences tests passed!")
