"""Regression checks for short-horizon preference helpers."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyze_producer_action_ranking import ActionCandidate
import orbit_wars_rl.research.build_short_horizon_action_preferences as short_horizon



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


def test_same_source_alternatives_only_return_same_source():
    candidates = [
        _cand(1, 7, 10, 9.0),
        _cand(1, 8, 12, 8.0),
        _cand(2, 9, 10, 7.0),
        _cand(1, 10, 14, 6.0),
    ]
    alts = short_horizon._same_source_alternatives(
        candidates,
        source_id=1,
        target_id=7,
        ships=10,
        limit=3,
    )
    assert [(c.source_id, c.target_id, c.ships) for c in alts] == [(1, 8, 12), (1, 10, 14)]
    print("test_same_source_alternatives_only_return_same_source: PASS")


def test_counterfactual_alternatives_only_compare_different_targets():
    candidates = [
        _cand(1, 7, 9, 9.0),
        _cand(1, 8, 12, 8.0),
        _cand(2, 9, 10, 7.0),
        _cand(1, 10, 14, 6.0),
    ]
    alts = short_horizon._counterfactual_alternatives(
        candidates,
        source_id=1,
        target_id=7,
        ships=10,
        limit=3,
    )
    got = [(c.source_id, c.target_id, c.ships) for c in alts]
    assert (1, 8, 12) in got
    assert (1, 10, 14) in got
    assert all(target_id != 7 for _, target_id, _ in got)
    assert len(got) == len(set(got))
    print("test_counterfactual_alternatives_only_compare_different_targets: PASS")


def test_is_sane_positive_rejects_far_low_score_targets():
    candidates = [
        _cand(1, 8, 12, 9.0),
        ActionCandidate(
            source_idx=0,
            source_id=1,
            target_idx=7,
            target_id=7,
            ships=10,
            eta=12,
            score=4.5,
            valid=True,
            target_is_mine=False,
            target_is_neutral=True,
            source_ships=20,
            target_prod=2,
            floor_at_arrival=10,
        ),
    ]
    assert not short_horizon._is_sane_positive(candidates, source_id=1, target_id=7)
    assert short_horizon._is_sane_positive(candidates, source_id=1, target_id=8)
    print("test_is_sane_positive_rejects_far_low_score_targets: PASS")


def test_plausible_target_negatives_keep_only_close_rivals():
    candidates = [
        ActionCandidate(
            source_idx=0,
            source_id=1,
            target_idx=7,
            target_id=7,
            ships=10,
            eta=5,
            score=9.0,
            valid=True,
            target_is_mine=False,
            target_is_neutral=True,
            source_ships=20,
            target_prod=2,
            floor_at_arrival=10,
        ),
        ActionCandidate(
            source_idx=0,
            source_id=1,
            target_idx=8,
            target_id=8,
            ships=12,
            eta=7,
            score=8.5,
            valid=True,
            target_is_mine=False,
            target_is_neutral=True,
            source_ships=20,
            target_prod=2,
            floor_at_arrival=10,
        ),
        ActionCandidate(
            source_idx=0,
            source_id=1,
            target_idx=10,
            target_id=10,
            ships=14,
            eta=12,
            score=8.8,
            valid=True,
            target_is_mine=False,
            target_is_neutral=True,
            source_ships=20,
            target_prod=2,
            floor_at_arrival=10,
        ),
        ActionCandidate(
            source_idx=0,
            source_id=1,
            target_idx=11,
            target_id=11,
            ships=14,
            eta=6,
            score=5.0,
            valid=True,
            target_is_mine=False,
            target_is_neutral=True,
            source_ships=20,
            target_prod=2,
            floor_at_arrival=10,
        ),
    ]
    alts = short_horizon._plausible_target_negatives(candidates, source_id=1, target_id=7, limit=4)
    got = [(c.target_id, c.eta, c.score) for c in alts]
    assert got == [(8, 7, 8.5)]
    print("test_plausible_target_negatives_keep_only_close_rivals: PASS")


if __name__ == "__main__":
    print("Running build_short_horizon_action_preferences tests...\n")
    test_same_source_alternatives_only_return_same_source()
    test_counterfactual_alternatives_only_compare_different_targets()
    test_is_sane_positive_rejects_far_low_score_targets()
    test_plausible_target_negatives_keep_only_close_rivals()
    print("\nAll build_short_horizon_action_preferences tests passed!")
