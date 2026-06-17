"""Regression checks for replay-reranker dataset construction helpers."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyze_producer_action_ranking import ActionCandidate  # noqa: E402
from build_replay_reranker import _negative_candidates  # noqa: E402


def _candidate(source_id: int, target_idx: int, target_id: int, ships: int, score: float) -> ActionCandidate:
    return ActionCandidate(
        source_idx=0,
        source_id=source_id,
        target_idx=target_idx,
        target_id=target_id,
        ships=ships,
        eta=4,
        score=score,
        valid=True,
        target_is_mine=False,
        target_is_neutral=False,
        source_ships=60,
        target_prod=2,
        floor_at_arrival=10,
    )


def test_negative_candidates_match_target_owner_mode():
    planets = [
        [0, 0, 20.0, 20.0, 1.5, 60, 2],
        [1, 0, 35.0, 20.0, 1.5, 20, 2],
        [2, -1, 50.0, 20.0, 1.5, 10, 2],
        [3, 1, 70.0, 20.0, 1.5, 30, 2],
    ]
    candidates = [
        _candidate(0, 1, 1, 20, 12.0),  # own target
        _candidate(0, 2, 2, 20, 12.0),  # neutral target
        _candidate(0, 3, 3, 20, 12.0),  # enemy target
    ]

    own_negs = _negative_candidates(
        candidates,
        planets=planets,
        player=0,
        source_id=0,
        target_id=1,
        ships=10,
        eta=4,
        target_owner="own",
        score_floor=10.0,
        score_slack=5.0,
        max_eta_gap=-1,
        limit=8,
    )
    not_own_negs = _negative_candidates(
        candidates,
        planets=planets,
        player=0,
        source_id=0,
        target_id=2,
        ships=10,
        eta=4,
        target_owner="not-own",
        score_floor=10.0,
        score_slack=5.0,
        max_eta_gap=-1,
        limit=8,
    )

    assert [c.target_id for c in own_negs] == [1]
    assert [c.target_id for c in not_own_negs] == [2, 3]
    print("test_negative_candidates_match_target_owner_mode: PASS")


if __name__ == "__main__":
    print("Running build_replay_reranker tests...\n")
    test_negative_candidates_match_target_owner_mode()
    print("\nAll build_replay_reranker tests passed!")
