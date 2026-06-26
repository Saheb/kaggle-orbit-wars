"""Regression checks for replay head-label audit projection."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orbit_wars_rl.research.audit_replay_head_labels import (
    _copy_replay_obs,
    _project_action,
    _same_source_nearest_baseline,
    _winner_loser_seats,
)
from action_mask import compute_action_masks


def _obs(step=None):
    data = {
        "player": 0,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 10.0, 50.0, 2.0, 100, 1],
            [1, -1, 30.0, 50.0, 2.0, 10, 1],
            [2, -1, 50.0, 50.0, 2.0, 10, 1],
            [3, 0, 70.0, 50.0, 2.0, 5, 1],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 10.0, 50.0, 2.0, 100, 1],
            [1, -1, 30.0, 50.0, 2.0, 10, 1],
            [2, -1, 50.0, 50.0, 2.0, 10, 1],
            [3, 0, 70.0, 50.0, 2.0, 5, 1],
        ],
    }
    if step is not None:
        data["step"] = step
    return data


def test_winner_loser_split_uses_rewards():
    assert _winner_loser_seats({"rewards": [1, -1]}) == {0: "winner", 1: "loser"}
    assert _winner_loser_seats({"rewards": [0, 0]}) == {}
    print("test_winner_loser_split_uses_rewards: PASS")


def test_replay_obs_missing_step_uses_previous_timing_index():
    copied = _copy_replay_obs(_obs(step=None), seat=1, step=7)
    assert copied["step"] == 7
    assert copied["player"] == 0
    print("test_replay_obs_missing_step_uses_previous_timing_index: PASS")


def test_project_action_keeps_largest_per_source_and_accounts_mass():
    obs = _obs(step=3)
    masks = compute_action_masks(obs, 0)
    projected, stats = _project_action(
        obs,
        [
            [0, 0.0, 10],
            [0, 0.0, 4],
            [3, math.pi, 2],
        ],
        masks,
        max_planets=48,
    )
    assert len(projected) == 2
    assert sorted((p["source_id"], p["ships"]) for p in projected) == [(0, 10), (3, 2)]
    assert stats["owned_moves"] == 3
    assert stats["projected_moves"] == 2
    assert stats["same_source_lost_moves"] == 1
    assert stats["owned_ship_mass"] == 16
    assert stats["projected_ship_mass"] == 12
    assert stats["same_source_lost_ship_mass"] == 4
    print("test_project_action_keeps_largest_per_source_and_accounts_mass: PASS")


def test_same_source_baseline_preserves_label_kind():
    obs = _obs(step=3)
    label = {
        "source_id": 0,
        "slot": 0,
        "target_idx": 3,
        "target_id": 3,
        "ships": 12,
        "kind": "save",
    }
    baseline = _same_source_nearest_baseline(obs, label, max_planets=48)
    assert baseline is not None
    assert baseline["source_id"] == 0
    assert baseline["slot"] == 0
    assert baseline["ships"] == 12
    assert baseline["kind"] == "save"
    assert baseline["target_id"] == 1
    print("test_same_source_baseline_preserves_label_kind: PASS")


if __name__ == "__main__":
    print("Running replay head-label audit tests...\n")
    test_winner_loser_split_uses_rewards()
    test_replay_obs_missing_step_uses_previous_timing_index()
    test_project_action_keeps_largest_per_source_and_accounts_mass()
    test_same_source_baseline_preserves_label_kind()
    print("\nAll replay head-label audit tests passed!")
