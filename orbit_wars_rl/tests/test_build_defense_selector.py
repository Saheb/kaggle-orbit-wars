"""Regression checks for defense-overlay selector dataset building."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from build_defense_selector import build_records  # noqa: E402


def _obs(step: int, target_owner: int = 0):
    planets = [
        [0, 0, 20.0, 20.0, 1.5, 60, 2],
        [1, target_owner, 50.0, 20.0, 1.5, 4, 2],
        [2, 1, 80.0, 20.0, 1.5, 20, 2],
    ]
    inbound = [[9, 1, 40.0, 20.0, 0.0, 2, 30]]
    return {
        "step": step,
        "player": 0,
        "planets": planets,
        "fleets": inbound,
        "angular_velocity": 0.0,
        "initial_planets": planets,
        "comet_planet_ids": [],
    }


def _agent(obs):
    return {"observation": obs, "action": []}


def test_defense_selector_labels_future_loss(tmp_path):
    replay = {
        "info": {"TeamNames": ["Strong", "Other"]},
        "rewards": [1, -1],
        "steps": [
            [_agent(_obs(0, target_owner=1)), _agent(_obs(0, target_owner=1))],
            [_agent(_obs(1, target_owner=0)), _agent(_obs(1, target_owner=0))],
            [_agent(_obs(2, target_owner=0)), _agent(_obs(2, target_owner=0))],
            [_agent(_obs(3, target_owner=1)), _agent(_obs(3, target_owner=1))],
        ],
    }
    (tmp_path / "r.json").write_text(json.dumps(replay))

    records, summary = build_records(
        replay_dirs=[str(tmp_path)],
        seat_mode="slot",
        player_slot=0,
        steps_min=2,
        steps_max=2,
        hold_horizon=4,
        garrison_floor=10,
        min_need=5,
        max_target_age=40,
    )

    assert len(records) == 1
    assert records[0]["label"] == 0
    assert records[0]["future_loss"] is True
    assert summary["stats"]["negative_lost"] == 1
    print("test_defense_selector_labels_future_loss: PASS")


if __name__ == "__main__":
    print("Running build_defense_selector tests...\n")
    with tempfile.TemporaryDirectory() as d:
        test_defense_selector_labels_future_loss(Path(d))
    print("\nAll build_defense_selector tests passed!")
