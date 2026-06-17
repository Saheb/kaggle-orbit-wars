"""Regression checks for curated supervised BC dataset building."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from build_supervised_bc import build


def _obs(step: int, p1_owner: int = -1):
    planets = [
        [0, 0, 20.0, 20.0, 1.5, 30, 2],
        [1, p1_owner, 35.0, 20.0, 1.5, 8, 2],
        [2, 1, 80.0, 80.0, 1.5, 20, 2],
    ]
    return {
        "step": step,
        "player": 0,
        "planets": planets,
        "fleets": [],
        "angular_velocity": 0.0,
        "initial_planets": planets,
        "comet_planet_ids": [],
    }


def _agent(obs, action=None):
    return {"observation": obs, "action": action or []}


def test_build_supervised_bc_pairs_prev_obs_and_rebalances(tmp_path):
    replay = {
        "info": {"TeamNames": ["Strong", "Other"]},
        "rewards": [1, -1],
        "steps": [
            [_agent(_obs(0)), _agent(_obs(0))],
            [_agent(_obs(1), []), _agent(_obs(1), [])],
            [_agent(_obs(2), [[0, 0.0, 10]]), _agent(_obs(2), [])],
            [_agent(_obs(3), []), _agent(_obs(3), [])],
        ],
    }
    (tmp_path / "r.json").write_text(json.dumps(replay))

    samples, summary = build(
        [str(tmp_path)],
        noop_keep_prob=0.0,
        fire_repeat=2,
        seed=123,
    )

    assert len(samples) == 2
    assert summary["stats"]["replays_selected"] == 1
    assert summary["stats"]["decision_frames_seen"] == 1
    assert summary["stats"]["noop_frames_seen"] == 2
    assert summary["stats"].get("noop_samples_added", 0) == 0
    assert samples[0]["fire_target"].sum().item() == 1
    assert samples[0]["target_target"][0].item() == 1
    assert summary["decision_sample_share"] == 1.0
    print("test_build_supervised_bc_pairs_prev_obs_and_rebalances: PASS")


def test_build_supervised_bc_steps_min_and_reinforce_repeat(tmp_path):
    replay = {
        "info": {"TeamNames": ["Strong", "Other"]},
        "rewards": [1, -1],
        "steps": [
            [_agent(_obs(0)), _agent(_obs(0))],
            [_agent(_obs(1), [[0, 0.0, 10]]), _agent(_obs(1), [])],
            [_agent(_obs(2, p1_owner=0), []), _agent(_obs(2), [])],
            [_agent(_obs(3), [[1, math.pi, 4]]), _agent(_obs(3), [])],
        ],
    }
    (tmp_path / "r.json").write_text(json.dumps(replay))

    samples, summary = build(
        [str(tmp_path)],
        steps_min=3,
        noop_keep_prob=0.0,
        fire_repeat=1,
        reinforce_repeat=3,
        seed=123,
    )

    assert len(samples) == 3
    assert summary["stats"]["decision_frames_seen"] == 1
    assert summary["stats"]["reinforce_frames_seen"] == 1
    assert summary["stats"]["reinforce_samples_added"] == 3
    fired_slots = samples[0]["fire_target"].nonzero().flatten().tolist()
    assert len(fired_slots) == 1
    assert samples[0]["target_target"][fired_slots[0]].item() == 0
    print("test_build_supervised_bc_steps_min_and_reinforce_repeat: PASS")


if __name__ == "__main__":
    print("Running build_supervised_bc tests...\n")
    with tempfile.TemporaryDirectory() as d:
        test_build_supervised_bc_pairs_prev_obs_and_rebalances(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_build_supervised_bc_steps_min_and_reinforce_repeat(Path(d))
    print("\nAll build_supervised_bc tests passed!")
