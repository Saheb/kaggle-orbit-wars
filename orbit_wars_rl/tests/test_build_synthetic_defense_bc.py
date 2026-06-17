"""Regression checks for synthetic-defense BC dataset building."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from build_synthetic_defense_bc import build  # noqa: E402


def _obs(step: int):
    planets = [
        [0, 0, 20.0, 20.0, 1.5, 60, 2],   # rear source
        [1, 0, 50.0, 20.0, 1.5, 4, 2],    # threatened target
        [2, 1, 80.0, 20.0, 1.5, 20, 2],   # enemy
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


def _with_owner(obs, pid, owner):
    out = json.loads(json.dumps(obs))
    for planet in out["planets"]:
        if int(planet[0]) == pid:
            planet[1] = owner
    return out


def test_synthetic_defense_builder_emits_rear_support(tmp_path):
    replay = {
        "info": {"TeamNames": ["Strong", "Other"]},
        "rewards": [1, -1],
        "steps": [
            [_agent(_obs(0)), _agent(_obs(0))],
            [_agent(_obs(1)), _agent(_obs(1))],
        ],
    }
    (tmp_path / "r.json").write_text(json.dumps(replay))

    samples, summary = build(
        replay_dirs=[str(tmp_path)],
        steps_max=1,
        garrison_floor=10,
        min_need=5,
        seed=123,
    )

    assert len(samples) == 1
    assert summary["stats"]["synthetic_defense_frames"] == 1
    assert summary["stats"]["synthetic_defense_moves"] == 1
    sample = samples[0]
    fired_slots = sample["fire_target"].nonzero().flatten().tolist()
    assert len(fired_slots) == 1
    slot = fired_slots[0]
    assert sample["owned_indices"][slot].item() == 0
    assert sample["target_target"][slot].item() == 1
    assert sample["ship_target"][slot].item() >= 4
    print("test_synthetic_defense_builder_emits_rear_support: PASS")


def test_hold_success_filter_skips_future_loss(tmp_path):
    replay = {
        "info": {"TeamNames": ["Strong", "Other"]},
        "rewards": [1, -1],
        "steps": [
            [_agent(_obs(0)), _agent(_obs(0))],
            [_agent(_obs(1)), _agent(_obs(1))],
            [_agent(_with_owner(_obs(2), 1, 1)), _agent(_obs(2))],
            [_agent(_with_owner(_obs(3), 1, 1)), _agent(_obs(3))],
        ],
    }
    (tmp_path / "r.json").write_text(json.dumps(replay))

    samples, summary = build(
        replay_dirs=[str(tmp_path)],
        steps_max=1,
        garrison_floor=10,
        min_need=5,
        hold_success_horizon=4,
        seed=123,
    )

    assert samples == []
    assert summary["stats"]["hold_success_skipped_future_loss"] >= 1
    print("test_hold_success_filter_skips_future_loss: PASS")


if __name__ == "__main__":
    print("Running build_synthetic_defense_bc tests...\n")
    with tempfile.TemporaryDirectory() as d:
        test_synthetic_defense_builder_emits_rear_support(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_hold_success_filter_skips_future_loss(Path(d))
    print("\nAll build_synthetic_defense_bc tests passed!")
