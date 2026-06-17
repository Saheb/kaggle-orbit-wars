"""Regression checks for policy-teacher supervised BC dataset building."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from build_policy_teacher_bc import build  # noqa: E402


def _obs(step: int, owners=None, fleets=None):
    owners = owners or {0: 0, 1: -1, 2: 1}
    planets = [
        [0, owners.get(0, 0), 20.0, 20.0, 1.5, 30, 2],
        [1, owners.get(1, -1), 35.0, 20.0, 1.5, 8, 2],
        [2, owners.get(2, 1), 80.0, 80.0, 1.5, 20, 2],
    ]
    return {
        "step": step,
        "player": 0,
        "planets": planets,
        "fleets": fleets or [],
        "angular_velocity": 0.0,
        "initial_planets": planets,
        "comet_planet_ids": [],
    }


def _agent(obs, action=None):
    return {"observation": obs, "action": action or []}


def test_policy_teacher_pairs_prev_obs_and_warms_teacher(tmp_path):
    teacher_path = tmp_path / "teacher.py"
    teacher_path.write_text(
        "CALLS = []\n"
        "def agent(obs):\n"
        "    CALLS.append(obs['step'])\n"
        "    if obs['step'] == 1:\n"
        "        return [[0, 0.0, 10]]\n"
        "    return []\n"
    )
    replay = {
        "info": {"TeamNames": ["Strong", "Other"]},
        "rewards": [1, -1],
        "steps": [
            [_agent(_obs(0)), _agent(_obs(0))],
            [_agent(_obs(1)), _agent(_obs(1))],
            [_agent(_obs(2)), _agent(_obs(2))],
        ],
    }
    (tmp_path / "r.json").write_text(json.dumps(replay))

    samples, summary = build(
        replay_dirs=[str(tmp_path)],
        teacher_agent=str(teacher_path),
        steps_min=2,
        steps_max=2,
        noop_keep_prob=0.0,
        action_repeat=2,
        seed=123,
    )

    assert len(samples) == 2
    assert summary["stats"]["teacher_warmup_frames"] == 1
    assert summary["stats"]["teacher_decision_frames_seen"] == 1
    assert summary["stats"]["teacher_decision_samples_added"] == 2
    assert summary["subjects"] == {"Strong": 1}
    assert samples[0]["fire_target"].sum().item() == 1
    assert samples[0]["target_target"][0].item() == 1
    print("test_policy_teacher_pairs_prev_obs_and_warms_teacher: PASS")


def test_policy_teacher_split_moves_emits_single_move_samples(tmp_path):
    teacher_path = tmp_path / "teacher.py"
    teacher_path.write_text(
        "def agent(obs):\n"
        "    return [[0, 0.0, 10], [0, 0.0, 5]] if obs['step'] == 0 else []\n"
    )
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
        teacher_agent=str(teacher_path),
        steps_max=1,
        noop_keep_prob=0.0,
        split_moves=True,
        seed=123,
    )

    assert len(samples) == 2
    assert summary["stats"]["teacher_split_move_labels"] == 2
    assert summary["stats"]["teacher_split_move_samples_added"] == 2
    assert samples[0]["fire_target"].sum().item() == 1
    assert samples[1]["fire_target"].sum().item() == 1
    print("test_policy_teacher_split_moves_emits_single_move_samples: PASS")


def test_policy_teacher_filters_multi_move_frames(tmp_path):
    teacher_path = tmp_path / "teacher.py"
    teacher_path.write_text(
        "def agent(obs):\n"
        "    return [[0, 0.0, 10], [0, 0.0, 5]] if obs['step'] == 0 else []\n"
    )
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
        teacher_agent=str(teacher_path),
        steps_max=1,
        noop_keep_prob=0.0,
        split_moves=True,
        max_teacher_moves_per_frame=1,
        seed=123,
    )

    assert len(samples) == 0
    assert summary["stats"]["frames_skipped_too_many_teacher_moves"] == 1
    assert summary["stats"]["moves_skipped_too_many_teacher_moves"] == 2
    print("test_policy_teacher_filters_multi_move_frames: PASS")


def test_policy_teacher_filters_multi_move_source(tmp_path):
    teacher_path = tmp_path / "teacher.py"
    teacher_path.write_text(
        "def agent(obs):\n"
        "    return [[0, 0.0, 10], [0, 0.5, 5]] if obs['step'] == 0 else []\n"
    )
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
        teacher_agent=str(teacher_path),
        steps_max=1,
        noop_keep_prob=0.0,
        split_moves=True,
        max_teacher_moves_per_source=1,
        seed=123,
    )

    assert len(samples) == 0
    assert summary["stats"]["frames_filtered_too_many_teacher_moves_per_source"] == 1
    assert summary["stats"]["moves_skipped_too_many_teacher_moves_per_source"] == 2
    assert summary["stats"]["frames_skipped_no_moves_after_source_filter"] == 1
    print("test_policy_teacher_filters_multi_move_source: PASS")


def test_policy_teacher_filters_target_owner(tmp_path):
    teacher_path = tmp_path / "teacher.py"
    teacher_path.write_text(
        "def agent(obs):\n"
        "    return [[0, 0.0, 10]] if obs['step'] == 0 else []\n"
    )
    replay = {
        "info": {"TeamNames": ["Strong", "Other"]},
        "rewards": [1, -1],
        "steps": [
            [_agent(_obs(0)), _agent(_obs(0))],
            [_agent(_obs(1)), _agent(_obs(1))],
        ],
    }
    (tmp_path / "r.json").write_text(json.dumps(replay))

    enemy_samples, enemy_summary = build(
        replay_dirs=[str(tmp_path)],
        teacher_agent=str(teacher_path),
        steps_max=1,
        noop_keep_prob=0.0,
        split_moves=True,
        target_owner="enemy",
        seed=123,
    )
    neutral_samples, neutral_summary = build(
        replay_dirs=[str(tmp_path)],
        teacher_agent=str(teacher_path),
        steps_max=1,
        noop_keep_prob=0.0,
        split_moves=True,
        target_owner="neutral",
        seed=123,
    )

    assert len(enemy_samples) == 0
    assert enemy_summary["stats"]["samples_skipped_target_owner_enemy"] == 1
    assert len(neutral_samples) == 1
    assert neutral_summary["stats"]["target_labels"] == 1
    print("test_policy_teacher_filters_target_owner: PASS")


def test_policy_teacher_filters_threatened_own_targets(tmp_path):
    teacher_path = tmp_path / "teacher.py"
    teacher_path.write_text(
        "def agent(obs):\n"
        "    return [[0, 0.0, 10]] if obs['step'] == 0 else []\n"
    )
    inbound = [[9, 1, 25.0, 20.0, 0.0, 2, 12]]
    replay = {
        "info": {"TeamNames": ["Strong", "Other"]},
        "rewards": [1, -1],
        "steps": [
            [_agent(_obs(0, owners={0: 0, 1: 0, 2: 1}, fleets=inbound)),
             _agent(_obs(0, owners={0: 0, 1: 0, 2: 1}))],
            [_agent(_obs(1, owners={0: 0, 1: 0, 2: 1}, fleets=inbound)),
             _agent(_obs(1, owners={0: 0, 1: 0, 2: 1}))],
        ],
    }
    (tmp_path / "r.json").write_text(json.dumps(replay))

    kept, kept_summary = build(
        replay_dirs=[str(tmp_path)],
        teacher_agent=str(teacher_path),
        steps_max=1,
        noop_keep_prob=0.0,
        split_moves=True,
        target_owner="own",
        inbound_threat_horizon=20,
        seed=123,
    )
    dropped, dropped_summary = build(
        replay_dirs=[str(tmp_path)],
        teacher_agent=str(teacher_path),
        steps_max=1,
        noop_keep_prob=0.0,
        split_moves=True,
        target_owner="own",
        inbound_threat_horizon=1,
        seed=123,
    )

    assert len(kept) == 1
    assert kept_summary["stats"]["inbound_threat_frames_seen"] == 1
    assert kept_summary["stats"]["threat_target_samples_added"] == 1
    assert len(dropped) == 0
    assert dropped_summary["stats"]["frames_skipped_no_inbound_threat"] == 1
    print("test_policy_teacher_filters_threatened_own_targets: PASS")


if __name__ == "__main__":
    print("Running build_policy_teacher_bc tests...\n")
    with tempfile.TemporaryDirectory() as d:
        test_policy_teacher_pairs_prev_obs_and_warms_teacher(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_policy_teacher_split_moves_emits_single_move_samples(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_policy_teacher_filters_multi_move_frames(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_policy_teacher_filters_multi_move_source(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_policy_teacher_filters_target_owner(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_policy_teacher_filters_threatened_own_targets(Path(d))
    print("\nAll build_policy_teacher_bc tests passed!")
