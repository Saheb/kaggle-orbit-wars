"""Regression checks for good-play replay scoring."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from score_good_play_replays import (  # noqa: E402
    QualityThresholds,
    build_samples_from_rows,
    score_replay_dirs,
    select_rows_for_samples,
)


def _obs(step: int, owners: dict[int, int], fleets=None):
    planets = [
        [0, owners.get(0, 0), 20.0, 20.0, 1.5, 40, 2],
        [1, owners.get(1, -1), 35.0, 20.0, 1.5, 8, 2],
        [2, owners.get(2, 1), 80.0, 80.0, 1.5, 30, 2],
        [3, owners.get(3, -1), 50.0, 20.0, 1.5, 8, 2],
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


def _good_replay():
    steps = []
    for t in range(21):
        owners = {0: 0, 2: 1}
        if t >= 1:
            owners[1] = 0
        if t >= 3:
            owners[3] = 0
        action = []
        if t == 1:
            action = [[0, 0.0, 10]]
        elif t == 3:
            action = [[1, 0.0, 10]]
        steps.append([_agent(_obs(t, owners), action), _agent(_obs(t, owners), [])])
    return {"info": {"TeamNames": ["Isaiah @ Tufa Labs", "Other"]}, "rewards": [1, -1], "steps": steps}


def _bad_replay():
    steps = [[_agent(_obs(t, {0: 0, 2: 1})), _agent(_obs(t, {0: 0, 2: 1}))] for t in range(21)]
    return {"info": {"TeamNames": ["Idle Winner", "Other"]}, "rewards": [1, -1], "steps": steps}


def test_good_play_scoring_accepts_conversion_and_rejects_idle(tmp_path):
    (tmp_path / "good.json").write_text(json.dumps(_good_replay()))
    (tmp_path / "bad.json").write_text(json.dumps(_bad_replay()))
    rows = score_replay_dirs(
        [str(tmp_path)],
        "*.json",
        QualityThresholds(
            min_attack_launches=1,
            min_cap_attack=0.5,
            min_planets16=2,
            min_planets50=0,
            max_lost_cap=1.0,
            min_score=5.0,
        ),
        winner_filters=[],
        strong_names=["Isaiah"],
    )

    by_name = {r.get("name"): r for r in rows}
    assert by_name["Isaiah @ Tufa Labs"]["accepted"]
    assert not by_name["Idle Winner"]["accepted"]
    assert "too_few_attack_launches:0" in by_name["Idle Winner"]["hard_fails"]

    samples, summary = build_samples_from_rows(
        rows,
        steps_min=1,
        steps_max=5,
        noop_keep_prob=0.0,
        fire_repeat=2,
        reinforce_repeat=1,
        contest_window=0,
        answer_inbound_only=False,
        seed=0,
    )
    assert len(samples) == 4
    assert summary["accepted_replays"] == 1
    assert summary["decision_sample_share"] == 1.0
    print("test_good_play_scoring_accepts_conversion_and_rejects_idle: PASS")


def test_contest_window_keeps_recent_capture_frames(tmp_path):
    (tmp_path / "good.json").write_text(json.dumps(_good_replay()))
    rows = score_replay_dirs(
        [str(tmp_path)],
        "*.json",
        QualityThresholds(
            min_attack_launches=1,
            min_cap_attack=0.5,
            min_planets16=2,
            min_planets50=0,
            max_lost_cap=1.0,
            min_score=5.0,
        ),
        winner_filters=[],
        strong_names=["Isaiah"],
    )

    samples, summary = build_samples_from_rows(
        rows,
        steps_min=2,
        steps_max=4,
        noop_keep_prob=0.0,
        fire_repeat=2,
        reinforce_repeat=1,
        contest_window=3,
        answer_inbound_only=False,
        seed=0,
    )

    assert len(samples) == 2
    assert summary["stats"]["contest_frames_seen"] == 3
    assert summary["stats"]["contest_recent_capture"] == 3
    assert summary["stats"]["decision_frames_seen"] == 1
    print("test_contest_window_keeps_recent_capture_frames: PASS")


def test_answer_inbound_filters_unrelated_moves(tmp_path):
    inbound = [[9, 1, 25.0, 20.0, 0.0, 2, 12]]
    replay = {
        "info": {"TeamNames": ["Strong", "Other"]},
        "rewards": [1, -1],
        "steps": [
            [_agent(_obs(0, {0: 0, 1: 0, 2: 1})), _agent(_obs(0, {0: 0, 1: 0, 2: 1}))],
            [_agent(_obs(1, {0: 0, 1: 0, 2: 1}, fleets=inbound)), _agent(_obs(1, {0: 0, 1: 0, 2: 1}))],
            [
                _agent(_obs(2, {0: 0, 1: 0, 2: 1}), [[1, math.pi, 4], [0, math.pi / 4, 10]]),
                _agent(_obs(2, {0: 0, 1: 0, 2: 1})),
            ],
        ],
    }
    replay_path = tmp_path / "inbound.json"
    replay_path.write_text(json.dumps(replay))
    rows = [{
        "accepted": True,
        "replay_path": str(replay_path),
        "seat": 0,
        "name": "Strong",
    }]

    samples, summary = build_samples_from_rows(
        rows,
        steps_min=2,
        steps_max=2,
        noop_keep_prob=0.0,
        fire_repeat=1,
        reinforce_repeat=1,
        contest_window=0,
        answer_inbound_only=True,
        seed=0,
    )

    assert len(samples) == 1
    assert summary["stats"]["answer_frames_kept"] == 1
    assert summary["stats"]["answer_source_threatened"] == 1
    assert samples[0]["fire_target"].sum().item() == 1
    print("test_answer_inbound_filters_unrelated_moves: PASS")


def test_answer_inbound_repeat_preserves_full_label(tmp_path):
    inbound = [[9, 1, 25.0, 20.0, 0.0, 2, 12]]
    replay = {
        "info": {"TeamNames": ["Strong", "Other"]},
        "rewards": [1, -1],
        "steps": [
            [_agent(_obs(0, {0: 0, 1: 0, 2: 1})), _agent(_obs(0, {0: 0, 1: 0, 2: 1}))],
            [_agent(_obs(1, {0: 0, 1: 0, 2: 1}, fleets=inbound)), _agent(_obs(1, {0: 0, 1: 0, 2: 1}))],
            [
                _agent(_obs(2, {0: 0, 1: 0, 2: 1}), [[1, math.pi, 4], [0, math.pi / 4, 10]]),
                _agent(_obs(2, {0: 0, 1: 0, 2: 1})),
            ],
        ],
    }
    replay_path = tmp_path / "inbound_soft.json"
    replay_path.write_text(json.dumps(replay))
    rows = [{
        "accepted": True,
        "replay_path": str(replay_path),
        "seat": 0,
        "name": "Strong",
    }]

    samples, summary = build_samples_from_rows(
        rows,
        steps_min=2,
        steps_max=2,
        noop_keep_prob=0.0,
        fire_repeat=1,
        reinforce_repeat=1,
        contest_window=0,
        answer_inbound_only=False,
        answer_inbound_repeat=3,
        seed=0,
    )

    assert len(samples) == 3
    assert summary["stats"]["answer_frames_weighted"] == 1
    assert summary["stats"]["answer_source_threatened"] == 1
    assert samples[0]["fire_target"].sum().item() == 2
    print("test_answer_inbound_repeat_preserves_full_label: PASS")


def test_held_capture_repeat_uses_future_hold_success(tmp_path):
    replay = {
        "info": {"TeamNames": ["Strong", "Other"]},
        "rewards": [1, -1],
        "steps": [
            [_agent(_obs(0, {0: 0, 2: 1})), _agent(_obs(0, {0: 0, 2: 1}))],
            [_agent(_obs(1, {0: 0, 1: 0, 2: 1})), _agent(_obs(1, {0: 0, 1: 0, 2: 1}))],
            [
                _agent(_obs(2, {0: 0, 1: 0, 2: 1}), [[0, 0.0, 8], [1, math.pi, 4]]),
                _agent(_obs(2, {0: 0, 1: 0, 2: 1})),
            ],
            [_agent(_obs(3, {0: 0, 1: 0, 2: 1})), _agent(_obs(3, {0: 0, 1: 0, 2: 1}))],
            [_agent(_obs(4, {0: 0, 1: 0, 2: 1})), _agent(_obs(4, {0: 0, 1: 0, 2: 1}))],
        ],
    }
    replay_path = tmp_path / "held_capture.json"
    replay_path.write_text(json.dumps(replay))
    rows = [{
        "accepted": True,
        "replay_path": str(replay_path),
        "seat": 0,
        "name": "Strong",
    }]

    samples, summary = build_samples_from_rows(
        rows,
        steps_min=2,
        steps_max=2,
        noop_keep_prob=0.0,
        fire_repeat=1,
        reinforce_repeat=1,
        contest_window=0,
        answer_inbound_only=False,
        held_capture_window=4,
        hold_success_horizon=2,
        held_capture_repeat=3,
        seed=0,
    )

    assert len(samples) == 3
    assert summary["stats"]["held_capture_frames_weighted"] == 1
    assert summary["stats"]["held_capture_target_moves"] == 1
    assert summary["stats"]["held_capture_source_moves"] == 1
    assert samples[0]["fire_target"].sum().item() == 2

    replay["steps"][3][0]["observation"] = _obs(3, {0: 0, 1: 1, 2: 1})
    replay["steps"][4][0]["observation"] = _obs(4, {0: 0, 1: 1, 2: 1})
    replay_path.write_text(json.dumps(replay))
    samples, summary = build_samples_from_rows(
        rows,
        steps_min=2,
        steps_max=2,
        noop_keep_prob=0.0,
        fire_repeat=1,
        reinforce_repeat=1,
        contest_window=0,
        answer_inbound_only=False,
        held_capture_window=4,
        hold_success_horizon=2,
        held_capture_repeat=3,
        held_capture_only=True,
        seed=0,
    )

    assert len(samples) == 0
    assert summary["stats"]["held_capture_rejected_future_loss"] == 1
    assert summary["stats"]["frames_skipped_no_held_capture"] == 1
    print("test_held_capture_repeat_uses_future_hold_success: PASS")


def test_synthetic_defense_adds_reinforce_label(tmp_path):
    inbound = [[9, 1, 25.0, 20.0, 0.0, 2, 30]]
    replay = {
        "info": {"TeamNames": ["Strong", "Other"]},
        "rewards": [1, -1],
        "steps": [
            [_agent(_obs(0, {0: 0, 1: 0, 2: 1})), _agent(_obs(0, {0: 0, 1: 0, 2: 1}))],
            [_agent(_obs(1, {0: 0, 1: 0, 2: 1}, fleets=inbound)), _agent(_obs(1, {0: 0, 1: 0, 2: 1}))],
            [_agent(_obs(2, {0: 0, 1: 0, 2: 1}), []), _agent(_obs(2, {0: 0, 1: 0, 2: 1}))],
        ],
    }
    replay_path = tmp_path / "synthetic_defense.json"
    replay_path.write_text(json.dumps(replay))
    rows = [{
        "accepted": True,
        "replay_path": str(replay_path),
        "seat": 0,
        "name": "Strong",
    }]

    samples, summary = build_samples_from_rows(
        rows,
        steps_min=2,
        steps_max=2,
        noop_keep_prob=0.0,
        fire_repeat=1,
        reinforce_repeat=1,
        contest_window=0,
        answer_inbound_only=False,
        synthetic_defense_repeat=3,
        seed=0,
    )

    assert len(samples) == 3
    assert summary["stats"]["synthetic_defense_frames"] == 1
    assert summary["stats"]["synthetic_defense_moves"] == 1
    fired_slots = samples[0]["fire_target"].nonzero().flatten().tolist()
    assert len(fired_slots) == 1
    assert samples[0]["target_target"][fired_slots[0]].item() == 1
    print("test_synthetic_defense_adds_reinforce_label: PASS")


def test_threat_horizon_labels_future_owned_planet_loss(tmp_path):
    replay = {
        "info": {"TeamNames": ["Strong", "Other"]},
        "rewards": [1, -1],
        "steps": [
            [_agent(_obs(0, {0: 0, 1: 0, 2: 1})), _agent(_obs(0, {0: 0, 1: 0, 2: 1}))],
            [_agent(_obs(1, {0: 0, 1: 0, 2: 1})), _agent(_obs(1, {0: 0, 1: 0, 2: 1}))],
            [_agent(_obs(2, {0: 0, 1: 1, 2: 1}), [[0, 0.0, 10]]), _agent(_obs(2, {0: 0, 1: 1, 2: 1}))],
        ],
    }
    replay_path = tmp_path / "threat.json"
    replay_path.write_text(json.dumps(replay))
    rows = [{
        "accepted": True,
        "replay_path": str(replay_path),
        "seat": 0,
        "name": "Strong",
    }]

    samples, summary = build_samples_from_rows(
        rows,
        steps_min=2,
        steps_max=2,
        noop_keep_prob=0.0,
        fire_repeat=1,
        reinforce_repeat=1,
        contest_window=0,
        answer_inbound_only=False,
        threat_horizon=2,
        seed=0,
    )

    assert len(samples) == 1
    assert summary["stats"]["threat_slots"] == 2
    assert summary["stats"]["threat_pos_slots"] == 1
    assert samples[0]["threat_target"].sum().item() == 1
    threatened_slots = samples[0]["threat_target"].nonzero().flatten().tolist()
    assert samples[0]["owned_indices"][threatened_slots[0]].item() == 1
    print("test_threat_horizon_labels_future_owned_planet_loss: PASS")


def test_select_rows_for_samples_caps_subjects_after_score_sort():
    rows = [
        {"accepted": True, "name": "A", "score": 9.0, "replay_path": "a1.json"},
        {"accepted": True, "name": "A", "score": 8.0, "replay_path": "a2.json"},
        {"accepted": True, "name": "B", "score": 7.0, "replay_path": "b1.json"},
        {"accepted": True, "name": "A", "score": 6.0, "replay_path": "a3.json"},
        {"accepted": False, "name": "C", "score": 10.0, "replay_path": "c1.json"},
    ]

    selected, summary = select_rows_for_samples(rows, max_accepted_per_subject=2)

    assert [r["replay_path"] for r in selected] == ["a1.json", "a2.json", "b1.json"]
    assert summary["candidate_accepted_replays"] == 4
    assert summary["selected_accepted_replays"] == 3
    assert summary["skipped_by_subject_cap"] == 1
    assert summary["selected_subjects"] == {"A": 2, "B": 1}
    print("test_select_rows_for_samples_caps_subjects_after_score_sort: PASS")


def test_build_samples_caps_final_samples_per_subject(tmp_path):
    replay_path = tmp_path / "good.json"
    replay_path.write_text(json.dumps(_good_replay()))
    rows = [{
        "accepted": True,
        "replay_path": str(replay_path),
        "seat": 0,
        "name": "A",
    }]

    samples, summary = build_samples_from_rows(
        rows,
        steps_min=1,
        steps_max=1,
        noop_keep_prob=0.0,
        fire_repeat=4,
        reinforce_repeat=1,
        contest_window=0,
        answer_inbound_only=False,
        max_samples_per_subject=2,
        seed=0,
    )

    assert len(samples) == 2
    assert summary["subject_samples"] == {"A": 2}
    assert summary["subject_decision_samples"] == {"A": 2}
    assert summary["stats"]["decision_samples_added"] == 2
    assert summary["stats"]["samples_skipped_subject_sample_cap"] == 2
    print("test_build_samples_caps_final_samples_per_subject: PASS")


if __name__ == "__main__":
    print("Running score_good_play_replays tests...\n")
    with tempfile.TemporaryDirectory() as d:
        test_good_play_scoring_accepts_conversion_and_rejects_idle(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_contest_window_keeps_recent_capture_frames(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_answer_inbound_filters_unrelated_moves(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_answer_inbound_repeat_preserves_full_label(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_held_capture_repeat_uses_future_hold_success(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_synthetic_defense_adds_reinforce_label(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_threat_horizon_labels_future_owned_planet_loss(Path(d))
    test_select_rows_for_samples_caps_subjects_after_score_sort()
    with tempfile.TemporaryDirectory() as d:
        test_build_samples_caps_final_samples_per_subject(Path(d))
    print("\nAll score_good_play_replays tests passed!")
