"""Regression checks for top-player sharded BC extraction."""

from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from build_best_player_bc_shards import build_shards


def _obs(step: int, player: int = 0):
    planets = [
        [0, player, 20.0, 20.0, 1.5, 30, 2],
        [1, -1, 35.0, 20.0, 1.5, 8, 2],
        [2, 1 - player, 80.0, 80.0, 1.5, 20, 2],
    ]
    return {
        "step": step,
        "player": player,
        "planets": planets,
        "fleets": [],
        "angular_velocity": 0.0,
        "initial_planets": planets,
        "comet_planet_ids": [],
    }


def _agent(obs, action=None):
    return {"observation": obs, "action": action or []}


def test_build_best_player_bc_shards_filters_and_writes_manifest(tmp_path):
    replay = {
        "info": {"TeamNames": ["Jake Will", "Other"]},
        "rewards": [1, -1],
        "steps": [
            [_agent(_obs(0, 0)), _agent(_obs(0, 1))],
            [_agent(_obs(1, 0), []), _agent(_obs(1, 1), [])],
            [_agent(_obs(2, 0), [[0, 0.0, 10]]), _agent(_obs(2, 1), [[2, 3.14, 10]])],
        ],
    }
    (tmp_path / "r.json").write_text(json.dumps(replay))

    summary = build_shards(
        [str(tmp_path)],
        out_dir=str(tmp_path / "shards"),
        player_filters=["Jake"],
        noop_keep_prob=0.0,
        samples_per_shard=1,
        seed=7,
    )

    assert summary["samples"] == 1
    assert summary["stats"]["replays_selected"] == 1
    assert summary["subjects"] == {"Jake Will": 1}
    assert len(summary["sample_paths"]) == 1
    assert Path(summary["manifest_path"]).exists()
    samples = pickle.load(open(summary["sample_paths"][0], "rb"))
    assert samples[0]["fire_target"].sum().item() == 1
    print("test_build_best_player_bc_shards_filters_and_writes_manifest: PASS")


def test_build_best_player_bc_shards_compact_frame_format(tmp_path):
    replay = {
        "info": {"TeamNames": ["Jake Will", "Other"]},
        "rewards": [1, -1],
        "steps": [
            [_agent(_obs(0, 0)), _agent(_obs(0, 1))],
            [_agent(_obs(1, 0), [[0, 0.0, 10]]), _agent(_obs(1, 1), [])],
        ],
    }
    (tmp_path / "r.json").write_text(json.dumps(replay))

    summary = build_shards(
        [str(tmp_path)],
        out_dir=str(tmp_path / "shards"),
        player_filters=["Jake"],
        noop_keep_prob=0.0,
        output_format="frame",
        seed=7,
    )

    assert summary["config"]["format"] == "frame"
    records = pickle.load(open(summary["sample_paths"][0], "rb"))
    assert "obs" in records[0]
    assert "action" in records[0]
    assert "planet_features" not in records[0]
    assert records[0]["player_name"] == "Jake Will"
    print("test_build_best_player_bc_shards_compact_frame_format: PASS")


def test_build_best_player_bc_shards_require_win(tmp_path):
    replay = {
        "info": {"TeamNames": ["Isaiah @ Tufa Labs", "Other"]},
        "rewards": [-1, 1],
        "steps": [
            [_agent(_obs(0, 0)), _agent(_obs(0, 1))],
            [_agent(_obs(1, 0), [[0, 0.0, 10]]), _agent(_obs(1, 1), [])],
        ],
    }
    (tmp_path / "r.json").write_text(json.dumps(replay))

    summary = build_shards(
        [str(tmp_path)],
        out_dir=str(tmp_path / "shards"),
        player_filters=["Isaiah"],
        require_win=True,
        noop_keep_prob=0.0,
        seed=7,
    )

    assert summary["samples"] == 0
    assert summary["stats"]["replays_without_matching_player"] == 1
    print("test_build_best_player_bc_shards_require_win: PASS")


def test_build_best_player_bc_shards_keeps_nonwins_with_metadata(tmp_path):
    replay = {
        "info": {"TeamNames": ["Jake Will", "Other"]},
        "rewards": [-1, 1],
        "steps": [
            [_agent(_obs(0, 0)), _agent(_obs(0, 1))],
            [_agent(_obs(1, 0), [[0, 0.0, 10]]), _agent(_obs(1, 1), [])],
        ],
    }
    (tmp_path / "r.json").write_text(json.dumps(replay))

    summary = build_shards(
        [str(tmp_path)],
        out_dir=str(tmp_path / "shards"),
        player_filters=["Jake"],
        noop_keep_prob=0.0,
        output_format="frame",
        nonwin_keep_prob=1.0,
        nonwin_repeat=2,
        seed=7,
    )

    assert summary["samples"] == 2
    assert summary["stats"]["nonwin_seats_selected"] == 1
    records = pickle.load(open(summary["sample_paths"][0], "rb"))
    assert records[0]["won"] is False
    assert records[0]["reward"] == -1.0
    print("test_build_best_player_bc_shards_keeps_nonwins_with_metadata: PASS")


if __name__ == "__main__":
    print("Running build_best_player_bc_shards tests...\n")
    with tempfile.TemporaryDirectory() as d:
        test_build_best_player_bc_shards_filters_and_writes_manifest(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_build_best_player_bc_shards_compact_frame_format(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_build_best_player_bc_shards_require_win(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_build_best_player_bc_shards_keeps_nonwins_with_metadata(Path(d))
    print("\nAll build_best_player_bc_shards tests passed!")
