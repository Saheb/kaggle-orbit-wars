"""Regression checks for BC loss weighting."""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bc import bc_loss
from bc import _assert_supervised_init_checkpoint
from bc import _find_ship_bin
from bc import _load_compatible_init
from bc import _metric_improved
from bc import _records_to_training_samples
from bc import _sample_streaming_validation_records
from config import Config
from model import EntityTransformer


def _obs(step: int):
    planets = [
        [0, 0, 20.0, 20.0, 1.5, 30, 2],
        [1, -1, 35.0, 20.0, 1.5, 8, 2],
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


def test_fire_pos_weight_changes_fire_bce():
    batch = {
        "slot_valid": torch.tensor([[1, 1]], dtype=torch.bool),
        "fire_target": torch.tensor([[1, 0]], dtype=torch.long),
        "ship_target": torch.tensor([[0, 0]], dtype=torch.long),
        "target_target": torch.tensor([[0, -1]], dtype=torch.long),
    }
    outputs = {
        "fire_logits": torch.tensor([[-2.0, 0.0]], dtype=torch.float32),
        "ship_logits": torch.zeros((1, 2, 3), dtype=torch.float32),
        "target_logits": torch.zeros((1, 2, 4), dtype=torch.float32),
    }

    _, plain = bc_loss(outputs, batch, fire_pos_weight=1.0)
    _, weighted = bc_loss(outputs, batch, fire_pos_weight=5.0)

    assert weighted["fire_loss"] > plain["fire_loss"]
    assert weighted["fire_pos_weight"] == 5.0
    print("test_fire_pos_weight_changes_fire_bce: PASS")


def test_threat_loss_uses_threat_labels():
    batch = {
        "slot_valid": torch.tensor([[1, 1]], dtype=torch.bool),
        "fire_target": torch.tensor([[1, 0]], dtype=torch.long),
        "ship_target": torch.tensor([[0, 0]], dtype=torch.long),
        "target_target": torch.tensor([[0, -1]], dtype=torch.long),
        "threat_target": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "threat_mask": torch.tensor([[1, 1]], dtype=torch.bool),
    }
    outputs = {
        "fire_logits": torch.tensor([[2.0, -2.0]], dtype=torch.float32),
        "ship_logits": torch.zeros((1, 2, 3), dtype=torch.float32),
        "target_logits": torch.zeros((1, 2, 4), dtype=torch.float32),
        "threat_logits": torch.tensor([[-2.0, 0.0]], dtype=torch.float32),
    }

    _, plain = bc_loss(outputs, batch, threat_loss_weight=1.0, threat_pos_weight=1.0)
    _, weighted = bc_loss(outputs, batch, threat_loss_weight=1.0, threat_pos_weight=5.0)

    assert "threat_loss" in plain
    assert weighted["threat_loss"] > plain["threat_loss"]
    assert weighted["threat_pos_weight"] == 5.0
    print("test_threat_loss_uses_threat_labels: PASS")


def test_rl_init_checkpoint_is_rejected_by_default(tmp_path):
    ckpt = tmp_path / "ppo.pt"
    torch.save({
        "model": {},
        "optimizer": {"state": {}, "param_groups": []},
        "total_steps": 123,
        "update_count": 4,
    }, ckpt)

    try:
        _assert_supervised_init_checkpoint(str(ckpt), allow_rl_init=False)
    except SystemExit as e:
        assert "PPO/RL" in str(e)
    else:
        raise AssertionError("PPO/RL checkpoint was not rejected")

    _assert_supervised_init_checkpoint(str(ckpt), allow_rl_init=True)
    print("test_rl_init_checkpoint_is_rejected_by_default: PASS")


def test_partial_init_skips_fraction_ship_head_shape(tmp_path):
    src_cfg = Config()
    src_model = EntityTransformer(src_cfg.model)
    ckpt = tmp_path / "absolute.pt"
    torch.save({"model": src_model.state_dict(), "config": {"ship_bin_mode": "absolute"}}, ckpt)

    dst_cfg = Config()
    dst_cfg.model.ship_bin_mode = "fraction"
    dst_cfg.model.num_ship_bins = 10
    dst_model = EntityTransformer(dst_cfg.model)
    report = _load_compatible_init(dst_model, str(ckpt))

    skipped = {name for name, _, _ in report["skipped_shape"]}
    assert "ship_head.weight" in skipped
    assert "ship_head.bias" in skipped
    assert "fire_head.weight" in report["loaded"]
    assert "target_scorer.0.weight" in report["loaded"]
    print("test_partial_init_skips_fraction_ship_head_shape: PASS")


def test_select_metric_direction():
    assert _metric_improved("val_loss", 4.0, 4.02)
    assert not _metric_improved("val_loss", 4.02, 4.0)
    assert _metric_improved("val_target_top3", 0.51, 0.50)
    assert not _metric_improved("val_target_top3", 0.50, 0.51)
    assert _metric_improved("target_red", 0.402, 0.400)
    print("test_select_metric_direction: PASS")


def test_compact_records_convert_to_training_samples():
    records = [{
        "obs": _obs(0),
        "action": [[0, 0.0, 10]],
    }]
    samples = _records_to_training_samples(records)
    assert len(samples) == 1
    assert samples[0]["fire_target"].sum().item() == 1
    assert samples[0]["target_target"][0].item() == 1
    print("test_compact_records_convert_to_training_samples: PASS")


def test_fraction_ship_labels_use_source_ship_ratio():
    assert _find_ship_bin(10, max_ships=30, mode="fraction") == 2
    assert _find_ship_bin(30, max_ships=30, mode="fraction") == 9

    records = [{
        "obs": _obs(0),
        "action": [[0, 0.0, 10]],
    }]
    samples = _records_to_training_samples(records, ship_bin_mode="fraction")
    assert len(samples) == 1
    assert samples[0]["ship_target"][0].item() == 2
    print("test_fraction_ship_labels_use_source_ship_ratio: PASS")


def test_fraction_mode_rejects_legacy_tensor_ship_labels():
    legacy = {
        "planet_features": torch.zeros((48, 20)),
        "ship_target": torch.zeros((16,), dtype=torch.long),
    }
    try:
        _records_to_training_samples([legacy], ship_bin_mode="fraction")
    except ValueError as e:
        assert "pre-materialized" in str(e)
    else:
        raise AssertionError("fraction mode accepted legacy tensor ship labels")
    print("test_fraction_mode_rejects_legacy_tensor_ship_labels: PASS")


def test_streaming_validation_samples_across_shards(tmp_path):
    paths = []
    for shard in range(2):
        path = tmp_path / f"samples_{shard}.pkl"
        with open(path, "wb") as f:
            pickle.dump([{"shard": shard, "i": i} for i in range(10)], f)
        paths.append(str(path))

    val_samples, val_indices, val_path_count = _sample_streaming_validation_records(
        paths, val_frac=0.2, max_val_samples=10, rng=np.random.default_rng(0)
    )

    assert val_path_count == 2
    assert set(val_indices) == set(paths)
    assert all(len(indices) == 2 for indices in val_indices.values())
    assert len(val_samples) == 4
    assert {sample["shard"] for sample in val_samples} == {0, 1}
    print("test_streaming_validation_samples_across_shards: PASS")


if __name__ == "__main__":
    print("Running bc_loss tests...\n")
    test_fire_pos_weight_changes_fire_bce()
    test_threat_loss_uses_threat_labels()
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        test_rl_init_checkpoint_is_rejected_by_default(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_partial_init_skips_fraction_ship_head_shape(Path(d))
    test_select_metric_direction()
    test_compact_records_convert_to_training_samples()
    test_fraction_ship_labels_use_source_ship_ratio()
    test_fraction_mode_rejects_legacy_tensor_ship_labels()
    with tempfile.TemporaryDirectory() as d:
        test_streaming_validation_samples_across_shards(Path(d))
    print("\nAll bc_loss tests passed!")
