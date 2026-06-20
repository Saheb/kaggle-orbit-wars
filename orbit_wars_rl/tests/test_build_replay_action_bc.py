"""Regression checks for replay-action BC sample filtering."""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bc import bc_loss
from build_replay_action_bc import _dedupe_by_source, _passes_filters, _winner_seat


def _args(**overrides):
    base = dict(
        drop_saves=False,
        save_quality_filter=True,
        reinforce_gate_min_planets=2,
        reverse_edge_cooldown=3,
        reinforce_garrison_floor=0.0,
        save_beta=0.0,
        save_horizon=18.0,
        save_overhead=1.0,
        max_save_cost_ratio=1.0,
        min_save_ship_fraction=0.0,
        min_attack_value=0.05,
        min_reactive_roi=0.0,
        min_keepability=0.0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _save_label(enemy_fleet=True, target_garrison=5, source_garrison=50, owned_count=2, step=5):
    planets = [
        [0, 0, 0.0, 0.0, 1.0, source_garrison, 1.0],
        [1, 0, 10.0, 0.0, 1.0, target_garrison, 4.0],
        [2, 1, 20.0, 0.0, 1.0, 0.0, 1.0],
    ]
    fleets = []
    if enemy_fleet:
        fleets.append([7, 1, 20.0, 0.0, math.pi, 2, 20])
    obs = {
        "step": step,
        "player": 0,
        "planets": planets,
        "fleets": fleets,
        "initial_planets": planets,
        "angular_velocity": 0.0,
    }
    return {
        "kind": "save",
        "obs": obs,
        "source_id": 0,
        "source_idx": 0,
        "source_slot": 0,
        "target_id": 1,
        "target_idx": 1,
        "target_owner": 0,
        "owned_count": owned_count,
        "max_ships": source_garrison,
        "step": step,
        "ships": 20,
        "move": [0, 0.0, 20],
    }


def test_winner_seat_requires_1v1_rewards():
    assert _winner_seat({"rewards": [1, -1]}) == 0
    assert _winner_seat({"rewards": [-1, 1]}) == 1
    assert _winner_seat({"rewards": [1, 0, -1]}) is None
    print("test_winner_seat_requires_1v1_rewards: PASS")


def test_attack_value_filters_are_move_level():
    args = _args()
    stats = Counter()
    assert not _passes_filters(
        {"kind": "attack", "capture_value": 0.04, "reactive_roi": 1.0, "keepability": 1.0},
        args,
        stats,
    )
    assert not _passes_filters(
        {"kind": "attack", "capture_value": 0.06, "reactive_roi": -0.1, "keepability": 1.0},
        args,
        stats,
    )
    assert not _passes_filters(
        {"kind": "attack", "capture_value": 0.06, "reactive_roi": 0.1, "keepability": -0.01},
        args,
        stats,
    )
    assert _passes_filters(
        {"kind": "attack", "capture_value": 0.06, "reactive_roi": 0.1, "keepability": 0.0},
        args,
        stats,
    )
    assert stats["filtered_low_value"] == 1
    assert stats["filtered_low_roi"] == 1
    assert stats["filtered_low_keep"] == 1
    print("test_attack_value_filters_are_move_level: PASS")


def test_save_quality_keeps_cheap_holdable_save():
    stats = Counter()
    assert _passes_filters(_save_label(), _args(), stats)
    assert stats["kept_quality_save_moves"] == 1
    print("test_save_quality_keeps_cheap_holdable_save: PASS")


def test_save_quality_drops_safe_hopeless_gate_and_reverse_edges():
    stats = Counter()
    assert not _passes_filters(_save_label(enemy_fleet=False), _args(), stats)
    assert stats["filtered_save_no_threat"] == 1

    assert not _passes_filters(_save_label(source_garrison=1), _args(), stats)
    assert stats["filtered_save_hopeless"] == 1

    assert not _passes_filters(_save_label(owned_count=1), _args(), stats)
    assert stats["filtered_save_gate"] == 1

    cooldown_last = {(1, 0): 4}
    assert not _passes_filters(_save_label(step=5), _args(), stats, cooldown_last)
    assert stats["filtered_save_reverse_edge"] == 1
    print("test_save_quality_drops_safe_hopeless_gate_and_reverse_edges: PASS")


def test_dedupe_keeps_largest_move_per_source():
    stats = Counter()
    labels = _dedupe_by_source(
        [
            {"source_id": 7, "ships": 5, "kind": "attack", "move": [7, 0.0, 5]},
            {"source_id": 7, "ships": 12, "kind": "save", "move": [7, 1.0, 12]},
            {"source_id": 8, "ships": 3, "kind": "attack", "move": [8, 2.0, 3]},
        ],
        stats,
    )
    moves = [label["move"] for label in labels]
    assert sorted(m[0] for m in moves) == [7, 8]
    assert [m for m in moves if m[0] == 7][0][2] == 12
    assert stats["same_source_groups"] == 1
    assert stats["same_source_dropped_moves"] == 1
    assert stats["kept_save_moves"] == 1
    assert stats["kept_attack_moves"] == 1
    print("test_dedupe_keeps_largest_move_per_source: PASS")


def test_bc_loss_accepts_target_conditioned_fire_ship_logits():
    B, MO, MP, C = 2, 3, 4, 5
    outputs = {
        "fire_logits": torch.randn(B, MO, MP, requires_grad=True),
        "ship_logits": torch.randn(B, MO, MP, C, requires_grad=True),
        "target_logits": torch.randn(B, MO, MP, requires_grad=True),
    }
    outputs["target_logits"].data[:, :, -1] = -100.0
    batch = {
        "slot_valid": torch.ones(B, MO),
        "fire_target": torch.tensor([[1, 0, 1], [0, 1, 0]]),
        "ship_target": torch.tensor([[2, 0, 3], [0, 1, 0]]),
        "target_target": torch.tensor([[1, -1, 2], [-1, 0, -1]]),
    }
    loss, metrics = bc_loss(outputs, batch)
    assert torch.isfinite(loss)
    assert metrics["target_top3"] >= metrics["target_top1"]
    loss.backward()
    assert outputs["fire_logits"].grad is not None
    assert outputs["ship_logits"].grad is not None
    assert outputs["target_logits"].grad is not None
    print("test_bc_loss_accepts_target_conditioned_fire_ship_logits: PASS")


if __name__ == "__main__":
    print("Running replay-action BC builder tests...\n")
    test_winner_seat_requires_1v1_rewards()
    test_attack_value_filters_are_move_level()
    test_save_quality_keeps_cheap_holdable_save()
    test_save_quality_drops_safe_hopeless_gate_and_reverse_edges()
    test_dedupe_keeps_largest_move_per_source()
    test_bc_loss_accepts_target_conditioned_fire_ship_logits()
    print("\nAll replay-action BC builder tests passed!")
