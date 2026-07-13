"""Intent telemetry must report semantics and resolved counts, not legacy ship-bin values."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import PAIRWISE_FEATURE_NAMES
from train_torch import decode_ship_bins, intent_rollout_metrics


def test_pairwise_feature_names_cover_intent_channels():
    assert len(PAIRWISE_FEATURE_NAMES) == 26
    assert PAIRWISE_FEATURE_NAMES[-4:] == (
        "intent_capture_ships",
        "intent_capture_defend_ships",
        "intent_maintain_ships",
        "intent_all_in_ships",
    )


def test_decode_ship_bins_reads_selected_target_and_intent():
    intent_sizes = torch.tensor([[
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
            [100.0, 200.0, 300.0, 400.0],
        ],
        [
            [5.0, 6.0, 7.0, 8.0],
            [50.0, 60.0, 70.0, 80.0],
            [500.0, 600.0, 700.0, 800.0],
        ],
    ]])
    decoded = decode_ship_bins(
        torch.tensor([[2, 3]]), torch.zeros(1, 2), "intent",
        target_bins=torch.tensor([[1, 2]]), intent_sizes=intent_sizes,
    )
    assert torch.equal(decoded, torch.tensor([[30.0, 800.0]]))


def test_intent_rollout_metrics_use_sampled_actions_and_required_capture_mass():
    pairwise = torch.zeros(1, 4, 3, 26)
    targets = torch.tensor([[0, 1, 2, 0]])

    # Neutral target: required=10+1=11.
    pairwise[0, 0, 0, 7] = 1.0
    pairwise[0, 0, 0, 10] = 10.0 / 200.0
    # Enemy target: required=10 + production(2)*3 + 1 = 17.
    pairwise[0, 1, 1, 6] = 1.0
    pairwise[0, 1, 1, 8] = 2.0 / 5.0
    pairwise[0, 1, 1, 10] = 10.0 / 200.0
    # Own target: excluded from attack commitment metrics.
    pairwise[0, 2, 2, 5] = 1.0
    # Neutral target under-committed: required=50, resolved=40.
    pairwise[0, 3, 0, 7] = 1.0
    pairwise[0, 3, 0, 10] = 49.0 / 200.0

    metrics = intent_rollout_metrics({
        "fire_a": torch.ones(1, 4, dtype=torch.long),
        "slot_valid": torch.ones(1, 4, dtype=torch.bool),
        "ship_a": torch.tensor([[0, 1, 2, 3]]),
        "ship_count_a": torch.tensor([[11.0, 20.0, 6.0, 40.0]]),
        "target_a": targets,
        "pairwise_features": pairwise,
    })

    for name in ("capture", "capture_defend", "maintain", "all_in"):
        assert metrics[f"intent_{name}_share"] == pytest.approx(0.25)
    assert metrics["intent_resolved_ships_mean"] == pytest.approx(19.25)
    assert metrics["intent_attack_resolved_ships_mean"] == pytest.approx(71.0 / 3.0)
    assert metrics["intent_attack_commit_ratio_capped2"] == pytest.approx(
        (1.0 + 20.0 / 17.0 + 0.8) / 3.0)
    assert metrics["intent_attack_undercommit_rate"] == pytest.approx(1.0 / 3.0)
