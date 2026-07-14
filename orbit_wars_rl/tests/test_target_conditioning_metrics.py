"""Focused tests for target-conditioned fire diagnostics."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo import _fire_target_conditioning_metrics


def test_fire_flip_is_policy_weighted_directional_and_actionable_only():
    # Source 0: prior NOOP; target probabilities 0.75/0.25; only target 0 flips to COMMIT.
    # Source 1: prior COMMIT; probabilities 0.25/0.75; only target 0 flips to NOOP.
    # Source 2: both targets flip, but fire_mask=False must exclude it entirely.
    target_logits = torch.log(torch.tensor([
        [[0.75, 0.25], [0.25, 0.75], [0.50, 0.50]],
    ]))
    fire_prior = torch.tensor([[[-1.0, -1.0], [1.0, 1.0], [1.0, 1.0]]])
    fire_residual = torch.tensor([[[2.0, 0.0], [-2.0, 0.0], [-2.0, -2.0]]])
    fire_mask = torch.tensor([[True, True, False]])
    slot_valid = torch.ones_like(fire_mask)

    metrics = _fire_target_conditioning_metrics(
        target_logits, fire_prior, fire_residual, fire_mask, slot_valid)

    assert torch.isclose(metrics["noop_to_commit_prob"], torch.tensor(0.375))
    assert torch.isclose(metrics["commit_to_noop_prob"], torch.tensor(0.125))
    assert torch.isclose(metrics["flip_prob"], torch.tensor(0.500))
    assert torch.isclose(
        metrics["flip_prob"],
        metrics["noop_to_commit_prob"] + metrics["commit_to_noop_prob"],
    )
    assert torch.isclose(metrics["straddle_rate"], torch.tensor(1.0))


def test_fire_flip_ignores_invalid_targets_and_empty_actionable_set():
    target_logits = torch.tensor([[[0.0, -1e9]]])
    fire_prior = torch.tensor([[[-1.0, -1.0]]])
    fire_residual = torch.tensor([[[2.0, 2.0]]])
    off = torch.tensor([[False]])

    metrics = _fire_target_conditioning_metrics(
        target_logits, fire_prior, fire_residual, off, torch.tensor([[True]]))

    assert all(value == 0 for value in metrics.values())
