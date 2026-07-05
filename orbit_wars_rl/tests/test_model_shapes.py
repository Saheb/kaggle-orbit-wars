"""Test model forward pass shapes and feature extraction normalization.

Uses the PyTorch EntityTransformer API (model(planet_features, ...)).
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ModelConfig, EnvConfig
from model import EntityTransformer, NUM_SHIP_BINS
from features import extract_features
from action_mask import compute_action_masks, ANGLE_BIN_WIDTH, NUM_ANGLE_BINS, actions_from_target_policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs():
    return {
        "step": 10,
        "player": 0,
        "angular_velocity": 0.03,
        "planets": [
            [0, -1, 60.0, 60.0, 1.69, 20, 3],
            [1,  0, 50.0, 80.0, 1.39, 50, 2],
            [2,  1, 75.0, 30.0, 1.10, 30, 1],
        ],
        "fleets": [
            [0, 0, 52.0, 79.0, 1.57, 1, 15],
        ],
        "initial_planets": [
            [0, -1, 60.0, 60.0, 1.69, 20, 3],
            [1, -1, 50.0, 80.0, 1.39, 10, 2],
            [2, -1, 75.0, 30.0, 1.10,  5, 1],
        ],
        "comet_planet_ids": [],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_angle_bins_cover_full_circle():
    """Angle bins span [0, 2π) without gaps."""
    bins = np.array([(i + 0.5) * ANGLE_BIN_WIDTH for i in range(NUM_ANGLE_BINS)])
    assert len(bins) == NUM_ANGLE_BINS
    assert abs(bins[0] - ANGLE_BIN_WIDTH / 2) < 1e-6
    for i in range(NUM_ANGLE_BINS - 1):
        assert abs(bins[i + 1] - bins[i] - ANGLE_BIN_WIDTH) < 1e-6
    print("test_angle_bins_cover_full_circle: PASS")


def test_feature_shapes():
    """extract_features returns tensors with the right shapes."""
    obs = _make_obs()
    cfg = ModelConfig()
    feats = extract_features(obs, player=0, num_players=2,
                             max_planets=cfg.max_entities,
                             max_fleets=128)

    assert feats["planet_features"].shape == (cfg.max_entities, cfg.planet_feature_dim), \
        feats["planet_features"].shape
    assert feats["fleet_features"].shape == (128, cfg.fleet_feature_dim), \
        feats["fleet_features"].shape
    assert feats["global_features"].shape == (cfg.global_feature_dim,), \
        feats["global_features"].shape
    assert feats["pairwise_features"].shape == (cfg.max_owned_planets, cfg.max_entities, cfg.pairwise_feature_dim), \
        feats["pairwise_features"].shape
    assert feats["planet_mask"].shape == (cfg.max_entities,)
    assert feats["fleet_mask"].shape == (128,)
    print("test_feature_shapes: PASS")


def test_feature_normalization():
    """Extracted features should be roughly in [-3, 3]."""
    obs = _make_obs()
    feats = extract_features(obs, player=0, num_players=2)

    p_max = float(feats["planet_features"].abs().max())
    f_max = float(feats["fleet_features"].abs().max())
    g_max = float(feats["global_features"].abs().max())

    print(f"  Planet max abs: {p_max:.3f}")
    print(f"  Fleet max abs:  {f_max:.3f}")
    print(f"  Global max abs: {g_max:.3f}")

    assert p_max < 5.0, f"Planet features seem unnormalized: max={p_max}"
    assert f_max < 5.0, f"Fleet features seem unnormalized: max={f_max}"
    assert g_max < 5.0, f"Global features seem unnormalized: max={g_max}"
    print("test_feature_normalization: PASS")


def test_model_forward_shapes():
    """Model forward pass produces correct output shapes."""
    cfg = ModelConfig()
    model = EntityTransformer(cfg)
    model.eval()

    B, N_p, N_f = 2, 20, 10
    planet_features = torch.randn(B, N_p, cfg.planet_feature_dim)
    fleet_features = torch.randn(B, N_f, cfg.fleet_feature_dim)
    global_features = torch.randn(B, cfg.global_feature_dim)
    planet_mask = torch.ones(B, N_p, dtype=torch.bool)
    fleet_mask = torch.ones(B, N_f, dtype=torch.bool)

    with torch.no_grad():
        out = model(planet_features, fleet_features, global_features, planet_mask, fleet_mask)

    assert out["fire_logits"].shape == (B, cfg.max_owned_planets, cfg.max_planets), out["fire_logits"].shape
    assert out["ship_logits"].shape == (B, cfg.max_owned_planets, cfg.max_planets, NUM_SHIP_BINS), out["ship_logits"].shape
    assert out["target_logits"].shape == (B, cfg.max_owned_planets, cfg.max_planets), out["target_logits"].shape
    assert out["value"].shape == (B,), out["value"].shape
    print("test_model_forward_shapes: PASS")


def test_model_with_masks():
    """Model forward correctly applies fire/angle/slot masks."""
    cfg = ModelConfig()
    model = EntityTransformer(cfg)
    model.eval()

    B, N_p, N_f = 1, 20, 5
    max_owned = cfg.max_owned_planets

    planet_features = torch.randn(B, N_p, cfg.planet_feature_dim)
    fleet_features = torch.randn(B, N_f, cfg.fleet_feature_dim)
    global_features = torch.randn(B, cfg.global_feature_dim)
    planet_mask = torch.ones(B, N_p, dtype=torch.bool)
    fleet_mask = torch.ones(B, N_f, dtype=torch.bool)

    fire_mask = torch.zeros(B, max_owned, dtype=torch.bool)
    fire_mask[0, 0] = True  # only slot 0 can fire

    slot_valid = torch.zeros(B, max_owned, dtype=torch.bool)
    slot_valid[0, 0] = True  # only 1 owned planet

    owned_indices = torch.zeros(B, max_owned, dtype=torch.long)

    with torch.no_grad():
        out = model(
            planet_features, fleet_features, global_features, planet_mask, fleet_mask,
            fire_mask=fire_mask, slot_valid=slot_valid,
            owned_indices=owned_indices,
        )

    # Masked-out slots should have fire_logit << 0
    assert out["fire_logits"][0, 1, 0].item() <= -100.0, "Slot 1 should be masked (slot_valid=False)"
    assert out["ship_logits"][0, 1, 0, 0].item() <= -100.0, "Slot 1 ship logits should be masked"
    print("test_model_with_masks: PASS")


def test_model_forward_with_pairwise_target_head():
    """Forward pass with pairwise target scoring should run and return target logits."""
    cfg = ModelConfig()
    model = EntityTransformer(cfg)
    model.eval()

    B, N_p, N_f = 1, 20, 5
    max_owned = cfg.max_owned_planets

    planet_features = torch.randn(B, N_p, cfg.planet_feature_dim)
    fleet_features = torch.randn(B, N_f, cfg.fleet_feature_dim)
    global_features = torch.randn(B, cfg.global_feature_dim)
    planet_mask = torch.ones(B, N_p, dtype=torch.bool)
    fleet_mask = torch.ones(B, N_f, dtype=torch.bool)
    fire_mask = torch.zeros(B, max_owned, dtype=torch.bool)
    fire_mask[0, 0] = True
    slot_valid = torch.zeros(B, max_owned, dtype=torch.bool)
    slot_valid[0, 0] = True
    owned_indices = torch.zeros(B, max_owned, dtype=torch.long)
    pairwise_features = torch.randn(B, max_owned, N_p, cfg.pairwise_feature_dim)

    with torch.no_grad():
        out = model(
            planet_features, fleet_features, global_features, planet_mask, fleet_mask,
            fire_mask=fire_mask, slot_valid=slot_valid,
            owned_indices=owned_indices, pairwise_features=pairwise_features,
        )

    assert out["target_logits"] is not None
    assert out["target_logits"].shape == (B, max_owned, cfg.max_planets), out["target_logits"].shape
    assert out["fire_logits"].shape == (B, max_owned, cfg.max_planets), out["fire_logits"].shape
    assert out["ship_logits"].shape == (B, max_owned, cfg.max_planets, NUM_SHIP_BINS), out["ship_logits"].shape
    assert out["value"].shape == (B,), out["value"].shape
    print("test_model_forward_with_pairwise_target_head: PASS")


def test_residual_broadcast_parity():
    """Zero-init residuals should broadcast legacy slot logits across targets."""
    cfg = ModelConfig()
    model = EntityTransformer(cfg)
    model.eval()

    B, N_p, N_f = 1, 8, 4
    max_owned = cfg.max_owned_planets
    planet_features = torch.randn(B, N_p, cfg.planet_feature_dim)
    fleet_features = torch.randn(B, N_f, cfg.fleet_feature_dim)
    global_features = torch.randn(B, cfg.global_feature_dim)
    planet_mask = torch.ones(B, N_p, dtype=torch.bool)
    fleet_mask = torch.ones(B, N_f, dtype=torch.bool)
    fire_mask = torch.ones(B, max_owned, dtype=torch.bool)
    slot_valid = torch.zeros(B, max_owned, dtype=torch.bool)
    slot_valid[:, :2] = True
    owned_indices = torch.zeros(B, max_owned, dtype=torch.long)
    pairwise_features = torch.randn(B, max_owned, N_p, cfg.pairwise_feature_dim)

    with torch.no_grad():
        encoded = model.encode_state(
            planet_features, fleet_features, global_features,
            planet_mask, fleet_mask, slot_valid=slot_valid,
            owned_indices=owned_indices, pairwise_features=pairwise_features,
        )
        old_fire = model.fire_head(encoded["owned_enriched"]).squeeze(-1)
        old_ship = model.ship_head(encoded["owned_enriched"])
        out = model(
            planet_features, fleet_features, global_features, planet_mask, fleet_mask,
            fire_mask=fire_mask, slot_valid=slot_valid,
            owned_indices=owned_indices, pairwise_features=pairwise_features,
        )

    valid = slot_valid.unsqueeze(-1).expand(-1, -1, N_p)
    assert torch.allclose(
        out["fire_logits"][..., :N_p][valid],
        old_fire.unsqueeze(-1).expand(-1, -1, N_p)[valid],
    )
    valid_ship = slot_valid.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N_p, old_ship.shape[-1])
    assert torch.allclose(
        out["ship_logits"][..., :N_p, :][valid_ship],
        old_ship.unsqueeze(2).expand(-1, -1, N_p, -1)[valid_ship],
    )
    print("test_residual_broadcast_parity: PASS")


def test_residual_small_init_wakes_output_layer():
    """Small nonzero residual init should keep the output layer off dead-zero."""
    cfg = ModelConfig(phase4_residual_init_std=0.01)
    model = EntityTransformer(cfg)

    fire_norm = model.fire_scorer[-1].weight.norm().item()
    ship_norm = model.ship_scorer[-1].weight.norm().item()

    assert fire_norm > 0.0
    assert ship_norm > 0.0
    assert model.fire_scorer[-1].bias.abs().sum().item() == 0.0
    assert model.ship_scorer[-1].bias.abs().sum().item() == 0.0
    print("test_residual_small_init_wakes_output_layer: PASS")


def test_end_to_end_obs_to_actions():
    """Full pipeline: obs → features → masks → model → action shapes."""
    obs = _make_obs()
    cfg = ModelConfig()
    model = EntityTransformer(cfg)
    model.eval()

    feats = extract_features(obs, player=0, num_players=2)
    masks = compute_action_masks(obs, player=0)

    with torch.no_grad():
        out = model(
            feats["planet_features"].unsqueeze(0),
            feats["fleet_features"].unsqueeze(0),
            feats["global_features"].unsqueeze(0),
            feats["planet_mask"].unsqueeze(0),
            feats["fleet_mask"].unsqueeze(0),
            fire_mask=masks["fire_mask"],
            slot_valid=masks["slot_valid"],
            owned_indices=masks["owned_indices"].unsqueeze(0),
            pairwise_features=feats["pairwise_features"].unsqueeze(0),
        )

    actions = actions_from_target_policy(
        out["fire_logits"], out["target_logits"], out["ship_logits"],
        {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()},
        obs, 0,
    )
    assert isinstance(actions, list), "Actions should be a list"
    for move in actions:
        assert len(move) == 3, f"Move should be [planet_id, angle, ships], got {move}"
    print(f"  Produced {len(actions)} action(s)")
    print("test_end_to_end_obs_to_actions: PASS")


if __name__ == "__main__":
    print("Running model shape tests...\n")
    test_angle_bins_cover_full_circle()
    test_feature_shapes()
    test_feature_normalization()
    test_model_forward_shapes()
    test_model_with_masks()
    test_model_forward_with_pairwise_target_head()
    test_residual_broadcast_parity()
    test_residual_small_init_wakes_output_layer()
    test_end_to_end_obs_to_actions()
    print("\nAll model shape tests passed!")
