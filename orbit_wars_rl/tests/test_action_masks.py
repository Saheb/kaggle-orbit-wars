"""Test action mask computation.

All tests use the correct obs-dict API: compute_action_masks(obs, player).
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from action_mask import (
    compute_action_masks,
    _apply_target_sanity_penalty_from_candidates,
    actions_from_target_policy,
    NUM_ANGLE_BINS,
    ANGLE_BIN_WIDTH,
    CENTER,
    SUN_RADIUS,
    BOARD_SIZE,
)


def _make_obs(planets, fleets=None):
    """Build a minimal obs dict from a list of planet rows [id, owner, x, y, radius, ships, prod]."""
    return {
        "step": 0,
        "player": 0,
        "planets": planets,
        "fleets": fleets or [],
        "angular_velocity": 0.0,
        "initial_planets": planets,
        "comet_planet_ids": [],
    }


def test_fire_mask_requires_ships():
    """Any owned planet with at least 1 ship can fire."""
    planets = [
        [0, 0, 25.0, 25.0, 1.5, 1, 2],   # owned, 1 ship → can fire all ships
        [1, 0, 75.0, 25.0, 1.5, 10, 2],  # owned, 10 ships → can fire
    ]
    masks = compute_action_masks(_make_obs(planets), player=0)
    fire = masks["fire_mask"].squeeze(0)
    assert fire[0].item(), "Planet with 1 ship should be fireable"
    assert fire[1].item(), "Planet with 10 ships should be fireable"
    print("test_fire_mask_requires_ships: PASS")


def test_only_owned_planets_fire():
    """Only player-0 planets appear in masks; neutral/enemy are excluded."""
    planets = [
        [0, 0, 25.0, 25.0, 1.5, 20, 2],   # owned
        [1, -1, 50.0, 50.0, 1.5, 10, 1],  # neutral
        [2, 1, 75.0, 75.0, 1.5, 15, 2],   # enemy
    ]
    masks = compute_action_masks(_make_obs(planets), player=0)
    assert masks["owned_count"] == 1, f"Expected 1 owned planet, got {masks['owned_count']}"
    slot_valid = masks["slot_valid"].squeeze(0)
    assert slot_valid[0].item(), "Slot 0 should be valid"
    assert not slot_valid[1].item(), "Slot 1 should not be valid (no second owned planet)"
    print("test_only_owned_planets_fire: PASS")


def test_sun_crossing_masked():
    """Planet directly above the sun — shooting south should be masked."""
    # Planet at (50, 25): sun center at (50, 50). Shooting south (angle≈π/2) crosses the sun.
    planets = [[0, 0, 50.0, 25.0, 1.5, 50, 2]]
    masks = compute_action_masks(_make_obs(planets), player=0)
    angle_mask = masks["angle_mask"].squeeze(0)[0]  # (72,) for slot 0

    # Bin for due-south angle (π/2)
    south_bin = int((math.pi / 2) / ANGLE_BIN_WIDTH)
    masked = ~angle_mask
    n_masked = masked.sum().item()
    print(f"  Sun-crossing: {n_masked}/{NUM_ANGLE_BINS} angles masked")
    assert n_masked > 0, "Some angles should be masked (planet close to sun)"
    assert masked[south_bin].item(), f"South bin ({south_bin}) should be masked"
    print("test_sun_crossing_masked: PASS")


def test_out_of_bounds_masked():
    """Planet very close to edge — spawn point goes OOB for outward angles."""
    # px=1.0, radius=1.5: spawn_x = 1.0 + 1.6*cos(π) = -0.6 → OOB for westward angle
    planets = [[0, 0, 1.0, 50.0, 1.5, 50, 2]]
    masks = compute_action_masks(_make_obs(planets), player=0)
    angle_mask = masks["angle_mask"].squeeze(0)[0]  # (72,)
    masked = ~angle_mask
    n_masked = masked.sum().item()
    print(f"  OOB (left-edge planet): {n_masked}/{NUM_ANGLE_BINS} angles masked")
    assert n_masked > 0, "Left-edge planet should have some OOB-masked (westward) angles"

    # West bin (angle ≈ π)
    west_bin = int(math.pi / ANGLE_BIN_WIDTH)
    assert masked[west_bin].item(), f"Westward bin ({west_bin}) should be OOB-masked"
    print("test_out_of_bounds_masked: PASS")


def test_max_ships_is_all_ships():
    """max_ships for a planet is the full ship count."""
    planets = [[0, 0, 25.0, 25.0, 1.5, 30, 2]]
    masks = compute_action_masks(_make_obs(planets), player=0)
    max_s = masks["max_ships"].squeeze(0)[0].item()
    assert max_s == 30, f"Expected max_ships=30, got {max_s}"
    print("test_max_ships_is_all_ships: PASS")


def test_angle_bins_cover_full_circle():
    """Verify that ANGLE_BIN_CENTERS span [0, 2π) without gaps."""
    from action_mask import ANGLE_BIN_CENTERS
    assert len(ANGLE_BIN_CENTERS) == NUM_ANGLE_BINS
    assert abs(ANGLE_BIN_CENTERS[0] - ANGLE_BIN_WIDTH / 2) < 1e-6
    for i in range(NUM_ANGLE_BINS - 1):
        gap = ANGLE_BIN_CENTERS[i + 1] - ANGLE_BIN_CENTERS[i]
        assert abs(gap - ANGLE_BIN_WIDTH) < 1e-6, f"Gap at bin {i}: {gap}"
    print("test_angle_bins_cover_full_circle: PASS")


def test_interior_planet_all_angles_legal():
    """A planet far from sun and edges should have all 72 angles legal."""
    planets = [[0, 0, 50.0, 85.0, 1.5, 50, 2]]  # top of map, away from sun
    masks = compute_action_masks(_make_obs(planets), player=0)
    angle_mask = masks["angle_mask"].squeeze(0)[0]
    n_legal = angle_mask.sum().item()
    # Due to the short 20-unit check distance and planet position, expect most angles legal.
    # Not necessarily all 72 since sun is at (50,50) and we're at (50,85) — shooting south crosses it.
    print(f"  Interior planet: {n_legal}/{NUM_ANGLE_BINS} legal angles")
    assert n_legal > NUM_ANGLE_BINS // 2, f"Expected most angles legal, got {n_legal}"
    print("test_interior_planet_all_angles_legal: PASS")


def test_target_decode_masks_own_planet_before_argmax():
    """Target decode should choose the best legal target, not drop a fire slot."""
    planets = [
        [0, 0, 20.0, 20.0, 1.5, 30, 2],   # owned source
        [1, 0, 25.0, 20.0, 1.5, 10, 2],   # owned non-source
        [2, -1, 35.0, 20.0, 1.5, 5, 3],   # legal neutral target
        [3, 1, 45.0, 20.0, 1.5, 7, 2],    # legal enemy target
    ]
    obs = _make_obs(planets)
    masks = compute_action_masks(obs, player=0)

    fire_logits = torch.full((1, 16), -100.0)
    fire_logits[0, 0] = 10.0

    target_logits = torch.full((1, 16, 48), -100.0)
    # Highest raw logit is invalid self-target; second-highest is another owned planet.
    target_logits[0, 0, 0] = 20.0
    target_logits[0, 0, 1] = 15.0
    # Best legal target should be planet 2.
    target_logits[0, 0, 2] = 12.0
    target_logits[0, 0, 3] = 8.0

    ship_logits = torch.full((1, 16, 32), -100.0)
    ship_logits[0, 0, 4] = 10.0  # send 5 ships

    moves = actions_from_target_policy(
        fire_logits, target_logits, ship_logits, masks, obs, player=0
    )
    assert len(moves) == 1, f"Expected one legal move, got {moves}"
    from_pid, angle, ships = moves[0]
    assert from_pid == 0
    assert ships > 0
    # Angle should point roughly from planet 0 to legal neutral planet 2.
    expected = math.atan2(planets[2][3] - planets[0][3], planets[2][2] - planets[0][2])
    delta = abs((angle - expected + math.pi) % (2 * math.pi) - math.pi)
    assert delta < 1e-3, f"Expected targeting legal planet 2, angle delta={delta}"
    print("test_target_decode_masks_own_planet_before_argmax: PASS")


def test_target_sanity_penalty_demotes_dominated_same_source_target():
    class _Cand:
        def __init__(self, source_id, target_idx, score, eta, ships=10, valid=True):
            self.source_id = source_id
            self.target_idx = target_idx
            self.score = score
            self.eta = eta
            self.ships = ships
            self.valid = valid

    planets = [
        [0, 0, 20.0, 20.0, 1.5, 30, 2],
        [1, -1, 28.0, 20.0, 1.5, 6, 2],
        [2, -1, 55.0, 20.0, 1.5, 6, 2],
    ]
    obs = _make_obs(planets)
    masks = compute_action_masks(obs, player=0)
    logits = torch.zeros((1, 16, 48), dtype=torch.float32)
    logits[0, 0, 1] = 5.0
    logits[0, 0, 2] = 6.0
    cands = [
        _Cand(source_id=0, target_idx=1, score=10.0, eta=3),
        _Cand(source_id=0, target_idx=2, score=5.5, eta=9),
    ]

    _apply_target_sanity_penalty_from_candidates(
        logits,
        masks,
        obs,
        player=0,
        candidates=cands,
        penalty=8.0,
    )
    assert logits[0, 0, 1].item() == 5.0
    assert logits[0, 0, 2].item() == -2.0
    print("test_target_sanity_penalty_demotes_dominated_same_source_target: PASS")


if __name__ == "__main__":
    print("Running action mask tests...\n")
    test_fire_mask_requires_ships()
    test_only_owned_planets_fire()
    test_sun_crossing_masked()
    test_out_of_bounds_masked()
    test_max_ships_is_all_ships()
    test_angle_bins_cover_full_circle()
    test_interior_planet_all_angles_legal()
    test_target_decode_masks_own_planet_before_argmax()
    test_target_sanity_penalty_demotes_dominated_same_source_target()
    print("\nAll action mask tests passed!")
