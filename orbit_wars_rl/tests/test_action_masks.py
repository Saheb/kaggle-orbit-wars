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

from action_mask import compute_action_masks, actions_from_target_policy


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


def test_max_ships_is_all_ships():
    """max_ships for a planet is the full ship count."""
    planets = [[0, 0, 25.0, 25.0, 1.5, 30, 2]]
    masks = compute_action_masks(_make_obs(planets), player=0)
    max_s = masks["max_ships"].squeeze(0)[0].item()
    assert max_s == 30, f"Expected max_ships=30, got {max_s}"
    print("test_max_ships_is_all_ships: PASS")


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

    fire_logits = torch.full((1, 16, 48), -100.0)
    fire_logits[0, 0, 2] = 10.0

    target_logits = torch.full((1, 16, 48), -100.0)
    # Highest raw logit is invalid self-target; second-highest is another owned planet.
    target_logits[0, 0, 0] = 20.0
    target_logits[0, 0, 1] = 15.0
    # Best legal target should be planet 2.
    target_logits[0, 0, 2] = 12.0
    target_logits[0, 0, 3] = 8.0

    ship_logits = torch.full((1, 16, 48, 32), -100.0)
    ship_logits[0, 0, 2, 4] = 10.0  # send 5 ships to legal target

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


if __name__ == "__main__":
    print("Running action mask tests...\n")
    test_fire_mask_requires_ships()
    test_only_owned_planets_fire()
    test_max_ships_is_all_ships()
    test_target_decode_masks_own_planet_before_argmax()
    print("\nAll action mask tests passed!")
