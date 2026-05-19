"""Action mask computation for Orbit Wars.

For each owned planet, determines which of the 72 angle bins are legal
(don't cross sun, don't go out of bounds). Also computes per-planet
max sendable ships and ownership masks.

Uses numpy for computation, torch tensors for model input.
"""

from __future__ import annotations

import math
import numpy as np
import torch

NUM_ANGLE_BINS = 72
ANGLE_BIN_WIDTH = 2 * math.pi / NUM_ANGLE_BINS
CENTER = 50.0
SUN_RADIUS = 10.0
BOARD_SIZE = 100.0
MAX_OWNED_PLANETS = 10

ANGLE_BIN_CENTERS = np.array([(i + 0.5) * ANGLE_BIN_WIDTH for i in range(NUM_ANGLE_BINS)])


def compute_action_masks(obs, player, max_owned=MAX_OWNED_PLANETS):
    """Compute action masks from observation dict.

    Returns dict with torch tensors (batch dim 0=1):
        - fire_mask: (1, max_owned) bool
        - angle_mask: (1, max_owned, 72) bool
        - max_ships: (1, max_owned) int
        - owned_indices: (max_owned,) int — indices into planet array
        - owned_count: int
        - slot_valid: (1, max_owned) bool
    """
    planets = obs["planets"]
    fleets = obs["fleets"]

    # Find owned planets
    my_planets = [(i, p) for i, p in enumerate(planets) if p[1] == player]
    n_owned = min(len(my_planets), max_owned)

    owned_indices = np.zeros(max_owned, dtype=np.int64)
    fire_mask = np.zeros(max_owned, dtype=np.bool_)
    angle_mask = np.zeros((max_owned, NUM_ANGLE_BINS), dtype=np.bool_)
    max_ships_arr = np.zeros(max_owned, dtype=np.int64)
    slot_valid = np.zeros(max_owned, dtype=np.bool_)

    # Compute incoming fleet pressure per planet (for future use)
    fleet_cos = np.array([math.cos(f[4]) for f in fleets]) if fleets else np.array([])
    fleet_sin = np.array([math.sin(f[4]) for f in fleets]) if fleets else np.array([])

    for slot, (idx, p) in enumerate(my_planets[:max_owned]):
        slot_valid[slot] = True
        owned_indices[slot] = idx
        px, py, pr, ps = p[2], p[3], p[4], p[5]

        fire_mask[slot] = ps > 1
        max_ships_arr[slot] = max(0, int(ps) - 1)

        # Compute legal angles: launch from just outside planet radius
        spawn_x = px + (pr + 0.1) * np.cos(ANGLE_BIN_CENTERS)
        spawn_y = py + (pr + 0.1) * np.sin(ANGLE_BIN_CENTERS)

        # Check 1) out of bounds
        in_bounds = (spawn_x >= 0) & (spawn_x <= BOARD_SIZE) & (spawn_y >= 0) & (spawn_y <= BOARD_SIZE)

        # Check 2) sun crossing — check line segment from spawn to spawn + direction * some distance
        # Use a segment length of ~20 (typical fleet travel) for sun crossing check
        check_dist = 20.0
        end_x = spawn_x + check_dist * np.cos(ANGLE_BIN_CENTERS)
        end_y = spawn_y + check_dist * np.sin(ANGLE_BIN_CENTERS)

        # Point-to-segment distance from sun center to spawn->end segment
        sun_dist = _point_segment_distance_array(CENTER, CENTER, spawn_x, spawn_y, end_x, end_y)
        not_cross_sun = sun_dist > SUN_RADIUS

        angle_mask[slot] = in_bounds & not_cross_sun

    # Convert to torch tensors with batch dim
    return {
        "fire_mask": torch.from_numpy(fire_mask).unsqueeze(0),
        "angle_mask": torch.from_numpy(angle_mask).unsqueeze(0),
        "max_ships": torch.from_numpy(max_ships_arr).unsqueeze(0),
        "owned_indices": torch.from_numpy(owned_indices),
        "owned_count": n_owned,
        "slot_valid": torch.from_numpy(slot_valid).unsqueeze(0),
    }


def _point_segment_distance_array(px, py, ax_arr, ay_arr, bx_arr, by_arr):
    """Vectorized point-segment distance from (px,py) to segments (ax,ay)->(bx,by)."""
    dx = bx_arr - ax_arr
    dy = by_arr - ay_arr
    l2 = dx * dx + dy * dy
    t = np.where(l2 < 1e-12, 0.0, np.clip(((px - ax_arr) * dx + (py - ay_arr) * dy) / np.maximum(l2, 1e-12), 0, 1))
    proj_x = ax_arr + t * dx
    proj_y = ay_arr + t * dy
    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def actions_from_policy(fire_probs, angle_logits, ship_logits, masks, obs, player):
    """Convert policy outputs to environment action format.

    Returns: list of [from_planet_id, angle_radians, ship_count]
    """
    planets = obs["planets"]
    my_planets = [(i, p) for i, p in enumerate(planets) if p[1] == player]
    owned_indices = masks["owned_indices"].cpu().numpy()
    max_ships = masks["max_ships"].cpu().numpy().squeeze(0)

    fire_decisions = (torch.sigmoid(fire_probs) > 0.5).cpu().numpy().squeeze(0)
    angle_bins = torch.argmax(angle_logits, dim=-1).cpu().numpy().squeeze(0)
    ship_bins = torch.argmax(ship_logits, dim=-1).cpu().numpy().squeeze(0)

    moves = []
    max_moves = 8
    for slot in range(min(masks["owned_count"], fire_decisions.shape[0])):
        if len(moves) >= max_moves:
            break
        if not fire_decisions[slot]:
            continue

        pidx = int(owned_indices[slot])
        if pidx >= len(planets):
            continue
        from_id = int(planets[pidx][0])

        angle = float(angle_bins[slot] * ANGLE_BIN_WIDTH + ANGLE_BIN_WIDTH / 2)
        ships = _ship_bin_to_count(int(ship_bins[slot]), int(max_ships[slot]))
        if ships > 0 and planets[pidx][5] > ships:
            moves.append([from_id, angle, ships])

    return moves


def _ship_bin_to_count(bin_idx, max_ships):
    _NUM_SHIP_BINS = 16
    counts = [int(math.ceil(2 ** ((i + 1) / 2.0))) for i in range(_NUM_SHIP_BINS)]
    return min(counts[bin_idx], max(1, max_ships))