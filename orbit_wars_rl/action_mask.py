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

NUM_ANGLE_BINS = 144
ANGLE_BIN_WIDTH = 2 * math.pi / NUM_ANGLE_BINS
CENTER = 50.0
SUN_RADIUS = 10.0
BOARD_SIZE = 100.0
MAX_OWNED_PLANETS = 16

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


def actions_from_policy(fire_probs, angle_logits, ship_logits, masks, obs, player,
                        fire_threshold=0.5, sample: bool = False,
                        ship_bin_mode: str = "absolute"):
    """Convert policy outputs to environment action format.

    sample=False (default): fire by threshold, angle/ship by argmax (mode).
    sample=True: fire by Bernoulli sample, angle/ship by Categorical sample.
      Useful when training-time distribution is multi-modal and the mode
      collapses to a useless bin while majority mass is on competent bins.

    Returns: list of [from_planet_id, angle_radians, ship_count]
    """
    planets = obs["planets"]
    my_planets = [(i, p) for i, p in enumerate(planets) if p[1] == player]
    owned_indices = masks["owned_indices"].cpu().numpy()
    max_ships = masks["max_ships"].cpu().numpy().squeeze(0)

    if sample:
        fire_dist = torch.distributions.Bernoulli(logits=fire_probs)
        angle_dist = torch.distributions.Categorical(logits=angle_logits)
        ship_dist = torch.distributions.Categorical(logits=ship_logits)
        fire_decisions = (fire_dist.sample() > 0.5).cpu().numpy().squeeze(0)
        angle_bins = angle_dist.sample().cpu().numpy().squeeze(0)
        ship_bins = ship_dist.sample().cpu().numpy().squeeze(0)
    else:
        fire_decisions = (torch.sigmoid(fire_probs) > fire_threshold).cpu().numpy().squeeze(0)
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

        angle_bin_width = 2 * math.pi / max(1, angle_logits.shape[-1])
        angle = float(angle_bins[slot] * angle_bin_width + angle_bin_width / 2)
        ships = _ship_bin_to_count(int(ship_bins[slot]), int(max_ships[slot]), mode=ship_bin_mode)
        if ships > 0 and planets[pidx][5] > ships:
            moves.append([from_id, angle, ships])

    return moves


_MAX_SHIP_SPEED = 6.0
_ROTATION_LIMIT = 50.0


def _fleet_speed(ships: int) -> float:
    if ships <= 0:
        return 1.0
    s = 1.0 + (_MAX_SHIP_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000.0)) ** 1.5
    return min(s, _MAX_SHIP_SPEED)


def _target_intercept_angle(src_planet, target_planet, ships: int, obs) -> float:
    """Aim from src at target's ETA-predicted position."""
    sx, sy = float(src_planet[2]), float(src_planet[3])
    tx, ty = float(target_planet[2]), float(target_planet[3])
    tgt_pid = int(target_planet[0])
    tgt_radius = float(target_planet[4])
    speed = _fleet_speed(ships)
    angular_velocity = float(obs.get("angular_velocity", 0.0))
    current_step = int(obs.get("step", 0))

    initial_planets = obs.get("initial_planets", obs["planets"])
    init_by_id = {int(p[0]): p for p in initial_planets}
    ip = init_by_id.get(tgt_pid)
    if ip is not None:
        irx = float(ip[2]) - CENTER
        iry = float(ip[3]) - CENTER
        init_angle = math.atan2(iry, irx)
        orbital_r = math.hypot(irx, iry)
        is_orbiting = (orbital_r + tgt_radius) < _ROTATION_LIMIT
    else:
        init_angle = 0.0
        orbital_r = 0.0
        is_orbiting = False

    ax, ay = tx, ty
    for _ in range(4):
        dist = math.hypot(ax - sx, ay - sy)
        eta = max(1, int(math.ceil(dist / speed)))
        if is_orbiting:
            ang = init_angle + angular_velocity * (current_step + eta)
            nax = CENTER + orbital_r * math.cos(ang)
            nay = CENTER + orbital_r * math.sin(ang)
        else:
            nax, nay = tx, ty
        if abs(nax - ax) < 0.5 and abs(nay - ay) < 0.5:
            ax, ay = nax, nay
            break
        ax, ay = nax, nay

    return float(math.atan2(ay - sy, ax - sx))


def actions_from_target_policy(fire_probs, target_logits, ship_logits, masks, obs, player,
                               fire_threshold=0.5, sample: bool = False,
                               ship_bin_mode: str = "absolute"):
    """Convert policy outputs to actions using target planet logits for aiming."""
    planets = obs["planets"]
    owned_indices = masks["owned_indices"].cpu().numpy()
    max_ships = masks["max_ships"].cpu().numpy().squeeze(0)
    target_logits = target_logits.clone()

    # Restrict target choice to legal launch targets before argmax / sampling.
    # The prior path argmaxed over all planets and then dropped own/self picks,
    # turning many fire-positive slots into silent no-ops at inference.
    for slot in range(min(masks["owned_count"], target_logits.shape[1])):
        pidx = int(owned_indices[slot])
        if pidx >= len(planets):
            continue
        for tidx, tgt in enumerate(planets[:target_logits.shape[-1]]):
            if int(tgt[1]) == player or int(tgt[0]) == int(planets[pidx][0]):
                target_logits[:, slot, tidx] = -1e9

    if sample:
        fire_dist = torch.distributions.Bernoulli(logits=fire_probs)
        target_dist = torch.distributions.Categorical(logits=target_logits)
        ship_dist = torch.distributions.Categorical(logits=ship_logits)
        fire_decisions = (fire_dist.sample() > 0.5).cpu().numpy().squeeze(0)
        target_indices = target_dist.sample().cpu().numpy().squeeze(0)
        ship_bins = ship_dist.sample().cpu().numpy().squeeze(0)
    else:
        fire_decisions = (torch.sigmoid(fire_probs) > fire_threshold).cpu().numpy().squeeze(0)
        target_indices = torch.argmax(target_logits, dim=-1).cpu().numpy().squeeze(0)
        ship_bins = torch.argmax(ship_logits, dim=-1).cpu().numpy().squeeze(0)

    moves = []
    max_moves = 8
    for slot in range(min(masks["owned_count"], fire_decisions.shape[0])):
        if len(moves) >= max_moves:
            break
        if not fire_decisions[slot]:
            continue

        pidx = int(owned_indices[slot])
        tidx = int(target_indices[slot])
        if pidx >= len(planets) or tidx >= len(planets):
            continue
        if int(planets[tidx][1]) == player or int(planets[pidx][0]) == int(planets[tidx][0]):
            continue

        ships = _ship_bin_to_count(int(ship_bins[slot]), int(max_ships[slot]), mode=ship_bin_mode)
        if ships <= 0 or planets[pidx][5] <= ships:
            continue

        angle = _target_intercept_angle(planets[pidx], planets[tidx], ships, obs)
        moves.append([int(planets[pidx][0]), angle, ships])

    return moves


def actions_from_sampled_policy(fire_action, angle_action, ship_action, masks, obs, player,
                                ship_bin_mode: str = "absolute"):
    """Convert sampled policy action tensors to environment action format."""
    planets = obs["planets"]
    owned_indices = masks["owned_indices"].cpu().numpy()
    max_ships = masks["max_ships"].cpu().numpy().squeeze(0)

    fire_decisions = fire_action.cpu().numpy().squeeze(0).astype(bool)
    angle_bins = angle_action.cpu().numpy().squeeze(0)
    ship_bins = ship_action.cpu().numpy().squeeze(0)

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

        angle = float(int(angle_bins[slot]) * ANGLE_BIN_WIDTH + ANGLE_BIN_WIDTH / 2)
        ships = _ship_bin_to_count(int(ship_bins[slot]), int(max_ships[slot]), mode=ship_bin_mode)
        if ships > 0 and planets[pidx][5] > ships:
            moves.append([from_id, angle, ships])

    return moves


SHIP_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 19, 22, 26, 30, 35, 42, 50, 60, 72, 86, 102, 122, 145, 173, 206, 245, 290, 350, 420]
# Fraction-bin values for ship_bin_mode="fraction" (10 bins on (0,1]):
# bin i → (i+1)/10 fraction of source's max_ships.
FRACTION_BIN_VALUES = [(i + 1) / 10 for i in range(10)]


def _ship_bin_to_count(bin_idx, max_ships, mode: str = "absolute"):
    """Convert a ship-bin index to an absolute ship count.

    mode:
      "absolute" — bin_idx indexes the 32-entry SHIP_COUNTS lookup
      "fraction" — bin_idx indexes the 10-entry FRACTION_BIN_VALUES; the
                   returned count is round(frac * max_ships), floored to 1
                   so the fleet is always non-empty.
    """
    max_ships = max(1, int(max_ships))
    if mode == "fraction":
        n = len(FRACTION_BIN_VALUES)
        b = max(0, min(int(bin_idx), n - 1))
        frac = FRACTION_BIN_VALUES[b]
        ships = int(round(frac * max_ships))
        return max(1, min(ships, max_ships))
    # absolute (legacy)
    return min(SHIP_COUNTS[bin_idx], max_ships)
