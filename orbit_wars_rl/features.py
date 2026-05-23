"""Feature extraction for Orbit Wars Entity Transformer.

Converts raw observations into padded entity tensors with baked-in
geometric features (ADR-003: geometry is exact, strategy is learned).

Features per entity type:
- Planet (18 features): position, owner, radius, ships, production, orbit info,
  pressure, capture cost, distance to home, is_home, active mask
- Fleet (9 features): position, owner, angle, ships, speed, dist_to_sun, mask
- Global (10 features): player, step, angular_velocity, economy stats, mode
"""

from __future__ import annotations

import math
import numpy as np
import torch

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT

BOARD_SIZE = 100.0
SUN_RADIUS = 10.0
MAX_SPEED = 6.0


def fleet_speed(ships):
    ships = max(1, int(ships))
    scale = (math.log(ships) / math.log(1000.0)) ** 1.5
    return 1.0 + (MAX_SPEED - 1.0) * min(scale, 1.0)


def extract_features(obs, player, num_players=2, max_planets=48, max_fleets=128):
    """Extract entity features from observation dict.

    Returns dict of torch tensors (no batch dim).
    """
    planets = obs["planets"]
    fleets = obs["fleets"]
    step = obs.get("step", 0)
    angular_velocity = obs.get("angular_velocity", 0.0)
    initial_planets = obs.get("initial_planets", planets)
    comet_planet_ids = set(obs.get("comet_planet_ids", []))

    n_planets = len(planets)
    n_fleets = min(len(fleets), max_fleets)

    # --- Planet features (18) ---
    planet_feats = np.zeros((max_planets, 18), dtype=np.float32)
    planet_mask = np.zeros(max_planets, dtype=np.bool_)
    owned_indices_count = 0
    owned_indices = np.zeros(10, dtype=np.int64)  # max 10 owned planets
    owned_count = 0

    # Pre-compute initial planet lookup for orbit prediction
    init_by_id = {}
    for p in initial_planets:
        init_by_id[int(p[0])] = p

    # Pre-compute fleet pressure arrays
    fleet_x = np.array([f[2] for f in fleets[:n_fleets]], dtype=np.float32) if n_fleets > 0 else np.array([], dtype=np.float32)
    fleet_y = np.array([f[3] for f in fleets[:n_fleets]], dtype=np.float32)
    fleet_cos = np.array([math.cos(f[4]) for f in fleets[:n_fleets]], dtype=np.float32)
    fleet_sin = np.array([math.sin(f[4]) for f in fleets[:n_fleets]], dtype=np.float32)
    fleet_owner = np.array([f[1] for f in fleets[:n_fleets]], dtype=np.int32)
    fleet_ships_arr = np.array([f[6] for f in fleets[:n_fleets]], dtype=np.float32)

    for i, p in enumerate(planets[:max_planets]):
        planet_mask[i] = True
        pid, owner, x, y, radius, ships, production = p[0], p[1], p[2], p[3], p[4], p[5], p[6]

        dist_to_sun = math.hypot(x - CENTER, y - CENTER)
        is_orbiting = (dist_to_sun + radius) < ROTATION_RADIUS_LIMIT
        is_comet = int(pid) in comet_planet_ids
        is_home = (owner == player) and (ships <= 10 + production * 5) and (ships >= 10 - production)

        # Orbit prediction: position 5 turns ahead
        init_p = init_by_id.get(int(pid), p)
        rx = init_p[2] - CENTER
        ry = init_p[3] - CENTER
        init_angle = math.atan2(ry, rx)
        orbital_r = math.hypot(rx, ry)
        future_angle = init_angle + angular_velocity * (step + 5)
        pred_x = CENTER + orbital_r * math.cos(future_angle) if is_orbiting else x
        pred_y = CENTER + orbital_r * math.sin(future_angle) if is_orbiting else y

        # Incoming fleet pressure
        friendly_pressure = 0.0
        enemy_pressure = 0.0
        if n_fleets > 0:
            vx = x - fleet_x
            vy = y - fleet_y
            along = vx * fleet_cos + vy * fleet_sin
            perp = np.abs(vx * fleet_sin - vy * fleet_cos)
            incoming = (along > 0) & (perp < radius + 1.5)
            friendly_pressure = float(np.sum(fleet_ships_arr[incoming & (fleet_owner == player)]))
            enemy_pressure = float(np.sum(fleet_ships_arr[incoming & (fleet_owner != player) & (fleet_owner >= 0)]))

        # Capture cost
        if owner == -1:
            capture_cost = ships + 1
        elif owner != player:
            capture_cost = ships + production * 3 + 1
        else:
            capture_cost = 0.0

        # Distance to nearest owned planet
        min_owned_dist = 100.0
        for j, q in enumerate(planets[:max_planets]):
            if q[1] == player:
                d = math.hypot(x - q[2], y - q[3])
                min_owned_dist = min(min_owned_dist, d)
        if not any(q[1] == player for q in planets[:max_planets]):
            min_owned_dist = 0.0

        # Owner encoding: player=1, neutral=0, enemy=-1
        owner_emb = 1.0 if owner == player else (-1.0 if owner >= 0 else 0.0)

        planet_feats[i] = [
            (x - CENTER) / CENTER,       # 0: normalized x
            (y - CENTER) / CENTER,       # 1: normalized y
            owner_emb,                    # 2: owner encoding
            radius / 2.0,                 # 3: normalized radius
            math.log1p(ships) / 8.0,      # 4: log-normalized ships
            production / 5.0,             # 5: normalized production
            float(is_orbiting),           # 6: is_orbiting
            float(is_comet),              # 7: is_comet
            dist_to_sun / CENTER,         # 8: normalized dist to sun
            orbital_r / CENTER,           # 9: normalized orbital radius
            (pred_x - CENTER) / CENTER,   # 10: predicted x
            (pred_y - CENTER) / CENTER,   # 11: predicted y
            friendly_pressure / 100.0,    # 12: incoming friendly pressure
            enemy_pressure / 100.0,       # 13: incoming enemy pressure
            math.log1p(capture_cost) / 8.0,  # 14: capture cost
            min_owned_dist / BOARD_SIZE,  # 15: dist to nearest owned
            float(is_home),               # 16: is_home
            1.0,                          # 17: active mask
        ]

        # Track owned planets
        if owner == player and owned_count < 10:
            owned_indices[owned_count] = i
            owned_count += 1

    # --- Fleet features (9) ---
    fleet_feats = np.zeros((max_fleets, 9), dtype=np.float32)
    fleet_mask_arr = np.zeros(max_fleets, dtype=np.bool_)

    for i, f in enumerate(fleets[:max_fleets]):
        fleet_mask_arr[i] = True
        fid, owner, x, y, angle, from_pid, ships = f[0], f[1], f[2], f[3], f[4], f[5], f[6]
        speed = fleet_speed(ships)
        dist_sun = math.hypot(x - CENTER, y - CENTER)
        owner_emb = 1.0 if owner == player else (-1.0 if owner >= 0 else -0.5)

        fleet_feats[i] = [
            (x - CENTER) / CENTER,   # 0: normalized x
            (y - CENTER) / CENTER,  # 1: normalized y
            owner_emb,               # 2: owner encoding
            math.cos(angle),         # 3: angle cos
            math.sin(angle),         # 4: angle sin
            math.log1p(ships) / 8.0, # 5: log ships
            speed / MAX_SPEED,       # 6: normalized speed
            dist_sun / CENTER,       # 7: dist to sun
            1.0,                     # 8: active mask
        ]

    # --- Global features (10) ---
    total_owned_ships = sum(p[5] for p in planets if p[1] == player)
    total_owned_production = sum(p[6] for p in planets if p[1] == player)
    num_owned = sum(1 for p in planets if p[1] == player)
    total_enemy_ships = sum(p[5] for p in planets if p[1] >= 0 and p[1] != player) + \
                        sum(f[6] for f in fleets if f[1] >= 0 and f[1] != player)
    owned_fleet_ships = sum(f[6] for f in fleets if f[1] == player)
    fleet_commitment = owned_fleet_ships / max(total_owned_ships + owned_fleet_ships, 1)

    mode_2p = 1.0 if num_players == 2 else 0.0
    mode_4p = 1.0 if num_players == 4 else 0.0

    global_feats = np.array([
        player / max(num_players - 1, 1),
        step / 500.0,
        angular_velocity / 0.05,
        num_owned / 10.0,
        total_owned_ships / 500.0,
        total_owned_production / 20.0,
        total_enemy_ships / 2000.0,
        fleet_commitment,
        mode_2p,
        mode_4p,
    ], dtype=np.float32)

    pairwise = compute_pairwise_features(
        planets, owned_indices, owned_count, player, max_planets=max_planets,
    )

    return {
        "planet_features": torch.from_numpy(planet_feats),
        "fleet_features": torch.from_numpy(fleet_feats),
        "global_features": torch.from_numpy(global_feats),
        "planet_mask": torch.from_numpy(planet_mask),
        "fleet_mask": torch.from_numpy(fleet_mask_arr),
        "owned_indices": torch.from_numpy(owned_indices),
        "owned_count": owned_count,
        "pairwise_features": torch.from_numpy(pairwise),
    }


# Number of pairwise features per (owned-slot, target-planet) pair.
# Keep in sync with compute_pairwise_features() and ModelConfig.pairwise_feature_dim.
PAIRWISE_FEATURE_DIM = 10

# Typical fleet size used to estimate an ETA prior (matches teacher MIN_SHIP_FLOOR ~ 10
# but in practice ETA varies modestly with size since speed is log-shaped).
_ETA_PROBE_SHIPS = 20
_ETA_PROBE_SPEED = 1.0 + (MAX_SPEED - 1.0) * (math.log(_ETA_PROBE_SHIPS) / math.log(1000.0)) ** 1.5


def compute_pairwise_features(planets, owned_indices, owned_count, player,
                              max_planets: int = 48, max_owned: int = 10):
    """For each (owned-slot, target-planet) pair return geometric + ownership features.

    These are exactly the quantities the model cannot easily compute from raw (x, y)
    via attention: angle direction (sin/cos), distance, ETA-at-typical-ships, and
    sun-cross flag. Prior BC angle-head failure (0.08 reduction vs 0.40 gate) was
    driven by the model trying to learn trig from gradients; this fills the gap.

    Output: (max_owned, max_planets, PAIRWISE_FEATURE_DIM) float32 numpy array.
    Invalid (slot, target) entries are zero.
    """
    out = np.zeros((max_owned, max_planets, PAIRWISE_FEATURE_DIM), dtype=np.float32)
    if owned_count == 0 or len(planets) == 0:
        return out

    # Vectorize over targets once
    n_p = min(len(planets), max_planets)
    tgt_x = np.array([planets[j][2] for j in range(n_p)], dtype=np.float32)
    tgt_y = np.array([planets[j][3] for j in range(n_p)], dtype=np.float32)
    tgt_radius = np.array([planets[j][4] for j in range(n_p)], dtype=np.float32)
    tgt_owner = np.array([planets[j][1] for j in range(n_p)], dtype=np.int32)
    tgt_prod = np.array([planets[j][6] for j in range(n_p)], dtype=np.float32)

    is_mine = (tgt_owner == player).astype(np.float32)
    is_enemy = ((tgt_owner != player) & (tgt_owner >= 0)).astype(np.float32)
    is_neutral = (tgt_owner == -1).astype(np.float32)

    for slot in range(min(owned_count, max_owned)):
        src_idx = int(owned_indices[slot])
        if src_idx >= len(planets):
            continue
        src = planets[src_idx]
        sx, sy = float(src[2]), float(src[3])

        dx = tgt_x - sx
        dy = tgt_y - sy
        dist = np.sqrt(dx * dx + dy * dy)
        # Avoid div-by-zero for self-pair (slot's own planet)
        dist_safe = np.maximum(dist, 1e-6)
        sin_a = dy / dist_safe
        cos_a = dx / dist_safe
        # ETA at typical ship count
        eta = np.maximum(1.0, np.ceil(dist / _ETA_PROBE_SPEED))

        # Sun-cross: point (CENTER, CENTER) to segment (src → tgt) distance
        seg_len2 = np.maximum(dx * dx + dy * dy, 1e-9)
        t = ((CENTER - sx) * dx + (CENTER - sy) * dy) / seg_len2
        t = np.clip(t, 0.0, 1.0)
        proj_x = sx + t * dx
        proj_y = sy + t * dy
        sun_d = np.sqrt((proj_x - CENTER) ** 2 + (proj_y - CENTER) ** 2)
        sun_safe = (sun_d >= SUN_RADIUS).astype(np.float32)

        out[slot, :n_p, 0] = sin_a
        out[slot, :n_p, 1] = cos_a
        out[slot, :n_p, 2] = dist / BOARD_SIZE
        out[slot, :n_p, 3] = 1.0 / (eta + 1.0)            # close-fast preference
        out[slot, :n_p, 4] = sun_safe
        out[slot, :n_p, 5] = is_mine
        out[slot, :n_p, 6] = is_enemy
        out[slot, :n_p, 7] = is_neutral
        out[slot, :n_p, 8] = tgt_prod / 5.0
        out[slot, :n_p, 9] = 1.0                            # target valid flag

    return out