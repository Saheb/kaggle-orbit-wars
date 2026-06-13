"""Feature extraction for Orbit Wars Entity Transformer.

Converts raw observations into padded entity tensors with baked-in
geometric features (ADR-003: geometry is exact, strategy is learned).

Features per entity type:
- Planet (20 features): position, owner, radius, ships, production, orbit info,
  pressure, capture cost, distance to nearest owned, connectivity counts,
  is_home, active mask
- Fleet (13 features): position, owner, angle, ships, speed, dist_to_sun,
  fleet destination ETA/dist, threatens_owned, target_production, mask
- Global (11 features): player, step, angular_velocity, economy stats,
  enemy ships split (on_planets / in_fleets), mode

Pairwise features (15 per owned-slot × target-planet pair):
  0: sin of arrival direction   (corrected for rotation on orbiting targets)
  1: cos of arrival direction   (corrected for rotation on orbiting targets)
  2: arrival dist / BOARD_SIZE  (corrected for rotation on orbiting targets)
  3: 1/(arrival_eta+1)          (corrected for rotation on orbiting targets)
  4: sun_safe flag
  5-7: is_mine / is_enemy / is_neutral
  8: target production / 5
  9: target valid flag
  10: ships-at-arrival / 200    (current ships + production * eta)
  11: capture-gap / 200         (ships_at_arrival - current_capture_cost; + = harder)
  12: roi_20                    (production*20 - cap_cost_at_arrival) / cap_cost_at_arrival, clipped [-1,1]
  13: roi_50                    same with horizon 50
  14: enemy_contest / 100       total enemy fleet ships racing toward this target
"""

from __future__ import annotations

import math
import os
import numpy as np
import torch

# Eval-only escape hatch: measure a policy trained BEFORE the friendly-coverage roi-deflation
# in its NATIVE feature regime (deflation off). Default off → deflation active (production path).
_NO_FRIENDLY_DEFLATION = os.environ.get("ORBIT_NO_FRIENDLY_DEFLATION") == "1"

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT

BOARD_SIZE = 100.0
SUN_RADIUS = 10.0
MAX_SPEED = 6.0

# Comet observation features — must stay in sync with torch_env (COMET_FEAT_LOOKAHEAD /
# COMET_LIFE_NORM). For comet planets the orbital channels (9 orb_r, 10 pred_x, 11 pred_y)
# are overloaded with path-aware values: position COMET_FEAT_LOOKAHEAD steps ahead along the
# comet path (pred_x/pred_y) and normalized steps-to-departure (orb_r channel).
COMET_FEAT_LOOKAHEAD = 5
COMET_LIFE_NORM = 40.0


def fleet_speed(ships):
    ships = max(1, int(ships))
    scale = (math.log(ships) / math.log(1000.0)) ** 1.5
    return 1.0 + (MAX_SPEED - 1.0) * min(scale, 1.0)


MAX_OWNED_PLANETS = 16  # hard cap for owned-planet slots; matches config.ModelConfig.max_owned_planets


def extract_features(obs, player, num_players=2, max_planets=48, max_fleets=128,
                     max_owned=MAX_OWNED_PLANETS):
    """Extract entity features from observation dict.

    Returns dict of torch tensors (no batch dim).
    """
    planets = obs["planets"]
    fleets = obs["fleets"]
    step = obs.get("step", 0)
    angular_velocity = obs.get("angular_velocity", 0.0)
    initial_planets = obs.get("initial_planets", planets)
    comet_planet_ids = set(obs.get("comet_planet_ids", []))
    # Map comet planet id -> (path, path_index) so comet slots get path-aware position/expiry
    # features instead of the (meaningless) circular-orbit prediction. Same data the real kaggle
    # obs exposes via observation.comets; mirrored in torch_env.get_features for train/eval parity.
    comet_info = {}
    for _grp in (obs.get("comets") or []):
        _pidx = _grp.get("path_index", -1)
        for _ci, _pid in enumerate(_grp.get("planet_ids", [])):
            comet_info[int(_pid)] = (_grp["paths"][_ci], _pidx)

    n_planets = len(planets)
    n_fleets = min(len(fleets), max_fleets)

    # --- Planet features (20) ---
    planet_feats = np.zeros((max_planets, 20), dtype=np.float32)
    planet_mask = np.zeros(max_planets, dtype=np.bool_)
    owned_indices = np.zeros(max_owned, dtype=np.int64)
    owned_count = 0

    # Pre-compute initial planet lookup for orbit prediction
    init_by_id = {}
    for p in initial_planets:
        init_by_id[int(p[0])] = p

    # Pre-compute fleet arrays for pressure computation
    fleet_x = np.array([f[2] for f in fleets[:n_fleets]], dtype=np.float32) if n_fleets > 0 else np.array([], dtype=np.float32)
    fleet_y = np.array([f[3] for f in fleets[:n_fleets]], dtype=np.float32)
    fleet_cos = np.array([math.cos(f[4]) for f in fleets[:n_fleets]], dtype=np.float32)
    fleet_sin = np.array([math.sin(f[4]) for f in fleets[:n_fleets]], dtype=np.float32)
    fleet_owner = np.array([f[1] for f in fleets[:n_fleets]], dtype=np.int32)
    fleet_ships_arr = np.array([f[6] for f in fleets[:n_fleets]], dtype=np.float32)

    # Pre-compute owned planet positions for connectivity features (vectorised)
    owned_pos = [(p[2], p[3]) for p in planets[:max_planets] if p[1] == player]
    if owned_pos:
        _owned_xy = np.array(owned_pos, dtype=np.float32)  # (K, 2)
    else:
        _owned_xy = None

    for i, p in enumerate(planets[:max_planets]):
        planet_mask[i] = True
        pid, owner, x, y, radius, ships, production = p[0], p[1], p[2], p[3], p[4], p[5], p[6]

        dist_to_sun = math.hypot(x - CENTER, y - CENTER)
        is_orbiting = (dist_to_sun + radius) < ROTATION_RADIUS_LIMIT
        is_comet = int(pid) in comet_planet_ids
        is_home = (owner == player) and (ships <= 10 + production * 5) and (ships >= 10 - production)

        # Orbit prediction: position 5 turns ahead (rough lookahead for entity-level features;
        # pairwise features compute per-slot ETA-matched arrival position precisely)
        init_p = init_by_id.get(int(pid), p)
        rx = init_p[2] - CENTER
        ry = init_p[3] - CENTER
        init_angle = math.atan2(ry, rx)
        orbital_r = math.hypot(rx, ry)
        future_angle = init_angle + angular_velocity * (step + 5)
        pred_x = CENTER + orbital_r * math.cos(future_angle) if is_orbiting else x
        pred_y = CENTER + orbital_r * math.sin(future_angle) if is_orbiting else y

        # Comet overlay: comets don't orbit circularly, so override the orbital channels with
        # path-aware values (pred = path position LOOKAHEAD steps ahead; orb_r channel =
        # normalized steps-to-departure). Mirrors torch_env.get_features' comet overlay.
        orb_r_feat = orbital_r / CENTER
        if is_comet and int(pid) in comet_info:
            _cpath, _cpidx = comet_info[int(pid)]
            _L = len(_cpath)
            if _L > 0:
                _ahead = _cpath[min(_cpidx + COMET_FEAT_LOOKAHEAD, _L - 1)]
                pred_x, pred_y = float(_ahead[0]), float(_ahead[1])
                orb_r_feat = min(max(_L - _cpidx, 0) / COMET_LIFE_NORM, 1.0)

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

        # Distance / connectivity to owned planets — one vectorised pass
        if _owned_xy is not None:
            _dists = np.hypot(_owned_xy[:, 0] - x, _owned_xy[:, 1] - y)
            min_owned_dist = float(_dists.min())
            owned_within_15 = int((_dists < 15.0).sum())
            owned_within_30 = int((_dists < 30.0).sum())
        else:
            min_owned_dist = 0.0
            owned_within_15 = 0
            owned_within_30 = 0

        # Owner encoding: player=1, neutral=0, enemy=-1
        owner_emb = 1.0 if owner == player else (-1.0 if owner >= 0 else 0.0)

        planet_feats[i] = [
            (x - CENTER) / CENTER,            # 0: normalized x
            (y - CENTER) / CENTER,            # 1: normalized y
            owner_emb,                         # 2: owner encoding
            radius / 2.0,                      # 3: normalized radius
            math.log1p(ships) / 8.0,           # 4: log-normalized ships
            production / 5.0,                  # 5: normalized production
            float(is_orbiting),                # 6: is_orbiting
            float(is_comet),                   # 7: is_comet
            dist_to_sun / CENTER,              # 8: normalized dist to sun
            orb_r_feat,                        # 9: orbital radius (or comet steps-to-departure)
            (pred_x - CENTER) / CENTER,        # 10: predicted x (5-turn lookahead)
            (pred_y - CENTER) / CENTER,        # 11: predicted y (5-turn lookahead)
            friendly_pressure / 100.0,         # 12: incoming friendly pressure
            enemy_pressure / 100.0,            # 13: incoming enemy pressure
            math.log1p(capture_cost) / 8.0,   # 14: capture cost
            min_owned_dist / BOARD_SIZE,       # 15: dist to nearest owned
            float(is_home),                    # 16: is_home
            min(owned_within_15, 8) / 8.0,    # 17: owned planets within r=15
            min(owned_within_30, 12) / 12.0,  # 18: owned planets within r=30
            1.0,                               # 19: active mask
        ]

        # Track owned planets
        if owner == player and owned_count < max_owned:
            owned_indices[owned_count] = i
            owned_count += 1

    # --- Fleet features (13) ---
    # New features 9-12: fleet destination ETA, distance, threatens_owned flag,
    # target production. These tell the model which planets each fleet is heading
    # toward and how urgent the threat / opportunity is.
    fleet_feats = np.zeros((max_fleets, 13), dtype=np.float32)
    fleet_mask_arr = np.zeros(max_fleets, dtype=np.bool_)

    # Pre-compute planet arrays for destination decoding
    n_p_dest = min(len(planets), max_planets)
    if n_p_dest > 0:
        pl_x = np.array([planets[j][2] for j in range(n_p_dest)], dtype=np.float32)
        pl_y = np.array([planets[j][3] for j in range(n_p_dest)], dtype=np.float32)
        pl_r = np.array([planets[j][4] for j in range(n_p_dest)], dtype=np.float32)
        pl_owner = np.array([planets[j][1] for j in range(n_p_dest)], dtype=np.int32)
        pl_prod = np.array([planets[j][6] for j in range(n_p_dest)], dtype=np.float32)
    else:
        pl_x = pl_y = pl_r = pl_prod = np.array([], dtype=np.float32)
        pl_owner = np.array([], dtype=np.int32)

    for i, f in enumerate(fleets[:max_fleets]):
        fleet_mask_arr[i] = True
        fid, owner, x, y, angle, from_pid, ships = f[0], f[1], f[2], f[3], f[4], f[5], f[6]
        speed = fleet_speed(ships)
        dist_sun = math.hypot(x - CENTER, y - CENTER)
        owner_emb = 1.0 if owner == player else (-1.0 if owner >= 0 else -0.5)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Fleet destination: find the planet most along the fleet's trajectory.
        # A planet is a candidate if it's ahead (dot > 0) and within lateral capture margin.
        eta_feat = 0.0
        dist_feat = 1.0
        threatens_owned = 0.0
        tgt_prod_feat = 0.0
        if n_p_dest > 0:
            dvx = pl_x - x
            dvy = pl_y - y
            along = dvx * cos_a + dvy * sin_a           # signed projection onto travel dir
            perp  = np.abs(dvx * sin_a - dvy * cos_a)  # lateral distance
            margin = pl_r + 2.0                         # fleet capture radius margin
            candidates = (along > 0) & (perp < margin)
            if candidates.any():
                # Among candidates, pick earliest arrival (minimum distance / speed)
                dists = np.sqrt(dvx ** 2 + dvy ** 2)
                dists_c = np.where(candidates, dists, np.inf)
                best = int(np.argmin(dists_c))
                best_dist = float(dists[best])
                best_eta = max(1.0, best_dist / max(speed, 1e-3))
                eta_feat = 1.0 / (best_eta + 1.0)
                dist_feat = best_dist / BOARD_SIZE
                threatens_owned = 1.0 if pl_owner[best] == player else 0.0
                tgt_prod_feat = float(pl_prod[best]) / 5.0

        fleet_feats[i] = [
            (x - CENTER) / CENTER,   # 0: normalized x
            (y - CENTER) / CENTER,   # 1: normalized y
            owner_emb,               # 2: owner encoding
            cos_a,                   # 3: angle cos
            sin_a,                   # 4: angle sin
            math.log1p(ships) / 8.0, # 5: log ships
            speed / MAX_SPEED,       # 6: normalized speed
            dist_sun / CENTER,       # 7: dist to sun
            eta_feat,                # 8: 1/(eta_to_target+1) — urgency
            dist_feat,               # 9: dist to target / BOARD_SIZE
            threatens_owned,         # 10: target is an owned planet
            tgt_prod_feat,           # 11: target production / 5
            1.0,                     # 12: active mask
        ]

    # --- Global features (11) ---
    # Enemy ships split into on-planets (static) vs in-fleets (already committed).
    total_owned_ships = sum(p[5] for p in planets if p[1] == player)
    total_owned_production = sum(p[6] for p in planets if p[1] == player)
    num_owned = sum(1 for p in planets if p[1] == player)
    enemy_ships_on_planets = sum(p[5] for p in planets if p[1] >= 0 and p[1] != player)
    enemy_ships_in_fleets  = sum(f[6] for f in fleets if f[1] >= 0 and f[1] != player)
    owned_fleet_ships = sum(f[6] for f in fleets if f[1] == player)
    fleet_commitment = owned_fleet_ships / max(total_owned_ships + owned_fleet_ships, 1)

    mode_2p = 1.0 if num_players == 2 else 0.0
    mode_4p = 1.0 if num_players == 4 else 0.0

    global_feats = np.array([
        player / max(num_players - 1, 1),    # 0: player index
        np.clip(step / 500.0, 0.0, 1.0),      # 1: game progress
        np.clip(angular_velocity / 0.05, -1.0, 1.0), # 2: orbital speed
        num_owned / float(MAX_OWNED_PLANETS), # 3: owned planet count, normalised to [0,1]
        np.clip(total_owned_ships / 500.0, 0.0, 1.0), # 4: friendly ships on planets
        np.clip(total_owned_production / 20.0, 0.0, 1.0), # 5: friendly production
        np.clip(enemy_ships_on_planets / 2000.0, 0.0, 1.0), # 6: enemy ships on planets (static)
        np.clip(enemy_ships_in_fleets / 2000.0, 0.0, 1.0), # 7: enemy ships in fleets (committed)
        np.clip(fleet_commitment, 0.0, 1.0),  # 8: fraction of own ships in transit
        mode_2p,                             # 9: 2-player mode flag
        mode_4p,                             # 10: 4-player mode flag
    ], dtype=np.float32)

    # Precompute enemy fleet ships racing toward each target planet (pairwise feat 14).
    # Result shape: (n_p_pair,) — independent of source slot, broadcast in compute_pairwise_features.
    n_p_pair = min(len(planets), max_planets)
    enemy_contest = np.zeros(n_p_pair, dtype=np.float32)
    if n_fleets > 0 and n_p_pair > 0:
        enemy_fleet_mask = (fleet_owner != player) & (fleet_owner >= 0)
        if enemy_fleet_mask.any():
            efx = fleet_x[enemy_fleet_mask]
            efy = fleet_y[enemy_fleet_mask]
            efcos = fleet_cos[enemy_fleet_mask]
            efsin = fleet_sin[enemy_fleet_mask]
            efships = fleet_ships_arr[enemy_fleet_mask]
            tgt_x_p = np.array([planets[j][2] for j in range(n_p_pair)], dtype=np.float32)
            tgt_y_p = np.array([planets[j][3] for j in range(n_p_pair)], dtype=np.float32)
            tgt_r_p = np.array([planets[j][4] for j in range(n_p_pair)], dtype=np.float32)
            # (E, n_p) broadcast: is each enemy fleet e heading toward planet p?
            vx_ep = tgt_x_p[np.newaxis, :] - efx[:, np.newaxis]
            vy_ep = tgt_y_p[np.newaxis, :] - efy[:, np.newaxis]
            along_ep = vx_ep * efcos[:, np.newaxis] + vy_ep * efsin[:, np.newaxis]
            perp_ep  = np.abs(vx_ep * efsin[:, np.newaxis] - vy_ep * efcos[:, np.newaxis])
            headed = (along_ep > 0) & (perp_ep < tgt_r_p[np.newaxis, :] + 1.5)
            enemy_contest = (efships[:, np.newaxis] * headed).sum(axis=0).astype(np.float32)

    # Precompute FRIENDLY fleet ships already racing toward each target — the missing
    # symmetric counterpart of enemy_contest. Used in compute_pairwise_features to deflate
    # capture roi for a target we ALREADY have a fleet inbound to (the planet still reads
    # neutral/enemy until our fleet lands, so it otherwise stays the top target and we
    # redundantly re-launch at it). Not its own channel: it modulates roi_20/roi_50 (warm
    # features) so no new input dimension is added.
    friendly_contest = np.zeros(n_p_pair, dtype=np.float32)
    if n_fleets > 0 and n_p_pair > 0:
        friend_fleet_mask = (fleet_owner == player)
        if friend_fleet_mask.any():
            ffx = fleet_x[friend_fleet_mask]
            ffy = fleet_y[friend_fleet_mask]
            ffcos = fleet_cos[friend_fleet_mask]
            ffsin = fleet_sin[friend_fleet_mask]
            ffships = fleet_ships_arr[friend_fleet_mask]
            tgt_x_pf = np.array([planets[j][2] for j in range(n_p_pair)], dtype=np.float32)
            tgt_y_pf = np.array([planets[j][3] for j in range(n_p_pair)], dtype=np.float32)
            tgt_r_pf = np.array([planets[j][4] for j in range(n_p_pair)], dtype=np.float32)
            vx_fp = tgt_x_pf[np.newaxis, :] - ffx[:, np.newaxis]
            vy_fp = tgt_y_pf[np.newaxis, :] - ffy[:, np.newaxis]
            along_fp = vx_fp * ffcos[:, np.newaxis] + vy_fp * ffsin[:, np.newaxis]
            perp_fp  = np.abs(vx_fp * ffsin[:, np.newaxis] - vy_fp * ffcos[:, np.newaxis])
            headed_f = (along_fp > 0) & (perp_fp < tgt_r_pf[np.newaxis, :] + 1.5)
            friendly_contest = (ffships[:, np.newaxis] * headed_f).sum(axis=0).astype(np.float32)

    pairwise = compute_pairwise_features(
        planets, owned_indices, owned_count, player, max_planets=max_planets,
        max_owned=max_owned, angular_velocity=angular_velocity, step=step,
        init_by_id=init_by_id, enemy_contest=enemy_contest,
        friendly_contest=None if _NO_FRIENDLY_DEFLATION else friendly_contest,
        comet_ids=comet_planet_ids,
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
PAIRWISE_FEATURE_DIM = 15

# Typical fleet size used to estimate an ETA prior (matches teacher MIN_SHIP_FLOOR ~ 10
# but in practice ETA varies modestly with size since speed is log-shaped).
_ETA_PROBE_SHIPS = 20
_ETA_PROBE_SPEED = 1.0 + (MAX_SPEED - 1.0) * (math.log(_ETA_PROBE_SHIPS) / math.log(1000.0)) ** 1.5


def _planet_arrival_pos(init_angle: float, orbital_r: float, is_orbiting: bool,
                         current_x: float, current_y: float,
                         eta: float, angular_velocity: float, step: int) -> tuple:
    """Return predicted (x, y) of a planet at time step+eta.

    For non-orbiting planets returns current position unchanged.
    Uses the same orbit formula as extract_features().
    """
    if not is_orbiting or orbital_r == 0.0:
        return current_x, current_y
    future_angle = init_angle + angular_velocity * (step + eta)
    return CENTER + orbital_r * math.cos(future_angle), CENTER + orbital_r * math.sin(future_angle)


def compute_pairwise_features(planets, owned_indices, owned_count, player,
                              max_planets: int = 48, max_owned: int = MAX_OWNED_PLANETS,
                              angular_velocity: float = 0.0, step: int = 0,
                              init_by_id: dict | None = None,
                              enemy_contest: np.ndarray | None = None,
                              friendly_contest: np.ndarray | None = None,
                              comet_ids=None):
    """For each (owned-slot, target-planet) pair return geometric + ownership features.

    These are exactly the quantities the model cannot easily compute from raw (x, y)
    via attention: angle direction (sin/cos), distance, ETA-at-typical-ships, and
    sun-cross flag. Prior BC angle-head failure (0.08 reduction vs 0.40 gate) was
    driven by the model trying to learn trig from gradients; this fills the gap.

    For orbiting target planets the direction/distance/ETA are corrected to the
    predicted arrival position (one Newton step): current ETA → predicted arrival
    planet position → corrected distance → corrected ETA.  Non-orbiting planets
    are unchanged.

    Output: (max_owned, max_planets, PAIRWISE_FEATURE_DIM) float32 numpy array.
    Invalid (slot, target) entries are zero.
    """
    out = np.zeros((max_owned, max_planets, PAIRWISE_FEATURE_DIM), dtype=np.float32)
    if owned_count == 0 or len(planets) == 0:
        return out

    # Vectorize over targets once — current positions
    n_p = min(len(planets), max_planets)
    tgt_x = np.array([planets[j][2] for j in range(n_p)], dtype=np.float32)
    tgt_y = np.array([planets[j][3] for j in range(n_p)], dtype=np.float32)
    tgt_owner  = np.array([planets[j][1] for j in range(n_p)], dtype=np.int32)
    tgt_prod   = np.array([planets[j][6] for j in range(n_p)], dtype=np.float32)
    tgt_ships  = np.array([planets[j][5] for j in range(n_p)], dtype=np.float32)
    # Current capture cost: how many ships needed to take each planet right now
    tgt_cap_cost = np.where(
        tgt_owner == -1,      tgt_ships + 1,
        np.where(tgt_owner != player, tgt_ships + tgt_prod * 3 + 1, 0.0)
    ).astype(np.float32)

    # Precompute orbit info for each target planet
    tgt_is_orbiting = np.zeros(n_p, dtype=np.bool_)
    tgt_init_angle = np.zeros(n_p, dtype=np.float64)
    tgt_orbital_r = np.zeros(n_p, dtype=np.float64)
    comet_id_set = set(int(c) for c in (comet_ids or []))
    if init_by_id is not None:
        for j in range(n_p):
            pid = int(planets[j][0])
            # Comets are not circular orbiters → no arrival correction (use current position).
            # Matches torch_env, where comet slots have _planet_is_orbiting=False.
            if pid in comet_id_set:
                continue
            init_p = init_by_id.get(pid, planets[j])
            rx = float(init_p[2]) - CENTER
            ry = float(init_p[3]) - CENTER
            r = math.hypot(rx, ry)
            radius = float(planets[j][4])
            if (r + radius) < ROTATION_RADIUS_LIMIT and r > 0:
                tgt_is_orbiting[j] = True
                tgt_init_angle[j] = math.atan2(ry, rx)
                tgt_orbital_r[j] = r

    is_mine = (tgt_owner == player).astype(np.float32)
    is_enemy = ((tgt_owner != player) & (tgt_owner >= 0)).astype(np.float32)
    is_neutral = (tgt_owner == -1).astype(np.float32)

    for slot in range(min(owned_count, max_owned)):
        src_idx = int(owned_indices[slot])
        if src_idx >= len(planets):
            continue
        src = planets[src_idx]
        sx, sy = float(src[2]), float(src[3])

        # --- Step 1: distance/ETA to CURRENT planet positions ---
        dx0 = tgt_x - sx
        dy0 = tgt_y - sy
        dist0 = np.sqrt(dx0 * dx0 + dy0 * dy0)
        eta0 = np.maximum(1.0, np.ceil(dist0 / _ETA_PROBE_SPEED))

        # --- Step 2: for orbiting planets, correct to arrival position ---
        arr_x = tgt_x.copy()
        arr_y = tgt_y.copy()
        for j in range(n_p):
            if tgt_is_orbiting[j]:
                ax, ay = _planet_arrival_pos(
                    tgt_init_angle[j], tgt_orbital_r[j], True,
                    float(tgt_x[j]), float(tgt_y[j]),
                    float(eta0[j]), angular_velocity, step,
                )
                arr_x[j] = ax
                arr_y[j] = ay

        dx = arr_x - sx
        dy = arr_y - sy
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 1e-6)
        sin_a = dy / dist_safe
        cos_a = dx / dist_safe
        eta = np.maximum(1.0, np.ceil(dist / _ETA_PROBE_SPEED))

        # Sun-cross uses current target positions (route planning, not arrival)
        seg_len2 = np.maximum(dx0 * dx0 + dy0 * dy0, 1e-9)
        t = ((CENTER - sx) * dx0 + (CENTER - sy) * dy0) / seg_len2
        t = np.clip(t, 0.0, 1.0)
        proj_x = sx + t * dx0
        proj_y = sy + t * dy0
        sun_d = np.sqrt((proj_x - CENTER) ** 2 + (proj_y - CENTER) ** 2)
        sun_safe = (sun_d >= SUN_RADIUS).astype(np.float32)

        # Projected ships at arrival: current ships + production accrued during ETA.
        # Tells the model how hard this planet will be to capture by the time we get there.
        ships_at_arrival = np.minimum(tgt_ships + tgt_prod * eta, 500.0)
        # Capture gap: how much harder (positive) or easier (negative) vs right now.
        cap_gap = ships_at_arrival - tgt_cap_cost

        # ROI at horizons 20 and 50: (prod * H - cap_cost_at_arrival) / cap_cost_at_arrival.
        # cap_cost_at_arrival accounts for ships accumulated during flight (ships_at_arrival).
        # Positive = attack pays off within H steps; negative = not worth it yet.
        cap_cost_at_arrival = np.where(
            tgt_owner == -1,       ships_at_arrival + 1,
            np.where(tgt_owner != player, ships_at_arrival + tgt_prod * 3 + 1, 0.0)
        )
        safe_cap = np.maximum(cap_cost_at_arrival, 1.0)
        roi_20 = np.clip((tgt_prod * 20 - cap_cost_at_arrival) / safe_cap, -1.0, 1.0)
        roi_50 = np.clip((tgt_prod * 50 - cap_cost_at_arrival) / safe_cap, -1.0, 1.0)

        # Deflate capture roi by friendly ships already inbound: a target we are already
        # capturing (our fleet en route) offers ~0 marginal return to a NEW launch, so it
        # should stop reading as attractive. coverage in [0,1] = inbound / capture-cost;
        # own (reinforce) targets are never deflated. Keeps the head from re-launching at a
        # planet it is already taking (the opening over-fire).
        if friendly_contest is not None:
            coverage = np.where(tgt_owner != player,
                                np.minimum(friendly_contest[:n_p] / safe_cap, 1.0), 0.0)
            roi_20 = roi_20 * (1.0 - coverage)
            roi_50 = roi_50 * (1.0 - coverage)

        out[slot, :n_p, 0]  = sin_a                         # arrival direction sin
        out[slot, :n_p, 1]  = cos_a                         # arrival direction cos
        out[slot, :n_p, 2]  = dist / BOARD_SIZE             # arrival dist
        out[slot, :n_p, 3]  = 1.0 / (eta + 1.0)            # arrival close-fast preference
        out[slot, :n_p, 4]  = sun_safe
        out[slot, :n_p, 5]  = is_mine
        out[slot, :n_p, 6]  = is_enemy
        out[slot, :n_p, 7]  = is_neutral
        out[slot, :n_p, 8]  = tgt_prod / 5.0
        out[slot, :n_p, 9]  = 1.0                            # target valid flag
        out[slot, :n_p, 10] = ships_at_arrival / 200.0       # projected ships at arrival
        out[slot, :n_p, 11] = np.clip(cap_gap / 200.0, -1.0, 5.0)  # capture gap (+ = harder by ETA)
        out[slot, :n_p, 12] = roi_20                          # ROI at horizon 20
        out[slot, :n_p, 13] = roi_50                          # ROI at horizon 50
        if enemy_contest is not None:
            out[slot, :n_p, 14] = np.minimum(enemy_contest[:n_p], 500.0) / 100.0  # contested ships

    return out