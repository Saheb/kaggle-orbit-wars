"""Feature extraction for Orbit Wars Entity Transformer.

Converts raw observations into padded entity tensors with baked-in
geometric features (ADR-003: geometry is exact, strategy is learned).

Features per entity type:
- Planet (116 features): 20 base — position, owner, radius, ships, production, orbit
  info, pressure, capture cost, distance to nearest owned, connectivity counts,
  is_home, active mask — plus 96 projected-timeline channels (timeline.py: owner
  one-hot + log-garrison over the next 24 steps assuming no new launches)
- Fleet (13 features): position, owner, angle, ships, speed, dist_to_sun,
  fleet destination ETA/dist, threatens_owned, target_production, mask
- Global (15 features): player, step, angular_velocity, economy stats,
  enemy ships split (on_planets / in_fleets), mode, game-phase channels

Pairwise features (26 per owned-slot × target-planet pair):
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
  15: reachable_enemy_mass /100 distance-decayed enemy garrison that could reach this target
  16: capture_value_40          production value over a capped 40-step horizon, normalized by board prod
  17: reactive_roi_40           value vs cap_cost_at_arrival + contest + reachable enemy mass
  18: friendly_reachable_mass/100 distance-decayed friendly garrison that can support this target
  19: keepability_margin/100    friendly support minus enemy contest/reaction, clipped [-5,5]
  20: enemy_mass_soon/100       enemy fleet mass landing within _THREAT_ETA_WINDOW steps (clamp 5)
  21: threat_imminence          1/(min_enemy_eta+1); urgency in (0,0.5], 0 if no enemy inbound
  22-25: resolved ships for capture / capture-defend / maintain / all-in, divided by 200
"""

from __future__ import annotations

import math
import numpy as np
import torch

from timeline import project_timeline, timeline_features
from action_mask import resolve_intent_sizes_np

# Canonical feature semantics:
# - friendly-coverage roi-deflation: ALWAYS ON
# - pressure channels resolved per-fleet by the lead-aware swept-collision resolver
#   (torch_env._fleet_target_idx mirror below): ALWAYS ON
# - game-phase global channels (global dim 15): ALWAYS ON
# - roi enemy-deflation / zero-roi / surface threat-ETA: removed (never in a blessed run)
# Older toggle-based variants are intentionally unsupported in this code path.
_COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)  # MUST match torch_env.COMET_SPAWN_STEPS


def _resolve_fleet_targets(fx, fy, fcos, fsin, fspeed, px, py, pr, angvel):
    """Per-fleet lead-aware swept-collision target index, (E,) int (-1 if it hits nothing).

    Numpy mirror of torch_env._fleet_target_idx: project each candidate planet to its orbit
    position at the fleet ETA (4 Newton iters), keep the min-ETA planet the heading reaches
    within radius (perp < pr+0.5). All real planets are alive (orbit wars never destroys them),
    so no alive mask is needed. Returns a one-target-per-fleet assignment (no double-counting)."""
    E = fx.shape[0]
    P = px.shape[0]
    if E == 0 or P == 0:
        return np.full(E, -1, dtype=np.int64)
    orbit_r = np.sqrt((px - CENTER) ** 2 + (py - CENTER) ** 2)          # (P,)
    static = (orbit_r + pr) >= ROTATION_RADIUS_LIMIT
    phase0 = np.arctan2(py - CENTER, px - CENTER)                       # (P,)
    fxc = fx[:, np.newaxis]; fyc = fy[:, np.newaxis]                    # (E,1)
    spd = np.maximum(fspeed[:, np.newaxis], 1e-6)
    eta = np.clip((np.sqrt((px[np.newaxis, :] - fxc) ** 2 + (py[np.newaxis, :] - fyc) ** 2)
                   - pr[np.newaxis, :]) / spd, 0.0, None)               # (E,P)
    for _ in range(4):                                                  # converge ETA vs the moving target
        a = phase0[np.newaxis, :] + angvel * eta
        lx = np.where(static[np.newaxis, :], px[np.newaxis, :], CENTER + orbit_r[np.newaxis, :] * np.cos(a))
        ly = np.where(static[np.newaxis, :], py[np.newaxis, :], CENTER + orbit_r[np.newaxis, :] * np.sin(a))
        eta = np.clip((np.sqrt((lx - fxc) ** 2 + (ly - fyc) ** 2) - pr[np.newaxis, :]) / spd, 0.0, None)
    a = phase0[np.newaxis, :] + angvel * eta
    lx = np.where(static[np.newaxis, :], px[np.newaxis, :], CENTER + orbit_r[np.newaxis, :] * np.cos(a))
    ly = np.where(static[np.newaxis, :], py[np.newaxis, :], CENTER + orbit_r[np.newaxis, :] * np.sin(a))
    vx = lx - fxc; vy = ly - fyc
    along = vx * fcos[:, np.newaxis] + vy * fsin[:, np.newaxis]
    perp = np.abs(vx * fsin[:, np.newaxis] - vy * fcos[:, np.newaxis])
    candidate = (along > 0) & (perp < pr[np.newaxis, :] + 0.5)          # (E,P)
    eta_masked = np.where(candidate, eta, np.inf)
    best = np.argmin(eta_masked, axis=1)                               # (E,)
    has = candidate.any(axis=1)
    return np.where(has, best, -1).astype(np.int64)


def _resolved_headed(fx, fy, fcos, fsin, fspeed, px, py, pr, angvel):
    """One-hot (E,P) bool: fleet e attributed ONLY to its resolved target planet."""
    res = _resolve_fleet_targets(fx, fy, fcos, fsin, fspeed, px, py, pr, angvel)
    headed = np.zeros((fx.shape[0], px.shape[0]), dtype=bool)
    valid = res >= 0
    headed[np.nonzero(valid)[0], res[valid]] = True
    return headed


def game_phase_channels(step):
    """The 4 game-phase global channels from the integer game step:
    [phase_early (step<50), phase_mid (50<=step<100), phase_late (step>=100),
     norm_steps_to_next_comet_spawn]. Comet spawns at _COMET_SPAWN_STEPS; the last channel is
     (next_spawn - step)/100 in (0,1], or 1.0 once no spawn remains. Single source of truth for
     the scalar path; torch_env mirrors it vectorized (parity-tested)."""
    early = 1.0 if step < 50 else 0.0
    mid = 1.0 if (50 <= step < 100) else 0.0
    late = 1.0 if step >= 100 else 0.0
    nxt = next((S for S in _COMET_SPAWN_STEPS if S > step), None)
    comet_cycle = (nxt - step) / 100.0 if nxt is not None else 1.0
    return [early, mid, late, comet_cycle]

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
                     max_owned=MAX_OWNED_PLANETS, timeline=True):
    """Extract entity features from observation dict.

    Returns dict of torch tensors (no batch dim).

    timeline=True appends the 96 projected-timeline channels to planet_features
    (dim 20 → 116). Pass False only for pre-timeline checkpoints (planet_proj
    20-wide) — eval.py infers this from the checkpoint's weight shapes.
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
    owned_cands = []  # (array_idx, round(ships)) for every owned planet; top-MAX_OWNED chosen below

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

    # Planet ch12/13 attribution: each fleet resolved to its single lead-aware swept-collision
    # target (mirrors torch_env incoming_pw). Resolved once over ALL fleets × planets →
    # per-fleet target array-index (matches the loop's planet index i since both index
    # planets[:max_planets]).
    pp_resolved_tgt = None
    if n_fleets > 0 and n_planets > 0:
        _n_pp = min(n_planets, max_planets)
        _pp_px = np.array([planets[j][2] for j in range(_n_pp)], dtype=np.float32)
        _pp_py = np.array([planets[j][3] for j in range(_n_pp)], dtype=np.float32)
        _pp_pr = np.array([planets[j][4] for j in range(_n_pp)], dtype=np.float32)
        pp_resolved_tgt = _resolve_fleet_targets(
            fleet_x, fleet_y, fleet_cos, fleet_sin, _ship_speed_np(fleet_ships_arr),
            _pp_px, _pp_py, _pp_pr, angular_velocity)             # (n_fleets,) target idx, -1 none

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
        if pp_resolved_tgt is not None:
            incoming = (pp_resolved_tgt == i)
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

        # Track owned planets: collect ALL owned (idx, garrison); the top-MAX_OWNED by
        # garrison are selected after the loop (source ranking — mirrors owned_indices_for).
        if owner == player:
            owned_cands.append((i, int(round(ships))))

    # Source selection: the highest-GARRISON owned planets fill the MAX_OWNED source slots
    # (ties -> lowest array index). Parity-exact with VecTorchEnv.owned_indices_for and
    # action_mask (-round(ships)*P + idx ordering). No-op at <=max_owned owned. Owning >16 is
    # ~16% of steps (up to 30 owned) — firing only from the first-16-by-index left force-
    # bearing planets inert; rank by garrison so the planets that can contribute get the slots.
    owned_cands.sort(key=lambda t: (-t[1], t[0]))
    owned_count = min(len(owned_cands), max_owned)
    for _slot in range(owned_count):
        owned_indices[_slot] = owned_cands[_slot][0]

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

    global_list = [
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
    ]
    global_list.extend(game_phase_channels(step))  # 11-13 phase one-hot, 14 comet-cycle
    global_feats = np.array(global_list, dtype=np.float32)

    # Precompute enemy fleet ships racing toward each target planet (pairwise feat 14).
    # Result shape: (n_p_pair,) — independent of source slot, broadcast in compute_pairwise_features.
    n_p_pair = min(len(planets), max_planets)
    enemy_contest = np.zeros(n_p_pair, dtype=np.float32)
    enemy_mass_soon = np.zeros(n_p_pair, dtype=np.float32)
    threat_imminence = np.zeros(n_p_pair, dtype=np.float32)
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
            # (E, n_p) broadcast geometry for threat ETAs below
            vx_ep = tgt_x_p[np.newaxis, :] - efx[:, np.newaxis]
            vy_ep = tgt_y_p[np.newaxis, :] - efy[:, np.newaxis]
            efspeed = _ship_speed_np(efships)                                            # (E,)
            headed = _resolved_headed(efx, efy, efcos, efsin, efspeed,
                                      tgt_x_p, tgt_y_p, tgt_r_p, angular_velocity)
            enemy_contest = (efships[:, np.newaxis] * headed).sum(axis=0).astype(np.float32)
            # Threat timing (ch 20-21): ETA-profile the inbound enemy mass. eta = planet-fleet
            # center dist / fleet speed (mirrors the torch_env training path byte-for-byte).
            # ch14 sums ALL inbound mass; these add the WHEN — the only urgency signal in the
            # pairwise bundle.
            dist_ep = np.sqrt(vx_ep * vx_ep + vy_ep * vy_ep)                              # (E, n_p)
            eta_ep = np.clip(dist_ep / np.maximum(efspeed[:, np.newaxis], 1e-3), 1.0, None)  # (E, n_p)
            soon = headed & (eta_ep <= _THREAT_ETA_WINDOW)                               # (E, n_p)
            enemy_mass_soon = (efships[:, np.newaxis] * soon).sum(axis=0).astype(np.float32)
            eta_for_min = np.where(headed, eta_ep, 1e6)                                  # (E, n_p)
            threat_imminence = (1.0 / (eta_for_min.min(axis=0) + 1.0)).astype(np.float32)

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
            ffspeed = _ship_speed_np(ffships)
            headed_f = _resolved_headed(ffx, ffy, ffcos, ffsin, ffspeed,
                                        tgt_x_pf, tgt_y_pf, tgt_r_pf, angular_velocity)
            friendly_contest = (ffships[:, np.newaxis] * headed_f).sum(axis=0).astype(np.float32)

    pairwise = compute_pairwise_features(
        planets, owned_indices, owned_count, player, max_planets=max_planets,
        max_owned=max_owned, angular_velocity=angular_velocity, step=step,
        init_by_id=init_by_id, enemy_contest=enemy_contest,
        friendly_contest=friendly_contest,
        enemy_mass_soon=enemy_mass_soon, threat_imminence=threat_imminence,
        comet_ids=comet_planet_ids,
    )

    # --- Projected-future timeline (96 = 4 ch × 24 steps; planet dim 20 → 116) ---
    # Runs the SAME timeline.py code the training path uses (batch of 1), so both paths
    # encode identically by construction. Projects over ALL fleets in the obs (no
    # max_fleets truncation — matches torch_env, which projects its full fleet set).
    # timeline=False serves pre-timeline checkpoints (planet_proj 20-wide; eval infers).
    if timeline:
        pl_arr = np.zeros((max_planets, 7), dtype=np.float32)
        n_tp = min(n_planets, max_planets)
        if n_tp > 0:
            pl_arr[:n_tp] = np.array([p[:7] for p in planets[:n_tp]], dtype=np.float32)
        if len(fleets) > 0:
            fl_arr = np.array([f[:7] for f in fleets], dtype=np.float32)
        else:
            fl_arr = np.zeros((1, 7), dtype=np.float32)  # one dead slot; masked out below
        own_ts, garr_ts = project_timeline(
            torch.from_numpy(pl_arr).unsqueeze(0),
            torch.from_numpy(planet_mask).unsqueeze(0),
            torch.from_numpy(fl_arr).unsqueeze(0),
            torch.tensor([[len(fleets) > 0] * fl_arr.shape[0]], dtype=torch.bool),
            torch.tensor([angular_velocity], dtype=torch.float32),
            num_players=num_players,
        )
        tl_feats = timeline_features(own_ts, garr_ts, player)[0].numpy()  # (max_planets, 96)
        tl_feats *= planet_mask[:, np.newaxis]
        planet_feats = np.concatenate([planet_feats, tl_feats], axis=1)  # (max_planets, 116)

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


# Stable names for telemetry and audits. Position is the feature contract: keep this tuple in
# lock-step with compute_pairwise_features(), the torch_env twin, and ModelConfig.
PAIRWISE_FEATURE_NAMES = (
    "arrival_sin", "arrival_cos", "arrival_distance", "arrival_closeness",
    "sun_safe", "is_mine", "is_enemy", "is_neutral", "target_production",
    "target_valid", "ships_at_arrival", "capture_gap", "roi_20", "roi_50",
    "enemy_contest", "reachable_enemy_mass", "capture_value", "reactive_roi",
    "friendly_reachable_mass", "keepability_margin", "enemy_mass_soon",
    "threat_imminence", "intent_capture_ships", "intent_capture_defend_ships",
    "intent_maintain_ships", "intent_all_in_ships",
)
PAIRWISE_FEATURE_DIM = len(PAIRWISE_FEATURE_NAMES)

# Typical fleet size used to estimate an ETA prior (matches teacher MIN_SHIP_FLOOR ~ 10
# but in practice ETA varies modestly with size since speed is log-shaped).
_ETA_PROBE_SHIPS = 20
_ETA_PROBE_SPEED = 1.0 + (MAX_SPEED - 1.0) * (math.log(_ETA_PROBE_SHIPS) / math.log(1000.0)) ** 1.5

# Horizon (steps) over which enemy garrison counts as "reachable" to a target, scaled by
# a garrison-dependent fleet speed. Mirrors torch_env._DM_HORIZON / _ship_speed so the
# reachable_enemy_mass pairwise channel (ch 15) matches the GPU training path exactly.
_REACH_HORIZON = 18.0
_VALUE_HORIZON = 40.0
# Threat-timing window for ch20-21 (enemy mass landing "soon"). Must match torch_env._THREAT_ETA_WINDOW.
_THREAT_ETA_WINDOW = 6.0
_EPISODE_STEPS = 500.0


def _ship_speed_np(ships: np.ndarray) -> np.ndarray:
    """Garrison-dependent fleet speed (kaggle formula), numpy counterpart of
    torch_env._ship_speed: speed = 1 + (MAX-1)*(log(ships)/log(1000))**1.5, clamped [1, MAX]."""
    s = np.maximum(ships, 1.0)
    base = (np.log(s) / math.log(1000.0)) ** 1.5
    return np.minimum(1.0 + (MAX_SPEED - 1.0) * base, MAX_SPEED).astype(np.float32)


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
                              enemy_mass_soon: np.ndarray | None = None,
                              threat_imminence: np.ndarray | None = None,
                              comet_ids=None):
    """For each (owned-slot, target-planet) pair return geometric + ownership features.

    These are quantities the model cannot easily compute from raw (x, y) via
    attention: direction (sin/cos), distance, ETA-at-typical-ships, and sun-cross
    flag. Supplying them avoids reconstructing this geometry from gradients.

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
    comet_id_set = set(int(c) for c in (comet_ids or []))
    is_comet_target = np.array([int(planets[j][0]) in comet_id_set for j in range(n_p)], dtype=np.bool_)
    regular_target = ~is_comet_target
    # Current capture cost: how many ships needed to take each planet right now
    tgt_cap_cost = np.where(
        tgt_owner == -1,      tgt_ships + 1,
        np.where(tgt_owner != player, tgt_ships + tgt_prod * 3 + 1, 0.0)
    ).astype(np.float32)

    total_board_prod = float(np.sum(tgt_prod[regular_target]))
    total_board_prod = max(total_board_prod, 1.0)
    value_horizon = min(max(_EPISODE_STEPS - float(step), 0.0), _VALUE_HORIZON)
    owner_value_weight = np.where(
        tgt_owner == player,
        1.0,
        np.where(tgt_owner == -1, 1.0, 2.0),
    ).astype(np.float32)
    capture_value_mass = (owner_value_weight * tgt_prod * value_horizon).astype(np.float32)
    capture_value = np.clip(capture_value_mass / (total_board_prod * _VALUE_HORIZON), 0.0, 2.0)
    capture_value = np.where(regular_target, capture_value, 0.0).astype(np.float32)

    # Reachable enemy planet mass per target (ch 15): distance-decayed enemy garrison
    # that could reinforce/contest each target within the horizon. Source-slot independent
    # (depends only on enemy planets vs target), so computed once and broadcast. Raw (no
    # rho/eta scaling) — the per-target head learns its own reaction coefficient. Mirrors
    # torch_env._compute_pairwise ch15 and the dm-floor enemy_mass (producer_v2 cheap_enemy_pressure).
    reach_em = np.zeros(n_p, dtype=np.float32)
    enemy_src_mask = (tgt_owner != player) & (tgt_owner >= 0)
    if enemy_src_mask.any():
        s_idx = np.where(enemy_src_mask)[0]
        sx_e, sy_e, sg_e = tgt_x[s_idx], tgt_y[s_idx], tgt_ships[s_idx]
        src_reach_e = np.maximum(_ship_speed_np(sg_e) * _REACH_HORIZON, 1e-6)
        dxe = tgt_x[np.newaxis, :] - sx_e[:, np.newaxis]
        dye = tgt_y[np.newaxis, :] - sy_e[:, np.newaxis]
        dec = np.clip(1.0 - np.sqrt(dxe * dxe + dye * dye) / src_reach_e[:, np.newaxis], 0.0, None)
        dec[np.arange(len(s_idx)), s_idx] = 0.0   # a planet does not reinforce itself
        reach_em = (sg_e[:, np.newaxis] * dec).sum(axis=0).astype(np.float32)

    # Reachable friendly planet mass per target (ch 18): same distance-decayed support
    # calculation as reachable_enemy_mass, but from our planets. This is a keepability
    # signal: after capture, can nearby friendly garrisons support the target?
    reach_fm = np.zeros(n_p, dtype=np.float32)
    friendly_src_mask = (tgt_owner == player) & regular_target
    if friendly_src_mask.any():
        s_idx = np.where(friendly_src_mask)[0]
        sx_f, sy_f, sg_f = tgt_x[s_idx], tgt_y[s_idx], tgt_ships[s_idx]
        src_reach_f = np.maximum(_ship_speed_np(sg_f) * _REACH_HORIZON, 1e-6)
        dxf = tgt_x[np.newaxis, :] - sx_f[:, np.newaxis]
        dyf = tgt_y[np.newaxis, :] - sy_f[:, np.newaxis]
        dec = np.clip(1.0 - np.sqrt(dxf * dxf + dyf * dyf) / src_reach_f[:, np.newaxis], 0.0, None)
        dec[np.arange(len(s_idx)), s_idx] = 0.0
        dec[:, ~regular_target] = 0.0
        reach_fm = (sg_f[:, np.newaxis] * dec).sum(axis=0).astype(np.float32)

    # Precompute orbit info for each target planet
    tgt_is_orbiting = np.zeros(n_p, dtype=np.bool_)
    tgt_init_angle = np.zeros(n_p, dtype=np.float64)
    tgt_orbital_r = np.zeros(n_p, dtype=np.float64)
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
        # NEUTRALS DON'T REGROW (engine applies production only to owner != -1), so they
        # accrue NO ships during flight — adding prod*eta to them was phantom production that
        # priced cheap rotating neutrals as far more expensive than they are.
        prod_growth = np.where(tgt_owner == -1, 0.0, tgt_prod * eta)
        ships_at_arrival = np.minimum(tgt_ships + prod_growth, 500.0)
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

        enemy_contest_raw = enemy_contest[:n_p] if enemy_contest is not None else np.zeros(n_p, dtype=np.float32)
        friendly_contest_raw = friendly_contest[:n_p] if friendly_contest is not None else np.zeros(n_p, dtype=np.float32)
        enemy_pressure = reach_em + enemy_contest_raw
        friendly_support = reach_fm + friendly_contest_raw
        reactive_cost = cap_cost_at_arrival + enemy_pressure
        reactive_roi_40 = np.where(
            (tgt_owner != player) & regular_target,
            np.clip((capture_value_mass - reactive_cost) / np.maximum(reactive_cost, 1.0), -1.0, 1.0),
            0.0,
        ).astype(np.float32)
        friendly_reach = np.where(regular_target, np.minimum(reach_fm, 500.0) / 100.0, 0.0).astype(np.float32)
        keepability_margin = np.where(
            regular_target,
            np.clip((friendly_support - enemy_pressure) / 100.0, -5.0, 5.0),
            0.0,
        ).astype(np.float32)

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
        out[slot, :n_p, 15] = np.minimum(reach_em, 500.0) / 100.0  # reachable enemy planet mass
        out[slot, :n_p, 16] = capture_value
        out[slot, :n_p, 17] = reactive_roi_40
        out[slot, :n_p, 18] = friendly_reach
        out[slot, :n_p, 19] = keepability_margin
        if enemy_mass_soon is not None:
            out[slot, :n_p, 20] = np.minimum(enemy_mass_soon[:n_p], 500.0) / 100.0  # enemy mass landing soon
        if threat_imminence is not None:
            out[slot, :n_p, 21] = threat_imminence[:n_p]                            # 1/(min_enemy_eta+1)

        # Intent-sizing resolved sizes (ch 22-25): exact ships for capture / capture-defend /
        # maintain / all-in, clamped to source garrison. The head chooses the semantic; these
        # feed it each option's resolved cost and are read back at decode.
        S_arr = np.full(n_p, float(src[5]), dtype=np.float32)
        mass_soon_arr = (enemy_mass_soon[:n_p].astype(np.float32)
                         if enemy_mass_soon is not None else np.zeros(n_p, dtype=np.float32))
        intent_sizes = resolve_intent_sizes_np(
            cap_cost_at_arrival, reach_em[:n_p], mass_soon_arr,
            S_arr, is_mine[:n_p].astype(bool))                                      # (n_p, 4)
        out[slot, :n_p, 22:26] = np.minimum(intent_sizes, 500.0) / 200.0

    return out
