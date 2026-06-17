"""Action mask computation for Orbit Wars.

For each owned planet, determines which of the angle bins are legal
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
        - angle_mask: (1, max_owned, 144) bool
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

        # Orbit Wars allows launching all ships from a planet.
        fire_mask[slot] = ps > 0
        max_ships_arr[slot] = max(0, int(ps))

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
        if ships > 0 and planets[pidx][5] >= ships:
            moves.append([from_id, angle, ships])

    return moves


_MAX_SHIP_SPEED = 6.0
_ROTATION_LIMIT = 50.0
_LAUNCH_OFFSET = 0.1


def _fleet_speed(ships: int, max_speed: float = _MAX_SHIP_SPEED) -> float:
    if ships <= 0:
        return 1.0
    s = 1.0 + (max_speed - 1.0) * (math.log(max(ships, 1)) / math.log(1000.0)) ** 1.5
    return min(s, max_speed)


def _target_intercept_angle(src_planet, target_planet, ships: int, obs) -> float:
    """Aim from src at target's lead (intercept) position.

    Iterative continuous lead, matching the engine. Adopted from the aim-benchmark
    reference (~95% vs our prior ~73% on reachable shots): predict the target from
    its CURRENT orbit position, subtract the source+target surface gap from the
    flight distance, and run 8 continuous (non-quantised) lead iterations. The old
    aimer over-led (full centre-to-centre distance, integer-ceil ETA, 4 iters).
    """
    sx, sy, s_r = float(src_planet[2]), float(src_planet[3]), float(src_planet[4])
    tx0, ty0, t_r = float(target_planet[2]), float(target_planet[3]), float(target_planet[4])
    max_speed = float(obs.get("ship_speed", _MAX_SHIP_SPEED))
    speed = _fleet_speed(ships, max_speed)
    omega = float(obs.get("angular_velocity", 0.0))

    # Orbit (radius + phase) about the centre, from the target's CURRENT position.
    # Static if it sits at/beyond the rotation-radius limit (engine leaves it fixed).
    dx0, dy0 = tx0 - CENTER, ty0 - CENTER
    orbit_r = math.hypot(dx0, dy0)
    static = (orbit_r + t_r) >= _ROTATION_LIMIT
    phase0 = math.atan2(dy0, dx0)

    def target_at(t):
        if static:
            return tx0, ty0
        a = phase0 + omega * t
        return CENTER + orbit_r * math.cos(a), CENTER + orbit_r * math.sin(a)

    gap = s_r + _LAUNCH_OFFSET + t_r
    t = max(0.0, (math.hypot(tx0 - sx, ty0 - sy) - gap) / speed)
    for _ in range(8):
        px, py = target_at(t)
        t = max(0.0, (math.hypot(px - sx, py - sy) - gap) / speed)
    px, py = target_at(t)
    return float(math.atan2(py - sy, px - sx))


def _fleet_eta_to_planet(fleet, planet) -> int | None:
    tx, ty, tr = float(planet[2]), float(planet[3]), float(planet[4])
    fx, fy, angle = float(fleet[2]), float(fleet[3]), float(fleet[4])
    c, s = math.cos(angle), math.sin(angle)
    vx, vy = tx - fx, ty - fy
    along = vx * c + vy * s
    perp = abs(vx * s - vy * c)
    if along <= 0 or perp >= tr + 1.5:
        return None
    speed = _fleet_speed(int(fleet[6]) if len(fleet) > 6 else 1)
    return max(1, int(math.ceil(max(0.0, along - tr) / speed)))


def _nearest_enemy_dist_global(planet, planets, player: int) -> float:
    enemies = [p for p in planets if int(p[1]) >= 0 and int(p[1]) != player]
    if not enemies:
        return float("inf")
    px, py = float(planet[2]), float(planet[3])
    return min(math.hypot(px - float(e[2]), py - float(e[3])) for e in enemies)


def defensive_overlay_moves(obs: dict, player: int, existing_moves: list,
                            garrison_floor: int = 10,
                            min_need: int = 5,
                            max_moves: int = 1,
                            eligible_target_pids: set[int] | None = None,
                            target_ages: dict[int, int] | None = None,
                            selector: dict | None = None,
                            selector_threshold: float = 0.5,
                            selector_mode: str = "survive",
                            multi_source_per_target: bool = False) -> list:
    """Generate post-policy rear-source support moves for threatened own planets.

    This mirrors the supervised synthetic-defense label rule, but keeps it
    isolated from the learned action heads. It is intentionally conservative:
    only rear sources may support more-forward owned targets, and already-used
    launch sources are not reused.
    """
    if max_moves <= 0:
        return []

    planets = obs.get("planets") or []
    own_planets = [p for p in planets if int(p[1]) == player]
    if len(own_planets) < 2:
        return []

    used_sources = {int(m[0]) for m in existing_moves if isinstance(m, list) and len(m) >= 1}
    overlay: list[list] = []
    for target in own_planets:
        if len(overlay) >= max_moves:
            break
        target_pid = int(target[0])
        if target_pid in used_sources:
            continue
        if eligible_target_pids is not None and target_pid not in eligible_target_pids:
            continue

        inbound_ships = 0
        min_eta: int | None = None
        for fleet in obs.get("fleets") or []:
            if int(fleet[1]) == player:
                continue
            eta = _fleet_eta_to_planet(fleet, target)
            if eta is None:
                continue
            inbound_ships += int(fleet[6]) if len(fleet) > 6 else 0
            min_eta = eta if min_eta is None else min(min_eta, eta)
        if inbound_ships <= 0 or min_eta is None:
            continue

        projected_garrison = float(target[5]) + float(target[6]) * min_eta
        need = int(math.ceil(inbound_ships + min_need - projected_garrison))
        if need < min_need:
            continue

        target_enemy_dist = _nearest_enemy_dist_global(target, planets, player)
        candidates = []
        for src in own_planets:
            src_pid = int(src[0])
            if src_pid == target_pid or src_pid in used_sources:
                continue
            sendable = int(src[5]) - int(garrison_floor)
            if sendable < min_need:
                continue
            source_enemy_dist = _nearest_enemy_dist_global(src, planets, player)
            if source_enemy_dist <= target_enemy_dist:
                continue
            source_target_dist = math.hypot(float(src[2]) - float(target[2]), float(src[3]) - float(target[3]))
            candidate_ships = min(sendable, max(need, min_need))
            support_gap = float(src[4]) + _LAUNCH_OFFSET + float(target[4])
            support_eta = max(1, int(math.ceil(
                max(0.0, source_target_dist - support_gap) / max(_fleet_speed(candidate_ships), 1e-6)
            )))
            candidates.append((sendable, source_enemy_dist, source_target_dist, support_eta, src))

        if not candidates:
            continue

        if multi_source_per_target:
            candidates.sort(key=lambda x: (x[3] > min_eta, x[3], -x[0]))
        else:
            candidates.sort(key=lambda x: x[0], reverse=True)

        remaining_need = max(need, min_need)
        for sendable, source_enemy_dist, source_target_dist, support_eta, src in candidates:
            if len(overlay) >= max_moves:
                break
            ships = min(sendable, max(remaining_need, min_need))
            if selector is not None:
                target_age = 0.0 if target_ages is None else float(target_ages.get(target_pid, 0))
                eta_margin = float(min_eta - support_eta)
                features = torch.tensor([
                    float(obs.get("step", 0)),
                    float(len(own_planets)),
                    target_age,
                    float(target[5]),
                    float(target[6]),
                    float(inbound_ships),
                    float(min_eta),
                    float(projected_garrison),
                    float(need),
                    float(src[5]),
                    float(sendable),
                    float(need) / max(float(sendable), 1.0),
                    float(target_enemy_dist),
                    float(source_enemy_dist),
                    float(source_enemy_dist - target_enemy_dist),
                    float(source_target_dist),
                    float(support_eta),
                    eta_margin,
                    float(support_eta <= min_eta),
                ], dtype=torch.float32)
                mean = selector["mean"].float()
                std = selector["std"].float().clamp(min=1e-6)
                weights = selector["weights"].float()
                bias = selector["bias"].float()
                if features.numel() > mean.numel():
                    features = features[:mean.numel()]
                elif features.numel() < mean.numel():
                    pad = torch.zeros(mean.numel() - features.numel(), dtype=features.dtype)
                    features = torch.cat([features, pad])
                raw_score = ((features - mean) / std * weights).sum() + bias
                if selector.get("activation", "sigmoid") == "linear":
                    score = raw_score.item()
                else:
                    score = torch.sigmoid(raw_score).item()
                if selector_mode == "risk":
                    if score > selector_threshold:
                        continue
                elif score < selector_threshold:
                    continue
            angle = _target_intercept_angle(src, target, ships, obs)
            overlay.append([int(src[0]), angle, int(ships)])
            used_sources.add(int(src[0]))
            remaining_need -= ships
            if not multi_source_per_target or remaining_need <= 0:
                break
        if len(overlay) >= max_moves:
            break

    return overlay


def actions_from_target_policy(fire_probs, target_logits, ship_logits, masks, obs, player,
                               fire_threshold=0.5, sample: bool = False,
                               ship_bin_mode: str = "absolute",
                               target_sanity_penalty: float = 0.0,
                               reserve_frac: float = 0.0,
                               allow_reinforce: bool = False,
                               reinforce_gate_min_planets: int = 0,
                               reinforce_forward_only: bool = False,
                               reinforce_garrison_floor: float = 0.0,
                               threat_logits=None,
                               threat_target_bias: float = 0.0,
                               reinforce_target_bias: float = 0.0,
                               defense_overlay: bool = False,
                               defense_overlay_garrison_floor: int = 10,
                               defense_overlay_min_need: int = 5,
                               defense_overlay_max_moves: int = 1,
                               defense_overlay_eligible_target_pids: set[int] | None = None,
                               defense_overlay_target_ages: dict[int, int] | None = None,
                               defense_overlay_selector: dict | None = None,
                               defense_overlay_selector_threshold: float = 0.5,
                               defense_overlay_selector_mode: str = "survive",
                               defense_overlay_multi_source_per_target: bool = False,
                               veto_stats: dict = None):
    """Convert policy outputs to actions using target planet logits for aiming.

    allow_reinforce: must MATCH the env's setting the checkpoint was trained with.
    False (default) = own planets are illegal targets. True = own planets are legal
    (reinforcement), only the launch source planet is excluded.

    reinforce_gate_min_planets / reinforce_forward_only / reinforce_garrison_floor:
    the three reinforce-DISCIPLINE masks from torch_env. They constrain only own
    (reinforce) targets; enemy/neutral are never affected. MUST match training, else
    the policy emits reinforce moves it was masked from at train time (e.g. reinforcing
    a 1-2 planet opening instead of expanding) and self-sabotages at inference.
    """
    planets = obs["planets"]
    owned_indices = masks["owned_indices"].cpu().numpy()
    max_ships = masks["max_ships"].cpu().numpy().squeeze(0)
    target_logits = target_logits.clone()
    if allow_reinforce and reinforce_target_bias != 0.0:
        for tidx, tgt in enumerate(planets[:target_logits.shape[-1]]):
            if int(tgt[1]) == player:
                target_logits[:, :, tidx] += float(reinforce_target_bias)
    if allow_reinforce and threat_logits is not None and threat_target_bias != 0.0:
        threat = torch.sigmoid(threat_logits).detach().cpu().squeeze(0)
        planet_idx_to_threat = {}
        for slot in range(min(masks["owned_count"], len(threat))):
            pidx = int(owned_indices[slot])
            if pidx < len(planets) and int(planets[pidx][1]) == player:
                planet_idx_to_threat[pidx] = float(threat[slot])
        for tidx, risk in planet_idx_to_threat.items():
            if tidx < target_logits.shape[-1]:
                target_logits[:, :, tidx] += float(threat_target_bias) * risk
    # Pre-discipline-mask copy: lets veto_stats see what the policy WANTS (unmasked argmax)
    # vs what the masks allow — i.e. which mask is actually binding on intended reinforces.
    _pre_mask_logits = target_logits.detach().clone() if veto_stats is not None else None

    # ----- reinforce-discipline precompute (parity with torch_env) -----
    owned_count = int(masks["owned_count"])
    gate_block_own = (allow_reinforce and reinforce_gate_min_planets > 0
                      and owned_count < reinforce_gate_min_planets)
    # enemy = owner >= 0 and != player (neutrals owner < 0 excluded), matching torch_env.
    enemy_xy = ([(float(p[2]), float(p[3])) for p in planets
                 if int(p[1]) >= 0 and int(p[1]) != player]
                if (allow_reinforce and reinforce_forward_only) else [])

    def _nearest_enemy_dist(p):
        px, py = float(p[2]), float(p[3])
        return min(math.hypot(px - ex, py - ey) for ex, ey in enemy_xy)

    def _own_reinforce_illegal(src_planet, tgt_planet):
        """True if an own (reinforce) target is barred by gate/forward-staging."""
        if gate_block_own:
            return True
        if reinforce_forward_only and enemy_xy:  # no live enemy -> forward moot
            if not (_nearest_enemy_dist(tgt_planet) < _nearest_enemy_dist(src_planet)):
                return True
        return False

    # Restrict target choice to legal launch targets before argmax / sampling.
    # The prior path argmaxed over all planets and then dropped own/self picks,
    # turning many fire-positive slots into silent no-ops at inference.
    for slot in range(min(masks["owned_count"], target_logits.shape[1])):
        pidx = int(owned_indices[slot])
        if pidx >= len(planets):
            continue
        for tidx, tgt in enumerate(planets[:target_logits.shape[-1]]):
            is_source = int(tgt[0]) == int(planets[pidx][0])
            is_own = int(tgt[1]) == player
            illegal = is_source or (is_own and not allow_reinforce)
            if not illegal and is_own and allow_reinforce:
                illegal = _own_reinforce_illegal(planets[pidx], tgt)
            if illegal:
                target_logits[:, slot, tidx] = -1e9

    if target_sanity_penalty > 0.0:
        cand_info = _enumerate_sane_target_candidates(obs)
        _apply_target_sanity_penalty_from_candidates(
            target_logits,
            masks,
            obs,
            player,
            cand_info["candidates"],
            penalty=float(target_sanity_penalty),
        )

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
        is_source = int(planets[pidx][0]) == int(planets[tidx][0])
        is_own_target = int(planets[tidx][1]) == player
        if is_source or (is_own_target and not allow_reinforce):
            continue
        # Reinforce-discipline parity: a slot whose every target was logit-masked
        # still argmaxes to one of them; reject gated/backward own reinforces here too.
        if is_own_target and allow_reinforce and _own_reinforce_illegal(planets[pidx], planets[tidx]):
            continue

        ships = _ship_bin_to_count(int(ship_bins[slot]), int(max_ships[slot]), mode=ship_bin_mode)
        # Reserve cap (probe): keep at least reserve_frac of the source planet's
        # current ships at home — forces a defensive garrison instead of
        # committing the whole army forward. reserve_frac=0.0 = no change.
        if reserve_frac > 0.0:
            ships = min(ships, int(planets[pidx][5] * (1.0 - reserve_frac)))
        if ships <= 0 or planets[pidx][5] < ships:
            continue
        # Garrison floor parity (torch_env): a reinforce must not drain the source
        # below the floor. Attacks (enemy/neutral) are never garrison-limited.
        if (is_own_target and allow_reinforce and reinforce_garrison_floor > 0.0
                and (planets[pidx][5] - ships) < reinforce_garrison_floor):
            continue

        angle = _target_intercept_angle(planets[pidx], planets[tidx], ships, obs)
        moves.append([int(planets[pidx][0]), angle, ships])

    # ----- veto diagnostics: of the reinforces the policy WANTS, which mask blocks them? -----
    # "want" = the unmasked-argmax target (over real planets, source excluded) for a fire-positive
    # slot. If that want is an own planet (reinforce intent), attribute the block to gate / forward /
    # garrison-floor (mutually exclusive, in production-check order). No effect on `moves`.
    if veto_stats is not None and allow_reinforce:
        nP = len(planets)
        for slot in range(min(masks["owned_count"], fire_decisions.shape[0])):
            if not fire_decisions[slot]:
                continue
            pidx = int(owned_indices[slot])
            if pidx >= nP:
                continue
            src = planets[pidx]
            row = _pre_mask_logits[0, slot, :nP].clone()
            for ti in range(nP):
                if int(planets[ti][0]) == int(src[0]):
                    row[ti] = -1e30                      # exclude source (physically illegal)
            want = int(torch.argmax(row))
            wt = planets[want]
            veto_stats["fire_slots"] = veto_stats.get("fire_slots", 0) + 1
            if int(wt[1]) != player:
                veto_stats["attack_intent"] = veto_stats.get("attack_intent", 0) + 1
                continue
            veto_stats["reinforce_intent"] = veto_stats.get("reinforce_intent", 0) + 1
            if gate_block_own:
                veto_stats["blocked_gate"] = veto_stats.get("blocked_gate", 0) + 1
            elif (reinforce_forward_only and enemy_xy
                  and not (_nearest_enemy_dist(wt) < _nearest_enemy_dist(src))):
                veto_stats["blocked_forward"] = veto_stats.get("blocked_forward", 0) + 1
            else:
                sent = _ship_bin_to_count(int(ship_bins[slot]), int(max_ships[slot]), mode=ship_bin_mode)
                if reserve_frac > 0.0:
                    sent = min(sent, int(src[5] * (1.0 - reserve_frac)))
                if reinforce_garrison_floor > 0.0 and (src[5] - sent) < reinforce_garrison_floor:
                    veto_stats["blocked_floor"] = veto_stats.get("blocked_floor", 0) + 1
                else:
                    veto_stats["reinforce_allowed"] = veto_stats.get("reinforce_allowed", 0) + 1

    if defense_overlay and len(moves) < max_moves:
        moves.extend(defensive_overlay_moves(
            obs,
            player,
            moves,
            garrison_floor=defense_overlay_garrison_floor,
            min_need=defense_overlay_min_need,
            max_moves=min(defense_overlay_max_moves, max_moves - len(moves)),
            eligible_target_pids=defense_overlay_eligible_target_pids,
            target_ages=defense_overlay_target_ages,
            selector=defense_overlay_selector,
            selector_threshold=defense_overlay_selector_threshold,
            selector_mode=defense_overlay_selector_mode,
            multi_source_per_target=defense_overlay_multi_source_per_target,
        ))

    return moves


def _enumerate_sane_target_candidates(obs: dict) -> dict:
    from orbit_wars_rl.analyze_producer_action_ranking import _enumerate_attack_candidates

    return _enumerate_attack_candidates(obs)


def _apply_target_sanity_penalty_from_candidates(
    target_logits: torch.Tensor,
    masks,
    obs: dict,
    player: int,
    candidates,
    *,
    penalty: float,
    max_score_gap: float = 3.0,
    max_eta_gap: int = 4,
) -> None:
    """Penalize same-source targets that are clearly dominated locally."""
    if penalty <= 0.0:
        return

    planets = obs["planets"]
    owned_indices = masks["owned_indices"].cpu().numpy()
    per_source: dict[int, list] = {}
    for cand in candidates:
        if not bool(getattr(cand, "valid", False)) or int(getattr(cand, "ships", 0)) <= 0:
            continue
        per_source.setdefault(int(cand.source_id), []).append(cand)

    for slot in range(min(masks["owned_count"], target_logits.shape[1])):
        pidx = int(owned_indices[slot])
        if pidx >= len(planets):
            continue
        src_id = int(planets[pidx][0])
        source_cands = per_source.get(src_id)
        if not source_cands:
            continue
        best_score = max(float(c.score) for c in source_cands)
        best_eta = min(int(c.eta) for c in source_cands)
        for cand in source_cands:
            tidx = int(cand.target_idx)
            score_gap = best_score - float(cand.score)
            eta_gap = int(cand.eta) - best_eta
            if score_gap > max_score_gap or eta_gap > max_eta_gap:
                target_logits[:, slot, tidx] -= penalty


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
        if ships > 0 and planets[pidx][5] >= ships:
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
