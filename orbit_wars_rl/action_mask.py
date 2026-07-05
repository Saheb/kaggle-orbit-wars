"""Action mask computation for Orbit Wars.

For each owned planet, determines which of the angle bins are legal
(don't cross sun, don't go out of bounds). Also computes per-planet
max sendable ships and ownership masks.

Uses numpy for computation, torch tensors for model input.
"""

from __future__ import annotations

import math
import os
import numpy as np
import torch

from reinforce_cooldown import is_blocked as _cd_is_blocked, record as _cd_record, on_ownership_loss as _cd_on_loss

# --- ship-commitment audit (probe-only; enable via action_mask._SHIP_AUDIT["on"]=True) ---
_SHIP_AUDIT = {"on": False}
_SHIP_AUDIT_DATA = {
    "attack":    {"n": 0, "overflow": 0, "full": 0, "ratio_sum": 0.0, "hist": [0] * 11},
    "reinforce": {"n": 0, "overflow": 0, "full": 0, "ratio_sum": 0.0, "hist": [0] * 11},
}
def _reset_ship_audit():
    for c in _SHIP_AUDIT_DATA.values():
        c.update(n=0, overflow=0, full=0, ratio_sum=0.0, hist=[0] * 11)
def _ship_audit_record(nominal, garrison, is_reinforce):
    d = _SHIP_AUDIT_DATA["reinforce" if is_reinforce else "attack"]
    g = max(1, int(garrison)); d["n"] += 1
    if nominal > garrison: d["overflow"] += 1
    if nominal >= garrison: d["full"] += 1
    r = nominal / g; d["ratio_sum"] += r
    d["hist"][10 if r >= 1.0 else min(int(r * 10), 9)] += 1

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

    # Find owned planets. Source selection: the highest-GARRISON owned planets fill the
    # MAX_OWNED slots (ties -> lowest array index), NOT the first-16-by-index. Parity-exact
    # with features.py / VecTorchEnv.owned_indices_for (-round(ships)*P + idx). No-op at
    # <=max_owned owned; matters when >16 are owned (~16% of steps, up to 30) so the
    # force-bearing planets — not arbitrary low-index ones — get the action slots.
    my_planets = [(i, p) for i, p in enumerate(planets) if p[1] == player]
    my_planets.sort(key=lambda ip: (-int(round(ip[1][5])), ip[0]))
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
    max_moves = MAX_OWNED_PLANETS  # = model's owned-slot width (16); kaggle env has NO move cap,
    # the old 8 was a self-nerf + train/eval mismatch (torch_env fires all 16). Bounded by owned_count.
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


_DEF_HORIZON = 18.0
_DEF_OVERHEAD = 1.0


def _def_stat(stats: dict | None, key: str, value: float = 1.0):
    if stats is not None:
        stats[key] = stats.get(key, 0.0) + value


def _def_fleet_target(planets, fleet):
    """Loose current-heading target resolver, matching the eval hold/decisive diagnostics."""
    best = None
    best_d = None
    c, s = math.cos(float(fleet[4])), math.sin(float(fleet[4]))
    fx, fy = float(fleet[2]), float(fleet[3])
    for p in planets:
        px, py, pr = float(p[2]), float(p[3]), float(p[4])
        vx, vy = px - fx, py - fy
        along = vx * c + vy * s
        if along <= 0:
            continue
        perp = abs(vx * s - vy * c)
        if perp >= pr + 1.5:
            continue
        d = math.hypot(vx, vy)
        if best_d is None or d < best_d:
            best_d, best = d, p
    return best


def _def_eta(src, tgt, ships: int) -> float:
    sx, sy, sr = float(src[2]), float(src[3]), float(src[4])
    tx, ty, tr = float(tgt[2]), float(tgt[3]), float(tgt[4])
    dist = max(0.0, math.hypot(tx - sx, ty - sy) - (sr + _LAUNCH_OFFSET + tr))
    return dist / max(_fleet_speed(max(1, int(ships))), 1e-6)


def _def_ship_count_for_cap(desired: float, cap: int, mode: str) -> int:
    """Pick a policy-representable ship count, bounded by safe-drain cap."""
    cap = int(max(0, cap))
    if cap <= 0:
        return 0
    options = [_ship_bin_to_count(i, cap, mode=mode)
               for i in range(len(FRACTION_BIN_VALUES) if mode == "fraction" else len(SHIP_COUNTS))]
    options = sorted(set(int(o) for o in options if 0 < int(o) <= cap))
    if not options:
        return 0
    want = max(1, int(math.ceil(desired)))
    ge = [o for o in options if o >= want]
    return ge[0] if ge else options[-1]


def _def_rank(scores, idx: int):
    if scores is None or idx is None or idx < 0 or idx >= len(scores):
        return None
    val = float(scores[idx])
    if val <= -1e20:
        return None
    return 1 + sum(1 for s in scores if float(s) > val)


def _def_ship_adequacy_rank(scores, required: int, max_ships: int, mode: str):
    if scores is None:
        return None
    pairs = []
    for i, s in enumerate(scores):
        count = _ship_bin_to_count(i, max_ships, mode=mode)
        pairs.append((float(s), i, count))
    pairs.sort(key=lambda x: x[0], reverse=True)
    for rank, (_score, _idx, count) in enumerate(pairs, start=1):
        if count >= required:
            return rank
    return None


def _head_prefixes(step: int):
    ph = "open" if step < 50 else ("mid" if step < 100 else "late")
    return ("natural_all", f"natural_{ph}")


def _head_stat(stats: dict | None, key: str, value: float = 1.0):
    if stats is not None:
        stats[key] = stats.get(key, 0.0) + value


def _head_attack_floor(tgt, eta: float, player: int) -> float:
    owner = int(tgt[1])
    if owner == player:
        return math.inf
    ships_at = float(tgt[5]) + max(1.0, float(eta)) * float(tgt[6])
    return ships_at + (1.0 if owner < 0 else float(tgt[6]) * 3.0 + 1.0)


def _head_best_attack_candidate(src, planets, player: int, cap: int):
    if cap <= 0:
        return None
    best = None
    for tidx, tgt in enumerate(planets):
        if int(tgt[0]) == int(src[0]) or int(tgt[1]) == player:
            continue
        eta = _def_eta(src, tgt, cap)
        floor = _head_attack_floor(tgt, eta, player)
        required = int(math.ceil(floor))
        if required <= 0 or cap < required:
            continue
        # Lightweight planner-like value: production stream and enemy ships are useful,
        # but high cost and slow arrival are bad. This is a diagnostic target, not a policy.
        score = (
            float(tgt[6]) * _DEF_HORIZON
            + (0.25 * float(tgt[5]) if int(tgt[1]) >= 0 else 0.0)
            - float(required)
            - 0.5 * float(eta)
        )
        if best is None or score > best["score"]:
            best = {
                "target_idx": tidx,
                "target_id": int(tgt[0]),
                "required": required,
                "score": score,
            }
    return best


def _head_threat_maps(planets, fleets, player: int):
    enemy_in: dict[int, float] = {}
    enemy_eta: dict[int, float] = {}
    friendly_in: dict[int, list[tuple[float, float]]] = {}
    for f in fleets or []:
        owner = int(f[1])
        if owner < 0:
            continue
        tgt = _def_fleet_target(planets, f)
        if tgt is None:
            continue
        tid = int(tgt[0])
        dist = math.hypot(float(tgt[2]) - float(f[2]), float(tgt[3]) - float(f[3]))
        eta = dist / max(_fleet_speed(int(f[6])), 1e-6)
        if owner == player:
            friendly_in.setdefault(tid, []).append((float(f[6]), eta))
        else:
            enemy_in[tid] = enemy_in.get(tid, 0.0) + float(f[6])
            enemy_eta[tid] = min(enemy_eta.get(tid, math.inf), eta)
    return enemy_in, enemy_eta, friendly_in


def _head_best_save_candidate(
    src,
    planets,
    fleets,
    player: int,
    cap: int,
    beta: float,
    own_reinforce_illegal,
    threat_maps=None,
):
    if cap <= 0:
        return None
    enemy_in, enemy_eta, friendly_in = (
        threat_maps if threat_maps is not None else _head_threat_maps(planets, fleets, player)
    )
    enemy_planets = [p for p in planets if int(p[1]) >= 0 and int(p[1]) != player]
    best = None
    for tidx, tgt in enumerate(planets):
        if int(tgt[1]) != player or int(tgt[0]) == int(src[0]):
            continue
        if own_reinforce_illegal(src, tgt):
            continue
        tid = int(tgt[0])
        ein = enemy_in.get(tid, 0.0)
        if ein <= 0:
            continue
        threat_eta = enemy_eta.get(tid, math.inf)
        eta = _def_eta(src, tgt, cap)
        if math.isfinite(threat_eta) and eta > threat_eta:
            continue
        em = 0.0
        for ep in enemy_planets:
            if int(ep[0]) == tid:
                continue
            d = math.hypot(float(ep[2]) - float(tgt[2]), float(ep[3]) - float(tgt[3]))
            reach = max(_fleet_speed(int(ep[5])) * _DEF_HORIZON, 1e-6)
            em += float(ep[5]) * max(1.0 - d / reach, 0.0)
        floor = ein + float(beta) * em + _DEF_OVERHEAD
        have = float(tgt[5]) + sum(
            ships for ships, f_eta in friendly_in.get(tid, [])
            if (not math.isfinite(threat_eta)) or f_eta <= threat_eta
        )
        deficit = int(math.ceil(floor - have))
        if deficit <= 0 or cap < deficit:
            continue
        score = float(tgt[6]) * _DEF_HORIZON - float(deficit) - 0.5 * float(eta)
        if best is None or score > best["score"]:
            best = {
                "target_idx": tidx,
                "target_id": tid,
                "required": deficit,
                "score": score,
            }
    return best


def _record_natural_candidate(stats, prefix: str, kind: str, intent: dict, cand: dict):
    _head_stat(stats, f"{prefix}_{kind}_n")
    fp = intent.get("fire_prob")
    if fp is not None and fp >= 0.5:
        _head_stat(stats, f"{prefix}_{kind}_fire_ready")
    if intent.get("target_id") == cand["target_id"]:
        _head_stat(stats, f"{prefix}_{kind}_chosen")
    target_rank = _def_rank(intent.get("target_scores"), cand.get("target_idx"))
    if target_rank is not None:
        _head_stat(stats, f"{prefix}_{kind}_target_rank_n")
        _head_stat(stats, f"{prefix}_{kind}_target_rank_sum", float(target_rank))
        if target_rank <= 1:
            _head_stat(stats, f"{prefix}_{kind}_target_top1")
            if fp is not None and fp < 0.5:
                _head_stat(stats, f"{prefix}_{kind}_target_top1_veto")
        if target_rank <= 3:
            _head_stat(stats, f"{prefix}_{kind}_target_top3")
        if target_rank <= 5:
            _head_stat(stats, f"{prefix}_{kind}_target_top5")
    ship_rank = _def_ship_adequacy_rank(
        intent.get("ship_scores"),
        int(cand.get("required", 0)),
        int(intent.get("max_ships", 0)),
        intent.get("ship_bin_mode", "absolute"),
    )
    if ship_rank is not None:
        _head_stat(stats, f"{prefix}_{kind}_ship_rank_n")
        _head_stat(stats, f"{prefix}_{kind}_ship_rank_sum", float(ship_rank))
        if ship_rank <= 1:
            _head_stat(stats, f"{prefix}_{kind}_ship_top1")
        if ship_rank <= 3:
            _head_stat(stats, f"{prefix}_{kind}_ship_top3")
        if ship_rank <= 5:
            _head_stat(stats, f"{prefix}_{kind}_ship_top5")
    if fp is not None and target_rank is not None and ship_rank is not None:
        if fp >= 0.5 and target_rank <= 1 and ship_rank <= 1:
            _head_stat(stats, f"{prefix}_{kind}_joint_top1")
        if fp >= 0.5 and target_rank <= 3 and ship_rank <= 3:
            _head_stat(stats, f"{prefix}_{kind}_joint_top3")


def _audit_natural_heads(
    slot_intents: dict[int, dict],
    planets,
    fleets,
    player: int,
    owned_indices,
    owned_count: int,
    max_ships,
    ship_bin_mode: str,
    beta: float,
    own_reinforce_illegal,
    stats: dict | None,
    step: int,
):
    """Passive head-coordination audit for natural policy logits.

    Logs whether fire/target/ship heads agree with lightweight planner-like attack and
    save candidates. It never changes the decoded action list.
    """
    if stats is None:
        return
    prefixes = _head_prefixes(int(step))
    threat_maps = _head_threat_maps(planets, fleets, player)
    for slot in range(min(int(owned_count), len(owned_indices), len(max_ships))):
        pidx = int(owned_indices[slot])
        if pidx < 0 or pidx >= len(planets):
            continue
        src = planets[pidx]
        if int(src[1]) != player:
            continue
        sid = int(src[0])
        intent = slot_intents.get(sid)
        if not intent:
            continue
        intent["ship_bin_mode"] = ship_bin_mode
        fp = intent.get("fire_prob")
        owner = intent.get("target_owner")
        for prefix in prefixes:
            _head_stat(stats, f"{prefix}_slots")
            if fp is not None:
                _head_stat(stats, f"{prefix}_fire_prob_sum", float(fp))
                if fp < 0.1:
                    _head_stat(stats, f"{prefix}_fire_lt_01")
                if fp < 0.3:
                    _head_stat(stats, f"{prefix}_fire_lt_03")
                if fp < 0.5:
                    _head_stat(stats, f"{prefix}_fire_lt_05")
            if intent.get("fired", False):
                _head_stat(stats, f"{prefix}_fired")
                if owner == player:
                    _head_stat(stats, f"{prefix}_chosen_own")
                elif owner is None or owner < 0:
                    _head_stat(stats, f"{prefix}_chosen_neutral")
                else:
                    _head_stat(stats, f"{prefix}_chosen_enemy")
        cap = int(max_ships[slot])
        attack = _head_best_attack_candidate(src, planets, player, cap)
        save = _head_best_save_candidate(
            src, planets, fleets, player, cap, beta, own_reinforce_illegal, threat_maps)
        for prefix in prefixes:
            if attack is None:
                _head_stat(stats, f"{prefix}_attack_none")
            else:
                _record_natural_candidate(stats, prefix, "attack", intent, attack)
            if save is None:
                _head_stat(stats, f"{prefix}_save_none")
            else:
                _record_natural_candidate(stats, prefix, "save", intent, save)


def _apply_defensive_reinforce_overlay(
    move_records: list[dict],
    slot_intents: dict[int, dict],
    planets,
    fleets,
    player: int,
    obs,
    owned_indices,
    max_ships,
    ship_bin_mode: str,
    k_sources: int,
    beta: float,
    max_targets: int,
    reinforce_garrison_floor: float,
    value_margin: float | None,
    overfill: float,
    own_reinforce_illegal,
    stats: dict | None,
):
    """Eval-time defensive deficit-fill overlay.

    For threatened own planets, fill the defensive floor from nearest reachable safe-drain
    sources. This is a hard diagnostic overlay: it deliberately bypasses the learned target
    and ship heads, while logging what those heads would have done.
    """
    if k_sources <= 0 or max_targets <= 0:
        return move_records
    _def_stat(stats, "steps")
    pid_to_idx = {int(p[0]): i for i, p in enumerate(planets)}
    enemy_in: dict[int, float] = {}
    enemy_eta: dict[int, float] = {}
    friendly_in: dict[int, list[tuple[float, float]]] = {}
    enemy_in_src: dict[int, float] = {}

    for f in fleets or []:
        owner = int(f[1])
        if owner < 0:
            continue
        tgt = _def_fleet_target(planets, f)
        if tgt is None:
            continue
        tid = int(tgt[0])
        dist = math.hypot(float(tgt[2]) - float(f[2]), float(tgt[3]) - float(f[3]))
        eta = dist / max(_fleet_speed(int(f[6])), 1e-6)
        if owner == player:
            friendly_in.setdefault(tid, []).append((float(f[6]), eta))
        else:
            enemy_in[tid] = enemy_in.get(tid, 0.0) + float(f[6])
            enemy_eta[tid] = min(enemy_eta.get(tid, math.inf), eta)
            enemy_in_src[tid] = enemy_in_src.get(tid, 0.0) + float(f[6])

    owned_slot_pids = []
    for slot in range(min(len(owned_indices), len(max_ships))):
        pidx = int(owned_indices[slot])
        if 0 <= pidx < len(planets) and int(planets[pidx][1]) == player:
            owned_slot_pids.append(int(planets[pidx][0]))

    enemy_planets = [p for p in planets if int(p[1]) >= 0 and int(p[1]) != player]
    source_planets = [planets[pid_to_idx[pid]] for pid in owned_slot_pids if pid in pid_to_idx]
    candidates = []
    for tgt in planets:
        if int(tgt[1]) != player:
            continue
        tid = int(tgt[0])
        ein = enemy_in.get(tid, 0.0)
        if ein <= 0:
            continue
        _def_stat(stats, "threatened_targets")
        threat_eta = enemy_eta.get(tid, math.inf)
        em = 0.0
        for ep in enemy_planets:
            if int(ep[0]) == tid:
                continue
            d = math.hypot(float(ep[2]) - float(tgt[2]), float(ep[3]) - float(tgt[3]))
            reach = max(_fleet_speed(int(ep[5])) * _DEF_HORIZON, 1e-6)
            em += float(ep[5]) * max(1.0 - d / reach, 0.0)
        floor = ein + float(beta) * em + _DEF_OVERHEAD
        have = float(tgt[5]) + sum(
            ships for ships, eta in friendly_in.get(tid, [])
            if (not math.isfinite(threat_eta)) or eta <= threat_eta
        )
        deficit = floor - have
        if deficit <= 0:
            _def_stat(stats, "already_safe_targets")
            continue
        sources = []
        for src in source_planets:
            sid = int(src[0])
            if sid == tid:
                continue
            if own_reinforce_illegal(src, tgt):
                _def_stat(stats, "blocked_by_cooldown_or_mask")
                continue
            garr = int(float(src[5]))
            src_threat = enemy_in_src.get(sid, 0.0)
            spare = garr if src_threat >= garr else int(max(0.0, garr - src_threat))
            if reinforce_garrison_floor > 0.0:
                spare = min(spare, int(max(0.0, garr - reinforce_garrison_floor)))
            if spare <= 0:
                continue
            eta = _def_eta(src, tgt, spare)
            if math.isfinite(threat_eta) and eta > threat_eta:
                continue
            sources.append((eta, sid, src, spare))
        sources.sort(key=lambda x: (x[0], -x[3], x[1]))
        # Fillability is tested with the nearest-k cap; otherwise this diagnostic can
        # silently require more sources than the requested hard constraint permits.
        fillable = sum(s[3] for s in sources[:k_sources])
        if fillable + 1e-6 < deficit:
            _def_stat(stats, "hopeless_targets")
            _def_stat(stats, "hopeless_deficit", deficit)
            continue
        selected_sources = sources[:k_sources]
        save_value = (
            float(tgt[6]) * _DEF_HORIZON
            + 0.25 * float(tgt[5])
            + 0.25 * float(ein)
            - float(deficit)
            - 0.5 * float(threat_eta if math.isfinite(threat_eta) else _DEF_HORIZON)
        )
        opportunity = 0.0
        for _eta, _sid, src, spare in selected_sources:
            attack = _head_best_attack_candidate(src, planets, player, int(spare))
            if attack is not None:
                opportunity += max(0.0, float(attack["score"]))
        net_value = save_value - opportunity
        if value_margin is not None:
            _def_stat(stats, "value_gate_checked")
            _def_stat(stats, "value_gate_save_value", save_value)
            _def_stat(stats, "value_gate_opportunity", opportunity)
            _def_stat(stats, "value_gate_net", net_value)
            if net_value < float(value_margin):
                _def_stat(stats, "value_gate_skipped_targets")
                _def_stat(stats, "value_gate_skipped_deficit", deficit)
                continue
        sort_value = net_value if value_margin is not None else deficit
        candidates.append((threat_eta, -sort_value, tid, tgt, deficit, selected_sources))

    if not candidates:
        return move_records

    candidates.sort()
    forced: dict[int, dict] = {}
    forced_targets = 0
    for _threat_eta, _neg_def, tid, tgt, deficit, sources in candidates[:max_targets]:
        requested = float(deficit) * max(1.0, float(overfill))
        remaining = requested
        target_forced = 0
        target_forced_ships = 0.0
        _def_stat(stats, "fillable_targets")
        _def_stat(stats, "deficit_before", deficit)
        _def_stat(stats, "requested_deficit", requested)
        if requested > deficit + 1e-6:
            _def_stat(stats, "overfill_targets")
        for _eta, sid, src, spare in sources:
            if remaining <= 0:
                break
            if sid in forced:
                continue
            send = _def_ship_count_for_cap(remaining, int(spare), ship_bin_mode)
            if send <= 0:
                continue
            forced[sid] = {
                "move": [sid, _target_intercept_angle(src, tgt, send, obs), int(send)],
                "src_id": sid,
                "target_id": tid,
                "is_own_target": True,
                "forced": True,
            }
            remaining -= send
            target_forced_ships += float(send)
            target_forced += 1
            _def_stat(stats, "forced_moves")
            _def_stat(stats, "forced_ships", send)
            intent = slot_intents.get(sid, {})
            if not intent.get("fired", False):
                _def_stat(stats, "orig_no_fire")
            else:
                owner = intent.get("target_owner")
                if intent.get("target_id") == tid:
                    _def_stat(stats, "orig_same_target")
                elif owner == player:
                    _def_stat(stats, "orig_other_own")
                elif owner is None or owner < 0:
                    _def_stat(stats, "orig_neutral")
                else:
                    _def_stat(stats, "orig_enemy")
                _def_stat(stats, "orig_ships", float(intent.get("ships", 0)))
                if float(intent.get("ships", 0)) < send:
                    _def_stat(stats, "orig_undersent")
                elif float(intent.get("ships", 0)) > send:
                    _def_stat(stats, "orig_oversent")
            fp = intent.get("fire_prob")
            if fp is not None:
                _def_stat(stats, "head_fire_n")
                _def_stat(stats, "head_fire_prob_sum", float(fp))
                if fp < 0.1:
                    _def_stat(stats, "head_fire_lt_01")
                if fp < 0.3:
                    _def_stat(stats, "head_fire_lt_03")
                if fp < 0.5:
                    _def_stat(stats, "head_fire_lt_05")
            target_rank = _def_rank(intent.get("target_scores"), pid_to_idx.get(tid))
            if target_rank is not None:
                _def_stat(stats, "head_target_rank_n")
                _def_stat(stats, "head_target_rank_sum", float(target_rank))
                if target_rank <= 1:
                    _def_stat(stats, "head_target_top1")
                if target_rank <= 3:
                    _def_stat(stats, "head_target_top3")
                if target_rank <= 5:
                    _def_stat(stats, "head_target_top5")
            ship_rank = _def_ship_adequacy_rank(
                intent.get("ship_scores"), int(send), int(intent.get("max_ships", 0)), ship_bin_mode)
            if ship_rank is not None:
                _def_stat(stats, "head_ship_rank_n")
                _def_stat(stats, "head_ship_rank_sum", float(ship_rank))
                if ship_rank <= 1:
                    _def_stat(stats, "head_ship_top1_ge_send")
                if ship_rank <= 3:
                    _def_stat(stats, "head_ship_top3_ge_send")
                if ship_rank <= 5:
                    _def_stat(stats, "head_ship_top5_ge_send")
            if fp is not None and target_rank is not None and ship_rank is not None:
                if fp >= 0.5 and target_rank <= 1 and ship_rank <= 1:
                    _def_stat(stats, "head_all_top1_ready")
                if fp >= 0.5 and target_rank <= 3 and ship_rank <= 3:
                    _def_stat(stats, "head_all_top3_ready")
        if target_forced:
            forced_targets += 1
            _def_stat(stats, "forced_targets")
            _def_stat(stats, "deficit_after", max(0.0, remaining))
            _def_stat(stats, "realized_fill_requested_sum", requested)
            _def_stat(stats, "realized_fill_forced_sum", target_forced_ships)
            if target_forced_ships + 1e-6 >= requested:
                _def_stat(stats, "realized_fill_full_targets")

    if not forced:
        return move_records

    # Forced defensive moves have priority. Keep non-conflicting policy moves up to the
    # model width so the overlay does not exceed the old eval/train action budget.
    final_records = list(forced.values())
    for rec in move_records:
        if rec["src_id"] in forced:
            _def_stat(stats, "policy_move_replaced")
            continue
        if len(final_records) >= MAX_OWNED_PLANETS:
            _def_stat(stats, "policy_move_dropped_for_cap")
            continue
        final_records.append(rec)
    return final_records


def actions_from_target_policy(fire_logits_target, target_logits, ship_logits_target, masks, obs, player,
                               fire_threshold=0.5, sample: bool = False,
                               ship_bin_mode: str = "absolute",
                               target_sanity_penalty: float = 0.0,
                               reserve_frac: float = 0.0,
                               allow_reinforce: bool = False,
                               reinforce_gate_min_planets: int = 0,
                               reinforce_forward_only: bool = False,
                               reinforce_garrison_floor: float = 0.0,
                               reverse_edge_cooldown: int = 0,
                               cooldown_last: dict = None,
                               cooldown_step: int = 0,
                               sufficient_commit_factor: float = 0.0,
                               defensive_reinforce_k: int = 0,
                               defensive_reinforce_beta: float = 2.2,
                               defensive_reinforce_max_targets: int = 1,
                               defensive_reinforce_value_margin: float | None = None,
                               defensive_reinforce_overfill: float = 1.0,
                               defensive_reinforce_stats: dict = None,
                               natural_head_audit_stats: dict = None,
                               natural_head_audit_beta: float = 2.2,
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
    fleets = obs.get("fleets") or []   # needed by the sufficient-commit veto (inbound-aware)
    owned_indices = masks["owned_indices"].cpu().numpy()
    max_ships = masks["max_ships"].cpu().numpy().squeeze(0)
    target_logits = target_logits.clone()
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

    cd_on = (reverse_edge_cooldown > 0 and cooldown_last is not None)

    def _own_reinforce_illegal(src_planet, tgt_planet):
        """True if an own (reinforce) target is barred by gate / forward-staging / reverse-edge cooldown."""
        if gate_block_own:
            return True
        if reinforce_forward_only and enemy_xy:  # no live enemy -> forward moot
            if not (_nearest_enemy_dist(tgt_planet) < _nearest_enemy_dist(src_planet)):
                return True
        # Reverse-edge cooldown: block reinforce src->dst if the reverse dst->src fired within K steps.
        if cd_on and _cd_is_blocked(cooldown_last, cooldown_step,
                                    int(src_planet[0]), int(tgt_planet[0]), reverse_edge_cooldown):
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
        target_dist = torch.distributions.Categorical(logits=target_logits)
        target_indices = target_dist.sample().cpu().numpy().squeeze(0)
        target_idx_t = torch.as_tensor(target_indices, device=target_logits.device).unsqueeze(0)
        chosen_fire_logits = torch.gather(fire_logits_target, -1, target_idx_t.unsqueeze(-1)).squeeze(-1)
        chosen_ship_logits = torch.gather(
            ship_logits_target,
            2,
            target_idx_t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, ship_logits_target.shape[-1]),
        ).squeeze(2)
        fire_dist = torch.distributions.Bernoulli(logits=chosen_fire_logits)
        ship_dist = torch.distributions.Categorical(logits=chosen_ship_logits)
        fire_decisions = (fire_dist.sample() > 0.5).cpu().numpy().squeeze(0)
        ship_bins = ship_dist.sample().cpu().numpy().squeeze(0)
        fire_prob_values = torch.sigmoid(chosen_fire_logits).cpu().numpy().squeeze(0)
    else:
        target_indices = torch.argmax(target_logits, dim=-1).cpu().numpy().squeeze(0)
        target_idx_t = torch.as_tensor(target_indices, device=target_logits.device).unsqueeze(0)
        chosen_fire_logits = torch.gather(fire_logits_target, -1, target_idx_t.unsqueeze(-1)).squeeze(-1)
        chosen_ship_logits = torch.gather(
            ship_logits_target,
            2,
            target_idx_t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, ship_logits_target.shape[-1]),
        ).squeeze(2)
        fire_prob_values = torch.sigmoid(chosen_fire_logits).cpu().numpy().squeeze(0)
        fire_decisions = (torch.sigmoid(chosen_fire_logits) > fire_threshold).cpu().numpy().squeeze(0)
        ship_bins = torch.argmax(chosen_ship_logits, dim=-1).cpu().numpy().squeeze(0)

    move_records = []
    slot_intents: dict[int, dict] = {}
    max_moves = MAX_OWNED_PLANETS  # = model's owned-slot width (16); kaggle env has NO move cap,
    # the old 8 was a self-nerf + train/eval mismatch (torch_env fires all 16). Bounded by owned_count.
    for slot in range(min(masks["owned_count"], fire_decisions.shape[0])):
        if len(move_records) >= max_moves:
            break

        pidx = int(owned_indices[slot])
        if pidx >= len(planets):
            continue
        src_id = int(planets[pidx][0])
        tidx = int(target_indices[slot])
        decoded_ships = _ship_bin_to_count(int(ship_bins[slot]), int(max_ships[slot]), mode=ship_bin_mode)
        slot_intents[src_id] = {
            "fired": bool(fire_decisions[slot]),
            "fire_prob": float(fire_prob_values[slot]) if not sample else None,
            "target_id": int(planets[tidx][0]) if 0 <= tidx < len(planets) else None,
            "target_owner": int(planets[tidx][1]) if 0 <= tidx < len(planets) else None,
            "ships": int(decoded_ships),
            "max_ships": int(max_ships[slot]),
            "target_scores": [float(x) for x in target_logits[0, slot, :len(planets)].detach().cpu().tolist()],
            "fire_scores_by_target": [float(x) for x in fire_logits_target[0, slot, :len(planets)].detach().cpu().tolist()],
            "ship_scores": [float(x) for x in chosen_ship_logits[0, slot].detach().cpu().tolist()],
        }
        if not fire_decisions[slot]:
            continue

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

        ships = decoded_ships
        if _SHIP_AUDIT["on"]:
            _nom = SHIP_COUNTS[int(ship_bins[slot])] if ship_bin_mode == "absolute" else ships
            _ship_audit_record(_nom, int(planets[pidx][5]), is_own_target)
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
        # Sufficient-commit parity (torch_env): veto a NEUTRAL attack launch where
        # (ships + friendly inbound arriving before us) can't beat the target's defense
        # (current garrison + enemy inbound arriving before us). Neutrals DON'T regrow
        # (engine applies production only to owner != -1) so there is NO production×ETA
        # term. Enemy targets exempt (under-strength attacks can soften/feint).
        # Reinforces (own targets) untouched (garrison floor instead).
        is_neutral_target = int(planets[tidx][1]) < 0
        if is_neutral_target and sufficient_commit_factor > 0.0:
            src = planets[pidx]
            tgt = planets[tidx]
            dist = math.hypot(tgt[2] - src[2], tgt[3] - src[3])
            eta = max(1.0, math.ceil(dist / max(_fleet_speed(ships), 1e-6)))
            projected_defense = tgt[5]
            tgt_id = int(tgt[0])
            friendly_inbound = 0.0
            enemy_inbound = 0.0
            for f in fleets:
                ft = _def_fleet_target(planets, f)
                if ft is None or int(ft[0]) != tgt_id:
                    continue
                f_speed = max(_fleet_speed(int(f[6])), 1e-6)
                f_dist = math.hypot(tgt[2] - f[2], tgt[3] - f[3])
                f_eta = f_dist / f_speed
                if f_eta <= eta:
                    if int(f[1]) == player:
                        friendly_inbound += f[6]
                    elif int(f[1]) >= 0:
                        enemy_inbound += f[6]
            projected_defense += enemy_inbound
            if (ships + friendly_inbound) <= projected_defense * sufficient_commit_factor:
                continue

        angle = _target_intercept_angle(planets[pidx], planets[tidx], ships, obs)
        move_records.append({
            "move": [src_id, angle, ships],
            "src_id": src_id,
            "target_id": int(planets[tidx][0]),
            "is_own_target": bool(is_own_target),
            "forced": False,
        })

    if natural_head_audit_stats is not None and not sample:
        _audit_natural_heads(
            slot_intents,
            planets,
            obs.get("fleets", []),
            player,
            owned_indices,
            int(masks["owned_count"]),
            max_ships,
            ship_bin_mode,
            float(natural_head_audit_beta),
            _own_reinforce_illegal,
            natural_head_audit_stats,
            int(obs.get("step", 0)),
        )

    if defensive_reinforce_k > 0 and allow_reinforce:
        move_records = _apply_defensive_reinforce_overlay(
            move_records,
            slot_intents,
            planets,
            obs.get("fleets", []),
            player,
            obs,
            owned_indices,
            max_ships,
            ship_bin_mode,
            int(defensive_reinforce_k),
            float(defensive_reinforce_beta),
            int(defensive_reinforce_max_targets),
            float(reinforce_garrison_floor),
            defensive_reinforce_value_margin,
            float(defensive_reinforce_overfill),
            _own_reinforce_illegal,
            defensive_reinforce_stats,
        )

    moves = [rec["move"] for rec in move_records[:max_moves]]
    _new_reinf_edges = [(int(rec["src_id"]), int(rec["target_id"]))
                        for rec in move_records[:max_moves] if rec.get("is_own_target")]

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

    # Reverse-edge cooldown commit (mirrors torch_env._apply_actions): consult used PRIOR state,
    # so update only now — (1) clear edges touching any planet we don't currently own (ownership
    # reset → recaptured planets aren't mis-blocked), (2) arm this step's executed reinforces.
    if cd_on:
        for p in planets:
            if int(p[1]) != player:
                _cd_on_loss(cooldown_last, int(p[0]))
        for s_id, d_id in _new_reinf_edges:
            _cd_record(cooldown_last, cooldown_step, s_id, d_id)

    return moves


def _enumerate_sane_target_candidates(obs: dict) -> dict:
    from orbit_wars_rl.producer_action_ranking import _enumerate_attack_candidates

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
    max_moves = MAX_OWNED_PLANETS  # = model's owned-slot width (16); kaggle env has NO move cap,
    # the old 8 was a self-nerf + train/eval mismatch (torch_env fires all 16). Bounded by owned_count.
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


# SINGLE SOURCE OF TRUTH for the ship-bin action space. model.py and torch_env.py import these
# (NUM_SHIP_BINS = len(SHIP_COUNTS)); export_agent.py inlines this module body, so they must stay
# defined here (not imported) for the standalone kaggle agent. Do NOT re-copy these elsewhere.
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
