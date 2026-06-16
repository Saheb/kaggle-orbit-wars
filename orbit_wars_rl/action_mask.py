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

from reinforce_cooldown import is_blocked as _cd_is_blocked, record as _cd_record, \
    on_ownership_loss as _cd_on_loss

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


def actions_from_target_policy(fire_probs, target_logits, ship_logits, masks, obs, player,
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
    _new_reinf_edges = []   # (src_id, dst_id) executed reinforces this step → recorded into cooldown_last after the loop
    max_moves = MAX_OWNED_PLANETS  # = model's owned-slot width (16); kaggle env has NO move cap,
    # the old 8 was a self-nerf + train/eval mismatch (torch_env fires all 16). Bounded by owned_count.
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
        # Sufficient-commit parity (torch_env): veto an ATTACK launch that can't beat
        # the target's current defense × factor → forces concentration. Reinforces
        # (own targets) untouched (they have the garrison floor instead).
        if (not is_own_target and sufficient_commit_factor > 0.0
                and ships <= planets[tidx][5] * sufficient_commit_factor):
            continue

        angle = _target_intercept_angle(planets[pidx], planets[tidx], ships, obs)
        moves.append([int(planets[pidx][0]), angle, ships])
        if cd_on and is_own_target:   # executed reinforce → arm the reverse edge after the loop
            _new_reinf_edges.append((int(planets[pidx][0]), int(planets[tidx][0])))

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
