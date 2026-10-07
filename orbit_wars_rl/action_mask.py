"""Action masking and target-based action decoding for Orbit Wars.

Computes source/target legality, ship commitment constraints, and converts policy
outputs into engine moves. Uses NumPy for observation-side geometry and returns
Torch tensors for model input.
"""

from __future__ import annotations

import math
import numpy as np
import torch

from binary_policy import binary_action_log_probs
from reinforce_cooldown import is_blocked as _cd_is_blocked, record as _cd_record, on_ownership_loss as _cd_on_loss

NUM_ANGLE_BINS = 144
ANGLE_BIN_WIDTH = 2 * math.pi / NUM_ANGLE_BINS
CENTER = 50.0
SUN_RADIUS = 10.0
BOARD_SIZE = 100.0
MAX_OWNED_PLANETS = 16

def compute_action_masks(obs, player, max_owned=MAX_OWNED_PLANETS):
    """Compute action masks from observation dict.

    Returns dict with torch tensors (batch dim 0=1):
        - fire_mask: (1, max_owned) bool
        - max_ships: (1, max_owned) int
        - owned_indices: (max_owned,) int — indices into planet array
        - owned_count: int
        - slot_valid: (1, max_owned) bool
    """
    planets = obs["planets"]

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
    max_ships_arr = np.zeros(max_owned, dtype=np.int64)
    slot_valid = np.zeros(max_owned, dtype=np.bool_)

    for slot, (idx, p) in enumerate(my_planets[:max_owned]):
        slot_valid[slot] = True
        owned_indices[slot] = idx
        ps = p[5]

        # Orbit Wars allows launching all ships from a planet.
        fire_mask[slot] = ps > 0
        max_ships_arr[slot] = max(0, int(ps))

    # Convert to torch tensors with batch dim
    return {
        "fire_mask": torch.from_numpy(fire_mask).unsqueeze(0),
        "max_ships": torch.from_numpy(max_ships_arr).unsqueeze(0),
        "owned_indices": torch.from_numpy(owned_indices),
        "owned_count": n_owned,
        "slot_valid": torch.from_numpy(slot_valid).unsqueeze(0),
    }


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

    Iterative continuous lead, matching the engine: predict the target from its
    current orbit position, subtract the source and target surface gap from the
    flight distance, and run 8 non-quantised lead iterations.
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


def actions_from_target_policy(fire_logits_target, target_logits, ship_logits_target, masks, obs, player,
                               fire_threshold=0.5, sample: bool = False,
                               ship_bin_mode: str = "absolute",
                               binary_commit_gates: str = "full",
                               pairwise_features=None,   # (MO, P, >=26) — binary mode sizes/gates COMMIT from it
                               allow_reinforce: bool = False,
                               reinforce_gate_min_planets: int = 0,
                               reinforce_forward_only: bool = False,
                               reinforce_garrison_floor: float = 0.0,
                               reverse_edge_cooldown: int = 0,
                               cooldown_last: dict = None,
                               cooldown_step: int = 0,
                               sufficient_commit_factor: float = 0.0):
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
    binary_sizes = None
    binary_can_fire = np.ones(target_logits.shape[1], dtype=np.bool_)
    if ship_bin_mode == "binary":
        if pairwise_features is None:
            raise ValueError("binary ship mode requires pairwise_features")
        binary_sizes, binary_feasible = resolve_binary_commit_np(
            pairwise_features, max_ships, gates=binary_commit_gates)

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
        if ship_bin_mode == "binary":
            n_tgt = min(len(planets), target_logits.shape[-1], binary_feasible.shape[1])
            base_legal = target_logits[0, slot, :n_tgt] > -1e8
            legal_commit = base_legal.cpu().numpy() & binary_feasible[slot, :n_tgt]
            if legal_commit.any():
                illegal_commit = torch.as_tensor(~legal_commit, device=target_logits.device)
                target_logits[:, slot, :n_tgt].masked_fill_(illegal_commit.unsqueeze(0), -1e9)
            else:
                # Keep a valid categorical row for the unused target sample; force NOOP below.
                binary_can_fire[slot] = False

    if ship_bin_mode == "binary":
        actionable = torch.as_tensor(binary_can_fire, device=target_logits.device).unsqueeze(0)
        log_noop, log_commit = binary_action_log_probs(
            target_logits, fire_logits_target, fire_mask=actionable)
        action_logits = torch.cat([log_noop.unsqueeze(-1), log_commit], dim=-1)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = (action_dist.sample() if sample else torch.argmax(action_logits, dim=-1))
        fire_t = action > 0
        target_idx_t = (action - 1).clamp(min=0)
        target_indices = target_idx_t.cpu().numpy().squeeze(0)
        fire_decisions = fire_t.cpu().numpy().squeeze(0)
        ship_bins = np.zeros_like(target_indices)
    elif sample:
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
    else:
        target_indices = torch.argmax(target_logits, dim=-1).cpu().numpy().squeeze(0)
        target_idx_t = torch.as_tensor(target_indices, device=target_logits.device).unsqueeze(0)
        chosen_fire_logits = torch.gather(fire_logits_target, -1, target_idx_t.unsqueeze(-1)).squeeze(-1)
        chosen_ship_logits = torch.gather(
            ship_logits_target,
            2,
            target_idx_t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, ship_logits_target.shape[-1]),
        ).squeeze(2)
        fire_decisions = (torch.sigmoid(chosen_fire_logits) > fire_threshold).cpu().numpy().squeeze(0)
        ship_bins = torch.argmax(chosen_ship_logits, dim=-1).cpu().numpy().squeeze(0)

    move_records = []
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
        if ship_bin_mode == "binary":
            decoded_ships = int(round(float(binary_sizes[slot, tidx])))
        else:
            decoded_ships = _ship_bin_to_count(int(ship_bins[slot]), int(max_ships[slot]))
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
        })

    moves = [rec["move"] for rec in move_records[:max_moves]]
    _new_reinf_edges = [(int(rec["src_id"]), int(rec["target_id"]))
                        for rec in move_records[:max_moves] if rec.get("is_own_target")]

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


# SINGLE SOURCE OF TRUTH for the ship-bin action space. model.py and torch_env.py import these
# (NUM_SHIP_BINS = len(SHIP_COUNTS)); export_agent.py inlines this module body, so they must stay
# defined here (not imported) for the standalone kaggle agent. Do NOT re-copy these elsewhere.
SHIP_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 19, 22, 26, 30, 35, 42, 50, 60, 72, 86, 102, 122, 145, 173, 206, 245, 290, 350, 420]


def _ship_bin_to_count(bin_idx, max_ships):
    """Absolute ship-bin index → ship count: SHIP_COUNTS[bin_idx], capped at max_ships."""
    max_ships = max(1, int(max_ships))
    return min(SHIP_COUNTS[bin_idx], max_ships)


# --- Intent ship-sizing resolver (target-relative sizing; experiments.md #4) ---------------
# Resolves four SEMANTIC sizes (capture / capture-defend / maintain / all-in) to exact integer
# ship counts from the target's requirement. The intent ship MODE that let the policy pick one
# was removed in the 2026-10 cleanup (superseded by binary); the sizes live on as pairwise
# feature channels ch22-25 (and "full" binary gates read ch24). Inputs are the already
# parity-exact pairwise quantities (cap_cost_at_arrival, reachable_enemy_mass ch15,
# enemy_mass_soon ch20, source garrison); the SAME formula runs in torch_env._resolve_intent_sizes
# (training) and here (eval/export — inlined). Keep the two in lockstep; tests/ has a fuzz parity.
INTENT_CAPTURE, INTENT_CAPTURE_DEFEND, INTENT_MAINTAIN, INTENT_ALL_IN = 0, 1, 2, 3
NUM_INTENTS = 4
MIN_BINARY_COMMIT_SHIPS = 5
_INTENT_CEIL_EPS = 1e-3   # snap near-integer costs before ceil so GPU-float and numpy round identically


def resolve_intent_sizes_np(cap_cost, reach_em, mass_soon, src_ships, is_own):
    """Raw integer ships per intent → shape (..., 4), each clamped to [0, src_ships].

    capture       = cap_cost_at_arrival             (D+1 — exact ships to flip)
    capture-defend= cap_cost + reachable_enemy_mass (capture + survive the counter, ch15)
    maintain      = incoming_threat + 1  (own tgt)  (cover enemy_mass_soon, ch20; else 0)
    all-in        = source garrison                 (send everything — the un-under-committable floor)
    Args broadcast to a common shape; is_own is a bool array. `ceil(x - eps)` snaps costs within
    1e-3 of an integer so float32(GPU)/float64(numpy) produce identical integers.
    """
    S = np.clip(src_ships, 0.0, None).astype(np.float32)
    ceil_snap = lambda x: np.ceil(x - _INTENT_CEIL_EPS)
    capture = np.clip(ceil_snap(cap_cost), 0.0, S)
    cap_def = np.clip(ceil_snap(cap_cost + reach_em), 0.0, S)
    maintain = np.where(is_own, np.clip(ceil_snap(mass_soon) + 1.0, 0.0, S), 0.0)
    all_in = S
    return np.stack([capture, cap_def, maintain, all_in], axis=-1).astype(np.float32)


def resolve_binary_commit_np(pairwise_features, src_ships, gates="full"):
    """Deterministic NOOP/COMMIT plan from normalized pairwise features.

    ``gates="full"`` (legacy): non-owned targets commit the full source garrison, but only when
    that single source can afford the projected capture cost; owned targets commit the
    maintain/defend amount and are legal only when that amount is >= MIN_BINARY_COMMIT_SHIPS.

    ``gates="minimal"``: COMMIT means send the whole garrison at ANY target; the only gate is
    having MIN_BINARY_COMMIT_SHIPS ships. Measured on 758 real own-target / 12,192 attack cells,
    "full" removes 80.2% of the action space — capture_required alone blocks 62.2% of attacks
    (so multi-source pincers are inexpressible) and maintain<5 blocks 73.3% of reinforces (so a
    planet cannot be reinforced until >=4 enemy ships are already <=6 steps out, i.e. pre-emptive
    consolidation is impossible). Both gates compute a verdict from features the model already
    sees (ch10 cap-cost, ch20 mass-soon, ch22-25 resolved sizes) and then delete the actions it
    might disagree with — the pattern writeup_lessons §1 warns about, and Isaiah reported masking
    made his model WORSE. "minimal" is SimJeg's shipped design: two actions per body, no-op or
    all-in. See docs/training.md "THE REINFORCEMENT LEGALITY WALL".
    """
    if gates not in ("full", "minimal"):
        raise ValueError(f"unknown binary commit gates: {gates}")
    pw = np.asarray(pairwise_features, dtype=np.float32)
    S = np.asarray(src_ships, dtype=np.float32)[..., np.newaxis]
    is_own = pw[..., 5] > 0.5
    is_enemy = pw[..., 6] > 0.5
    capture_required = pw[..., 10] * 200.0 + is_enemy.astype(np.float32) * pw[..., 8] * 5.0 * 3.0 + 1.0
    defend = np.rint(pw[..., 24] * 200.0).astype(np.float32)
    if gates == "minimal":
        # Only gate: do you have ships. Own targets lose the maintain sizing too — COMMIT is
        # all-in everywhere, which is what Ender actually does (measured 99.1% all-in reinforce).
        feasible = np.broadcast_to(S >= MIN_BINARY_COMMIT_SHIPS, is_own.shape).copy()
        defend = np.broadcast_to(S, defend.shape).astype(np.float32)
    else:
        attack_ok = (S >= MIN_BINARY_COMMIT_SHIPS) & (S + 1e-3 >= capture_required)
        defend_ok = (defend >= MIN_BINARY_COMMIT_SHIPS) & (S + 1e-3 >= defend)
        feasible = np.where(is_own, defend_ok, attack_ok)
    ships = np.where(is_own, defend, S)
    return np.where(feasible, ships, 0.0).astype(np.float32), feasible
