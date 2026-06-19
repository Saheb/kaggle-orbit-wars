"""Shared Phase 5 wave-floor primitives.

This module is intentionally scalar/pure-Python. Evaluation, BC planners, and
unit tests can call it directly; the GPU env mirrors these formulas in vectorized
torch code where Python loops would be too slow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

MAX_SHIP_SPEED = 6.0
SHIP_COUNTS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 19, 22, 26,
    30, 35, 42, 50, 60, 72, 86, 102, 122, 145, 173, 206,
    245, 290, 350, 420,
)

WAVE_TOL_STEPS = 2.0
CAPTURE_OVERHEAD = 1.0
SAFETY_PAD = 1.0
WAVE_MARGIN = CAPTURE_OVERHEAD + SAFETY_PAD

DEFAULT_REACTIVE_BETA = 2.2
DM_ETA_FREE = 3.0
DM_ETA_SCALE = 12.0
DM_HORIZON = 18.0
VALUE_HORIZON = 40
MIN_RESERVE = 2.0
MATERIAL_FRAC = 0.10
MIN_MATERIAL_SHIPS = 5.0
EPS = 1e-6

HOLD_SAFE = "SAFE"
HOLD_HOLDABLE = "HOLDABLE"
HOLD_DOOMED = "DOOMED"


@dataclass(frozen=True)
class ShipOption:
    bin_idx: int
    count: int
    eta: float


@dataclass(frozen=True)
class ShipChoice:
    viable: bool
    ship_target: float
    chosen_count: int = 0
    chosen_bin: int | None = None
    arrival_tau: float | None = None
    eta_fast: float | None = None
    eta_slow: float | None = None


@dataclass(frozen=True)
class WaveAnchor:
    tau: float
    mode: str
    floor: float
    cover: float
    remaining: float
    source_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class QuotaPlan:
    remaining: float
    ready_safe: float
    ready_source_ids: tuple[int, ...]
    quotas: dict[int, float]
    marginal_needed: dict[int, float]
    crosses_if_all_ready_send: bool


@dataclass
class HoldInfo:
    planet_id: int
    hold_class: str
    safe_sendable: float
    base_safe_sendable: float
    remaining0: float = 0.0
    d_def_tau: float | None = None
    reserved_for_defense: float = 0.0
    claims: dict[int, float] = field(default_factory=dict)


def ship_speed(ships: float) -> float:
    """Kaggle Orbit Wars fleet speed for a ship count."""
    s = max(float(ships), 1.0)
    base = (math.log(s) / math.log(1000.0)) ** 1.5
    return min(1.0 + (MAX_SHIP_SPEED - 1.0) * base, MAX_SHIP_SPEED)


def planet_id(planet: list) -> int:
    return int(planet[0])


def planet_owner(planet: list) -> int:
    return int(planet[1])


def planet_ships(planet: list) -> float:
    return float(planet[5])


def distance_planets(a: list, b: list) -> float:
    return math.hypot(float(a[2]) - float(b[2]), float(a[3]) - float(b[3]))


def eta_between_planets(source: list, target: list, ships: float) -> float:
    return distance_planets(source, target) / max(ship_speed(ships), 1e-6)


def legal_ship_options(
    source: list,
    target: list,
    max_sendable: float,
    min_ship_bin: int = 0,
    ship_counts: tuple[int, ...] = SHIP_COUNTS,
) -> list[ShipOption]:
    """Unique emitted counts available under absolute ship-bin decode."""
    cap = int(math.floor(max(float(max_sendable), 0.0)))
    if cap <= 0 or planet_id(source) == planet_id(target):
        return []
    options: list[ShipOption] = []
    seen: set[int] = set()
    for b in range(max(0, int(min_ship_bin)), len(ship_counts)):
        count = min(int(ship_counts[b]), cap)
        if count <= 0 or count in seen:
            continue
        seen.add(count)
        options.append(ShipOption(bin_idx=b, count=count, eta=eta_between_planets(source, target, count)))
    return options


def ship_choice_for_quota(
    source: list,
    target: list,
    safe_sendable: float,
    quota: float,
    tau: float,
    tol: float = WAVE_TOL_STEPS,
    min_ship_bin: int = 0,
) -> ShipChoice:
    """Choose the smallest legal on-time ship count satisfying the quota when possible."""
    options = legal_ship_options(source, target, safe_sendable, min_ship_bin=min_ship_bin)
    if not options:
        return ShipChoice(False, ship_target=0.0)

    eta_fast = min(o.eta for o in options)
    eta_slow = max(o.eta for o in options)
    # ONE-SIDED arrival window (Phase 5 audit 2026-06-19): a source is on-time if it can arrive
    # BY the deadline (eta <= tau+tol); early arrival is allowed. The original two-sided window
    # [tau-tol, tau+tol] forced near sources to send slow/small fleets to land exactly in the
    # window, which starved the wave (reactive-cross 0.14 in §9). The loss-mode audit showed
    # staggered-arrival is ~1% of losses, so synchronization is dropped in favour of sufficiency
    # (crossing the reactive floor). See docs/phase5-blocked.md.
    hi = float(tau) + float(tol)
    on_time = [o for o in options if o.eta <= hi + EPS]
    ship_target = min(max(float(quota), 0.0), max(float(safe_sendable), 0.0))
    if not on_time:
        return ShipChoice(False, ship_target=ship_target, eta_fast=eta_fast, eta_slow=eta_slow)

    at_or_above = [o for o in on_time if o.count + EPS >= ship_target]
    chosen = min(at_or_above, key=lambda o: (o.count, o.bin_idx)) if at_or_above else max(
        on_time, key=lambda o: (o.count, -o.bin_idx)
    )
    return ShipChoice(
        True,
        ship_target=ship_target,
        chosen_count=chosen.count,
        chosen_bin=chosen.bin_idx,
        arrival_tau=chosen.eta,
        eta_fast=eta_fast,
        eta_slow=eta_slow,
    )


def ready_now(
    source: list,
    target: list,
    safe_sendable: float,
    tau: float,
    tol: float = WAVE_TOL_STEPS,
    min_ship_bin: int = 0,
) -> bool:
    return ship_choice_for_quota(
        source, target, safe_sendable, 0.0, tau, tol=tol, min_ship_bin=min_ship_bin
    ).viable


def reactive_rho(tau: float) -> float:
    """Reactive-defense ramp for a relative deadline."""
    return min(max((float(tau) - DM_ETA_FREE) / DM_ETA_SCALE, 0.0), 1.0)


def fleet_target_planet(planets: list, fleet: list, radius_pad: float = 2.0):
    """Planet a fleet is geometrically converging on, or None if no target resolves."""
    fx, fy, ang = float(fleet[2]), float(fleet[3]), float(fleet[4])
    c, sn = math.cos(ang), math.sin(ang)
    best, best_dist = None, None
    for p in planets:
        vx, vy = float(p[2]) - fx, float(p[3]) - fy
        along = vx * c + vy * sn
        if along <= 0:
            continue
        perp = abs(vx * sn - vy * c)
        if perp >= float(p[4]) + radius_pad:
            continue
        dist = math.hypot(vx, vy)
        if best_dist is None or dist < best_dist:
            best, best_dist = p, dist
    return best


def fleet_eta_to_target(fleet: list, target: list) -> float:
    """Remaining ETA from an in-flight fleet to its resolved target."""
    dist = math.hypot(float(target[2]) - float(fleet[2]), float(target[3]) - float(fleet[3]))
    return dist / max(ship_speed(float(fleet[6])), 1e-6)


def fleets_targeting(planets: list, fleets: list, target: list, owner: int | None = None) -> list[tuple[list, float]]:
    """Resolved fleets aimed at target, paired with remaining ETA."""
    target_id = planet_id(target)
    out: list[tuple[list, float]] = []
    for f in fleets or []:
        if owner is not None and int(f[1]) != int(owner):
            continue
        if int(f[1]) < 0:
            continue
        ft = fleet_target_planet(planets, f)
        if ft is None or planet_id(ft) != target_id:
            continue
        out.append((f, fleet_eta_to_target(f, ft)))
    return out


def reactive_enemy_mass_to_target(planets: list, target: list, attacker: int, tau: float) -> float:
    """Enemy planet garrison that can reach target by the relative deadline tau."""
    mass = 0.0
    target_id = int(target[0])
    for src in planets:
        owner = int(src[1])
        if owner < 0 or owner == attacker or int(src[0]) == target_id:
            continue
        dist = math.hypot(float(src[2]) - float(target[2]), float(src[3]) - float(target[3]))
        eta = dist / max(ship_speed(float(src[5])), 1e-6)
        if eta <= float(tau):
            mass += float(src[5])
    return mass


def enemy_inbound_to_target_by_deadline(
    planets: list,
    fleets: list,
    target: list,
    attacker: int,
    tau: float,
    tol: float = WAVE_TOL_STEPS,
) -> float:
    """Enemy fleet mass aimed at target and arriving no later than tau+tol."""
    deadline = float(tau) + float(tol)
    mass = 0.0
    for f, eta in fleets_targeting(planets, fleets, target):
        f_owner = int(f[1])
        if f_owner < 0 or f_owner == attacker:
            continue
        if eta <= deadline:
            mass += float(f[6])
    return mass


def recapture_risk_to_target(
    planets: list,
    fleets: list,
    target: list,
    attacker: int,
    tau: float,
    tol: float = WAVE_TOL_STEPS,
) -> float:
    """Enemy fleet mass aimed at target but arriving after the capture window."""
    deadline = float(tau) + float(tol)
    mass = 0.0
    for f, eta in fleets_targeting(planets, fleets, target):
        owner = int(f[1])
        if owner < 0 or owner == attacker:
            continue
        if eta > deadline:
            mass += float(f[6])
    return mass


def capture_floor(
    target: list,
    planets: list,
    fleets: list,
    attacker: int,
    tau: float,
    beta: float = DEFAULT_REACTIVE_BETA,
    tol: float = WAVE_TOL_STEPS,
    margin: float = WAVE_MARGIN,
) -> float:
    """Phase 5 capture floor at a relative arrival deadline tau."""
    owner = int(target[1])
    if owner == attacker:
        return 0.0

    tau_f = max(float(tau), 0.0)
    floor = float(target[5]) + float(target[6]) * tau_f + float(margin)
    if owner >= 0:
        inbound = enemy_inbound_to_target_by_deadline(planets, fleets, target, attacker, tau_f, tol)
        reactive = reactive_enemy_mass_to_target(planets, target, attacker, tau_f)
        floor += inbound + float(beta) * reactive_rho(tau_f) * reactive
    return max(floor, 1e-6)


def friendly_wave_cover(
    planets: list,
    fleets: list,
    target: list,
    player: int,
    tau: float,
    tol: float = WAVE_TOL_STEPS,
) -> float:
    """Friendly mass already inbound to target inside the wave arrival window."""
    lo, hi = float(tau) - float(tol), float(tau) + float(tol)
    mass = 0.0
    for f, eta in fleets_targeting(planets, fleets, target, owner=player):
        if lo - EPS <= eta <= hi + EPS:
            mass += float(f[6])
    return mass


def friendly_inbound_to_target_by_deadline(
    planets: list,
    fleets: list,
    target: list,
    player: int,
    tau: float,
) -> float:
    mass = 0.0
    for f, eta in fleets_targeting(planets, fleets, target, owner=player):
        if eta <= float(tau) + EPS:
            mass += float(f[6])
    return mass


def attack_remaining(
    target: list,
    planets: list,
    fleets: list,
    attacker: int,
    tau: float,
    beta: float = DEFAULT_REACTIVE_BETA,
    tol: float = WAVE_TOL_STEPS,
) -> tuple[float, float, float]:
    floor = capture_floor(target, planets, fleets, attacker, tau, beta=beta, tol=tol)
    cover = friendly_wave_cover(planets, fleets, target, attacker, tau, tol=tol)
    return floor, cover, max(0.0, floor - cover)


def defense_floor(
    planet: list,
    planets: list,
    fleets: list,
    player: int,
    tau: float,
    tol: float = WAVE_TOL_STEPS,
    margin: float = WAVE_MARGIN,
) -> float:
    enemy_mass = 0.0
    for f, eta in fleets_targeting(planets, fleets, planet):
        owner = int(f[1])
        if owner >= 0 and owner != player and eta <= float(tau) + float(tol) + EPS:
            enemy_mass += float(f[6])
    return max(enemy_mass + float(margin), 0.0)


def defense_cover(planet: list, planets: list, fleets: list, player: int, tau: float) -> float:
    return (
        float(planet[5])
        + float(planet[6]) * max(float(tau), 0.0)
        + friendly_inbound_to_target_by_deadline(planets, fleets, planet, player, tau)
    )


def defense_remaining(
    planet: list,
    planets: list,
    fleets: list,
    player: int,
    tau: float,
    tol: float = WAVE_TOL_STEPS,
) -> tuple[float, float, float]:
    floor = defense_floor(planet, planets, fleets, player, tau, tol=tol)
    cover = defense_cover(planet, planets, fleets, player, tau)
    return floor, cover, max(0.0, floor - cover)


def choose_defense_anchor(
    planet: list,
    planets: list,
    fleets: list,
    player: int,
    tol: float = WAVE_TOL_STEPS,
) -> WaveAnchor | None:
    """Earliest enemy-arrival deadline where cumulative floor-cover is positive."""
    if planet_owner(planet) != player:
        return None
    etas = sorted(
        eta for f, eta in fleets_targeting(planets, fleets, planet)
        if int(f[1]) >= 0 and int(f[1]) != player
    )
    for tau in etas:
        floor, cover, remaining = defense_remaining(planet, planets, fleets, player, tau, tol=tol)
        if remaining > EPS:
            return WaveAnchor(tau=tau, mode="defense", floor=floor, cover=cover, remaining=remaining)
    return None


def choose_attack_anchor(
    target: list,
    planets: list,
    fleets: list,
    attacker: int,
    safe_sendable_by_pid: dict[int, float],
    beta: float = DEFAULT_REACTIVE_BETA,
    tol: float = WAVE_TOL_STEPS,
    material_frac: float = MATERIAL_FRAC,
    min_material_ships: float = MIN_MATERIAL_SHIPS,
) -> WaveAnchor | None:
    """Deterministic v1 attack anchor: sticky material inbound first, else tightest feasible."""
    if planet_owner(target) == attacker:
        return None

    arrivals = sorted(
        (eta, float(f[6])) for f, eta in fleets_targeting(planets, fleets, target, owner=attacker)
    )
    for eta, _ships in arrivals:
        group = [(e, s) for e, s in arrivals if eta - tol <= e <= eta + tol]
        if not group:
            continue
        tau = max(e for e, _s in group)
        floor, cover, remaining = attack_remaining(target, planets, fleets, attacker, tau, beta=beta, tol=tol)
        threshold = max(float(min_material_ships), float(material_frac) * floor)
        mass = sum(s for _e, s in group)
        if mass + EPS >= threshold:
            return WaveAnchor(tau=tau, mode="sticky", floor=floor, cover=cover, remaining=remaining)

    candidates: list[tuple[float, int, float]] = []
    source_by_pid = {planet_id(p): p for p in planets}
    for pid, safe in safe_sendable_by_pid.items():
        source = source_by_pid.get(int(pid))
        if source is None or planet_owner(source) != attacker or planet_id(source) == planet_id(target):
            continue
        safe_f = float(safe)
        if safe_f <= EPS:
            continue
        candidates.append((eta_between_planets(source, target, safe_f), int(pid), safe_f))
    candidates.sort(key=lambda x: (x[0], x[1]))

    prefix_safe = 0.0
    prefix_ids: list[int] = []
    for tau, pid, safe in candidates:
        prefix_safe += safe
        prefix_ids.append(pid)
        floor, cover, remaining = attack_remaining(target, planets, fleets, attacker, tau, beta=beta, tol=tol)
        if prefix_safe + EPS >= remaining:
            return WaveAnchor(
                tau=tau,
                mode="fresh",
                floor=floor,
                cover=cover,
                remaining=remaining,
                source_ids=tuple(prefix_ids),
            )
    return None


def wave_feasible_by_tau(
    target: list,
    planets: list,
    fleets: list,
    attacker: int,
    safe_sendable_by_pid: dict[int, float],
    tau: float,
    beta: float = DEFAULT_REACTIVE_BETA,
    tol: float = WAVE_TOL_STEPS,
) -> bool:
    floor, cover, remaining = attack_remaining(target, planets, fleets, attacker, tau, beta=beta, tol=tol)
    source_by_pid = {planet_id(p): p for p in planets}
    arrivable = 0.0
    for pid, safe in safe_sendable_by_pid.items():
        source = source_by_pid.get(int(pid))
        safe_f = float(safe)
        if source is None or safe_f <= EPS or planet_id(source) == planet_id(target):
            continue
        if eta_between_planets(source, target, safe_f) <= float(tau) + float(tol) + EPS:
            arrivable += safe_f
    return cover + arrivable + EPS >= floor and remaining <= arrivable + EPS


def ready_wave_quota(
    sources: list,
    target: list,
    safe_sendable_by_pid: dict[int, float],
    remaining: float,
    tau: float,
    tol: float = WAVE_TOL_STEPS,
    min_ship_bin: int = 0,
) -> QuotaPlan:
    ready_ids: list[int] = []
    ready_safe = 0.0
    for src in sources:
        pid = planet_id(src)
        safe = float(safe_sendable_by_pid.get(pid, 0.0))
        if safe <= EPS:
            continue
        if ready_now(src, target, safe, tau, tol=tol, min_ship_bin=min_ship_bin):
            ready_ids.append(pid)
            ready_safe += safe
    rem = max(float(remaining), 0.0)
    quotas: dict[int, float] = {}
    marginal: dict[int, float] = {}
    for src in sources:
        pid = planet_id(src)
        safe = float(safe_sendable_by_pid.get(pid, 0.0))
        if pid in ready_ids and ready_safe > EPS:
            quotas[pid] = rem * safe / ready_safe
            marginal[pid] = min(safe, rem)
        else:
            quotas[pid] = 0.0
            marginal[pid] = 0.0
    return QuotaPlan(
        remaining=rem,
        ready_safe=ready_safe,
        ready_source_ids=tuple(ready_ids),
        quotas=quotas,
        marginal_needed=marginal,
        crosses_if_all_ready_send=ready_safe + EPS >= rem,
    )


def hold_value(planet: list, current_step: int, episode_steps: int, value_horizon: int = VALUE_HORIZON) -> float:
    remaining_steps = max(0, int(episode_steps) - int(current_step))
    return float(planet[6]) * min(remaining_steps, int(value_horizon))


def classify_holds(
    planets: list,
    fleets: list,
    player: int,
    current_step: int = 0,
    episode_steps: int = 500,
    tol: float = WAVE_TOL_STEPS,
    min_reserve: float = MIN_RESERVE,
    value_horizon: int = VALUE_HORIZON,
) -> dict[int, HoldInfo]:
    """Single-pass deterministic hold_class/safe_sendable classification."""
    own = [p for p in planets if planet_owner(p) == player]
    by_pid = {planet_id(p): p for p in own}
    infos: dict[int, HoldInfo] = {}
    candidates: list[int] = []

    for p in own:
        pid = planet_id(p)
        anchor = choose_defense_anchor(p, planets, fleets, player, tol=tol)
        if anchor is None or anchor.remaining <= EPS:
            base = max(0.0, planet_ships(p) - float(min_reserve))
            infos[pid] = HoldInfo(pid, HOLD_SAFE, safe_sendable=base, base_safe_sendable=base)
            continue
        infos[pid] = HoldInfo(
            pid,
            "CANDIDATE",
            safe_sendable=0.0,
            base_safe_sendable=0.0,
            remaining0=anchor.remaining,
            d_def_tau=anchor.tau,
        )
        candidates.append(pid)

    pool: dict[int, float] = {
        pid: info.base_safe_sendable
        for pid, info in infos.items()
        if info.hold_class == HOLD_SAFE and info.base_safe_sendable > EPS
    }
    still_candidate: list[int] = []
    for pid in candidates:
        p = by_pid[pid]
        info = infos[pid]
        tau = float(info.d_def_tau or 0.0)
        # Review fix B: optimistic help is from SAFE planets ONLY — other CANDIDATEs may reserve
        # their garrison for their own defense, so counting them over-states available help and
        # can mis-classify a DOOMED planet as savable. Doomed planets that fall out below add their
        # full garrison to `pool`, which pass 3 then claims from (no over-count, still no fixed-point).
        optimistic = 0.0
        for q in own:
            qid = planet_id(q)
            if qid == pid or infos[qid].hold_class != HOLD_SAFE:
                continue
            cap = max(0.0, planet_ships(q) - float(min_reserve))
            if cap > EPS and eta_between_planets(q, p, cap) <= tau + tol + EPS:
                optimistic += cap
        if info.remaining0 > optimistic + EPS or hold_value(p, current_step, episode_steps, value_horizon) < info.remaining0:
            info.hold_class = HOLD_DOOMED
            info.base_safe_sendable = planet_ships(p)
            pool[pid] = pool.get(pid, 0.0) + info.base_safe_sendable
        else:
            still_candidate.append(pid)

    still_candidate.sort(key=lambda pid: (infos[pid].d_def_tau if infos[pid].d_def_tau is not None else float("inf"), pid))
    for pid in still_candidate:
        p = by_pid[pid]
        info = infos[pid]
        tau = float(info.d_def_tau or 0.0)
        need = info.remaining0
        eligible: list[tuple[float, int, float]] = []
        for qid, cap in pool.items():
            if cap <= EPS or qid == pid:
                continue
            q = by_pid.get(qid)
            if q is None:
                continue
            eta = eta_between_planets(q, p, cap)
            if eta <= tau + tol + EPS:
                eligible.append((eta, qid, cap))
        eligible.sort(key=lambda x: (x[0], x[1]))

        claims: dict[int, float] = {}
        for _eta, qid, cap in eligible:
            if need <= EPS:
                break
            take = min(pool[qid], need)
            if take <= EPS:
                continue
            pool[qid] -= take
            claims[qid] = claims.get(qid, 0.0) + take
            need -= take

        if need <= EPS:
            info.hold_class = HOLD_HOLDABLE
            info.safe_sendable = 0.0
            info.claims = claims
            for qid, take in claims.items():
                if qid in infos:
                    infos[qid].reserved_for_defense += take
        else:
            for qid, take in claims.items():
                pool[qid] = pool.get(qid, 0.0) + take
            info.hold_class = HOLD_DOOMED
            info.base_safe_sendable = planet_ships(p)
            pool[pid] = pool.get(pid, 0.0) + info.base_safe_sendable

    for pid, info in infos.items():
        if info.hold_class == HOLD_HOLDABLE:
            info.safe_sendable = 0.0
        elif info.hold_class in (HOLD_SAFE, HOLD_DOOMED):
            info.safe_sendable = max(0.0, pool.get(pid, info.base_safe_sendable))
    return infos
