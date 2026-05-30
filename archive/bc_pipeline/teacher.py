"""Essential heuristic teacher for Orbit Wars BC warmstart.

8 primitives, no strategy. Its only job is to demonstrate env rules so BC can
clone basic competent play and PPO doesn't waste training discovering them
through sparse reward.

Primitives encoded:
  P1  iterate every owned planet as a source each turn
  P2  min-ship floor so fleets travel at usable speed (speed = f(ships))
  P3  sun-safe chord test (skip launches whose path crosses the sun)
  P4  speed-aware ETA using log-shaped speed formula
  P5  orbital target prediction at ETA (aim where it will be, not where it is)
  P6  capture cost = enemy_garrison + production*ETA + percent margin
  P7  defense reserve from forecast inbound enemy fleets
  P8  production-weighted target ranking
  +   in-flight commitment tracking (self + enemy scans feed P6/P7/skip-winning)

Deliberately NOT here (PPO learns):
  - 2-source coordinated swarms
  - Snipe (time arrival to follow enemy attack)
  - Crash exploit (4-player tactics)
  - Same-turn combat top-1 vs top-2 awareness
  - Macro mode flags
  - Comet handling
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants — 10 named. Tunable in BC validation, especially CAPTURE_MARGIN_FRAC.
# ---------------------------------------------------------------------------
CENTER = 50.0
SUN_RADIUS = 10.0
BOARD_SIZE = 100.0
MAX_SHIP_SPEED = 6.0
ROTATION_RADIUS_LIMIT = 50.0       # (orbital_r + radius) < this  → planet orbits

MIN_SHIP_FLOOR = 5                 # speed(5) ≈ 1.56x speed(1); avoids 1-ship suicide fleets
CAPTURE_MARGIN_FRAC = 0.15         # 15% extra over exact need
CAPTURE_MARGIN_MIN = 3
CAPTURE_MARGIN_MAX = 50
DEFENSE_BUFFER = 0                 # legacy: rely on forecast inbound for dynamic reserve
# Option A: static defense reserve. Reserve a minimum fraction of source ships
# regardless of inbound enemies. Forces fractional sends architecturally —
# a 50-ship source can send at most 35, even if the attack would benefit from
# 50. Trade: less raw firepower per attack, but prevents teacher from going
# all-in everywhere (which empirically produced 96% bin=1.0 BC labels).
STATIC_RESERVE_FRAC = 0.30
MIN_COVERAGE_FRAC = 0.30           # skip launches that contribute <30% of capture need
MAX_FLEETS_PER_TURN = 10           # = MAX_OWNED, one launch per source max


# ---------------------------------------------------------------------------
# Physics helpers (match torch_env / fast_env)
# ---------------------------------------------------------------------------

def _fleet_speed(ships: int) -> float:
    if ships <= 0:
        return 1.0
    s = 1.0 + (MAX_SHIP_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000.0)) ** 1.5
    return min(s, MAX_SHIP_SPEED)


def _segment_crosses_sun(x0: float, y0: float, x1: float, y1: float) -> bool:
    dx, dy = x1 - x0, y1 - y0
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= 1e-9:
        return False
    t = ((CENTER - x0) * dx + (CENTER - y0) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    px = x0 + t * dx
    py = y0 + t * dy
    return math.hypot(px - CENTER, py - CENTER) < SUN_RADIUS


# ---------------------------------------------------------------------------
# Orbital prediction (P5)
# ---------------------------------------------------------------------------

def _build_orbit_table(initial_planets) -> Dict[int, Tuple[float, float, bool]]:
    """{pid: (orbital_r, initial_angle, is_orbiting)} derived from initial positions."""
    table: Dict[int, Tuple[float, float, bool]] = {}
    for ip in initial_planets:
        pid = int(ip[0])
        ix, iy, radius = float(ip[2]), float(ip[3]), float(ip[4])
        dx, dy = ix - CENTER, iy - CENTER
        orbital_r = math.hypot(dx, dy)
        initial_angle = math.atan2(dy, dx)
        is_orbiting = (orbital_r + radius) < ROTATION_RADIUS_LIMIT
        table[pid] = (orbital_r, initial_angle, is_orbiting)
    return table


def _position_at(pid: int, orbit_table, t: int, angular_velocity: float,
                 fallback_xy: Tuple[float, float]) -> Tuple[float, float]:
    entry = orbit_table.get(pid)
    if entry is None:
        return fallback_xy
    orbital_r, initial_angle, is_orbiting = entry
    if not is_orbiting:
        return fallback_xy
    ang = initial_angle + angular_velocity * t
    return (CENTER + orbital_r * math.cos(ang),
            CENTER + orbital_r * math.sin(ang))


# ---------------------------------------------------------------------------
# ETA solver (P4 + P5). Orbital targets need fixed-point iteration.
# ---------------------------------------------------------------------------

def _solve_eta(src_xy, target_pid, ships, current_step, orbit_table,
               angular_velocity, target_now_xy, max_iter: int = 4):
    speed = _fleet_speed(ships)
    tx, ty = target_now_xy
    for _ in range(max_iter):
        dist = math.hypot(tx - src_xy[0], ty - src_xy[1])
        eta = max(1, int(math.ceil(dist / speed)))
        nx, ny = _position_at(target_pid, orbit_table, current_step + eta,
                              angular_velocity, target_now_xy)
        if abs(nx - tx) < 0.5 and abs(ny - ty) < 0.5:
            tx, ty = nx, ny
            break
        tx, ty = nx, ny
    dist = math.hypot(tx - src_xy[0], ty - src_xy[1])
    eta = max(1, int(math.ceil(dist / speed)))
    return eta, tx, ty


# ---------------------------------------------------------------------------
# In-flight projection — scans ALL fleets (self + enemy). Used by:
#   - defense reserve (P7) for enemy arrivals at our planets
#   - skip-already-winning for our arrivals at enemy/neutral planets
# ---------------------------------------------------------------------------

def _project_arrivals(fleets, planets) -> Dict[int, List[Tuple[int, int, int]]]:
    """Each fleet projected to its likely-impact planet via ray-circle intersection.

    Returns {planet_id: [(arrival_turn_offset, ships, owner), ...]}.
    Approximate (assumes straight-line, ignores planet motion) — good enough for
    sizing defense reserve. PPO refines this.
    """
    arrivals: Dict[int, List[Tuple[int, int, int]]] = {int(p[0]): [] for p in planets}
    for f in fleets:
        owner = int(f[1])
        fx, fy = float(f[2]), float(f[3])
        angle, ships = float(f[4]), int(f[6])
        if ships <= 0:
            continue
        speed = _fleet_speed(ships)
        vx, vy = math.cos(angle) * speed, math.sin(angle) * speed
        best: Tuple[float, int] | None = None
        for p in planets:
            pid = int(p[0])
            px, py, pr = float(p[2]), float(p[3]), float(p[4])
            v2 = vx * vx + vy * vy
            if v2 <= 1e-9:
                continue
            t = ((px - fx) * vx + (py - fy) * vy) / v2
            if t <= 0:
                continue
            cx, cy = fx + t * vx, fy + t * vy
            d = math.hypot(cx - px, cy - py)
            if d < pr + 1.0:
                if best is None or t < best[0]:
                    best = (t, pid)
        if best is not None:
            t, pid = best
            arrivals.setdefault(pid, []).append((max(1, int(math.ceil(t))), ships, owner))
    return arrivals


# ---------------------------------------------------------------------------
# P6 (capture cost) and P7 (defense reserve)
# ---------------------------------------------------------------------------

def _capture_cost(target_garrison: float, target_production: float,
                  target_owner: int, eta: int) -> int:
    base = target_garrison + (target_production if target_owner != -1 else 0.0) * eta
    margin = max(CAPTURE_MARGIN_MIN,
                 min(CAPTURE_MARGIN_MAX, int(CAPTURE_MARGIN_FRAC * base)))
    return int(math.ceil(base + margin))


def _defense_reserve(inbound_enemy: List[Tuple[int, int, int]], production: float) -> int:
    reserve = 0
    for eta, ships, _owner in inbound_enemy:
        needed = ships - int(production * eta)
        if needed > reserve:
            reserve = needed
    return max(0, reserve + DEFENSE_BUFFER)


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

def agent(obs) -> List[List]:
    me = int(obs["player"])
    planets = obs["planets"]
    fleets = obs["fleets"]
    initial_planets = obs.get("initial_planets") or planets
    angular_velocity = float(obs.get("angular_velocity", 0.0))
    current_step = int(obs["step"])

    orbit_table = _build_orbit_table(initial_planets)
    arrivals = _project_arrivals(fleets, planets)
    by_id = {int(p[0]): p for p in planets}

    # P7: defense reserve per owned planet — uses enemy in-flight scans
    reserves: Dict[int, int] = {}
    owned: List[Tuple[int, list]] = []
    for p in planets:
        pid, owner, prod = int(p[0]), int(p[1]), float(p[6])
        if owner == me:
            owned.append((pid, p))
            inbound_enemy = [a for a in arrivals.get(pid, []) if a[2] != me]
            reserves[pid] = _defense_reserve(inbound_enemy, prod)

    if not owned:
        return []

    # Skip-already-winning: if our own in-flight already captures a target, drop it
    def already_winning(tgt_pid: int) -> bool:
        ours = sorted([a for a in arrivals.get(tgt_pid, []) if a[2] == me])
        if not ours:
            return False
        target = by_id[tgt_pid]
        tgt_ships, tgt_owner, tgt_prod = float(target[5]), int(target[1]), float(target[6])
        my_eta = ours[0][0]
        my_total = sum(a[1] for a in ours if a[0] <= my_eta + 2)
        their_total = sum(
            a[1] for a in arrivals.get(tgt_pid, [])
            if a[2] != me and a[0] <= my_eta + 2
        )
        needed = tgt_ships + (tgt_prod if tgt_owner != -1 else 0.0) * my_eta
        return (my_total - their_total) > needed

    # P1 + P8: build (src, tgt) candidate list with score = prod / (eta + k)
    source_budget: Dict[int, int] = {pid: int(p[5]) for pid, p in owned}
    committed_to: Dict[int, int] = {}   # this-turn commits to each target

    # Option A: static defense reserve. Total reserve is max(dynamic_inbound,
    # STATIC_RESERVE_FRAC * source_ships). Forces fractional sends naturally.
    def _available_for(src_pid: int) -> int:
        ships = source_budget[src_pid]
        dyn_reserve = reserves.get(src_pid, 0)
        static_reserve = int(STATIC_RESERVE_FRAC * ships)
        total_reserve = max(dyn_reserve, static_reserve)
        return max(0, ships - total_reserve)

    candidates: List[Tuple[float, int, int, int, float]] = []

    for src_pid, src_p in owned:
        sx, sy, src_radius = float(src_p[2]), float(src_p[3]), float(src_p[4])
        available = _available_for(src_pid)
        if available < MIN_SHIP_FLOOR:
            continue

        for tgt_p in planets:
            tgt_pid = int(tgt_p[0])
            if tgt_pid == src_pid:
                continue
            tgt_owner = int(tgt_p[1])
            if tgt_owner == me:
                continue  # no reinforce missions in this teacher
            tgt_x, tgt_y = float(tgt_p[2]), float(tgt_p[3])
            tgt_ships, tgt_prod = float(tgt_p[5]), float(tgt_p[6])

            if already_winning(tgt_pid):
                continue

            # initial ship sizing — bound by available, floor by MIN_SHIP_FLOOR
            ships_try = max(MIN_SHIP_FLOOR, min(available, 30))

            # P4 + P5: ETA with orbital prediction
            eta, aim_x, aim_y = _solve_eta(
                (sx, sy), tgt_pid, ships_try, current_step,
                orbit_table, angular_velocity, (tgt_x, tgt_y),
            )

            # P6: refine cost at the actual ETA, less anything we've already committed.
            need = _capture_cost(tgt_ships, tgt_prod, tgt_owner, eta)
            need = max(MIN_SHIP_FLOOR, need - committed_to.get(tgt_pid, 0))
            ships_final = min(available, need)
            if ships_final < MIN_SHIP_FLOOR:
                continue
            if ships_final < MIN_COVERAGE_FRAC * need:
                continue

            # P3: sun-safe — chord from source boundary to predicted target
            angle = math.atan2(aim_y - sy, aim_x - sx)
            start_x = sx + math.cos(angle) * (src_radius + 0.1)
            start_y = sy + math.sin(angle) * (src_radius + 0.1)
            if _segment_crosses_sun(start_x, start_y, aim_x, aim_y):
                continue

            # P8: production-weighted, slight neutral bonus (no in-flight production loss)
            k = 5.0
            neutral_bonus = 1.2 if tgt_owner == -1 else 1.0
            score = (tgt_prod * neutral_bonus) / (eta + k)

            candidates.append((score, src_pid, tgt_pid, ships_final, angle))

    candidates.sort(key=lambda c: -c[0])

    # Commit best-first, one launch per source per turn (matches action-head shape)
    moves: List[List] = []
    launched_from: set = set()
    for score, src_pid, tgt_pid, ships, angle in candidates:
        if len(moves) >= MAX_FLEETS_PER_TURN:
            break
        if src_pid in launched_from:
            continue
        available = _available_for(src_pid)
        ships = min(ships, available)
        if ships < MIN_SHIP_FLOOR:
            continue
        moves.append([src_pid, float(angle), int(ships)])
        source_budget[src_pid] -= ships
        committed_to[tgt_pid] = committed_to.get(tgt_pid, 0) + ships
        launched_from.add(src_pid)

    return moves
