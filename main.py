"""
Orbit Wars - Learning Baseline Agent

This is a transparent model-based baseline, not a tuned leaderboard bot.

Core ideas:
  - Planet EV: production is future income; ships are capture/defense cost.
  - Contest risk: avoid neutral targets opponents can cheaply contest.
  - Selective defense: reinforce valuable owned planets, not every planet.
  - Punish overcommitment: low-garrison enemy planets are valid targets.

Set ORBIT_WARS_DEBUG=1 locally to print compact per-turn decision logs.
"""

import math
import os
import sys
from collections import defaultdict

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet


BOARD_SIZE = 100.0
CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0
MAX_SPEED = 6.0
EPISODE_STEPS = 500
DEBUG = os.environ.get("ORBIT_WARS_DEBUG") == "1"
MAX_MOVES = 8
ENABLE_WAIT_GATE = os.environ.get("ORBIT_WARS_ENABLE_WAIT_GATE", "0") != "0"
ENABLE_UNIFIED_SELECTION = os.environ.get("ORBIT_WARS_ENABLE_UNIFIED_SELECTION", "0") != "0"
ENABLE_LATENT_THREAT_RESERVE = os.environ.get("ORBIT_WARS_ENABLE_LATENT_THREAT_RESERVE", "0") != "0"
ENABLE_OPENING_RESERVE_V6 = os.environ.get("ORBIT_WARS_ENABLE_OPENING_RESERVE_V6", "0") == "1"
ENABLE_ORBITING_DEFENSE_V6 = os.environ.get("ORBIT_WARS_ENABLE_ORBITING_DEFENSE_V6", "0") != "0"
ENABLE_SPEED_OVERSEND = os.environ.get("ORBIT_WARS_ENABLE_SPEED_OVERSEND", "0") == "1"
ENABLE_RACE_OVERSEND = os.environ.get("ORBIT_WARS_ENABLE_RACE_OVERSEND", "0") == "1"
ENABLE_TIMELINE_NEED = os.environ.get("ORBIT_WARS_ENABLE_TIMELINE_NEED", "1") == "1"
ENABLE_PHASE_PRESSURE = os.environ.get("ORBIT_WARS_ENABLE_PHASE_PRESSURE", "1") == "1"
ENABLE_OPENING_IN_FLIGHT_LOCK = os.environ.get("ORBIT_WARS_ENABLE_OPENING_IN_FLIGHT_LOCK", "0") == "1"
TIMELINE_TOPUP_GUARD_UNTIL = int(os.environ.get("ORBIT_WARS_TIMELINE_TOPUP_GUARD_UNTIL", "40"))
PROD1_NEUTRAL_BLOCK_UNTIL = int(os.environ.get("ORBIT_WARS_PROD1_NEUTRAL_BLOCK_UNTIL", "0"))
PROD1_NEUTRAL_MAX_BLOCK_PROD = float(os.environ.get("ORBIT_WARS_PROD1_NEUTRAL_MAX_BLOCK_PROD", "0"))
PROD1_NEUTRAL_MIN_PROD_GAIN = float(os.environ.get("ORBIT_WARS_PROD1_NEUTRAL_MIN_PROD_GAIN", "0"))
WAIT_GATE_START = int(os.environ.get("ORBIT_WARS_WAIT_GATE_START", "90"))
WAIT_GATE_MIN_PRODUCTION = int(os.environ.get("ORBIT_WARS_WAIT_GATE_MIN_PRODUCTION", "2"))
WAIT_GATE_MARGIN = float(os.environ.get("ORBIT_WARS_WAIT_GATE_MARGIN", "8.0"))
WAIT_GATE_MAX_WAIT = int(os.environ.get("ORBIT_WARS_WAIT_GATE_MAX_WAIT", "15"))
ATTACK_VALUE_HORIZON = float(os.environ.get("ORBIT_WARS_ATTACK_VALUE_HORIZON", "160.0"))
ATTACK_TRAVEL_PENALTY = float(os.environ.get("ORBIT_WARS_ATTACK_TRAVEL_PENALTY", "0.55"))
OPENING_ATTACK_TRAVEL_LIMIT = float(os.environ.get("ORBIT_WARS_OPENING_ATTACK_TRAVEL_LIMIT", "999.0"))
MIDGAME_ATTACK_TRAVEL_LIMIT = float(os.environ.get("ORBIT_WARS_MIDGAME_ATTACK_TRAVEL_LIMIT", "999.0"))

# A1: Weakest-enemy targeting — focus fire on whoever is most vulnerable
WEAKEST_ENEMY_VALUE_MULT_2P = 1.25
WEAKEST_ENEMY_VALUE_MULT_4P = 1.50

# A2: Elimination bonus — push through kills when we have the advantage
ELIMINATION_BONUS = 55.0
ELIMINATION_WEAK_THRESHOLD = 0.90  # trigger when enemy total < 90% of ours

# A3: Gang-up missions — exploit inter-enemy battles
GANG_UP_VALUE_MULT = 1.4
GANG_UP_POST_BATTLE_DELAY = 2    # turns after battle to arrive
GANG_UP_ETA_WINDOW = 4           # ±turn tolerance on arrival timing


def get_field(obs, name, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def angle_to(a, b):
    return math.atan2(b.y - a.y, b.x - a.x)


def fleet_speed(ships):
    ships = max(1, int(ships))
    scale = (math.log(ships) / math.log(1000.0)) ** 1.5
    return 1.0 + (MAX_SPEED - 1.0) * scale


def travel_time(src, dst, ships):
    return max(1.0, distance(src, dst) / fleet_speed(ships))


def point_segment_distance(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = min(1.0, max(0.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def crosses_sun(src, dst):
    return point_segment_distance(
        CENTER[0], CENTER[1], src.x, src.y, dst.x, dst.y
    ) <= SUN_RADIUS + 0.25


def safe_to_launch(src, dst):
    if crosses_sun(src, dst):
        return False
    return 0.0 <= dst.x <= BOARD_SIZE and 0.0 <= dst.y <= BOARD_SIZE


def defense_reserve(planet, step):
    """Keep enough ships that ordinary opportunistic attacks are unattractive."""
    phase = step / EPISODE_STEPS
    if step < 90 and planet.production <= 1:
        # A production-1 start is a trap if we keep normal reserves. Spend down
        # enough to buy nearby income; otherwise stronger expanders snowball.
        if planet.ships <= 12:
            return 0
        return min(3, max(1, int(planet.ships * 0.15)))
    if step < 90 and ENABLE_OPENING_RESERVE_V6:
        # Opening tempo dominates. Leader replays redeploy ships aggressively
        # from both home and first captures; static garrisons delay compounding.
        return min(8 + planet.production, max(4, int(planet.ships * 0.35)))
    base = 8 + 2 * planet.production
    return int(min(max(base, 8), max(8, planet.ships * 0.65)))


def compute_enemy_stats(planets, fleets, player):
    """Return (my_total_ships, enemy_strength_by_id, weakest_enemy_id).

    enemy_strength_by_id maps owner_id -> total ships (planets + fleets).
    weakest_enemy_id is the owner with the fewest total ships, or None.
    """
    my_total = sum(p.ships for p in planets if p.owner == player)
    enemy_strength = {}
    for p in planets:
        if p.owner not in (-1, player):
            enemy_strength[p.owner] = enemy_strength.get(p.owner, 0) + p.ships
    for f in fleets:
        if f.owner not in (-1, player):
            enemy_strength[f.owner] = enemy_strength.get(f.owner, 0) + f.ships
    weakest = min(enemy_strength, key=enemy_strength.get) if enemy_strength else None
    return my_total, enemy_strength, weakest


def usable_ships(planet, committed, step):
    reserve = defense_reserve(planet, step)
    return max(0, int(planet.ships - committed[planet.id] - reserve))


def predict_planet_position(target, initial_by_id, angular_velocity, turns):
    """
    Approximate future position for orbiting non-comet planets.

    The current observation is enough for direct targeting. This helper gives
    moving inner planets a simple lead target when travel time is non-trivial.
    """
    rx = target.x - CENTER[0]
    ry = target.y - CENTER[1]
    orbital_radius = math.hypot(rx, ry)
    if orbital_radius + target.radius >= 50.0:
        return target

    theta = math.atan2(ry, rx) + angular_velocity * max(0.0, turns)
    x = CENTER[0] + orbital_radius * math.cos(theta)
    y = CENTER[1] + orbital_radius * math.sin(theta)
    return Planet(target.id, target.owner, x, y, target.radius, target.ships, target.production)


def is_orbiting_planet(target):
    rx = target.x - CENTER[0]
    ry = target.y - CENTER[1]
    return math.hypot(rx, ry) + target.radius < 50.0


def fleet_pressure_to_planet(planet, fleets, owner=None, excluded_owner=None, extra_radius=0.75):
    pressure = 0
    earliest = None
    for f in fleets:
        if owner is not None and f.owner != owner:
            continue
        if excluded_owner is not None and f.owner == excluded_owner:
            continue
        ux = math.cos(f.angle)
        uy = math.sin(f.angle)
        vx = planet.x - f.x
        vy = planet.y - f.y
        along = vx * ux + vy * uy
        if along <= 0:
            continue
        perp = abs(vx * uy - vy * ux)
        if perp > planet.radius + extra_radius:
            continue
        turns = along / fleet_speed(f.ships)
        pressure += f.ships
        earliest = turns if earliest is None else min(earliest, turns)
    return pressure, earliest


def fleet_eta_to_planet(fleet, planet, initial_by_id, angular_velocity, extra_radius=1.5):
    ux = math.cos(fleet.angle)
    uy = math.sin(fleet.angle)
    vx = planet.x - fleet.x
    vy = planet.y - fleet.y
    along = vx * ux + vy * uy
    if along <= 0:
        return None
    speed = fleet_speed(fleet.ships)
    eta = along / speed

    for candidate in (
        planet,
        predict_planet_position(planet, initial_by_id, angular_velocity, eta),
    ):
        vx = candidate.x - fleet.x
        vy = candidate.y - fleet.y
        along = vx * ux + vy * uy
        if along <= 0:
            continue
        perp = abs(vx * uy - vy * ux)
        if perp <= candidate.radius + extra_radius:
            return max(1, int(math.ceil(along / speed)))
    if is_orbiting_planet(planet):
        for turn in range(1, 81):
            candidate = predict_planet_position(planet, initial_by_id, angular_velocity, turn)
            fx = fleet.x + ux * speed * turn
            fy = fleet.y + uy * speed * turn
            if math.hypot(candidate.x - fx, candidate.y - fy) <= candidate.radius + extra_radius:
                return turn
    return None


def arrivals_to_planet(planet, fleets, initial_by_id, angular_velocity, horizon):
    arrivals = []
    for fleet in fleets:
        eta = fleet_eta_to_planet(fleet, planet, initial_by_id, angular_velocity)
        if eta is not None and eta <= horizon:
            arrivals.append((eta, fleet.owner, int(fleet.ships)))
    return arrivals


def resolve_arrivals(owner, garrison, arrivals):
    forces = defaultdict(int)
    for _, arrival_owner, ships in arrivals:
        if arrival_owner != -1 and ships > 0:
            forces[arrival_owner] += int(ships)
    if not forces:
        return owner, max(0.0, garrison)
    ranked = sorted(forces.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return owner, max(0.0, garrison)
    survivor_owner, ships = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    survivor_ships = ships - runner_up
    if survivor_ships <= 0:
        return owner, max(0.0, garrison)
    if owner == survivor_owner:
        return owner, garrison + survivor_ships
    garrison -= survivor_ships
    if garrison < 0:
        return survivor_owner, -garrison
    return owner, garrison


def owns_with_extra_arrival(planet, arrivals, horizon, player, extra_ships):
    by_turn = defaultdict(list)
    for eta, owner, ships in arrivals:
        if ships > 0 and eta <= horizon:
            by_turn[max(1, int(math.ceil(eta)))].append((eta, owner, int(ships)))
    if extra_ships > 0:
        by_turn[horizon].append((horizon, player, int(extra_ships)))

    owner = planet.owner
    garrison = float(planet.ships)
    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production
        if by_turn.get(turn):
            owner, garrison = resolve_arrivals(owner, garrison, by_turn[turn])
    return owner == player


def timeline_ships_needed_to_own(planet, fleets, player, arrival_turn, initial_by_id, angular_velocity, upper_bound):
    horizon = max(1, int(math.ceil(arrival_turn)))
    arrivals = arrivals_to_planet(planet, fleets, initial_by_id, angular_velocity, horizon)
    if owns_with_extra_arrival(planet, arrivals, horizon, player, 0):
        return 0
    hi = max(1, int(upper_bound))
    if not owns_with_extra_arrival(planet, arrivals, horizon, player, hi):
        return hi + 1
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if owns_with_extra_arrival(planet, arrivals, horizon, player, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def incoming_enemy_pressure(planet, fleets, player):
    """
    Estimate enemy ships already committed to hit a planet.

    This is geometric and conservative: if a fleet's ray passes close to the
    planet and the planet is in front of it, count it as incoming.
    """
    return fleet_pressure_to_planet(planet, fleets, excluded_owner=player)


def incoming_enemy_pressure_orbiting(planet, fleets, player, initial_by_id, angular_velocity):
    """
    Estimate enemy pressure on a possibly moving planet.

    Fleets aimed at orbiting planets often do not pass close to the planet's
    current coordinates. Check the current position, then refine against the
    planet position at the fleet's approximate arrival time.
    """
    refined_pressure = 0
    refined_earliest = None

    for f in fleets:
        if f.owner == player:
            continue
        current_pressure, current_earliest = fleet_pressure_to_planet(
            planet, [f], excluded_owner=player, extra_radius=1.0
        )
        if current_pressure > 0:
            refined_pressure += current_pressure
            refined_earliest = (
                current_earliest
                if refined_earliest is None
                else min(refined_earliest, current_earliest or refined_earliest)
            )
            continue

        ux = math.cos(f.angle)
        uy = math.sin(f.angle)
        vx = planet.x - f.x
        vy = planet.y - f.y
        along = vx * ux + vy * uy
        if along <= 0:
            continue
        eta = along / fleet_speed(f.ships)
        lead = predict_planet_position(planet, initial_by_id, angular_velocity, eta)
        lead_pressure, lead_earliest = fleet_pressure_to_planet(
            lead, [f], excluded_owner=player, extra_radius=2.5
        )
        if lead_pressure <= 0:
            continue
        refined_pressure += lead_pressure
        refined_earliest = (
            lead_earliest
            if refined_earliest is None
            else min(refined_earliest, lead_earliest or refined_earliest)
        )

    return refined_pressure, refined_earliest


def opponent_contest_risk(target, planets, player, my_eta):
    """
    Capability model: which opponents can plausibly contest this target soon?
    """
    risk = 0.0
    strongest = 0
    for p in planets:
        if p.owner in (-1, player):
            continue
        reserve = defense_reserve(p, 0)
        available = max(0, int(p.ships - reserve))
        if available <= 0 or crosses_sun(p, target):
            continue
        eta = travel_time(p, target, max(1, available))
        if eta <= my_eta + 4:
            strongest = max(strongest, available)
            # High production targets attract everybody; close enemies matter more.
            timing = max(0.0, 1.0 - max(0.0, eta - my_eta) / 8.0)
            risk += timing * (available / max(1.0, target.ships + 1)) * (1.0 + 0.15 * target.production)
    return risk, strongest


def fastest_opponent_contester(target, planets, player):
    best_eta = None
    best_available = 0
    for p in planets:
        if p.owner in (-1, player):
            continue
        reserve = defense_reserve(p, 0)
        available = max(0, int(p.ships - reserve))
        if available <= target.ships or crosses_sun(p, target):
            continue
        eta = travel_time(p, target, available)
        if best_eta is None or eta < best_eta:
            best_eta = eta
            best_available = available
    return best_eta, best_available


def capture_ships_needed(src, target):
    """Ships needed to own target on arrival under the current coarse model."""
    ships = int(target.ships) + 1
    if target.owner == -1:
        return ships

    # Owned planets produce while we travel. Iterate because fleet size changes
    # speed, and therefore changes how many defensive ships appear before impact.
    for _ in range(3):
        eta = travel_time(src, target, ships)
        ships = int(target.ships + target.production * math.ceil(eta) + 2)
    return ships


def capture_ships_needed_after_wait(src, target, lead, wait_turns):
    """Ships needed if we wait before launching toward a predicted lead point."""
    ships = int(target.ships) + 1
    if target.owner == -1:
        return ships

    for _ in range(3):
        eta = travel_time(src, lead, ships)
        ships = int(target.ships + target.production * math.ceil(wait_turns + eta) + 2)
    return ships


def opening_anchor_target(src, planets, player, comet_ids):
    """
    Find a nearby high-production neutral worth saving for in the opening.

    Without this, production-1 homes spend scarce ships on cheap low-income
    planets and miss the compounding jump from a reachable production-4/5 planet.
    """
    best = None
    max_anchor_ships = 55 if src.production == 2 else 75
    for target in planets:
        if target.owner != -1 or target.id in comet_ids or target.production < 4:
            continue
        ships = int(target.ships) + 1
        if ships > max_anchor_ships or not safe_to_launch(src, target):
            continue
        eta = travel_time(src, target, ships)
        if eta > 24:
            continue
        score = 75 * target.production - ships - 1.5 * eta
        if best is None or score > best["score"]:
            best = {"id": target.id, "ships": ships, "score": score}
    return best


def strategic_phase(my_planets, planets, fleets, player, step):
    enemies = [p for p in planets if p.owner not in (-1, player)]
    neutrals = [p for p in planets if p.owner == -1]
    my_prod = sum(p.production for p in my_planets)
    enemy_prod = sum(p.production for p in enemies)
    my_ships = sum(p.ships for p in my_planets) + sum(f.ships for f in fleets if f.owner == player)
    enemy_ships = sum(p.ships for p in enemies) + sum(f.ships for f in fleets if f.owner not in (-1, player))
    prod_ratio = my_prod / enemy_prod if enemy_prod > 0 else 999.0
    ship_ratio = my_ships / enemy_ships if enemy_ships > 0 else 999.0
    nearby_neutrals = sum(
        1
        for target in neutrals
        if any(distance(src, target) < 35.0 and safe_to_launch(src, target) for src in my_planets)
    )
    threats = [
        predicted_enemy_threat(src, planets, player, step, horizon=24.0)[2]
        for src in my_planets
    ]
    threatened = any(strongest > src.ships * 0.45 for strongest, src in zip(threats, my_planets))

    early_enemy_commitment = step < 90 and any(f.owner not in (-1, player) for f in fleets)

    if my_ships > 120 and len(my_planets) < 4 and enemies:
        return "rush"
    if early_enemy_commitment:
        return "counter"
    if len(my_planets) < 3 or (nearby_neutrals > 0 and len(my_planets) < 5):
        return "expand"
    if threatened:
        return "counter"
    if prod_ratio > 4.0 and my_ships > 80 and len(my_planets) >= 3:
        return "crush"
    if prod_ratio > 2.0 or ship_ratio > 2.5:
        return "aggressive"
    if enemy_prod > 0 and my_prod < enemy_prod * 0.7:
        return "defend"
    if enemies and len(my_planets) >= 3 and my_prod > enemy_prod:
        return "dominate"
    return "grow"


def phase_target_bias(phase, target, step):
    if not ENABLE_PHASE_PRESSURE:
        return 0.0
    if target.owner == -1:
        if phase in ("dominate", "aggressive", "crush"):
            return 18.0
        if phase == "defend":
            return -10.0
        return 0.0
    if phase == "rush":
        return 60.0 - 0.10 * target.ships
    if phase == "crush":
        return 72.0 - 0.08 * target.ships
    if phase == "dominate":
        return 54.0 - 0.08 * target.ships
    if phase == "aggressive":
        return 42.0 - 0.12 * target.ships
    if phase == "counter":
        return 36.0 + 0.35 * target.production
    if phase == "expand" and step < 90:
        return -38.0
    if phase == "defend":
        return -30.0
    return 0.0


def attack_plan_for_wait(src, target, planets, fleets, player, step, initial_by_id, angular_velocity, wait_turns, phase, enemy_stats=None):
    ships_needed = int(target.ships) + 1
    lead = predict_planet_position(target, initial_by_id, angular_velocity, wait_turns)
    for _ in range(3):
        if ENABLE_TIMELINE_NEED:
            coarse_needed = capture_ships_needed_after_wait(src, target, lead, wait_turns)
            eta = travel_time(src, lead, ships_needed)
            upper = max(
                coarse_needed + 1,
                int(target.ships + target.production * math.ceil(wait_turns + eta) + 80),
            )
            ships_needed = timeline_ships_needed_to_own(
                target,
                fleets,
                player,
                wait_turns + eta,
                initial_by_id,
                angular_velocity,
                upper,
            )
            if step < TIMELINE_TOPUP_GUARD_UNTIL and 0 < ships_needed < coarse_needed:
                return None
        else:
            ships_needed = capture_ships_needed_after_wait(src, target, lead, wait_turns)
        if ships_needed <= 0:
            return None
        eta = travel_time(src, lead, ships_needed)
        lead = predict_planet_position(target, initial_by_id, angular_velocity, wait_turns + eta)

    if not ENABLE_TIMELINE_NEED:
        current_incoming, _ = fleet_pressure_to_planet(target, fleets, owner=player, extra_radius=2.5)
        lead_incoming, _ = fleet_pressure_to_planet(lead, fleets, owner=player, extra_radius=2.5)
        my_incoming = max(current_incoming, lead_incoming)
        if my_incoming >= ships_needed:
            return None
        ships_needed = max(1, int(ships_needed - my_incoming))

    if not safe_to_launch(src, lead):
        return None

    def score_for_send(send):
        eta = travel_time(src, lead, send)
        if step < 80 and eta > OPENING_ATTACK_TRAVEL_LIMIT:
            return None
        if step < 160 and eta > MIDGAME_ATTACK_TRAVEL_LIMIT:
            return None

        total_delay = wait_turns + eta
        useful_turns = max(0.0, EPISODE_STEPS - step - total_delay)
        production_value = target.production * min(useful_turns, ATTACK_VALUE_HORIZON)
        capture_cost = send
        travel_penalty = ATTACK_TRAVEL_PENALTY * eta + 0.8 * wait_turns
        owner_bonus = 0.0 if target.owner == -1 else 18.0
        owner_bonus += phase_target_bias(phase, target, step)
        opening_bonus = 0.0
        if step < 90:
            if target.owner == -1:
                payback_turns = send / max(1, target.production)
                opening_bonus = 3.0 * max(0.0, 80.0 - total_delay - payback_turns)
                opening_bonus += 18.0 * max(0, target.production - 1)
            elif phase not in ("rush", "counter"):
                owner_bonus = -28.0

        contest_risk, strongest_contester = opponent_contest_risk(lead, planets, player, total_delay)
        contest_penalty = 11.0 * contest_risk
        if strongest_contester >= send and target.owner == -1:
            contest_penalty += 10.0

        score = production_value + owner_bonus + opening_bonus - capture_cost - travel_penalty - contest_penalty

        # A1 + A2: enemy-targeting bonuses (applied after base score)
        if enemy_stats is not None and target.owner not in (-1, player):
            my_total, enemy_strength, weakest_id = enemy_stats
            t_owner = target.owner
            t_strength = enemy_strength.get(t_owner, 0)

            # A2: Elimination — flat bonus when the target enemy is on the back foot
            if my_total > 0 and t_strength < my_total * ELIMINATION_WEAK_THRESHOLD:
                score += ELIMINATION_BONUS

            # A1: Weakest targeting — extra weight on production value of vulnerable enemy
            if weakest_id is not None and t_owner == weakest_id:
                is_multi = len(enemy_strength) > 1
                mult = WEAKEST_ENEMY_VALUE_MULT_4P if is_multi else WEAKEST_ENEMY_VALUE_MULT_2P
                score += production_value * (mult - 1.0)

        return {
            "score": score,
            "target": target,
            "lead": lead,
            "ships": send,
            "eta": eta,
            "wait": wait_turns,
            "contest_risk": contest_risk,
        }

    send_options = [ships_needed]
    if (
        (ENABLE_SPEED_OVERSEND or ENABLE_RACE_OVERSEND)
        and wait_turns == 0
        and step < 120
        and target.owner == -1
        and target.production >= 3
    ):
        available = usable_ships(src, defaultdict(int), step)
        max_extra = 35 if ENABLE_SPEED_OVERSEND else max(6, min(18, int(math.ceil(ships_needed * 0.35))))
        max_mult = 1.8 if ENABLE_SPEED_OVERSEND else 1.35
        max_send = min(available, max(ships_needed + max_extra, int(math.ceil(ships_needed * max_mult))))
        if max_send > ships_needed:
            candidates = {
                ships_needed + 4,
                ships_needed + 8,
                int(math.ceil(ships_needed * 1.15)),
                int(math.ceil(ships_needed * 1.30)),
                max_send,
            }
            if ENABLE_SPEED_OVERSEND:
                candidates.update({
                    ships_needed + 16,
                    int(math.ceil(ships_needed * 1.50)),
                    int(math.ceil(ships_needed * 1.80)),
                })
            send_options.extend(sorted(send for send in candidates if ships_needed < send <= max_send))
    elif (
        ENABLE_PHASE_PRESSURE
        and wait_turns == 0
        and target.owner not in (-1, player)
        and phase in ("rush", "aggressive", "dominate", "crush", "counter")
    ):
        available = usable_ships(src, defaultdict(int), step)
        if phase in ("rush", "crush"):
            fraction = 0.85
        elif phase == "dominate":
            fraction = 0.70
        elif phase == "counter":
            fraction = 0.95
        else:
            fraction = 0.60
        max_send = min(available, int(src.ships * fraction))
        if max_send > ships_needed:
            send_options.extend(
                sorted({
                    int(math.ceil(ships_needed * 1.15)),
                    int(math.ceil(ships_needed * 1.35)),
                    max_send,
                } - {ships_needed})
            )

    scored = [plan for send in send_options for plan in [score_for_send(send)] if plan is not None]
    if not scored:
        return None
    if ENABLE_RACE_OVERSEND and len(scored) > 1:
        exact = next((plan for plan in scored if plan["ships"] == ships_needed), None)
        if exact is not None:
            opponent_eta, opponent_available = fastest_opponent_contester(lead, planets, player)
            race_window = opponent_eta is not None and exact["eta"] >= opponent_eta - 2.0
            slow_high_value = target.production >= 4 and exact["eta"] >= 16.0
            if race_window or slow_high_value:
                viable = [exact]
                for plan in scored:
                    if plan["ships"] == ships_needed:
                        continue
                    eta_gain = exact["eta"] - plan["eta"]
                    extra_ships = plan["ships"] - ships_needed
                    adjusted = dict(plan)
                    adjusted["score"] -= 0.75 * extra_ships
                    if eta_gain >= 1.8 and adjusted["score"] >= exact["score"] - 4.0:
                        if opponent_eta is None or plan["eta"] <= opponent_eta + 1.0 or plan["ships"] > opponent_available:
                            viable.append(adjusted)
                scored = viable
            else:
                scored = [exact]
    return max(scored, key=lambda plan: plan["score"])


def target_score(src, target, planets, fleets, player, step, initial_by_id, angular_velocity, comet_ids, phase, enemy_stats=None):
    if target.id in comet_ids:
        # Comets are temporary. Only take them when cheap, nearby, and safe.
        if target.production > 1 or distance(src, target) > 16 or target.ships > 8:
            return None

    if target.owner == player:
        return None

    if step < PROD1_NEUTRAL_BLOCK_UNTIL and src.production <= 1 and target.owner == -1:
        if (
            (PROD1_NEUTRAL_MAX_BLOCK_PROD > 0 and target.production <= PROD1_NEUTRAL_MAX_BLOCK_PROD)
            or target.production < src.production + PROD1_NEUTRAL_MIN_PROD_GAIN
        ):
            return None

    weak_opening_target = target.production <= (3 if src.production <= 1 else 2)
    if step < 80 and src.production <= 2 and target.owner == -1 and weak_opening_target:
        anchor = opening_anchor_target(src, planets, player, comet_ids)
        horizon = 60 if src.production <= 1 else 25
        affordable_soon = anchor is not None and anchor["ships"] <= src.ships + src.production * horizon
        cheap_local = (
            src.production <= 1
            and target.ships <= 12
            and target.production >= src.production
            and safe_to_launch(src, target)
            and travel_time(src, target, int(target.ships) + 1) <= 16.0
        )
        if affordable_soon and not cheap_local and target.id != anchor["id"] and src.ships < anchor["ships"]:
            return None

    waits = [0]
    if (
        ENABLE_WAIT_GATE
        and step >= WAIT_GATE_START
        and is_orbiting_planet(target)
        and target.production >= WAIT_GATE_MIN_PRODUCTION
    ):
        waits = [0, 3, 6, 10, 15]
        waits = [wait for wait in waits if wait <= WAIT_GATE_MAX_WAIT]

    plans = [
        plan
        for wait_turns in waits
        for plan in [attack_plan_for_wait(
            src, target, planets, fleets, player, step, initial_by_id, angular_velocity, wait_turns, phase,
            enemy_stats=enemy_stats,
        )]
        if plan is not None
    ]
    if not plans:
        return None

    now = next((plan for plan in plans if plan["wait"] == 0), None)
    best = max(plans, key=lambda plan: plan["score"])
    if now is None:
        return None
    if best["wait"] > 0 and best["score"] > now["score"] + WAIT_GATE_MARGIN:
        return None
    return now


def build_attack_candidates(my_planets, planets, fleets, player, step, initial_by_id, angular_velocity, comet_ids, phase, enemy_stats=None):
    candidates = []
    for src in my_planets:
        for target in planets:
            if target.owner == player:
                continue
            candidate = target_score(
                src, target, planets, fleets, player, step, initial_by_id, angular_velocity, comet_ids, phase,
                enemy_stats=enemy_stats,
            )
            if candidate is None:
                continue
            candidate["src"] = src
            candidates.append(candidate)
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def build_counter_capture_candidates(my_planets, planets, fleets, player, step, initial_by_id, angular_velocity, enemy_stats=None):
    if step >= 140:
        return []

    outgoing_by_source = defaultdict(int)
    for fleet in fleets:
        if fleet.owner not in (-1, player) and fleet.from_planet_id is not None:
            outgoing_by_source[fleet.from_planet_id] += int(fleet.ships)

    if not outgoing_by_source:
        return []

    candidates = []
    enemy_by_id = {p.id: p for p in planets if p.owner not in (-1, player)}
    for target_id, outgoing in outgoing_by_source.items():
        target = enemy_by_id.get(target_id)
        if target is None or outgoing < 18:
            continue
        if target.ships > max(22, outgoing * 0.45):
            continue
        depletion_bonus = min(85.0, 1.4 * outgoing + max(0.0, 24.0 - target.ships))
        for src in my_planets:
            if src.id == target.id:
                continue
            plan = attack_plan_for_wait(
                src,
                target,
                planets,
                fleets,
                player,
                step,
                initial_by_id,
                angular_velocity,
                0,
                "counter",
                enemy_stats=enemy_stats,
            )
            if plan is None:
                continue
            plan = dict(plan)
            plan["src"] = src
            plan["score"] += depletion_bonus
            plan["counter_outgoing"] = outgoing
            candidates.append(plan)
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def build_gang_up_candidates(my_planets, planets, fleets, player, step, initial_by_id, angular_velocity):
    """Exploit inter-enemy battles: schedule our fleet to arrive post-battle.

    When two enemies are fighting each other, the winner is depleted.
    We time our fleet to arrive GANG_UP_POST_BATTLE_DELAY turns after the
    battle resolves, capturing the weakened survivor cheaply.
    """
    if step > 420:
        return []

    candidates = []
    enemy_planets = {p.id: p for p in planets if p.owner not in (-1, player)}

    for fleet in fleets:
        if fleet.owner in (-1, player):
            continue
        for target_planet in enemy_planets.values():
            if target_planet.owner == fleet.owner:
                continue  # fleet and target are on same team
            eta = fleet_eta_to_planet(fleet, target_planet, initial_by_id, angular_velocity)
            if eta is None or eta > 50:
                continue

            # Estimate garrison when fleet arrives (production accrues during travel)
            garrison_at_battle = int(target_planet.ships + target_planet.production * eta)
            fleet_ships = int(fleet.ships)

            # Compute battle survivor (simplified: no other fleets considered)
            survivor_ships = abs(fleet_ships - garrison_at_battle)

            # Only worthwhile if the battle leaves a small survivor
            if survivor_ships > max(12, garrison_at_battle * 0.45):
                continue

            send_needed = max(1, survivor_ships + 2)
            desired_arrival = eta + GANG_UP_POST_BATTLE_DELAY

            for src in my_planets:
                if not safe_to_launch(src, target_planet):
                    continue
                actual_eta = travel_time(src, target_planet, send_needed)
                if abs(actual_eta - desired_arrival) > GANG_UP_ETA_WINDOW:
                    continue

                useful_turns = max(0.0, EPISODE_STEPS - step - actual_eta)
                production_value = target_planet.production * min(useful_turns, ATTACK_VALUE_HORIZON)
                owner_bonus = 18.0  # enemy planet
                score = (production_value + owner_bonus) * GANG_UP_VALUE_MULT \
                        - send_needed - ATTACK_TRAVEL_PENALTY * actual_eta

                if score <= 12.0:
                    continue

                candidates.append({
                    "src": src,
                    "target": target_planet,
                    "lead": target_planet,
                    "ships": send_needed,
                    "score": score,
                    "eta": actual_eta,
                    "wait": 0,
                    "contest_risk": 0.0,
                    "kind": "gang_up",
                })

    # Deduplicate — keep highest-score candidate per (src, target) pair
    seen = set()
    deduped = []
    for c in sorted(candidates, key=lambda c: c["score"], reverse=True):
        key = (c["src"].id, c["target"].id)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def fleet_commitment_ratio(planets, fleets, player):
    planet_ships = sum(p.ships for p in planets if p.owner == player)
    fleet_ships = sum(f.ships for f in fleets if f.owner == player)
    total = planet_ships + fleet_ships
    if total <= 0:
        return 0.0
    return fleet_ships / total


def source_fleet_pressure(fleets, player):
    pressure = defaultdict(int)
    for f in fleets:
        if f.owner == player:
            pressure[f.from_planet_id] += f.ships
    return pressure


def my_fleet_sources_and_targets(planets, fleets, player):
    """Approximate owned fleet commitments using the closest non-source planet."""
    sources = set()
    targets = set()
    for fleet in fleets:
        if fleet.owner != player:
            continue
        if fleet.from_planet_id is not None:
            sources.add(fleet.from_planet_id)
        best_target = None
        best_dist = None
        for planet in planets:
            if planet.id == fleet.from_planet_id:
                continue
            dist = math.hypot(planet.x - fleet.x, planet.y - fleet.y)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_target = planet.id
        if best_target is not None:
            targets.add(best_target)
    return sources, targets


def planet_keep_value(planet, step, horizon=120.0):
    useful_turns = max(0.0, min(horizon, EPISODE_STEPS - step))
    return planet.production * useful_turns + 0.25 * planet.ships


def opponent_available_ships(planet, step):
    reserve = defense_reserve(planet, step)
    return max(0, int(planet.ships - reserve))


def predicted_enemy_threat(planet, planets, player, step, horizon=24.0):
    """Estimate likely future enemy pressure if opponents launch soon."""
    pressure = 0
    earliest = None
    strongest = 0
    attackers = 0
    for enemy in planets:
        if enemy.owner in (-1, player):
            continue
        available = opponent_available_ships(enemy, step)
        if available <= 0 or not safe_to_launch(enemy, planet):
            continue
        eta = travel_time(enemy, planet, available)
        if eta > horizon:
            continue
        future_garrison = planet.ships + planet.production * max(1, int(eta))
        if available <= future_garrison + 2:
            continue
        pressure += available
        strongest = max(strongest, available)
        attackers += 1
        earliest = eta if earliest is None else min(earliest, eta)
    return pressure, earliest, strongest, attackers


def build_defense_candidates(my_planets, fleets, player, step, initial_by_id, angular_velocity):
    candidates = []
    owned_by_id = {p.id: p for p in my_planets}
    for threatened in my_planets:
        if ENABLE_ORBITING_DEFENSE_V6:
            incoming, eta = incoming_enemy_pressure_orbiting(
                threatened, fleets, player, initial_by_id, angular_velocity
            )
        else:
            incoming, eta = incoming_enemy_pressure(threatened, fleets, player)
        if incoming <= 0:
            continue
        future_garrison = threatened.ships + threatened.production * max(1, int(eta or 1))
        needed = int(incoming - future_garrison + 4)
        if needed <= 0:
            continue
        value = planet_keep_value(threatened, step, horizon=140.0)
        for src in my_planets:
            if src.id == threatened.id or not safe_to_launch(src, threatened):
                continue
            send = max(1, needed)
            arrive = travel_time(src, threatened, send)
            if eta is not None and arrive > eta + 1:
                continue
            source_cost = 0.25 * src.production * min(80.0, EPISODE_STEPS - step)
            urgency_bonus = 45.0 if eta is not None and eta <= 8 else 20.0
            score = value + urgency_bonus - send - 1.8 * arrive - 0.08 * source_cost
            candidates.append({
                "src": src,
                "target": owned_by_id[threatened.id],
                "ships": send,
                "score": score,
                "eta": arrive,
                "contest_risk": 0.0,
                "kind": "defend",
            })
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def latent_threat_reserves(my_planets, planets, player, step):
    """Extra ships to hold because an opponent can plausibly punish soon."""
    if not ENABLE_LATENT_THREAT_RESERVE or step < 90:
        return {planet.id: 0 for planet in my_planets}

    reserves = {}
    for threatened in my_planets:
        _, enemy_eta, strongest, attackers = predicted_enemy_threat(
            threatened, planets, player, step
        )
        if strongest <= 0 or enemy_eta is None:
            reserves[threatened.id] = 0
            continue
        future_garrison = threatened.ships + threatened.production * max(1, int(enemy_eta))
        needed = max(0, int(strongest - future_garrison + 8))
        if attackers > 1:
            needed += 4
        # Do not freeze the whole economy on speculation. This just raises the
        # floor under attack spending from planets that look punishable.
        reserves[threatened.id] = min(needed, max(0, int(threatened.ships * 0.45)))
    return reserves


def candidate_sort_key(candidate):
    ships = max(1, int(candidate["ships"]))
    kind = candidate.get("kind", "attack")
    urgency = 0.0
    if kind == "defend":
        urgency = 35.0
    elif kind == "predefend":
        urgency = 8.0
    return candidate["score"] + urgency + 0.15 * (candidate["score"] / ships)


def choose_moves(
    candidates,
    committed,
    step,
    max_moves,
    min_score=12.0,
    source_pressure=None,
    threat_reserve=None,
    in_flight_sources=None,
    in_flight_targets=None,
):
    source_pressure = source_pressure or {}
    threat_reserve = threat_reserve or {}
    in_flight_sources = in_flight_sources or set()
    in_flight_targets = in_flight_targets or set()
    moves = []
    attack_targets = set()
    defense_targets = set()
    for c in sorted(candidates, key=candidate_sort_key, reverse=True):
        if len(moves) >= max_moves:
            break
        kind = c.get("kind", "attack")
        if c["score"] <= min_score:
            continue
        src = c["src"]
        target = c["target"]
        ships = int(c["ships"])

        if kind == "attack" and target.id in attack_targets:
            continue
        if kind == "attack" and ENABLE_OPENING_IN_FLIGHT_LOCK and step < 90:
            if target.id in in_flight_targets:
                continue
            if src.production <= 1 and src.id in in_flight_sources and target.owner == -1:
                continue
        if kind in ("defend", "predefend") and target.id in defense_targets:
            continue
        if (
            kind == "attack"
            and step < 80
            and src.production <= 1
            and source_pressure.get(src.id, 0) > 0
            and src.ships < 25
        ):
            continue
        if (
            kind == "attack"
            and step < 45
            and step > 0
            and src.production <= 1
            and src.ships < 35
            and target.owner == -1
        ):
            continue
        if kind == "attack" and source_pressure.get(src.id, 0) > max(25, src.ships * 1.5):
            continue
        available = usable_ships(src, committed, step)
        if kind == "attack":
            available = max(0, available - threat_reserve.get(src.id, 0))
        if kind in ("defend", "predefend") and available < ships:
            if available < max(3, int(ships * 0.15)):
                continue
            ships = available
        elif available < ships:
            continue

        committed[src.id] += ships
        if kind == "attack":
            attack_targets.add(target.id)
        elif kind in ("defend", "predefend"):
            defense_targets.add(target.id)
        moves.append([src.id, angle_to(src, c.get("lead", target)), ships, c])
    return moves


def agent(obs):
    player = get_field(obs, "player", 0)
    step = int(get_field(obs, "step", 0) or get_field(obs, "turn", 0) or 0)
    raw_planets = get_field(obs, "planets", [])
    raw_fleets = get_field(obs, "fleets", [])
    raw_initial = get_field(obs, "initial_planets", raw_planets)
    angular_velocity = float(get_field(obs, "angular_velocity", 0.0) or 0.0)
    comet_ids = set(get_field(obs, "comet_planet_ids", []) or [])

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    initial_by_id = {p.id: p for p in [Planet(*p) for p in raw_initial]}
    my_planets = [p for p in planets if p.owner == player]

    if not my_planets:
        return []

    committed = defaultdict(int)
    phase = strategic_phase(my_planets, planets, fleets, player, step)
    enemy_stats = compute_enemy_stats(planets, fleets, player)
    defense = build_defense_candidates(my_planets, fleets, player, step, initial_by_id, angular_velocity)
    attacks = build_attack_candidates(
        my_planets, planets, fleets, player, step, initial_by_id, angular_velocity, comet_ids, phase,
        enemy_stats=enemy_stats,
    )
    counters = build_counter_capture_candidates(
        my_planets, planets, fleets, player, step, initial_by_id, angular_velocity,
        enemy_stats=enemy_stats,
    )
    pressure = source_fleet_pressure(fleets, player)
    in_flight_sources, in_flight_targets = my_fleet_sources_and_targets(planets, fleets, player)
    threat_reserve = latent_threat_reserves(my_planets, planets, player, step)
    gang_ups = build_gang_up_candidates(
        my_planets, planets, fleets, player, step, initial_by_id, angular_velocity
    )

    if ENABLE_UNIFIED_SELECTION:
        chosen = choose_moves(
            defense + counters + attacks + gang_ups,
            committed,
            step,
            max_moves=MAX_MOVES,
            source_pressure=pressure,
            threat_reserve=threat_reserve,
            in_flight_sources=in_flight_sources,
            in_flight_targets=in_flight_targets,
        )
    else:
        chosen = []
        chosen.extend(choose_moves(defense, committed, step, max_moves=3))
        chosen.extend(
            choose_moves(
                counters + attacks + gang_ups,
                committed,
                step,
                max_moves=MAX_MOVES - len(chosen),
                source_pressure=pressure,
                in_flight_sources=in_flight_sources,
                in_flight_targets=in_flight_targets,
            )
        )

    moves = [[src_id, theta, ships] for src_id, theta, ships, _ in chosen]

    if DEBUG and moves:
        notes = []
        for src_id, _, ships, c in chosen:
            notes.append(
                f"{c.get('kind', 'attack')} {src_id}->{c['target'].id} "
                f"ships={ships} ev={c['score']:.1f} eta={c['eta']:.1f} "
                f"risk={c['contest_risk']:.2f}"
            )
        print(f"step={step} player={player} phase={phase} " + " | ".join(notes), file=sys.stderr)

    return moves
