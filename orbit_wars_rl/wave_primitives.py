"""Shared Phase 5 wave-floor primitives.

This module is intentionally scalar/pure-Python. Evaluation, BC planners, and
unit tests can call it directly; the GPU env mirrors these formulas in vectorized
torch code where Python loops would be too slow.
"""

from __future__ import annotations

import math

MAX_SHIP_SPEED = 6.0

WAVE_TOL_STEPS = 2.0
CAPTURE_OVERHEAD = 1.0
SAFETY_PAD = 1.0
WAVE_MARGIN = CAPTURE_OVERHEAD + SAFETY_PAD

DEFAULT_REACTIVE_BETA = 2.2
DM_ETA_FREE = 3.0
DM_ETA_SCALE = 12.0
DM_HORIZON = 18.0


def ship_speed(ships: float) -> float:
    """Kaggle Orbit Wars fleet speed for a ship count."""
    s = max(float(ships), 1.0)
    base = (math.log(s) / math.log(1000.0)) ** 1.5
    return min(1.0 + (MAX_SHIP_SPEED - 1.0) * base, MAX_SHIP_SPEED)


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
    target_id = int(target[0])
    deadline = float(tau) + float(tol)
    mass = 0.0
    for f in fleets or []:
        owner = int(f[1])
        if owner < 0 or owner == attacker:
            continue
        ft = fleet_target_planet(planets, f)
        if ft is None or int(ft[0]) != target_id:
            continue
        if fleet_eta_to_target(f, ft) <= deadline:
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
    target_id = int(target[0])
    deadline = float(tau) + float(tol)
    mass = 0.0
    for f in fleets or []:
        owner = int(f[1])
        if owner < 0 or owner == attacker:
            continue
        ft = fleet_target_planet(planets, f)
        if ft is None or int(ft[0]) != target_id:
            continue
        if fleet_eta_to_target(f, ft) > deadline:
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
