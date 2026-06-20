"""Ground-truth regression guard for the lead-aware target resolvers.

Simulates each launch tick-by-tick with the ENGINE's OWN `swept_pair_hit` + orbit formula (fleets fly
straight at ship-speed; planets advance one orbit-tick at a time), records the planet the fleet actually
collides with, and checks the resolvers against it. Guards two things the 2026-06-20 fix established:

  1. bc._find_target_planet_index (the BC LABEL builder) and eval._resolve_launch_target (the eval METRIC
     resolver) must agree on EVERY launch — they are the same geometry and must never silently drift.
  2. The lead-aware logic must be present: it roughly DOUBLES target accuracy over the old angle-only
     matcher, and is high on near/clear launches (the regime real teacher launches live in).

torch_env._fleet_target_idx is the vectorised mirror, guarded separately by test_fleet_target_lead
(vec == scalar lead-aware == the geometry validated here). project_target_resolution_metric_bug.

Run:  orbit_wars_rl/.venv/bin/python -m pytest orbit_wars_rl/tests/test_target_resolution_groundtruth.py
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    generate_planets, swept_pair_hit, point_to_segment_distance,
    CENTER, BOARD_SIZE, SUN_RADIUS, ROTATION_RADIUS_LIMIT,
)
import bc
import eval as ev

MAX_SPEED = 6.0  # configuration.shipSpeed default


def _fleet_speed(ships):
    s = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000)) ** 1.5
    return min(s, MAX_SPEED)


def _rot(p, ang):
    """Planet p rotated about CENTER by `ang` rad (static at/beyond the rotation limit)."""
    dx, dy = p[2] - CENTER, p[3] - CENTER
    r = math.hypot(dx, dy)
    if r + p[4] >= ROTATION_RADIUS_LIMIT:
        return (p[2], p[3])
    a = math.atan2(dy, dx) + ang
    return (CENTER + r * math.cos(a), CENTER + r * math.sin(a))


def _true_target(planets, angvel, src, angle, ships, max_ticks=200):
    """Planet id a launch physically collides with per the engine's swept-pair test; None = void."""
    sx = src[2] + math.cos(angle) * (src[4] + 0.1)
    sy = src[3] + math.sin(angle) * (src[4] + 0.1)
    speed = _fleet_speed(ships)
    cx, cy = math.cos(angle) * speed, math.sin(angle) * speed
    for k in range(1, max_ticks + 1):
        old = (sx + cx * (k - 1), sy + cy * (k - 1))
        new = (sx + cx * k, sy + cy * k)
        for p in planets:
            if swept_pair_hit(old, new, _rot(p, angvel * (k - 1)), _rot(p, angvel * k), p[4]):
                return p[0]
        if not (0 <= new[0] <= BOARD_SIZE and 0 <= new[1] <= BOARD_SIZE):
            return None
        if point_to_segment_distance((CENTER, CENTER), old, new) < SUN_RADIUS:
            return None
    return None


def _intercept_angle(src, tgt, angvel, ships):
    """Lead aim: angle to where `tgt` will be at fleet ETA (the real teacher-launch regime)."""
    speed = _fleet_speed(ships)
    eta = max(0.0, (math.hypot(tgt[2] - src[2], tgt[3] - src[3]) - tgt[4]) / speed)
    for _ in range(6):
        lx, ly = _rot(tgt, angvel * eta)
        eta = max(0.0, (math.hypot(lx - src[2], ly - src[3]) - tgt[4]) / speed)
    lx, ly = _rot(tgt, angvel * eta)
    return math.atan2(ly - src[3], lx - src[2])


def _old_angle_target(planets, src, angle, max_planets=48):
    """The pre-fix min-angular-error matcher (distance/motion-blind), for the regression contrast."""
    best, berr = -1, float("inf")
    for j, p in enumerate(planets[:max_planets]):
        if p[0] == src[0]:
            continue
        d = abs(math.atan2(p[3] - src[3], p[2] - src[2]) - angle)
        d = min(d, 2 * math.pi - d)
        if d < berr:
            berr, best = d, j
    return -1 if berr > math.radians(15.0) else best


def test_target_resolution_vs_engine_groundtruth():
    n = bc_ok = old_ok = bc_ev_mismatch = 0
    near_n = near_bc_ok = 0
    rng = random.Random(0)

    for seed in range(3):
        planets = generate_planets(random.Random(1000 + seed))
        if len(planets) < 4:
            continue
        angvel = random.Random(2000 + seed).uniform(0.025, 0.05)
        for i, p in enumerate(planets):
            p[1] = i % 2
        for s0 in (0, 113):                    # two orbit phases — resolver must be step-invariant
            obs = [list(p[:2]) + list(_rot(p, angvel * s0)) + list(p[4:]) for p in planets]
            initial = [list(p) for p in obs]
            by_id = {p[0]: p for p in obs}
            for si, src in enumerate(obs):
                others = [t for t in range(len(obs)) if t != si]
                rng.shuffle(others)
                for ti in others[:3]:
                    other = obs[ti]
                    for ships in (3, 40):
                        for angle in (_intercept_angle(src, other, angvel, ships),
                                      math.atan2(other[3] - src[3], other[2] - src[2])):
                            truth = _true_target(obs, angvel, src, angle, ships)
                            if truth is None or truth == src[0]:
                                continue
                            n += 1

                            bidx = bc._find_target_planet_index(
                                (src[2], src[3]), angle, ships, obs, initial, angvel, s0)
                            bc_id = obs[bidx][0] if bidx >= 0 else None

                            ev._CONV_ANGVEL = angvel
                            ev_p = ev._resolve_launch_target(obs, src, angle, ships)
                            ev_id = ev_p[0] if ev_p is not None else None

                            if bc_id != ev_id:
                                bc_ev_mismatch += 1
                            if bc_id == truth:
                                bc_ok += 1
                            if _old_id_to_truth(obs, src, angle, truth):
                                old_ok += 1

                            d = math.hypot(by_id[truth][2] - src[2], by_id[truth][3] - src[3])
                            if d < 25:
                                near_n += 1
                                near_bc_ok += int(bc_id == truth)

    assert n > 500, f"too few resolvable launches sampled ({n})"
    # 1. bc (label builder) and eval (metric) resolvers must agree on EVERY launch.
    assert bc_ev_mismatch == 0, f"{bc_ev_mismatch}/{n} bc-vs-eval target disagreements"
    # 2. lead-aware must roughly double the old angle-only matcher and be high on near/clear launches.
    bc_acc, old_acc, near_acc = bc_ok / n, old_ok / n, near_bc_ok / max(near_n, 1)
    assert bc_acc > 0.75, f"fixed resolver accuracy regressed: {bc_acc:.3f}"
    assert bc_acc > old_acc + 0.20, f"fix not present? fixed {bc_acc:.3f} vs old {old_acc:.3f}"
    assert near_acc > 0.85, f"near-launch accuracy regressed: {near_acc:.3f} (n={near_n})"


def _old_id_to_truth(obs, src, angle, truth):
    idx = _old_angle_target(obs, src, angle)
    return idx >= 0 and obs[idx][0] == truth


if __name__ == "__main__":
    test_target_resolution_vs_engine_groundtruth()
    print("test_target_resolution_groundtruth: PASS")
