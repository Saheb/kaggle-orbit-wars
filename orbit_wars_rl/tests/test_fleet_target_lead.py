"""Parity test for the lead-aware fleet target resolver (torch_env._fleet_target_idx).

The vectorised resolver must match a straightforward scalar reference of the SAME lead-aware
geometry (project each candidate planet to its orbit position at the fleet's ETA, keep the min-ETA
planet the heading reaches within radius). The scalar form is validated at 98.4% vs the true
swept-collision on replay (eval._lead_collision_target); this test guards the vectorisation against
broadcasting/argmin bugs, and exercises ORBITING planets (the case the old along/perp-r+2 heuristic
got wrong by ignoring the lead). project_force_concentration_wall.

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_fleet_target_lead.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch_env import VecTorchEnv, CENTER, ROTATION_RADIUS_LIMIT, _ship_speed


def _ship_speed_scalar(ships):
    s = max(float(ships), 1.0)
    base = (math.log(s) / math.log(1000.0)) ** 1.5
    mx = float(_ship_speed(torch.tensor([1e9])).item())   # asymptote == MAX_SHIP_SPEED
    return min(1.0 + (mx - 1.0) * base, mx)


def _planet_pos_at(px, py, pr, angvel, t):
    dx, dy = px - CENTER, py - CENTER
    orb = math.hypot(dx, dy)
    if orb + pr >= ROTATION_RADIUS_LIMIT:
        return px, py
    ph = math.atan2(dy, dx) + angvel * t
    return CENTER + orb * math.cos(ph), CENTER + orb * math.sin(ph)


def _scalar_target(planets, alive, angvel, fx, fy, fang, fships):
    """planets: list of (x, y, r). Returns the planet index of the min-ETA lead hit, or -1."""
    c, sn = math.cos(fang), math.sin(fang)
    speed = max(_ship_speed_scalar(fships), 1e-6)
    best, best_eta = -1, None
    for i, (px, py, pr) in enumerate(planets):
        if not alive[i]:
            continue
        eta = max(0.0, (math.hypot(px - fx, py - fy) - pr) / speed)
        for _ in range(4):
            lx, ly = _planet_pos_at(px, py, pr, angvel, eta)
            eta = max(0.0, (math.hypot(lx - fx, ly - fy) - pr) / speed)
        lx, ly = _planet_pos_at(px, py, pr, angvel, eta)
        vx, vy = lx - fx, ly - fy
        if vx * c + vy * sn <= 0:
            continue
        if abs(vx * sn - vy * c) < pr + 0.5 and (best_eta is None or eta < best_eta):
            best_eta, best = eta, i
    return best


def test_lead_target_parity():
    torch.manual_seed(0)
    env = VecTorchEnv(num_envs=4, num_players=2, device="cpu", action_decode="target")
    env.reset(seeds=[1, 2, 3, 4])
    N, P = env.planets.shape[0], env.planets.shape[1]
    F = env.fleets.shape[1]

    # Place a spread of fleets: random positions across the board, random headings, varied sizes.
    n_fleet = min(F, 12)
    env.fleet_alive[:, :] = False
    for f in range(n_fleet):
        env.fleets[:, f, 1] = (f % 2)
        env.fleets[:, f, 2] = torch.rand(N) * 100.0
        env.fleets[:, f, 3] = torch.rand(N) * 100.0
        env.fleets[:, f, 4] = torch.rand(N) * (2 * math.pi)
        env.fleets[:, f, 6] = torch.randint(1, 200, (N,)).float()
        env.fleet_alive[:, f] = True

    vec = env._fleet_target_idx()                      # (N, F)

    mism = 0
    orbiting_hits = 0
    for n in range(N):
        angvel = float(env.angular_velocity[n].item())
        planets = [(float(env.planets[n, p, 2]), float(env.planets[n, p, 3]),
                    float(env.planets[n, p, 4])) for p in range(P)]
        alive = [bool(env.planet_alive[n, p]) for p in range(P)]
        for f in range(n_fleet):
            ref = _scalar_target(planets, alive, angvel,
                                 float(env.fleets[n, f, 2]), float(env.fleets[n, f, 3]),
                                 float(env.fleets[n, f, 4]), float(env.fleets[n, f, 6]))
            got = int(vec[n, f].item())
            if got != ref:
                mism += 1
                print(f"  MISMATCH env{n} fleet{f}: vec={got} scalar={ref}")
            elif ref >= 0:
                px, py, pr = planets[ref]
                if math.hypot(px - CENTER, py - CENTER) + pr < ROTATION_RADIUS_LIMIT:
                    orbiting_hits += 1
    assert mism == 0, f"{mism} vectorised/scalar target mismatches"
    print(f"OK: vectorised == scalar over {N * n_fleet} fleets "
          f"({orbiting_hits} resolved to ORBITING planets — lead path exercised)")


if __name__ == "__main__":
    test_target_resolves = None
    test_lead_target_parity()
    print("test_fleet_target_lead: PASS")
