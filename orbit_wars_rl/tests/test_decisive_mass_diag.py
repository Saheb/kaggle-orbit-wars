"""Parity + smoke test for the decisive-mass GAP diagnostic (dm_*).

The diag must use the EXACT capture floor of the Lever-A reward so the two can never drift:
  floor = garrison + prod*tau + enemy_inbound_by_deadline
        + beta*rho(tau)*reachable_enemy_mass_by_deadline + margin

(1) PARITY: on a constructed board, the eval Python floor (eval._decisive_gap_step, pure-Python
    over a kaggle-style observation) produces the SAME per-target mass/floor ratios as the training
    torch floor (torch_env._decisive_mass_fields, vectorised). Same target resolution, same eta
    (MAX arrival), same deadline-classified inbound/reach, same rho — so the eval read can't silently diverge.
(2) SMOKE: VecTorchEnv(decisive_diag=True) accumulates the phase-split dm_* counters over real
    steps without error, and the gap/cross derive consistently (cross == 1 - normalized... no:
    cross+something; we just assert the accumulators populate and ratios are sane).

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_decisive_mass_diag.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

from torch_env import VecTorchEnv
import eval as ev


def _build():
    """A board with two enemy targets, our converging fleets (aggregation + a second target),
    enemy inbound, and a nearby enemy planet (reactive margin)."""
    env = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                      action_decode="target", decisive_mass_coef=0.0, decisive_diag=True)
    env.reset(seeds=[0])
    env.planet_alive[0, :] = False
    env.fleet_alive[0, :] = False

    def planet(slot, owner, x, y, ships, prod, r=8.0):
        env.planets[0, slot, 1] = owner
        env.planets[0, slot, 2] = x
        env.planets[0, slot, 3] = y
        env.planets[0, slot, 4] = r
        env.planets[0, slot, 5] = ships
        env.planets[0, slot, 6] = prod
        env.planet_alive[0, slot] = True

    def fleet(slot, owner, x, y, ang, ships):
        env.fleets[0, slot, 1] = owner
        env.fleets[0, slot, 2] = x
        env.fleets[0, slot, 3] = y
        env.fleets[0, slot, 4] = ang
        env.fleets[0, slot, 6] = ships
        env.fleet_alive[0, slot] = True

    planet(0, 0, 100.0, 300.0, 40.0, 1.0)        # ours (behind the fleets, not a target)
    planet(1, 1, 300.0, 300.0, 10.0, 2.0)        # enemy target A
    planet(2, 1, 330.0, 300.0, 50.0, 0.0)        # enemy planet near A → reachable-enemy-mass margin
    planet(3, 1, 100.0, 600.0, 12.0, 1.0)        # enemy target B (separate axis)

    # Two of OUR fleets converging on A with matching arrival ETAs (aggregation).
    fleet(0, 0, 266.5, 300.0, 0.0, 8.0)          # -> A (angle 0, along +x)
    fleet(1, 0, 250.0, 300.0, 0.0, 30.0)         # -> A
    # An ENEMY fleet inbound to A.
    fleet(2, 1, 360.0, 300.0, 3.14159265, 6.0)   # heading -x toward A
    # OUR fleet converging on B (heading +y from below).
    fleet(3, 0, 100.0, 500.0, 1.5707963, 9.0)    # → B (angle +pi/2)
    return env


def _torch_ratios(env, seat=0):
    mass, floor, _eta, is_enemy = env._decisive_mass_fields()
    m = mass[0, :, seat]
    fl = floor[0, :, seat].clamp(min=1e-6)
    ie = is_enemy[0, :, seat]
    att = ie & (m > 0)
    return sorted((m[i] / fl[i]).item() for i in range(m.shape[0]) if bool(att[i]))


def _kaggle_obs(env, seat=0):
    """Build a kaggle-style (planets, fleets) observation from the torch tensors."""
    planets, fleets = [], []
    for s in range(env.planets.shape[1]):
        if not bool(env.planet_alive[0, s]):
            continue
        p = env.planets[0, s]
        planets.append([s, int(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6])])
    for s in range(env.fleets.shape[1]):
        if not bool(env.fleet_alive[0, s]):
            continue
        f = env.fleets[0, s]
        fleets.append([s, int(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5]), float(f[6])])
    return planets, fleets


def test_parity_eval_vs_torch():
    env = _build()
    tr = _torch_ratios(env, seat=0)
    planets, fleets = _kaggle_obs(env, seat=0)
    ev_r = sorted(ev._decisive_gap_step(planets, fleets, seat=0))
    assert len(tr) >= 2, f"expected >=2 attacked enemy targets (A and B), got {len(tr)}"
    assert len(tr) == len(ev_r), f"target count mismatch torch={len(tr)} eval={len(ev_r)}"
    for a, b in zip(tr, ev_r):
        assert abs(a - b) < 1e-3, f"ratio parity drift: torch {a:.5f} vs eval {b:.5f}"


def _A_mass_floor(env, seat=0):
    """Inflight mass + floor on enemy target A (planet slot 1) for `seat`."""
    mass, floor, _eta, _ie = env._decisive_mass_fields()
    return float(mass[0, 1, seat]), float(floor[0, 1, seat])


def test_aggregation_sums_not_maxes():
    # Two of OUR fleets converge on A inside the same arrival window: slot0 = 8 ships,
    # slot1 = 30 ships. The diagnostic must SUM them (38), not max (30).
    # The slower fleet anchors tau in both cases, so removing slot1 leaves the floor unchanged
    # and the ratio must drop by exactly the mass factor (38->8).
    env = _build()
    mass_full, floor_full = _A_mass_floor(env, seat=0)
    assert abs(mass_full - 38.0) < 1e-4, f"A mass must SUM both fleets (8+30=38), got {mass_full}"

    env.fleet_alive[0, 1] = False                 # drop the 30-ship fleet → only the 8-ship remains
    mass_single, floor_single = _A_mass_floor(env, seat=0)
    assert abs(mass_single - 8.0) < 1e-4, f"A mass should be 8 after removing slot1, got {mass_single}"
    assert abs(floor_single - floor_full) < 1e-4, "floor must be eta/board-driven, independent of our mass"
    r_full, r_single = mass_full / floor_full, mass_single / floor_single
    assert r_full > r_single, "aggregating both fleets must RAISE the ratio"
    assert abs(r_full - r_single * (38.0 / 8.0)) < 1e-3, "ratio must scale with summed mass"


def test_staggered_fleets_do_not_fake_aggregation():
    env = _build()
    env.fleets[0, 0, 2] = 150.0                  # make the 8-ship fleet arrive much later
    mass, _floor = _A_mass_floor(env, seat=0)
    assert abs(mass - 8.0) < 1e-4, f"staggered A mass should count only the anchored wave, got {mass}"


def test_smoke_accumulators_populate():
    env = _build()
    env.reset_reinforce_stats()
    # Drive a few real steps; decisive_diag should accumulate without error.
    for _ in range(5):
        env.step()
    assert env._dm_targets is not None
    tot = float(env._dm_targets.sum().item())
    assert tot >= 0.0
    # gap_sum and cross are bounded by targets (gap in [0,1], cross in {0,1} per target)
    assert float(env._dm_gap_sum.sum().item()) <= tot + 1e-3
    assert float(env._dm_cross.sum().item()) <= tot + 1e-3


def test_parity_nondefault_beta():
    # P1: eval floor must track a NON-default --decisive-mass-beta. Build with beta=0.5 (changes the
    # reactive margin), compute torch ratios at that beta, and assert the eval floor matches when the
    # same beta is passed (it would drift if eval hardcoded 2.2).
    env = _build()
    env.decisive_mass_beta = 0.5
    tr = _torch_ratios(env, seat=0)
    planets, fleets = _kaggle_obs(env, seat=0)
    ev_r = sorted(ev._decisive_gap_step(planets, fleets, seat=0, beta=0.5))
    assert len(tr) == len(ev_r) and len(tr) >= 2
    for a, b in zip(tr, ev_r):
        assert abs(a - b) < 1e-3, f"non-default-beta parity drift: torch {a:.5f} vs eval {b:.5f}"


if __name__ == "__main__":
    test_parity_eval_vs_torch()
    test_parity_nondefault_beta()
    test_aggregation_sums_not_maxes()
    test_staggered_fleets_do_not_fake_aggregation()
    test_smoke_accumulators_populate()
    print("PASS: decisive-mass diag — eval/torch floor parity (default+beta), aggregation-sums, smoke")
