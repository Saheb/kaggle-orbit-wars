"""Unit test for Lever A — the decisive-mass reward (torch_env._decisive_mass_bonus).

Signal: +decisive_mass_coef the step our INFLIGHT force converging on an ENEMY planet first
reaches producer_v2's capture floor:
    floor = garrison + prod*eta + enemy_inbound + beta*rho(eta)*reachable_enemy_mass + overhead
Board-grounded, not outcome-tied → the force-concentration gradient self-play can't price
(project_force_concentration_wall).

Asserts: (1) sub-floor inflight → no credit; (2) two sub-floor fleets that SUM >= floor → one
credit (aggregation, the actual wall); (3) credit fires only on the CROSSING; (4) own/neutral
gated out; (5) the v2 REACTIVE margin — a nearby enemy planet (reachable_enemy_mass>0, eta>eta_free)
RAISES the floor and blocks a strike that would clear the bare floor.

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_decisive_mass.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch_env import VecTorchEnv

COEF = 0.5


def _env():
    env = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                      action_decode="target", decisive_mass_coef=COEF)
    env.reset(seeds=[0])
    env.planet_alive[0, :] = False
    # planet 0: ours, behind the fleets (not a candidate target)
    env.planets[0, 0, 1] = 0
    env.planets[0, 0, 2] = 100.0
    env.planets[0, 0, 3] = 300.0
    env.planets[0, 0, 4] = 8.0
    env.planets[0, 0, 5] = 5.0
    env.planets[0, 0, 6] = 0.0
    env.planet_alive[0, 0] = True
    # planet 1: ENEMY target at (300,300), garrison 10, prod 0 → bare floor = 10 + 1 = 11
    env.planets[0, 1, 1] = 1
    env.planets[0, 1, 2] = 300.0
    env.planets[0, 1, 3] = 300.0
    env.planets[0, 1, 4] = 8.0
    env.planets[0, 1, 5] = 10.0
    env.planets[0, 1, 6] = 0.0
    env.planet_alive[0, 1] = True
    env.fleet_alive[0, :] = False
    return env


def _add_fleet(env, slot, owner, ships, x=150.0):
    # Fleet at (x,300) heading +x (angle 0) → target planet 1 at (300,300).
    env.fleets[0, slot, 1] = owner
    env.fleets[0, slot, 2] = x
    env.fleets[0, slot, 3] = 300.0
    env.fleets[0, slot, 4] = 0.0
    env.fleets[0, slot, 6] = ships
    env.fleet_alive[0, slot] = True


def _credit(env):
    tr = torch.zeros(1, 2)
    env._decisive_mass_bonus(tr)
    return tr[0, 0].item()


def test_target_resolves():
    env = _env()
    _add_fleet(env, 0, owner=0, ships=10)
    assert env._fleet_target_idx()[0, 0].item() == 1, "fleet should target enemy planet 1"


def test_subfloor_no_credit():
    env = _env()
    _add_fleet(env, 0, owner=0, ships=10)   # 10 < bare floor 11
    assert abs(_credit(env)) < 1e-6, "sub-floor inflight must not be credited"


def test_aggregation_credits_once_on_crossing():
    env = _env()
    _add_fleet(env, 0, owner=0, ships=10)   # 10 < 11
    assert abs(_credit(env)) < 1e-6
    _add_fleet(env, 1, owner=0, ships=10)   # 10 + 10 = 20 >= 11 → CROSS
    assert abs(_credit(env) - COEF) < 1e-6, "two sub-floor fleets summing >= floor must credit once"
    assert abs(_credit(env)) < 1e-6, "must credit the crossing only, not every sufficient step"


def test_own_and_neutral_gated():
    for tgt_owner in (0, -1):
        env = _env()
        env.planets[0, 1, 1] = tgt_owner
        _add_fleet(env, 0, owner=0, ships=100)
        assert abs(_credit(env)) < 1e-6, f"target owner {tgt_owner} must be gated out (enemy-only)"


def test_reactive_margin_raises_floor():
    # Far fleets (eta capped at horizon → rho=1) carrying mass=15 > bare floor 11.
    # Without a nearby enemy planet → credited. WITH one (reachable_enemy_mass>0) the v2
    # margin beta*rho*enemy_mass lifts the floor far above 15 → NOT credited.
    env = _env()
    _add_fleet(env, 0, owner=0, ships=15, x=100.0)   # far → eta>eta_free → rho=1
    assert abs(_credit(env) - COEF) < 1e-6, "mass 15 clears bare floor 11 → credited"

    env = _env()
    env.planets[0, 2, 1] = 1        # enemy planet near the target → reachable enemy mass
    env.planets[0, 2, 2] = 305.0
    env.planets[0, 2, 3] = 300.0
    env.planets[0, 2, 4] = 8.0
    env.planets[0, 2, 5] = 50.0
    env.planets[0, 2, 6] = 0.0
    env.planet_alive[0, 2] = True
    _add_fleet(env, 0, owner=0, ships=15, x=100.0)
    assert abs(_credit(env)) < 1e-6, "v2 reactive margin must lift the floor above 15 → no credit"


def test_rearm_on_episode_boundary():
    # Drive the REAL production path (step → _auto_reset), not a hand-replayed reset. A far fleet
    # keeps planet 1 "sufficient", so the per-step _decisive_mass_bonus that runs INSIDE step()
    # would KEEP prev_decisive_suff[1] True (crossing already armed) — only the auto-reset re-arm
    # should clear it. We force termination via time_up so it's ownership-independent.
    env = _env()
    _add_fleet(env, 0, owner=0, ships=15)        # far → stays inflight; mass 15 ≥ bare floor 11
    tr = torch.zeros(1, 2)
    env._decisive_mass_bonus(tr)                 # arm planet 1
    assert bool(env.prev_decisive_suff[0, 1, 0]), "planet 1 should be armed (sufficient inflight mass)"
    # Force the next step() to terminate (time_up) and run the auto-reset path.
    env.step_count[0] = int(env.episode_steps)
    _, _, done = env.step()
    assert bool(done[0]), "env should terminate via time_up"
    # The fresh board has no fleets; step()'s internal _decisive_mass_bonus would have re-set
    # prev[1]=True (planet 1 still enemy + sufficient pre-reset), so a clean prev here proves the
    # auto-reset re-arm — not the per-step bonus — cleared it.
    assert not bool(env.prev_decisive_suff[0].any()), \
        "step()/_auto_reset must re-arm (clear) prev_decisive_suff for the done env"


if __name__ == "__main__":
    test_target_resolves()
    test_subfloor_no_credit()
    test_aggregation_credits_once_on_crossing()
    test_own_and_neutral_gated()
    test_reactive_margin_raises_floor()
    test_rearm_on_episode_boundary()
    print("PASS: decisive-mass (producer_v2 floor) — aggregation, crossing-once, gate, reactive margin, boundary re-arm")
