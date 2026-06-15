"""Regression test for the fleet-saturation ship-burn bug (torch_env _apply_actions).

Bug: ships were debited for every `can_fire` launch BEFORE checking for a free fleet slot, so in
fleet-saturated states a launch with no free slot debited ships from its planet but created no
fleet → ships vanished. Fix: compute slot availability first, debit only `target_valid` (launches
that actually create a fleet). This asserts conservation (debit == fleets created) across a range
of free-slot counts, including 0 (fully saturated, the burn case).

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_fleet_saturation_burn.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch_env import VecTorchEnv, MAX_FLEETS, MAX_OWNED


def _trial(free_slots, n_planets=6, ships=10):
    env = VecTorchEnv(num_envs=1, num_players=2, device="cpu", action_decode="angle")
    env.reset(seeds=[0])
    # give player 0 several owned planets with ships
    for k in range(n_planets):
        env.planets[0, k, 1] = 0
        env.planets[0, k, 5] = ships
        env.planet_alive[0, k] = True
    # saturate fleet storage, leaving exactly `free_slots` free
    env.fleet_alive[0, :] = True
    if free_slots > 0:
        env.fleet_alive[0, MAX_FLEETS - free_slots:] = False
    # fire 1 ship from every owned slot (ship_bin 0 = 1 ship, never overasks)
    a = torch.zeros(1, MAX_OWNED, 3, dtype=torch.long)
    a[:, :, 0] = 1
    a[:, :, 2] = 0
    ships_before = env.planets[0, :, 5].sum().item()
    id_before = int(env.next_fleet_id[0])
    env._apply_actions(a, owner_id=0)
    debited = ships_before - env.planets[0, :, 5].sum().item()
    created = int(env.next_fleet_id[0]) - id_before
    return debited, created, free_slots


def test_no_ship_burn_under_saturation():
    n_owned = 7  # the 6 we set + 1 player-0 planet from reset
    for free in (0, 1, 3, 5, 8):
        debited, created, _ = _trial(free)
        assert abs(debited - created) < 1e-6, (
            f"ship BURN: free={free} debited {debited} but only {created} fleets created")
        # created is bounded by both free slots and the number of owned planets that fire
        assert created == min(free, n_owned), (
            f"free={free}: expected {min(free, n_owned)} fleets, got {created}")
    print("PASS: debit == fleets created at free slots 0/1/3/5/8 (no ship burn under saturation)")


if __name__ == "__main__":
    test_no_ship_burn_under_saturation()
