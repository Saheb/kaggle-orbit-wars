"""Unit test for the defensive hold-floor diagnostic (eval._hold_floor_step).

hold = (our_garrison + friendly_inbound) / (enemy_inbound + beta*reachable_enemy_mass + overhead)
Measured per OWN planet under an actual inbound threat (enemy fleet converging on it). ratio < 1 =
under-defended. age-after-capture comes from cap_step (None for home/initial planets).

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_hold_floor.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import eval as ev
from torch_env import _DM_BETA, _DM_OVERHEAD

SEAT = 0
# planet record: [id, owner, x, y, radius, ships, prod]
# fleet record:  [id, owner, x, y, angle, src, ships]


def test_threatened_only_and_ratio():
    # OUR planet A (id 1) at (300,300) garr 10. An enemy fleet (6 ships) inbound from the +x side.
    # No nearby enemy PLANET → reachable_enemy_mass≈0 → floor ≈ 6 + 1 = 7 → ratio = 10/7 ≈ 1.43.
    planets = [
        [1, SEAT, 300.0, 300.0, 8.0, 10.0, 1.0],     # ours, threatened
        [2, SEAT, 100.0, 100.0, 8.0, 50.0, 1.0],     # ours, NOT threatened (no inbound) → excluded
    ]
    fleets = [[0, 1, 360.0, 300.0, 3.14159265, 0, 6.0]]  # enemy fleet heading -x → planet 1
    out = ev._hold_floor_step(planets, fleets, SEAT, cap_step={}, t=10)
    assert len(out) == 1, f"only the threatened planet is scored, got {len(out)}"
    ratio, age = out[0]
    assert abs(ratio - 10.0 / (6.0 + _DM_OVERHEAD)) < 1e-6, f"ratio {ratio}"
    assert age is None, "no cap_step entry → home/initial → age None"


def test_friendly_inbound_raises_ratio_and_age():
    planets = [[1, SEAT, 300.0, 300.0, 8.0, 4.0, 1.0]]
    fleets = [
        [0, 1, 360.0, 300.0, 3.14159265, 0, 12.0],      # ENEMY inbound 12
        [1, SEAT, 240.0, 300.0, 0.0, 0, 9.0],           # OUR reinforcement inbound 9 (angle 0 → +x → planet 1)
    ]
    # floor = 12 + 0 + 1 = 13; mass = garr 4 + friendly 9 = 13 → ratio 1.0 exactly.
    out = ev._hold_floor_step(planets, fleets, SEAT, cap_step={1: 7}, t=10)
    assert len(out) == 1
    ratio, age = out[0]
    assert abs(ratio - 13.0 / 13.0) < 1e-6, f"friendly inbound must add to mass, ratio {ratio}"
    assert age == 3, f"age = t - cap_step = 10-7 = 3, got {age}"


def test_underdefended_when_outmassed():
    # garr 5, enemy inbound 40, no friendly → ratio = 5/41 << 1 (the wall: out-massed hold-loss).
    planets = [[1, SEAT, 300.0, 300.0, 8.0, 5.0, 1.0]]
    fleets = [[0, 1, 360.0, 300.0, 3.14159265, 0, 40.0]]
    ratio, _ = ev._hold_floor_step(planets, fleets, SEAT, cap_step={}, t=5)[0]
    assert ratio < 1.0, f"out-massed planet must read under-defended, ratio {ratio}"


def test_reachable_enemy_mass_lowers_ratio():
    # Same inbound, but add a nearby ENEMY planet (big garrison) → reachable_enemy_mass raises the
    # floor (forward-projected reinforcement) → ratio drops vs the bare-inbound case.
    base = [[1, SEAT, 300.0, 300.0, 8.0, 20.0, 1.0]]
    f = [[0, 1, 360.0, 300.0, 3.14159265, 0, 6.0]]
    r_bare, _ = ev._hold_floor_step(base, f, SEAT, cap_step={}, t=5)[0]
    # enemy planet near our planet 1 but OFF the fleet's -x path (perp > radius → not the fleet's
    # resolved target, so our planet stays the threatened one) → contributes reachable_enemy_mass.
    near = base + [[2, 1, 300.0, 320.0, 8.0, 80.0, 0.0]]
    r_near, _ = ev._hold_floor_step(near, f, SEAT, cap_step={}, t=5)[0]
    assert r_near < r_bare, f"reachable enemy mass must lift the floor → lower ratio ({r_near} vs {r_bare})"


if __name__ == "__main__":
    test_threatened_only_and_ratio()
    test_friendly_inbound_raises_ratio_and_age()
    test_underdefended_when_outmassed()
    test_reachable_enemy_mass_lowers_ratio()
    print("PASS: hold-floor — threatened-only gate, friendly-inbound+age, out-massed under-defended, reachable-mass margin")
