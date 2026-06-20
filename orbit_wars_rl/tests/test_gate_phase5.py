"""Phase 5 Step 6 — pre-PPO no-veto gate tests (spec §11, §14.6).

Pins the load-bearing property the gate depends on: the deterministic oracle (wave_planner) passes
EVERY synthetic state, and a no-op policy FAILS every state. If either breaks, the states no longer
discriminate a real BC policy and the gate is meaningless.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gate_phase5 as G


def test_battery_has_each_category():
    cats = {st.category for st in G.battery()}
    assert cats == {"attack", "defense_hold", "defense_doomed"}, cats


def test_oracle_passes_every_state():
    passed, results = G.run_gate(G.oracle_policy)
    failed = [r for r in results if not r.passed]
    assert passed, f"oracle must pass all states; failed: {[(r.name, r.detail) for r in failed]}"


def test_noop_fails_every_state():
    _, results = G.run_gate(G.noop_policy)
    survived = [r for r in results if r.passed]
    assert not survived, f"noop must fail all states; survived: {[r.name for r in survived]}"


def test_gate_is_load_bearing():
    oracle_ok, _ = G.run_gate(G.oracle_policy)
    noop_ok, noop_res = G.run_gate(G.noop_policy)
    assert oracle_ok and not noop_ok and not any(r.passed for r in noop_res)


if __name__ == "__main__":
    test_battery_has_each_category()
    test_oracle_passes_every_state()
    test_noop_fails_every_state()
    test_gate_is_load_bearing()
    print("All Phase 5 gate tests passed!")
