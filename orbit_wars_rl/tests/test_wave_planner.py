"""Phase 5 wave planner (§7) tests on synthetic states.

Pins the contract the BC labels depend on (post 2026-06-19 audit: synchronization dropped, the
arrival window is ONE-SIDED — arrive BY the deadline; see docs/phase5-blocked.md):
  - attack: a multi-source bundle CROSSES the reactive floor (sufficiency, not synchronization);
  - defense: a HOLDABLE owned planet under threat gets reinforced; a DOOMED one is drained, not fed;
  - post-round: every launched fleet arrives BY its deadline (arrival <= tau+tol).
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import wave_planner as WPL
import wave_primitives as wp

TOL = wp.WAVE_TOL_STEPS


def _post_round_ok(res):
    """Every emitted move must arrive BY its deadline (tau+tol). One-sided window: early arrival
    is allowed (synchronization dropped by the 2026-06-19 audit; see docs/phase5-blocked.md)."""
    for d in res.decisions.values():
        if d.kind == "noop":
            continue
        assert d.arrival_tau is not None and d.tau is not None
        assert d.arrival_tau <= d.tau + TOL + 1e-6, \
            f"src {d.src_pid} arrival {d.arrival_tau} later than deadline {d.tau+TOL}"


def test_attack_crosses_reactive_floor():
    # Two owned sources must COMBINE to cross an enemy target's REACTIVE floor (sufficiency is the
    # contract now; arrival synchronization was dropped by the audit).
    obs = {"step": 20, "planets": [
        [10, 1, 50, 50, 2, 30, 1],   # T enemy, 30 ships
        [0,  0, 30, 50, 2, 50, 1],   # S1 owned
        [1,  0, 70, 50, 2, 50, 1],   # S2 owned
    ], "fleets": []}
    res = WPL.plan(obs, player=0, episode_steps=500)
    atk = [d for d in res.decisions.values() if d.kind == "attack"]
    assert len(atk) >= 2, "expected a multi-source attack bundle"
    assert all(d.target_pid == 10 for d in atk)
    waves = [w for w in res.waves if w.kind == "attack" and w.target_pid == 10]
    assert waves, "no attack wave recorded"
    w = waves[0]
    assert w.launched + 1e-6 >= w.floor, f"launched {w.launched} must cross reactive floor {w.floor}"
    _post_round_ok(res)
    print("test_attack_crosses_reactive_floor: PASS")


def test_defense_reinforces_holdable():
    # Owned P threatened by an enemy fleet; a near owned planet R can reinforce in time -> HOLDABLE.
    obs = {"step": 20, "planets": [
        [5, 0, 50, 50, 2, 10, 1],    # P owned, under threat
        [6, 0, 40, 50, 2, 50, 1],    # R owned reinforcement, close
        [9, 1, 90, 90, 2, 80, 1],    # distant enemy (keeps R from being trivially safe-only context)
    ], "fleets": [
        [100, 1, 50, 20, math.pi / 2, 0, 30],   # enemy fleet inbound to P (30 ships)
    ]}
    res = WPL.plan(obs, player=0, episode_steps=500)
    holds = wp.classify_holds(obs["planets"], obs["fleets"], 0, current_step=20, episode_steps=500)
    assert holds[5].hold_class == wp.HOLD_HOLDABLE, f"P should be HOLDABLE, got {holds[5].hold_class}"
    defn = [d for d in res.decisions.values() if d.kind == "defense" and d.target_pid == 5]
    assert defn, "expected a defense reinforcement move toward P"
    assert defn[0].src_pid == 6
    _post_round_ok(res)
    print("test_defense_reinforces_holdable: PASS")


def test_doomed_is_drained_not_fed():
    # P is hopeless (huge enemy wave, no reachable rescue) -> DOOMED; its garrison must be a
    # drainable attack SOURCE, never a reinforce target.
    obs = {"step": 20, "planets": [
        [5, 0, 50, 50, 2, 40, 1],    # P owned, doomed
        [8, 1, 52, 50, 2, 20, 1],    # nearby enemy target P's drained garrison can attack
    ], "fleets": [
        [100, 1, 50, 10, math.pi / 2, 0, 200],   # overwhelming enemy wave inbound to P
    ]}
    res = WPL.plan(obs, player=0, episode_steps=500)
    holds = wp.classify_holds(obs["planets"], obs["fleets"], 0, current_step=20, episode_steps=500)
    assert holds[5].hold_class == wp.HOLD_DOOMED, f"P should be DOOMED, got {holds[5].hold_class}"
    # No defense move should target the doomed planet.
    assert not any(d.kind == "defense" and d.target_pid == 5 for d in res.decisions.values()), \
        "must not reinforce a DOOMED planet"
    # Its garrison is drainable (safe_sendable == full garrison).
    assert holds[5].safe_sendable >= 40 - 1e-6, "DOOMED must drain full garrison"
    _post_round_ok(res)
    print("test_doomed_is_drained_not_fed: PASS")


if __name__ == "__main__":
    print("Running Phase 5 wave planner tests...\n")
    test_attack_crosses_reactive_floor()
    test_defense_reinforces_holdable()
    test_doomed_is_drained_not_fed()
    print("\nAll wave planner tests passed!")
