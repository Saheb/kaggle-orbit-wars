"""Unit test for the reinforce-triage classifier (eval._threat_class + _reachable_friendly_mass).

Mirrors producer_v2 (safe_drain + roi gate): a threatened own planet is
  0 already-safe  (defense_cost <= 0)
  1 cheap-save    (savable, save_ratio = cost/(prod*H) < 1)
  2 expensive-save (savable, ratio >= 1)
  3 hopeless      (reachable friendly mass < defense_cost → planner abandons + recycles)
friendly_available is FAITHFUL: arrives before the threat ETA, owned sources contribute SPARE
(garrison − own threat), and DOOMED sources (own threat >= garrison) drain fully.

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_reinforce_triage.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import eval as ev

SEAT = 0
# planet: [id, owner, x, y, radius, ships, prod]   fleet: [id, owner, x, y, angle, src, ships]
PI = 3.14159265


def test_already_safe():
    # garrison 30 vs tiny inbound 3, no nearby enemy planet → defense_cost < 0 → already-safe.
    planets = [[1, SEAT, 300.0, 300.0, 8.0, 30.0, 2.0]]
    fleets = [[0, 1, 360.0, 300.0, PI, 0, 3.0]]
    assert ev._threat_class(planets, fleets, planets[0], SEAT) == 0


def test_hopeless_when_no_reachable_friendly():
    # garrison 5, big inbound 60, our only other planet is FAR (can't reach before arrival) → hopeless.
    planets = [
        [1, SEAT, 300.0, 300.0, 8.0, 5.0, 2.0],          # threatened
        [2, SEAT, 300.0, 300.0 - 1000.0, 8.0, 90.0, 2.0],  # ours but ~1000 away → unreachable in time
    ]
    fleets = [[0, 1, 330.0, 300.0, PI, 0, 60.0]]          # inbound 60, close → short threat ETA
    assert ev._threat_class(planets, fleets, planets[0], SEAT) == 3


def test_cheap_save_with_reachable_spare():
    # garrison 8, inbound 12 → cost ~5; a nearby own planet with big spare reachable in time → savable.
    # prod 4 → value 4*18=72, cost ~5 → ratio < 1 → cheap-save.
    planets = [
        [1, SEAT, 300.0, 300.0, 8.0, 8.0, 4.0],
        [2, SEAT, 300.0 - 30.0, 300.0, 8.0, 50.0, 1.0],   # close own planet, large spare
    ]
    fleets = [[0, 1, 360.0, 300.0, PI, 0, 12.0]]
    assert ev._threat_class(planets, fleets, planets[0], SEAT) == 1


def test_doomed_source_drains_fully():
    # The would-be helper (planet 2) is itself doomed (its own inbound 40 >= its garrison 30). safe_drain
    # says a doomed source has nothing to protect → it can send ALL 30. That spare should be counted.
    planets = [
        [1, SEAT, 300.0, 300.0, 8.0, 4.0, 1.0],           # threatened, needs help
        [2, SEAT, 300.0 - 25.0, 300.0, 8.0, 30.0, 1.0],   # doomed helper (40 inbound vs 30 garr)
    ]
    fleets = [
        [0, 1, 360.0, 300.0, PI, 0, 20.0],                # inbound to planet 1
        [1, 1, 250.0 - 80.0, 300.0, 0.0, 0, 40.0],        # inbound to planet 2 (its own doom)
    ]
    avail_doomed = ev._reachable_friendly_mass(planets, fleets, planets[0], SEAT, threat_eta=20.0)
    # If planet 2 were NOT doomed (drop its attacker), spare would be garr − 0 = 30 too; the point is
    # doomed still contributes its FULL garrison (not garr − 40 = negative → 0).
    assert avail_doomed >= 30.0, f"doomed source must drain fully (>=30), got {avail_doomed}"


def test_reachability_respects_threat_eta():
    # A huge-garrison own planet that is too FAR to arrive before a FAST/near inbound → not counted.
    planets = [
        [1, SEAT, 300.0, 300.0, 8.0, 5.0, 1.0],
        [2, SEAT, 300.0 - 200.0, 300.0, 8.0, 200.0, 1.0],  # far helper
    ]
    near_fast = [[0, 1, 312.0, 300.0, PI, 0, 50.0]]        # very close → tiny threat ETA
    avail = ev._reachable_friendly_mass(planets, near_fast, planets[1 - 1], SEAT,
                                        threat_eta=ev._enemy_threat(planets, near_fast, planets[0], SEAT)[1])
    assert avail < 200.0, f"far helper must not count under a short threat ETA, got {avail}"
    assert ev._threat_class(planets, near_fast, planets[0], SEAT) == 3   # → hopeless


class _Step:
    def __init__(self, planets, fleets=None, action=None):
        self.observation = {"planets": planets, "fleets": fleets or []}
        self.action = action or []


def _row(planets, fleets=None, action=None):
    return [_Step(planets, fleets=fleets, action=action)]


def test_game_conversion_triage_refinements():
    # Timeline:
    # t=1 observes planet 1 neutral, then t=2 captures it.
    # t=3 safely reinforces planet 1 from planet 2.
    # t=4 planet 1 launches an attack, so safe-reinf utility should be "attack".
    # t=5 planet 1 is lost while planet 2 stays ours, so it is a captured-only nonterminal loss.
    p_neutral = [1, -1, 300.0, 300.0, 8.0, 5.0, 2.0]
    p_held = [1, SEAT, 300.0, 300.0, 8.0, 30.0, 2.0]
    p_lost = [1, 1, 300.0, 300.0, 8.0, 30.0, 2.0]
    p_src = [2, SEAT, 250.0, 300.0, 8.0, 50.0, 1.0]
    p_enemy = [3, 1, 360.0, 300.0, 8.0, 0.0, 1.0]
    steps = [
        _row([p_neutral, p_src, p_enemy]),
        _row([p_neutral, p_src, p_enemy]),
        _row([p_held, p_src, p_enemy]),
        _row([p_held, p_src, p_enemy], action=[[2, 0.0, 10]]),  # p2 -> p1 reinforce
        _row([p_held, p_src, p_enemy], action=[[1, 0.0, 5]]),   # p1 -> p3 attack
        _row([p_lost, p_src, p_enemy]),
    ]

    conv = ev.game_conversion(steps, SEAT)
    assert conv["cap_born_class"][0] == 1, conv["cap_born_class"]
    assert conv["cap_born_lost_nt"][0] == 1, conv["cap_born_lost_nt"]
    assert conv["lost_by_class_cap_nt"][0] == 1, conv["lost_by_class_cap_nt"]
    assert conv["reinf_lost_within"] == [10.0, 10.0, 10.0], conv["reinf_lost_within"]
    assert conv["safe_reinf_util"] == [10.0, 0.0, 0.0], conv["safe_reinf_util"]
    assert conv["reinf_to_lost"] == 10.0, conv["reinf_to_lost"]


if __name__ == "__main__":
    test_already_safe()
    test_hopeless_when_no_reachable_friendly()
    test_cheap_save_with_reachable_spare()
    test_doomed_source_drains_fully()
    test_reachability_respects_threat_eta()
    test_game_conversion_triage_refinements()
    print("PASS: reinforce-triage — already-safe / hopeless / cheap-save / doomed-drains-fully / ETA-reachability")
