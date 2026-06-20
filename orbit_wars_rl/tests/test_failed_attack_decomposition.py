"""Unit tests for eval failed-attack decomposition.

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_failed_attack_decomposition.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import eval as ev

SEAT = 0


class _Step:
    def __init__(self, planets, fleets=None, action=None):
        self.observation = {"planets": planets, "fleets": fleets or []}
        self.action = action or []


def _row(planets, fleets=None, action=None):
    return [_Step(planets, fleets=fleets, action=action)]


def _neutral_case(src_ships, other_ships, target_ships=20.0, sent=10, captured=False):
    src = [1, SEAT, 0.0, 0.0, 8.0, float(src_ships), 1.0]
    other = [2, SEAT, 20.0, 0.0, 8.0, float(other_ships), 1.0]
    tgt = [3, -1, 100.0, 0.0, 8.0, float(target_ships), 1.0]
    planets = [src, other, tgt]
    return ev._attack_failure_bucket(planets, [], src, tgt, SEAT, sent, captured)


def test_attack_failure_buckets():
    assert _neutral_case(src_ships=50, other_ships=0, sent=10) == "single"
    assert _neutral_case(src_ships=10, other_ships=50, sent=10) == "wrong-src"
    assert _neutral_case(src_ships=10, other_ships=15, sent=10) == "aggregate"
    assert _neutral_case(src_ships=30, other_ships=30, target_ships=100, sent=10) == "unafford"
    assert _neutral_case(src_ships=50, other_ships=0, sent=25) == "sufficient"
    assert _neutral_case(src_ships=50, other_ships=0, sent=10, captured=True) is None


def test_redundant_bucket_uses_decision_time_inbound():
    src = [1, SEAT, 0.0, 0.0, 8.0, 50.0, 1.0]
    tgt = [3, -1, 100.0, 0.0, 8.0, 20.0, 1.0]
    planets = [src, tgt]
    fleets = [[0, SEAT, 80.0, 0.0, 0.0, 1, 25.0]]

    assert ev._attack_failure_bucket(planets, fleets, src, tgt, SEAT, sent=5, captured_soon=False) == "redundant"


def test_reachable_drainable_agg_tightens_raw_aggregate():
    tgt = [3, -1, 100.0, 0.0, 8.0, 20.0, 1.0]
    near_a = [1, SEAT, 0.0, 0.0, 8.0, 10.0, 1.0]
    near_b = [2, SEAT, 20.0, 0.0, 8.0, 15.0, 1.0]
    avail, nsrc = ev._reachable_drainable_attack_mass([near_a, near_b, tgt], [], tgt, SEAT, window=80)
    assert avail >= 25.0 and nsrc == 2, (avail, nsrc)

    far_b = [2, SEAT, -2000.0, 0.0, 8.0, 15.0, 1.0]
    avail_far, nsrc_far = ev._reachable_drainable_attack_mass([near_a, far_b, tgt], [], tgt, SEAT, window=80)
    assert avail_far == 10.0 and nsrc_far == 1, (avail_far, nsrc_far)

    threatened_a = [1, SEAT, 0.0, 0.0, 8.0, 10.0, 1.0]
    fleets = [[0, 1, 12.0, 0.0, 3.14159265, 9, 8.0]]
    avail_threat, _ = ev._reachable_drainable_attack_mass([threatened_a, near_b, tgt], fleets, tgt, SEAT, window=80, beta=0.0)
    assert avail_threat == 15.0, avail_threat


def test_game_conversion_routes_failed_attack_bucket():
    src = [1, SEAT, 0.0, 0.0, 8.0, 50.0, 1.0]
    tgt = [3, -1, 100.0, 0.0, 8.0, 20.0, 1.0]
    planets = [src, tgt]
    steps = [
        _row(planets),
        _row(planets, action=[[1, 0.0, 10]]),  # source had enough, sent too little, target never flips
        _row(planets),
    ]

    conv = ev.game_conversion(steps, SEAT)
    assert conv["atk_n_ph"] == [1, 0, 0], conv["atk_n_ph"]
    assert conv["atkfail_n_ph"] == [1, 0, 0], conv["atkfail_n_ph"]
    single_idx = ev._ATK_FAIL_LABELS.index("single")
    assert conv["atkfail_bucket_ph"][single_idx] == [1, 0, 0], conv["atkfail_bucket_ph"]


def test_game_conversion_routes_reachable_agg2():
    src = [1, SEAT, 0.0, 0.0, 8.0, 10.0, 1.0]
    # helper sits OFF the src->tgt firing line (perp 18 > radius) so the launch resolves to the
    # target rather than physically colliding with the helper, but is still a reachable drainable
    # source for the aggregate-failure bucket.
    helper = [2, SEAT, 20.0, 18.0, 8.0, 15.0, 1.0]
    tgt = [3, -1, 100.0, 0.0, 8.0, 20.0, 1.0]
    planets = [src, tgt, helper]
    steps = [
        _row(planets),
        _row(planets, action=[[1, 0.0, 10]]),  # no single source covers 21, aggregate can
        _row(planets),
    ]

    conv = ev.game_conversion(steps, SEAT)
    agg_idx = ev._ATK_FAIL_LABELS.index("aggregate")
    assert conv["atkfail_bucket_ph"][agg_idx] == [1, 0, 0], conv["atkfail_bucket_ph"]
    assert conv["atkagg_reach_ph"] == [1, 0, 0], conv["atkagg_reach_ph"]


if __name__ == "__main__":
    test_attack_failure_buckets()
    test_redundant_bucket_uses_decision_time_inbound()
    test_reachable_drainable_agg_tightens_raw_aggregate()
    test_game_conversion_routes_failed_attack_bucket()
    test_game_conversion_routes_reachable_agg2()
    print("PASS: failed-attack decomposition")
