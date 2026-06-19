"""Phase 5 shared wave primitive tests.

Run: orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_wave_primitives.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wave_primitives import (
    HOLD_DOOMED,
    HOLD_HOLDABLE,
    HOLD_SAFE,
    WAVE_MARGIN,
    attack_remaining,
    choose_attack_anchor,
    classify_holds,
    defense_remaining,
    eta_between_planets,
    ready_now,
    ready_wave_quota,
    ship_choice_for_quota,
)


def p(pid, owner, x, y=0.0, ships=0.0, prod=0.0, r=3.0):
    return [pid, owner, x, y, r, ships, prod]


def f(fid, owner, x, y=0.0, angle=0.0, ships=1.0):
    return [fid, owner, x, y, angle, -1, ships]


def test_ship_choice_uses_actual_count_to_land_on_time():
    src = p(1, 0, 0.0, ships=100)
    tgt = p(2, 1, 100.0, ships=10)
    tau = eta_between_planets(src, tgt, 100)

    choice = ship_choice_for_quota(src, tgt, safe_sendable=100, quota=10, tau=tau, tol=0.25)
    assert choice.viable
    assert choice.chosen_count > 10, "quota alone would arrive late; count must be raised for timing"
    assert abs(choice.arrival_tau - tau) <= 0.25

    assert ready_now(src, tgt, safe_sendable=100, tau=tau, tol=0.25)
    assert not ready_now(src, tgt, safe_sendable=100, tau=tau - 5.0, tol=0.25)


def test_sticky_anchor_ignores_submaterial_trickle():
    tgt = p(10, 1, 100.0, ships=8)
    planets = [p(1, 0, 0.0, ships=20), tgt]
    fleets = [
        f(1, 0, 96.0, ships=1),    # early but sub-material
        f(2, 0, 80.0, ships=6),    # later material wave
    ]

    anchor = choose_attack_anchor(tgt, planets, fleets, attacker=0, safe_sendable_by_pid={})
    assert anchor is not None
    assert anchor.mode == "sticky"
    assert anchor.tau > 10.0, f"submaterial trickle incorrectly anchored the wave: {anchor}"
    floor, cover, remaining = attack_remaining(tgt, planets, fleets, attacker=0, tau=anchor.tau)
    assert abs(anchor.floor - floor) < 1e-6 and abs(anchor.cover - cover) < 1e-6
    assert remaining < floor


def test_fresh_anchor_uses_farthest_needed_source():
    near = p(1, 0, 90.0, ships=20)
    far = p(2, 0, 0.0, ships=30)
    tgt = p(10, 1, 100.0, ships=40)
    planets = [near, far, tgt]
    safe = {1: 20.0, 2: 30.0}

    anchor = choose_attack_anchor(tgt, planets, [], attacker=0, safe_sendable_by_pid=safe)
    assert anchor is not None
    assert anchor.mode == "fresh"
    assert anchor.source_ids == (1, 2)
    assert abs(anchor.tau - eta_between_planets(far, tgt, 30.0)) < 1e-6


def test_ready_wave_quota_sums_to_remaining_without_overcommit():
    tgt = p(10, 1, 100.0, ships=20)
    sources = [p(i, 0, 0.0, ships=30) for i in range(1, 5)]
    safe = {i: 30.0 for i in range(1, 5)}
    tau = eta_between_planets(sources[0], tgt, 30.0)

    plan = ready_wave_quota(sources, tgt, safe, remaining=40.0, tau=tau, tol=0.5)
    assert plan.crosses_if_all_ready_send
    assert abs(plan.ready_safe - 120.0) < 1e-6
    assert abs(sum(plan.quotas.values()) - 40.0) < 1e-6
    assert all(abs(q - 10.0) < 1e-6 for q in plan.quotas.values())


def test_defense_remaining_counts_friendly_inbound_once():
    hold = p(10, 0, 100.0, ships=5, prod=0)
    planets = [hold]
    enemy = f(1, 1, 90.0, ships=12)
    friend = f(2, 0, 95.0, ships=4)
    tau = eta_between_planets(p(99, 1, 90.0), hold, 12)

    floor, cover, remaining = defense_remaining(hold, planets, [enemy, friend], player=0, tau=tau)
    assert abs(floor - (12.0 + WAVE_MARGIN)) < 1e-6
    assert abs(cover - 9.0) < 1e-6
    assert abs(remaining - 5.0) < 1e-6


def test_hold_class_safe_holdable_doomed_and_reserved_pool():
    helper = p(1, 0, 0.0, ships=30, prod=1)
    holdable = p(2, 0, 10.0, ships=5, prod=1)
    doomed = p(3, 0, 100.0, ships=5, prod=0)
    planets = [helper, holdable, doomed]
    fleets = [
        f(1, 1, 5.0, ships=12),     # threatens holdable
        f(2, 1, 95.0, ships=20),    # threatens doomed
    ]

    infos = classify_holds(planets, fleets, player=0, current_step=0, episode_steps=80)
    assert infos[1].hold_class == HOLD_SAFE
    assert infos[2].hold_class == HOLD_HOLDABLE
    assert infos[3].hold_class == HOLD_DOOMED
    assert infos[2].remaining0 > 0
    assert infos[2].claims.get(1, 0.0) > 0, "holdable planet should reserve helper mass"
    assert infos[1].reserved_for_defense == infos[2].claims[1]
    assert infos[1].safe_sendable < infos[1].base_safe_sendable
    assert abs(infos[3].safe_sendable - 5.0) < 1e-6


if __name__ == "__main__":
    test_ship_choice_uses_actual_count_to_land_on_time()
    test_sticky_anchor_ignores_submaterial_trickle()
    test_fresh_anchor_uses_farthest_needed_source()
    test_ready_wave_quota_sums_to_remaining_without_overcommit()
    test_defense_remaining_counts_friendly_inbound_once()
    test_hold_class_safe_holdable_doomed_and_reserved_pool()
    print("PASS: Phase 5 wave primitives")
