"""Regression tests for the small set of passive, decision-facing eval metrics."""

import math
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import (  # noqa: E402
    _already_covered_neutral,
    _relative_economy_milestones,
    _relative_economy_snapshot,
    add_conversion,
    new_conversion_acc,
)


def _planet(pid, owner, x, ships, production, radius=1.0):
    return [pid, owner, x, 0.0, radius, ships, production]


def _fleet(fid, owner, x, angle, ships):
    return [fid, owner, x, 0.0, angle, 1.0, ships]


def _state(obs):
    return SimpleNamespace(observation=obs)


def test_relative_economy_is_a_paired_difference_and_counts_fleets():
    obs = {
        "planets": [
            _planet(0, 0, 0.0, 10, 2),
            _planet(1, 1, 20.0, 20, 5),
            _planet(2, -1, 10.0, 100, 5),
        ],
        "fleets": [_fleet(0, 0, 2.0, 0.0, 7), _fleet(1, 1, 18.0, math.pi, 3)],
    }

    assert _relative_economy_snapshot(obs, 0) == (-3.0, -6.0)
    assert _relative_economy_snapshot(obs, 1) == (3.0, 6.0)


def test_relative_economy_carries_terminal_state_to_later_milestones():
    initial = {"planets": [_planet(0, 0, 0.0, 10, 2), _planet(1, 1, 20.0, 10, 2)], "fleets": []}
    terminal = {"planets": [_planet(0, 1, 0.0, 12, 2), _planet(1, 1, 20.0, 20, 2)], "fleets": []}
    steps = [[_state(initial), _state(initial)], [_state(terminal), _state(terminal)]]

    snapshots = _relative_economy_milestones(steps, 0)

    assert snapshots == {32: (-4.0, -32.0), 50: (-4.0, -32.0), 100: (-4.0, -32.0)}


def test_already_covered_neutral_requires_capture_mass_and_no_visible_enemy_inbound():
    target = _planet(1, -1, 10.0, 14, 1)
    planets = [target]

    assert _already_covered_neutral(
        planets, [_fleet(0, 0, 0.0, 0.0, 15)], target, seat=0)
    assert not _already_covered_neutral(
        planets, [_fleet(0, 0, 0.0, 0.0, 14)], target, seat=0)
    assert not _already_covered_neutral(
        planets,
        [_fleet(0, 0, 0.0, 0.0, 15), _fleet(1, 1, 20.0, math.pi, 1)],
        target,
        seat=0,
    )


def test_new_fields_are_outcome_split_and_old_panel_records_remain_readable():
    conv = {
        "captures": 0, "attack_launches": 0, "reinforce_launches": 0, "attack_ships": 0,
        "end_planets": 0, "atk_early": 0, "caps_early": 0, "atk_mid": 0, "caps_mid": 0,
        "reinf_early": 0, "lost_caps": 0, "hold_durations": [], "glen": 1,
        "launch_states": 0, "launch_count": 0, "fire_steps": 0, "fire_frac_sum": 0.0,
        "launches_ph": [0, 0, 0], "ship1_ph": [0, 0, 0], "ship_ph_sum": [0, 0, 0],
        "p16": None, "p32": None, "p50": None, "p100": None,
        "neutral_launches_u100": 2, "neutral_ships_u100": 30,
        "already_covered_neutral_launches_u100": 1, "already_covered_neutral_ships_u100": 20,
        "prod_delta_32": -1, "prod_delta_50": 2, "prod_delta_100": 3,
        "material_delta_32": -10, "material_delta_50": 20, "material_delta_100": 30,
    }
    acc = new_conversion_acc()
    add_conversion(acc, conv, won=False, material=0)

    assert acc["already_covered_neutral_ships_u100_lost"] == 20
    assert acc["prod_delta_50_lost"] == [2]
    assert acc["prod_delta_50_won"] == []

    old_conv = {k: v for k, v in conv.items()
                if not k.startswith(("neutral_", "already_covered_", "prod_delta_", "material_delta_"))}
    add_conversion(new_conversion_acc(), old_conv, won=True, material=10)
