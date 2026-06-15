"""Unit test for floor-aware attack aggregation diagnostics.

Run:  PYTHONPATH=orbit_wars_rl python3 orbit_wars_rl/tests/test_attack_aggregation_metrics.py
"""
import math
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval import game_conversion


def _steps(sent_a, sent_b, target_ships=60):
    planets = [
        [1, 0, 0.0, 0.0, 5.0, 100.0, 0.0],
        [2, 0, 0.0, 10.0, 5.0, 100.0, 0.0],
        [3, -1, 100.0, 0.0, 5.0, float(target_ships), 0.0],
    ]
    angle_a = 0.0
    angle_b = math.atan2(0.0 - 10.0, 100.0 - 0.0)
    obs = {"planets": planets, "fleets": []}
    return [
        [SimpleNamespace(observation=obs, action=[])],
        [SimpleNamespace(observation=obs, action=[[1, angle_a, sent_a], [2, angle_b, sent_b]])],
    ]


def test_split_source_essential_group():
    conv = game_conversion(_steps(40, 40), seat=0)
    assert conv["atk_turns"] == 1
    assert conv["atk_agg_turns"] == 1
    assert conv["atk_agg_moves"] == 2
    assert conv["atk_agg_groups"] == 1
    assert conv["atk_agg_essential"] == 1
    assert conv["atk_agg_solo"] == 0
    assert conv["atk_agg_under"] == 0
    assert conv["atk_agg_groups_ph"] == [1, 0, 0]
    assert conv["atk_agg_essential_ph"] == [1, 0, 0]


def test_split_source_under_floor_group():
    conv = game_conversion(_steps(20, 20), seat=0)
    assert conv["atk_agg_groups"] == 1
    assert conv["atk_agg_essential"] == 0
    assert conv["atk_agg_solo"] == 0
    assert conv["atk_agg_under"] == 1
    assert conv["atk_agg_under_ph"] == [1, 0, 0]


if __name__ == "__main__":
    test_split_source_essential_group()
    test_split_source_under_floor_group()
    print("ALL PASS")
