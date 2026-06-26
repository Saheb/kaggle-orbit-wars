"""Regression checks for conversion-BC dataset helpers."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orbit_wars_rl.scripts.build_conversion_bc import _capture_cost, _max_sendable_ships


def test_max_sendable_ships_uses_full_source_count():
    assert _max_sendable_ships(10) == 10
    assert _max_sendable_ships(1) == 1
    assert _max_sendable_ships(0) == 0
    print("test_max_sendable_ships_uses_full_source_count: PASS")


def test_capture_cost_allows_full_commit_on_neutral_ten():
    target = [7, -1, 40.0, 40.0, 1.5, 9, 2]  # neutral needs 10 ships
    required = _capture_cost(target, player_slot=0)
    assert required == 10
    assert required <= _max_sendable_ships(10)
    print("test_capture_cost_allows_full_commit_on_neutral_ten: PASS")


if __name__ == "__main__":
    print("Running build_conversion_bc tests...\n")
    test_max_sendable_ships_uses_full_source_count()
    test_capture_cost_allows_full_commit_on_neutral_ten()
    print("\nAll build_conversion_bc tests passed!")
