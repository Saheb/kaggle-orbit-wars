"""Pure-state unit test for the reverse-edge reinforce cooldown canonical rule
(reinforce_cooldown.py). Locks the edge/window/ownership/boundary semantics that train,
eval, and export all must share.

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_reinforce_cooldown.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reinforce_cooldown import is_blocked, record, on_ownership_loss, reset

K = 3
A, B, C = 10, 11, 12   # planet ids


def test_ordered_edges():
    last = {}
    record(last, 0, A, B)                      # reinforce A->B at t=0
    assert is_blocked(last, 1, B, A, K), "B->A (reverse) must be blocked"
    assert not is_blocked(last, 1, A, B, K), "A->B again must NOT be blocked (same direction)"
    assert not is_blocked(last, 1, B, C, K), "B->C must NOT be blocked (different edge)"
    assert not is_blocked(last, 1, C, A, K), "C->A must NOT be blocked (C->A's reverse A->C never fired)"


def test_window_expiry():
    last = {}
    record(last, 0, A, B)
    assert is_blocked(last, K, B, A, K), "B->A blocked at the edge of the window (t-0 == K)"
    assert not is_blocked(last, K + 1, B, A, K), "B->A free once the K-step window passes"


def test_ownership_change_resets_edge():
    # Stale cooldown must NOT block reinforcing a just-recaptured planet.
    last = {}
    record(last, 0, A, B)
    on_ownership_loss(last, A)                  # we lose A (then retake it)
    assert not is_blocked(last, 1, B, A, K), "losing A must clear the A<->B cooldown"
    # Symmetric: losing the OTHER endpoint also clears.
    last = {}
    record(last, 0, A, B)
    on_ownership_loss(last, B)
    assert not is_blocked(last, 1, B, A, K), "losing B must clear the A<->B cooldown too"


def test_episode_reset():
    last = {}
    record(last, 0, A, B)
    reset(last)
    assert not is_blocked(last, 1, B, A, K), "episode reset must clear all state"
    assert last == {}


if __name__ == "__main__":
    test_ordered_edges()
    test_window_expiry()
    test_ownership_change_resets_edge()
    test_episode_reset()
    print("PASS: reinforce-cooldown canonical rule — ordered edges, window, ownership reset, boundary")
