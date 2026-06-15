"""Unit test for the reinforce ping-pong diagnostic (_reinf_reciprocity).

A temporal A->B then B->A reinforce loop (ships oscillating between two own planets) is pure
waste and is NOT caught by a same-turn role mutex. The metric counts reinforces whose nearest
reverse edge lands within 1/2/3 steps (cumulative). Validated against Saheb's replay slice.

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_reinf_reciprocity.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval import _reinf_reciprocity


def test_pingpong_slice():
    # Saheb's observed loop: 10->22 @90/94/98, 22->10 @92/96/100 (period 2).
    edges = [(90, 10, 22), (92, 22, 10), (94, 10, 22),
             (96, 22, 10), (98, 10, 22), (100, 22, 10)]
    r = _reinf_reciprocity(edges)
    # every edge except the last (100, no reverse after) has its reverse exactly 2 steps later
    assert r == [0, 5, 5], f"expected [0,5,5], got {r}"
    print(f"ok: ping-pong slice -> within1/2/3 = {r}")


def test_distances_and_empty():
    assert _reinf_reciprocity([]) == [0, 0, 0]
    # reverse at +1, +3, and a never-reversed edge
    edges = [(10, 1, 2), (11, 2, 1),         # d=1
             (20, 3, 4), (23, 4, 3),         # d=3
             (30, 5, 6)]                      # no reverse
    r = _reinf_reciprocity(edges)
    # (10):d1 -> all three; (11):reverse 1->2? none after 11 -> skip; (20):d3 -> recip3 only;
    # (23):reverse 3->4 after 23? none -> skip; (30): none
    assert r == [1, 1, 2], f"expected [1,1,2], got {r}"
    # nearest-reverse only (not double-counted): A->B at t with two reverses picks the closest
    edges2 = [(0, 1, 2), (1, 2, 1), (2, 2, 1)]
    assert _reinf_reciprocity(edges2) == [1, 1, 1], "should use nearest reverse (d=1), count once"
    print("ok: distance buckets + empty + nearest-reverse")


if __name__ == "__main__":
    test_pingpong_slice()
    test_distances_and_empty()
    print("ALL PASS")
