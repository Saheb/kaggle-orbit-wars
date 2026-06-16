"""Eval-side wiring test for the reverse-edge cooldown: actions_from_target_policy must veto a
reinforce whose reverse edge is on cooldown, and must record an allowed reinforce into the
per-game state (so the next step blocks the reverse). Reuses the canonical reinforce_cooldown.py.

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_reverse_edge_cooldown_eval.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from action_mask import actions_from_target_policy, SHIP_COUNTS

PLAYER = 0
# planets: A(id0, ours), B(id1, ours) only — so B's sole non-source target is the own A
# (a reinforce). A veto therefore leaves B with NO legal target => no move from B (clean signal,
# vs B redirecting to an attack if an enemy were present). 7 cols [id,owner,x,y,r,ships,prod].
PLANETS = [
    [0, 0, 100.0, 300.0, 8.0, 20.0, 1.0],
    [1, 0, 300.0, 300.0, 8.0, 20.0, 1.0],
]
OBS = {"step": 6, "player": PLAYER, "planets": PLANETS, "fleets": [],
       "angular_velocity": 0.0}
K = 3
NBINS = len(SHIP_COUNTS)


def _logits():
    # Only slot 1 (source B=id1) fires, aiming at slot-target index 0 (A) → a B->A reinforce.
    fire = torch.tensor([[-10.0, 10.0]])
    tl = torch.full((1, 2, 2), -10.0); tl[0, 1, 0] = 10.0       # slot1 (B) -> planet index 0 (A)
    sl = torch.full((1, 2, NBINS), -10.0); sl[0, 1, 0] = 10.0   # ship bin 0 (small, valid)
    masks = {"owned_indices": torch.tensor([0, 1]),
             "max_ships": torch.tensor([[20.0, 20.0]]),
             "owned_count": 2}
    return fire, tl, sl, masks


def _run(cooldown_last, K_):
    fire, tl, sl, masks = _logits()
    return actions_from_target_policy(
        fire, tl, sl, masks, OBS, PLAYER,
        allow_reinforce=True,
        reverse_edge_cooldown=K_, cooldown_last=cooldown_last, cooldown_step=OBS["step"],
    )


def test_reverse_edge_vetoed():
    # We reinforced A->B at step 5; now (step 6) B->A must be vetoed (no move emitted).
    last = {(0, 1): 5}
    moves = _run(last, K)
    assert all(int(m[0]) != 1 for m in moves), f"B->A reinforce must be vetoed on cooldown, got {moves}"


def test_allowed_reinforce_is_recorded():
    # No prior reverse edge → B->A fires AND gets recorded so a later A->B would block.
    last = {}
    moves = _run(last, K)
    assert any(int(m[0]) == 1 for m in moves), f"B->A should fire when not on cooldown, got {moves}"
    assert (1, 0) in last, f"executed B->A must be recorded into cooldown state, got {last}"


def test_off_when_k_zero():
    last = {(0, 1): 5}
    moves = _run(last, 0)            # cooldown disabled → B->A fires despite the (0,1) record
    assert any(int(m[0]) == 1 for m in moves), "K=0 must disable the cooldown entirely"


if __name__ == "__main__":
    test_reverse_edge_vetoed()
    test_allowed_reinforce_is_recorded()
    test_off_when_k_zero()
    print("PASS: reverse-edge cooldown (eval) — reverse veto, allowed-record, K=0 off")
