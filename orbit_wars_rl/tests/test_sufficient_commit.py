"""Sufficient-commit mask: in torch_env._apply_actions, an ATTACK launch whose ship
count <= target's current defense × sufficient_commit_factor is VETOED (no fleet
leaves, source ships unchanged); a launch that beats the defense fires normally; and
with the factor OFF even the under-strength attack fires (proving the flag blocks it).

Reinforces (own targets) are NOT affected by this mask — they keep the garrison floor.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from orbit_wars_rl.torch_env import VecTorchEnv, SHIP_COUNTS

SRC, TGT = 0, 1            # planet indices: source (mine) and the neutral attack target
DEF = 30                  # target defense; SHIP_COUNTS[16]=30 (== def), [17]=35 (> def)
BIN_EQ, BIN_GT = 16, 17


def _board(factor):
    """1 env: player-0 source planet (100 ships) + a neutral target with DEF ships."""
    te = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                     action_decode="target", sufficient_commit_factor=factor)
    te.reset([7])
    te.planet_alive[0, :] = True
    te.planets[0, :, 1] = -1            # all neutral first
    te.planets[0, SRC, 1] = 0           # source owned by player 0
    te.planets[0, SRC, 5] = 100.0
    te.planets[0, TGT, 1] = -1          # target neutral
    te.planets[0, TGT, 5] = float(DEF)
    return te


def _fire(te, ship_bin):
    """Fire from the slot owning SRC at TGT with ship_bin; return source ships after."""
    owned_idx, slot_valid = te.owned_indices_for(0)
    slot = next(s for s in range(owned_idx.shape[1])
                if slot_valid[0, s] and int(owned_idx[0, s]) == SRC)
    act = torch.zeros((1, owned_idx.shape[1], 4), dtype=torch.long)
    act[0, slot, 0] = 1            # fire
    act[0, slot, 2] = ship_bin    # ship bin
    act[0, slot, 3] = TGT         # target idx
    te._apply_actions(act, owner_id=0)
    return float(te.planets[0, SRC, 5])


def test_sufficient_commit_vetoes_underpowered_attack():
    # factor 1.0: ships == defense (30) cannot take it -> VETOED, source untouched.
    te = _board(1.0)
    assert _fire(te, BIN_EQ) == 100.0, "ships<=defense attack must be vetoed (no debit)"


def test_sufficient_commit_allows_winning_attack():
    # factor 1.0: ships (35) > defense (30) -> fires, source debited by 35.
    te = _board(1.0)
    assert _fire(te, BIN_GT) == 100.0 - SHIP_COUNTS[BIN_GT], "ships>defense attack must fire"


def test_off_lets_underpowered_attack_fire():
    # factor 0.0 (off): the same ships==defense attack fires -> proves the flag blocks it.
    te = _board(0.0)
    assert _fire(te, BIN_EQ) == 100.0 - SHIP_COUNTS[BIN_EQ], "with mask off the attack must fire"


if __name__ == "__main__":
    test_sufficient_commit_vetoes_underpowered_attack()
    test_sufficient_commit_allows_winning_attack()
    test_off_lets_underpowered_attack_fire()
    print("sufficient-commit mask tests passed")
