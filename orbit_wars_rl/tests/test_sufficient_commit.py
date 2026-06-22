"""Sufficient-commit mask (arrival-aware): in torch_env._apply_actions, a NEUTRAL
attack launch whose (ship_count + friendly inbound) can't beat the target's PROJECTED
defense at arrival (current garrison + production×ETA + enemy inbound arriving before
us) × factor is VETOED. Enemy targets are exempt — under-strength attacks on enemies
can soften/feint. Reinforces (own targets) are NOT affected — garrison floor instead.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from orbit_wars_rl.torch_env import VecTorchEnv, SHIP_COUNTS

SRC, TGT = 0, 1            # planet indices: source (mine) and the attack target
DEF = 30                  # target defense; SHIP_COUNTS[16]=30 (== def), [17]=35 (> def)
BIN_EQ, BIN_GT = 16, 17


def _board(factor, target_owner=-1, target_prod=1):
    """1 env: player-0 source planet (100 ships) + a target with DEF ships.
    target_owner: -1 = neutral (default), 1 = enemy. target_prod: production rate."""
    te = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                     action_decode="target", sufficient_commit_factor=factor)
    te.reset([7])
    te.planet_alive[0, :] = True
    te.planets[0, :, 1] = -1            # all neutral first
    te.planets[0, SRC, 1] = 0           # source owned by player 0
    te.planets[0, SRC, 5] = 100.0
    te.planets[0, TGT, 1] = target_owner
    te.planets[0, TGT, 5] = float(DEF)
    te.planets[0, TGT, 6] = float(target_prod)
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


def test_vetoes_underpowered_neutral_attack():
    # factor 1.0: ships == defense (30) on NEUTRAL, close range (ETA=1, prod=1 -> proj 31).
    # 30 <= 31 -> VETOED, source untouched.
    te = _board(1.0)
    assert _fire(te, BIN_EQ) == 100.0, "ships<=projected_defense neutral attack must be vetoed (no debit)"


def test_allows_winning_neutral_attack():
    # factor 1.0: ships (50) vs projected defense. Close target (ETA=1, prod=1 -> proj 31).
    # 50 > 31 -> fires, source debited by 50.
    te = _board(1.0)
    # Place source and target adjacent (ETA=1, projected = 30 + 1*1 = 31)
    te.planets[0, SRC, 2] = 50.0; te.planets[0, SRC, 3] = 50.0
    te.planets[0, TGT, 2] = 52.0; te.planets[0, TGT, 3] = 50.0
    te.planets[0, TGT, 5] = 30.0; te.planets[0, TGT, 6] = 1.0
    # BIN 19 = 50 ships, speed ~3.2, ETA=1, projected=31, 50 > 31 -> fires
    result = _fire(te, 19)
    assert result == 100.0 - SHIP_COUNTS[19], f"ships>projected_defense neutral attack must fire, got {result}"


def test_vetoes_far_neutral_production_outgrows_attack():
    # Far target with high production: 30 ships at ETA~19, prod=5 -> projected 125.
    # Ship bin 19 = 50 ships. 50 <= 125 -> VETOED (production outgrows the attack).
    te = _board(1.0, target_prod=5)
    te.planets[0, SRC, 2] = 10.0; te.planets[0, SRC, 3] = 10.0
    te.planets[0, TGT, 2] = 90.0; te.planets[0, TGT, 3] = 90.0
    te.planets[0, TGT, 5] = 30.0
    result = _fire(te, 19)
    assert result == 100.0, f"far neutral with high prod must be vetoed (50 <= projected), got {result}"


def test_lets_underpowered_enemy_attack_fire():
    # factor 1.0: ships == defense (30) on ENEMY -> fires (enemy exempt from veto).
    te = _board(1.0, target_owner=1)
    assert _fire(te, BIN_EQ) == 100.0 - SHIP_COUNTS[BIN_EQ], "ships<=defense ENEMY attack must fire (enemy exempt)"


def test_off_lets_underpowered_attack_fire():
    # factor 0.0 (off): the same ships==defense attack fires -> proves the flag blocks it.
    te = _board(0.0)
    assert _fire(te, BIN_EQ) == 100.0 - SHIP_COUNTS[BIN_EQ], "with mask off the attack must fire"


if __name__ == "__main__":
    test_vetoes_underpowered_neutral_attack()
    print("PASS vetoes_underpowered_neutral_attack")
    test_allows_winning_neutral_attack()
    print("PASS allows_winning_neutral_attack")
    test_vetoes_far_neutral_production_outgrows_attack()
    print("PASS vetoes_far_neutral_production_outgrows_attack")
    test_lets_underpowered_enemy_attack_fire()
    print("PASS lets_underpowered_enemy_attack_fire")
    test_off_lets_underpowered_attack_fire()
    print("PASS off_lets_underpowered_attack_fire")
    print("All sufficient-commit tests passed (arrival-aware, neutral-only, enemy exempt)")
