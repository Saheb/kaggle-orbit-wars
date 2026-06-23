"""Sufficient-commit mask: in torch_env._apply_actions, a NEUTRAL attack launch whose
(ship_count + friendly inbound) can't beat the target's defense (current garrison +
enemy inbound arriving before us) × factor is VETOED. Neutrals DON'T regrow (the engine
applies production only to owner != -1), so there is NO production×ETA term. Enemy
targets are exempt — under-strength attacks on enemies can soften/feint. Reinforces
(own targets) are NOT affected — garrison floor instead.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from torch_env import VecTorchEnv, SHIP_COUNTS
from action_mask import compute_action_masks, actions_from_target_policy

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
    # factor 1.0: ships == garrison (30) on NEUTRAL -> 30 <= 30 -> VETOED, source untouched.
    te = _board(1.0)
    assert _fire(te, BIN_EQ) == 100.0, "ships<=garrison neutral attack must be vetoed (no debit)"


def test_allows_winning_neutral_attack():
    # factor 1.0: ships (50) > garrison (30) -> fires, source debited by 50.
    te = _board(1.0)
    te.planets[0, SRC, 2] = 50.0; te.planets[0, SRC, 3] = 50.0
    te.planets[0, TGT, 2] = 52.0; te.planets[0, TGT, 3] = 50.0
    te.planets[0, TGT, 5] = 30.0
    # BIN 19 = 50 ships, 50 > garrison 30 -> fires
    result = _fire(te, 19)
    assert result == 100.0 - SHIP_COUNTS[19], f"ships>garrison neutral attack must fire, got {result}"


def test_far_neutral_high_prod_still_fires():
    # Neutrals DON'T regrow: a far target with high production rate does NOT gain ships
    # before capture. Garrison 30, send 50 -> fires regardless of distance/prod (the old
    # production×ETA phantom term would have vetoed this — regression guard for that bug).
    te = _board(1.0, target_prod=5)
    te.planets[0, SRC, 2] = 10.0; te.planets[0, SRC, 3] = 10.0
    te.planets[0, TGT, 2] = 90.0; te.planets[0, TGT, 3] = 90.0
    te.planets[0, TGT, 5] = 30.0
    result = _fire(te, 19)
    assert result == 100.0 - SHIP_COUNTS[19], f"far neutral must fire (neutrals don't regrow), got {result}"


def test_lets_underpowered_enemy_attack_fire():
    # factor 1.0: ships == defense (30) on ENEMY -> fires (enemy exempt from veto).
    te = _board(1.0, target_owner=1)
    assert _fire(te, BIN_EQ) == 100.0 - SHIP_COUNTS[BIN_EQ], "ships<=defense ENEMY attack must fire (enemy exempt)"


def test_off_lets_underpowered_attack_fire():
    # factor 0.0 (off): the same ships==defense attack fires -> proves the flag blocks it.
    te = _board(0.0)
    assert _fire(te, BIN_EQ) == 100.0 - SHIP_COUNTS[BIN_EQ], "with mask off the attack must fire"


def test_target_decode_falls_through_to_sufficient_small_neutral():
    """Regression for top-1 target decode + post-hoc veto idling.

    The preferred neutral needs 63 ships but this slot's ship head chooses 35 for it,
    so sufficient-commit must mask that target before argmax and select the lower
    ranked clearable small neutral instead.
    """
    obs = {
        "player_id": 0,
        "step": 0,
        "ship_speed": 6.0,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 30.0, 50.0, 3.0, 100.0, 1.0],
            [1, -1, 70.0, 50.0, 3.0, 63.0, 1.0],
            [2, -1, 50.0, 70.0, 3.0, 18.0, 1.0],
        ],
        "fleets": [],
    }
    masks = compute_action_masks(obs, player=0)
    slots = masks["slot_valid"].shape[1]
    n_planets = len(obs["planets"])
    n_bins = len(SHIP_COUNTS)

    fire_logits = torch.full((1, slots, n_planets), -20.0)
    target_logits = torch.full((1, slots, n_planets), -20.0)
    ship_logits = torch.full((1, slots, n_planets, n_bins), -20.0)

    target_logits[0, 0, 1] = 10.0  # top target, but under-committed
    target_logits[0, 0, 2] = 9.0   # fallback target, sufficient
    fire_logits[0, 0, 1] = 10.0
    fire_logits[0, 0, 2] = 10.0
    ship_logits[0, 0, 1, 17] = 10.0  # 35 ships <= 63 neutral defense -> mask
    ship_logits[0, 0, 2, 13] = 10.0  # 19 ships > 18 neutral defense -> fire

    moves = actions_from_target_policy(
        fire_logits,
        target_logits,
        ship_logits,
        masks,
        obs,
        player=0,
        sufficient_commit_factor=1.0,
    )

    assert len(moves) == 1
    assert moves[0][0] == 0
    assert moves[0][2] == SHIP_COUNTS[13]


if __name__ == "__main__":
    test_vetoes_underpowered_neutral_attack()
    print("PASS vetoes_underpowered_neutral_attack")
    test_allows_winning_neutral_attack()
    print("PASS allows_winning_neutral_attack")
    test_far_neutral_high_prod_still_fires()
    print("PASS far_neutral_high_prod_still_fires")
    test_lets_underpowered_enemy_attack_fire()
    print("PASS lets_underpowered_enemy_attack_fire")
    test_off_lets_underpowered_attack_fire()
    print("PASS off_lets_underpowered_attack_fire")
    print("All sufficient-commit tests passed (arrival-aware, neutral-only, enemy exempt)")
