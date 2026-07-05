"""Redundant-target mask: in torch_env._apply_actions, a NEUTRAL attack launch whose
IN-FLIGHT friendly mass ALONE already clears the target (garrison + enemy inbound) ×
factor is VETOED — the launch would only reinforce an already-won capture, so the policy
is pushed to retarget the spare. Static floor (no reactive term). Enemy/own targets and
the no-inbound case are untouched.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from torch_env import VecTorchEnv, SHIP_COUNTS

SRC, TGT = 0, 1
DEF = 30


def _board(factor, with_inflight_friendly):
    """SRC (p0, 100 ships) at (10,10); neutral TGT (30 ships) at (50,50). Optionally a
    friendly fleet of 35 ships already inbound to TGT (35 > 30 garrison ⇒ redundant)."""
    te = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                     action_decode="target", redundant_target_factor=factor)
    te.reset([7])
    te.planet_alive[0, :] = True
    te.planets[0, :, 1] = -1
    te.planets[0, SRC, 1] = 0
    te.planets[0, SRC, 5] = 100.0
    te.planets[0, SRC, 2] = 10.0; te.planets[0, SRC, 3] = 10.0
    te.planets[0, TGT, 1] = -1
    te.planets[0, TGT, 5] = float(DEF)
    te.planets[0, TGT, 2] = 50.0; te.planets[0, TGT, 3] = 50.0
    te.fleet_alive[0, :] = False
    if with_inflight_friendly:
        ang = math.atan2(50.0 - 46.0, 50.0 - 46.0)        # heading from (46,46) at TGT
        te.fleets[0, 0, 0] = 999                            # id
        te.fleets[0, 0, 1] = 0                              # owner = player 0 (friendly)
        te.fleets[0, 0, 2] = 46.0; te.fleets[0, 0, 3] = 46.0
        te.fleets[0, 0, 4] = ang
        te.fleets[0, 0, 5] = SRC                            # from
        te.fleets[0, 0, 6] = 35.0                           # ships > garrison 30
        te.fleet_alive[0, 0] = True
    return te


def _fire(te, ship_bin=19):
    owned_idx, slot_valid = te.owned_indices_for(0)
    slot = next(s for s in range(owned_idx.shape[1])
                if slot_valid[0, s] and int(owned_idx[0, s]) == SRC)
    act = torch.zeros((1, owned_idx.shape[1], 4), dtype=torch.long)
    act[0, slot, 0] = 1
    act[0, slot, 2] = ship_bin
    act[0, slot, 3] = TGT
    te._apply_actions(act, owner_id=0)
    return float(te.planets[0, SRC, 5])


def test_vetoes_redundant_neutral_attack():
    # factor 1.0 + a friendly 35-ship fleet inbound to a 30-garrison neutral → redundant → VETOED.
    te = _board(1.0, with_inflight_friendly=True)
    assert _fire(te) == 100.0, "redundant neutral attack (friendly inbound already clears) must be vetoed"


def test_allows_attack_when_no_inflight_friendly():
    # Same factor, but NO friendly inbound → not redundant → fires (source debited).
    te = _board(1.0, with_inflight_friendly=False)
    assert _fire(te) == 100.0 - SHIP_COUNTS[19], "non-redundant neutral attack must fire"


def test_off_lets_redundant_attack_fire():
    # factor 0.0 (off): the redundant attack fires → proves the flag is what blocks it.
    te = _board(0.0, with_inflight_friendly=True)
    assert _fire(te) == 100.0 - SHIP_COUNTS[19], "with mask off the redundant attack must fire"


if __name__ == "__main__":
    test_vetoes_redundant_neutral_attack(); print("PASS vetoes_redundant_neutral_attack")
    test_allows_attack_when_no_inflight_friendly(); print("PASS allows_attack_when_no_inflight_friendly")
    test_off_lets_redundant_attack_fire(); print("PASS off_lets_redundant_attack_fire")
    print("All redundant-target tests passed")
