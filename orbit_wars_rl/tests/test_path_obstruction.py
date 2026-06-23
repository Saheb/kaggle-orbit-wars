"""Path-obstruction decode mask: a launch travels on a fixed heading and hits any planet
its path crosses. If a bigger, UNCAPTURABLE planet screens the intended target, the fleet
dies on the screen and the target is never reached. The decode mask zeroes such (slot,target)
pairs pre-argmax so the head falls through to a reachable target.

Geometry from replay submission_analysis/81509243.json (we are player 1, lost): home P3
(corner, 12 ships) keeps firing at cheap neutral P19 (11 ships), but the 86-ship neutral P11
sits on the straight path and annihilates every 12-ship launch — P19 never captured, we never
expand, eliminated at step 99.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from action_mask import _path_obstruction_blocked

# [id, owner, x, y, radius, ships, production]
P3  = [3,  1, 26.4, 3.5,  1.00, 12, 1]   # ours (source)
P19 = [19, -1, 9.2, 22.0, 2.39, 11, 4]   # cheap neutral target
P11 = [11, -1, 20.3, 11.7, 2.10, 86, 3]  # 86-ship neutral screening the path
PLANETS = [P3, P19, P11]


def test_screened_target_is_blocked():
    # Path P3->P19 crosses P11 (cost 87) and our full garrison (12) can't clear it -> blocked.
    assert _path_obstruction_blocked(
        P3[2], P3[3], P19[2], P19[3], PLANETS, src_id=3, tgt_id=19, player=1, max_send=12)


def test_direct_target_not_blocked():
    # P11 itself as the target: nothing between us and it -> not blocked (the sufficient-commit
    # veto, not this mask, handles whether 12 < 87 is enough).
    assert not _path_obstruction_blocked(
        P3[2], P3[3], P11[2], P11[3], PLANETS, src_id=3, tgt_id=11, player=1, max_send=12)


def test_capturable_blocker_not_masked():
    # If the screen were weak (5 ships, cost 6 < 12) it's a stepping stone, not a waste -> allowed.
    weak = [[3, 1, 26.4, 3.5, 1.0, 12, 1],
            [19, -1, 9.2, 22.0, 2.39, 11, 4],
            [11, -1, 20.3, 11.7, 2.10, 5, 3]]
    assert not _path_obstruction_blocked(
        26.4, 3.5, 9.2, 22.0, weak, src_id=3, tgt_id=19, player=1, max_send=12)


def test_blocker_off_the_line_not_masked():
    # Move the screen far off the segment -> path is clear.
    off = [[3, 1, 26.4, 3.5, 1.0, 12, 1],
           [19, -1, 9.2, 22.0, 2.39, 11, 4],
           [11, -1, 45.0, 45.0, 2.10, 86, 3]]
    assert not _path_obstruction_blocked(
        26.4, 3.5, 9.2, 22.0, off, src_id=3, tgt_id=19, player=1, max_send=12)


# ---- torch_env training-side veto (parity with the decode mask) ----
import torch
from torch_env import VecTorchEnv, SHIP_COUNTS

SRC, TGT, BLK = 0, 1, 2
BIN12 = SHIP_COUNTS.index(12)  # send 12 ships (replay's garrison)


def _obstruction_env(block_ships):
    """1 env, target-decode, path_obstruction_mask ON. Source (12 ships) -> cheap neutral TGT
    (11 ships) with a neutral BLK on the straight path. Returns source ships after the launch."""
    te = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                     action_decode="target", path_obstruction_mask=True)
    te.reset([7])
    te.planet_alive[0, :] = False
    te.planet_alive[0, :3] = True
    te.planets[0, :, 1] = -1
    # source (ours), 12 ships — like the replay corner planet
    te.planets[0, SRC, 1] = 0
    te.planets[0, SRC, 2] = 26.4; te.planets[0, SRC, 3] = 3.5
    te.planets[0, SRC, 4] = 1.0;  te.planets[0, SRC, 5] = 12.0; te.planets[0, SRC, 6] = 1
    # cheap neutral target (11 ships) — 12 would take it if reachable
    te.planets[0, TGT, 2] = 9.2;  te.planets[0, TGT, 3] = 22.0
    te.planets[0, TGT, 4] = 2.39; te.planets[0, TGT, 5] = 11.0; te.planets[0, TGT, 6] = 4
    # screen on the path
    te.planets[0, BLK, 2] = 20.3; te.planets[0, BLK, 3] = 11.7
    te.planets[0, BLK, 4] = 2.10; te.planets[0, BLK, 5] = float(block_ships); te.planets[0, BLK, 6] = 3
    owned_idx, slot_valid = te.owned_indices_for(0)
    slot = next(s for s in range(owned_idx.shape[1])
                if slot_valid[0, s] and int(owned_idx[0, s]) == SRC)
    act = torch.zeros((1, owned_idx.shape[1], 4), dtype=torch.long)
    act[0, slot, 0] = 1; act[0, slot, 2] = BIN12; act[0, slot, 3] = TGT
    te._apply_actions(act, owner_id=0)
    return float(te.planets[0, SRC, 5])


def test_torch_veto_blocks_screened_launch():
    # Strong screen (86, cost 87 > 12 garrison): launch vetoed, source ships untouched (no debit).
    assert _obstruction_env(86) == 12.0, "screened launch must be vetoed (no debit) in training"


def test_torch_fires_when_screen_capturable():
    # Weak screen (5, cost 6 < 12): not a waste -> launch fires, source debited.
    assert _obstruction_env(5) < 12.0, "capturable screen must NOT block the launch"


if __name__ == "__main__":
    test_screened_target_is_blocked();        print("PASS screened_target_is_blocked")
    test_direct_target_not_blocked();          print("PASS direct_target_not_blocked")
    test_capturable_blocker_not_masked();      print("PASS capturable_blocker_not_masked")
    test_blocker_off_the_line_not_masked();    print("PASS blocker_off_the_line_not_masked")
    test_torch_veto_blocks_screened_launch();  print("PASS torch_veto_blocks_screened_launch")
    test_torch_fires_when_screen_capturable(); print("PASS torch_fires_when_screen_capturable")
    print("All path-obstruction tests passed")
