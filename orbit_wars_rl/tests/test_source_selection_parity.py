"""Train/eval/export parity for GARRISON-ranked source selection (>16 owned).

When >MAX_OWNED planets are owned, the 16 source slots are the highest-garrison owned planets
(ties -> lowest array index). This MUST be identical across the three paths or it reintroduces a
train/eval mismatch:
  - VecTorchEnv.owned_indices_for  (training)
  - features.extract_features      (eval feature path)
  - action_mask.compute_action_masks (eval/export mask path)

Compared by PLANET ID per slot (not raw index): torch indexes the padded tensor, the obs paths
index the alive-only planet list — different index spaces, but slot k must resolve to the SAME
planet (so the model sees identical input). Tiebreak orders agree because to_legacy_obs preserves
tensor order. Covers >16 owned + an exact garrison tie (the index-tiebreak path).

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_source_selection_parity.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

from torch_env import VecTorchEnv, to_legacy_obs, MAX_OWNED
from features import extract_features
from action_mask import compute_action_masks


def _board(n_owned=20):
    """One env: n_owned player-0 planets with varied (and one tied) garrisons + a few enemies."""
    env = VecTorchEnv(num_envs=1, num_players=2, device="cpu", action_decode="angle")
    env.reset(seeds=[0])
    env.planet_alive[:] = False
    env.fleet_alive[:] = False
    env.fleets[:] = 0
    # Distinct ids, garrisons designed so the top-16 set + order is unambiguous, plus ONE tie
    # (two planets at 50 ships) to exercise the lowest-index tiebreak.
    ships = [5, 90, 12, 50, 7, 200, 33, 50, 8, 120, 3, 77, 19, 140, 2, 66, 41, 9, 150, 4]
    for i in range(n_owned):
        env.planets[0, i, 0] = 100 + i          # planet id
        env.planets[0, i, 1] = 0                # owner = player 0
        env.planets[0, i, 5] = ships[i]         # garrison
        env.planets[0, i, 2] = 20.0 + i         # x (nonzero positions)
        env.planets[0, i, 3] = 30.0
        env.planet_alive[0, i] = True
    # a couple of enemy planets (must never occupy our source slots)
    for j, idx in enumerate((30, 31)):
        env.planets[0, idx, 0] = 200 + j
        env.planets[0, idx, 1] = 1
        env.planets[0, idx, 5] = 999            # huge garrison, but enemy -> excluded
        env.planets[0, idx, 2] = 70.0
        env.planets[0, idx, 3] = 70.0
        env.planet_alive[0, idx] = True
    return env, ships


def _pids_torch(env, player):
    oi, sv = env.owned_indices_for(player)
    pids = []
    for k in range(MAX_OWNED):
        if bool(sv[0, k]):
            pids.append(int(env.planets[0, int(oi[0, k]), 0].item()))
    return pids


def _pids_from_obs(obs, owned_indices, owned_count):
    return [int(obs["planets"][int(owned_indices[k])][0]) for k in range(owned_count)]


def test_source_selection_parity():
    env, ships = _board(n_owned=20)
    player = 0
    obs = to_legacy_obs(env, 0, player)

    pids_torch = _pids_torch(env, player)
    feats = extract_features(obs, player)
    pids_feat = _pids_from_obs(obs, feats["owned_indices"].tolist(), int(feats["owned_count"]))
    masks = compute_action_masks(obs, player)
    pids_mask = _pids_from_obs(obs, masks["owned_indices"].tolist()
                               if torch.is_tensor(masks["owned_indices"]) else masks["owned_indices"],
                               int(masks["owned_count"]))

    assert len(pids_torch) == MAX_OWNED, f"expected 16 valid slots, got {len(pids_torch)}"
    assert pids_torch == pids_feat == pids_mask, (
        f"source-selection parity MISMATCH:\n torch={pids_torch}\n feat ={pids_feat}\n mask ={pids_mask}")

    # Ground truth: top-16 by (-round(ships), array index). ids are 100+i.
    expected = [100 + i for i, _ in sorted(enumerate(ships), key=lambda t: (-t[1], t[0]))[:MAX_OWNED]]
    assert pids_torch == expected, f"not top-16-by-garrison:\n got={pids_torch}\n exp={expected}"
    # the tie (idx 3 & 7, both 50 ships): 103 must come before 107 (lower index wins)
    assert pids_torch.index(103) < pids_torch.index(107), "garrison tie not broken by lowest index"
    # no enemy planet (id 200/201) leaked into our slots
    assert 200 not in pids_torch and 201 not in pids_torch, "enemy planet selected as source"
    print(f"ok: train/eval/export agree on top-16-by-garrison (20 owned). slots={pids_torch}")


def test_no_op_under_cap():
    """<=MAX_OWNED owned: every owned planet gets a slot, all three paths agree."""
    env, _ = _board(n_owned=10)
    player = 0
    obs = to_legacy_obs(env, 0, player)
    pids_torch = _pids_torch(env, player)
    feats = extract_features(obs, player)
    pids_feat = _pids_from_obs(obs, feats["owned_indices"].tolist(), int(feats["owned_count"]))
    assert len(pids_torch) == 10 and sorted(pids_torch) == [100 + i for i in range(10)]
    assert pids_torch == pids_feat, f"under-cap mismatch: {pids_torch} vs {pids_feat}"
    print("ok: <=16 owned — all owned planets slotted, paths agree")


if __name__ == "__main__":
    test_source_selection_parity()
    test_no_op_under_cap()
    print("ALL PASS")
