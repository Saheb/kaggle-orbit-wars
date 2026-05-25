"""Ship-bin decode semantics for absolute and fraction heads."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from action_mask import _ship_bin_to_count
from torch_env import MAX_OWNED, VecTorchEnv


def test_action_mask_absolute_decode_uses_ship_counts():
    assert _ship_bin_to_count(0, 100, mode="absolute") == 1
    assert _ship_bin_to_count(9, 100, mode="absolute") == 10
    assert _ship_bin_to_count(31, 100, mode="absolute") == 100


def test_action_mask_fraction_decode_uses_sendable_ships():
    assert _ship_bin_to_count(0, 100, mode="fraction") == 10
    assert _ship_bin_to_count(4, 100, mode="fraction") == 50
    assert _ship_bin_to_count(9, 100, mode="fraction") == 100


def test_torch_env_fraction_decode_leaves_one_ship_behind():
    env = VecTorchEnv(num_envs=1, num_players=2, device="cpu", ship_bin_mode="fraction")
    env.reset(seeds=[0])

    owned_idx, slot_valid = env.owned_indices_for(0)
    slot = int(torch.where(slot_valid[0])[0][0].item())
    pidx = int(owned_idx[0, slot].item())
    env.planets[0, pidx, 5] = 21.0

    actions = torch.zeros(1, MAX_OWNED, 3, dtype=torch.long)
    actions[0, slot, 0] = 1
    actions[0, slot, 1] = 0
    actions[0, slot, 2] = 9  # 100% of max sendable

    env._apply_actions(actions, owner_id=0)

    assert env.planets[0, pidx, 5].item() == 1.0


def test_torch_env_target_decode_aims_at_selected_planet():
    env = VecTorchEnv(
        num_envs=1,
        num_players=2,
        device="cpu",
        ship_bin_mode="absolute",
        action_decode="target",
    )
    env.reset(seeds=[0])

    # Static, easy geometry: source at (20, 20), neutral target due east.
    env.planet_alive.zero_()
    env.planets.zero_()
    env.init_planets.zero_()
    env.planets[0, 0] = torch.tensor([0, 0, 20.0, 20.0, 2.0, 20.0, 1.0])
    env.planets[0, 1] = torch.tensor([1, -1, 95.0, 20.0, 2.0, 5.0, 1.0])
    env.init_planets.copy_(env.planets)
    env.planet_alive[0, :2] = True
    env._precompute_orbital_params()

    actions = torch.zeros(1, MAX_OWNED, 4, dtype=torch.long)
    actions[0, 0, 0] = 1
    actions[0, 0, 1] = 36  # would be north if angle-bin decoded
    actions[0, 0, 2] = 9
    actions[0, 0, 3] = 1

    env._apply_actions(actions, owner_id=0)

    fleet_angle = env.fleets[0, 0, 4].item()
    assert abs(fleet_angle) < 1e-4


def test_torch_env_features_include_non_owned_target_mask():
    env = VecTorchEnv(num_envs=1, num_players=2, device="cpu")
    env.reset(seeds=[0])
    feats = env.get_features(0)
    owned_idx, slot_valid = env.owned_indices_for(0)
    slot = int(torch.where(slot_valid[0])[0][0].item())
    mine = int(owned_idx[0, slot].item())

    assert not feats["target_mask"][0, slot, mine].item()
    assert feats["target_mask"][0, slot].any().item()
