"""Phase 5 NO_OP target column — rollout / chain-rule contract.

NO_OP (target idx == max_planets) is the always-legal "do nothing" action. This
test pins the three guarantees the rest of the pipeline relies on:
  1. the env's get_features target_mask carries a NO_OP column == slot_valid;
  2. sampling a NO_OP target forces fire=0 (so its fire log-prob is well-defined
     and ship log-prob is excluded — the §4 chain-rule contract);
  3. an all-NO_OP action step launches no fleets in the env.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from model import EntityTransformer
from torch_env import VecTorchEnv
from train_torch import sample_action_batched


def test_target_mask_has_noop_column():
    cfg = Config()
    env = VecTorchEnv(num_envs=8, num_players=2, device="cpu", episode_steps=500)
    env.reset(seeds=list(range(8)))
    feats = env.get_features(0, max_planets=cfg.env.max_planets, max_fleets=128)
    assert feats["target_mask"].shape[-1] == cfg.env.max_planets + 1
    # NO_OP column (last) is legal exactly for valid source slots.
    assert torch.equal(feats["target_mask"][..., -1], feats["slot_valid"])
    print("test_target_mask_has_noop_column: PASS")


def test_sampled_noop_forces_no_fire():
    cfg = Config()
    m = EntityTransformer(cfg.model); m.eval()
    env = VecTorchEnv(num_envs=8, num_players=2, device="cpu", episode_steps=500)
    env.reset(seeds=list(range(8)))
    feats = env.get_features(0, max_planets=cfg.env.max_planets, max_fleets=128)
    with torch.no_grad():
        out = m(feats["planet_features"], feats["fleet_features"], feats["global_features"],
                feats["planet_mask"], feats["fleet_mask"], fire_mask=feats["fire_mask"],
                slot_valid=feats["slot_valid"], owned_indices=feats["owned_indices"],
                pairwise_features=feats["pairwise_features"])
    fire_a, _ang, _ship, target_a, _lpf, _lps, _lpt = sample_action_batched(
        out, feats["fire_mask"], feats.get("target_mask"))
    noop = (target_a == cfg.env.max_planets)
    assert int(fire_a[noop].sum()) == 0, "NO_OP picks must force fire=0"
    print("test_sampled_noop_forces_no_fire: PASS")


def test_all_noop_step_launches_nothing():
    cfg = Config()
    m = EntityTransformer(cfg.model); m.eval()
    env = VecTorchEnv(num_envs=8, num_players=2, device="cpu", episode_steps=500)
    env.reset(seeds=list(range(8)))
    feats = env.get_features(0, max_planets=cfg.env.max_planets, max_fleets=128)
    with torch.no_grad():
        out = m(feats["planet_features"], feats["fleet_features"], feats["global_features"],
                feats["planet_mask"], feats["fleet_mask"], fire_mask=feats["fire_mask"],
                slot_valid=feats["slot_valid"], owned_indices=feats["owned_indices"],
                pairwise_features=feats["pairwise_features"])
    fire_a, ang, ship, target_a, *_ = sample_action_batched(out, feats["fire_mask"], feats.get("target_mask"))
    NO_OP = cfg.env.max_planets
    noop_act = torch.stack(
        [torch.zeros_like(fire_a), ang, ship, torch.full_like(target_a, NO_OP)], dim=-1)
    before = env.fleet_alive.sum().item()
    env.step({0: noop_act, 1: noop_act})
    after = env.fleet_alive.sum().item()
    assert after <= before, "no new fleets may launch under an all-NO_OP step"
    print("test_all_noop_step_launches_nothing: PASS")


if __name__ == "__main__":
    print("Running Phase 5 NO_OP rollout tests...\n")
    test_target_mask_has_noop_column()
    test_sampled_noop_forces_no_fire()
    test_all_noop_step_launches_nothing()
    print("\nAll NO_OP rollout tests passed!")
