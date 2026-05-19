#!/usr/bin/env python3
"""End-to-end smoke test: env reset -> feature extraction -> model forward -> action."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from config import Config
from env import OrbitWarsEnv
from features import extract_features
from action_mask import compute_action_masks, actions_from_policy
from model import EntityTransformer, count_params

def main():
    print("=== Orbit Wars RL Smoke Test ===\n")

    cfg = Config()
    device = torch.device(cfg.device)
    print(f"Device: {device}")

    # 1. Environment
    print("\n--- 1. Environment Reset ---")
    env = OrbitWarsEnv(num_players=2, seed=42, debug=False)
    obs = env.reset(seed=42)
    print(f"  Step: {obs['step']}")
    print(f"  Planets: {len(obs['planets'])}")
    print(f"  Fleets: {len(obs['fleets'])}")
    print(f"  Angular velocity: {obs['angular_velocity']:.4f}")
    print(f"  Player: {obs['player']}")

    # 2. Feature extraction
    print("\n--- 2. Feature Extraction ---")
    features = extract_features(obs, player=0, num_players=2)
    print(f"  Planet features: {features['planet_features'].shape}")
    print(f"  Fleet features: {features['fleet_features'].shape}")
    print(f"  Global features: {features['global_features'].shape}")
    print(f"  Planet mask: {features['planet_mask'].sum()}/{features['planet_mask'].shape[0]}")
    print(f"  Fleet mask: {features['fleet_mask'].sum()}/{features['fleet_mask'].shape[0]}")
    print(f"  Owned planets: {features['owned_count']}")

    # 3. Action masks
    print("\n--- 3. Action Masks ---")
    masks = compute_action_masks(obs, player=0)
    print(f"  Fire mask: {masks['fire_mask'].sum()}/{masks['fire_mask'].shape[1]} can fire")
    print(f"  Angle mask: {masks['angle_mask'].shape}")
    total_legal_angles = masks['angle_mask'].sum().item()
    total_possible = masks['angle_mask'].shape[1] * masks['angle_mask'].shape[2]
    print(f"  Legal angles: {total_legal_angles}/{total_possible}")
    print(f"  Max ships per owned planet: {masks['max_ships'].squeeze(0).tolist()}")

    # 4. Model forward pass
    print("\n--- 4. Model Forward Pass ---")
    model = EntityTransformer(cfg.model).to(device)
    print(f"  Params: {count_params(model):,}")

    planet_feats = features["planet_features"].unsqueeze(0).to(device)
    fleet_feats = features["fleet_features"].unsqueeze(0).to(device)
    global_feats = features["global_features"].unsqueeze(0).to(device)
    planet_mask = features["planet_mask"].unsqueeze(0).to(device)
    fleet_mask = features["fleet_mask"].unsqueeze(0).to(device)

    fire_mask = masks["fire_mask"].to(device)
    angle_mask = masks["angle_mask"].to(device)
    slot_valid = masks["slot_valid"].to(device)
    owned_indices = masks["owned_indices"].to(device)

    with torch.no_grad():
        outputs = model(
            planet_feats, fleet_feats, global_feats,
            planet_mask, fleet_mask,
            fire_mask=fire_mask, angle_mask=angle_mask,
            slot_valid=slot_valid, owned_indices=owned_indices,
            owned_count=masks["owned_count"],
        )

    print(f"  Fire logits: {outputs['fire_logits'].shape}")
    print(f"  Angle logits: {outputs['angle_logits'].shape}")
    print(f"  Ship logits: {outputs['ship_logits'].shape}")
    print(f"  Value: {outputs['value'].item():.4f}")

    # 5. Action to environment
    print("\n--- 5. Policy -> Action -> Env Step ---")
    env_actions = actions_from_policy(
        outputs["fire_logits"].cpu(),
        outputs["angle_logits"].cpu(),
        outputs["ship_logits"].cpu(),
        {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()},
        obs, 0,
    )
    print(f"  Actions: {env_actions}")

    obs2, reward, done, info = env.step(env_actions)
    print(f"  After step: {len(obs2['planets'])} planets, {len(obs2['fleets'])} fleets")
    print(f"  Done: {done}, Reward: {reward}")

    # 6. Run a full random game
    print("\n--- 6. Full Game with Random Actions ---")
    env2 = OrbitWarsEnv(num_players=2, seed=123, debug=False)
    obs_game = env2.reset(seed=123)
    steps = 0
    total_reward = 0
    while not env2.done and steps < 500:
        # Random actions from model
        features_game = extract_features(obs_game, player=0, num_players=2)
        masks_game = compute_action_masks(obs_game, player=0)

        pf = features_game["planet_features"].unsqueeze(0).to(device)
        ff = features_game["fleet_features"].unsqueeze(0).to(device)
        gf = features_game["global_features"].unsqueeze(0).to(device)
        pm = features_game["planet_mask"].unsqueeze(0).to(device)
        fm = features_game["fleet_mask"].unsqueeze(0).to(device)
        fm_game = masks_game["fire_mask"].to(device)
        am_game = masks_game["angle_mask"].to(device)
        sv_game = masks_game["slot_valid"].to(device)
        oi_game = masks_game["owned_indices"].to(device)

        with torch.no_grad():
            out_game = model(pf, ff, gf, pm, fm,
                           fire_mask=fm_game, angle_mask=am_game,
                           slot_valid=sv_game, owned_indices=oi_game,
                           owned_count=masks_game["owned_count"])

        actions_game = actions_from_policy(
            out_game["fire_logits"].cpu(),
            out_game["angle_logits"].cpu(),
            out_game["ship_logits"].cpu(),
            {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks_game.items()},
            obs_game, 0,
        )

        obs_game, r, done, _ = env2.step(actions_game)
        steps += 1

    final_material = env2.compute_material(0)
    final_reward = env2._compute_reward()
    print(f"  Game finished in {steps} steps")
    print(f"  Final material (player 0): {final_material:.0f}")
    print(f"  Final reward: {final_reward}")

    print("\n=== ALL SMOKE TESTS PASSED ===")


if __name__ == "__main__":
    main()