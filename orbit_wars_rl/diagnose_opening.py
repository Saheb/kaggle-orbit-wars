"""
Diagnose opening paralysis: inspect model action probabilities at each step
of a downloaded LB loss episode to determine if the agent is:
  A) Confident about NOT firing (deliberate passivity)
  B) Confused / fractured logits (exploration failure)
  C) Actually firing but at wrong targets

Usage:
  python3 orbit_wars_rl/diagnose_opening.py \
    --checkpoint <path.pt> \
    --episode /tmp/rev28_losses/78612678.json \
    --steps 30 \
    --target-decode
"""
import argparse, json, sys, torch
import torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orbit_wars_rl.config import Config
from orbit_wars_rl.model import EntityTransformer
from orbit_wars_rl.eval import load_checkpoint
from orbit_wars_rl.features import extract_features
from orbit_wars_rl.action_mask import compute_action_masks

OUR_NAME = "Saheb"


def diagnose(checkpoint, episode_path, n_steps=30, target_decode=False):
    # Load model
    cfg = Config()
    sd, _ = load_checkpoint(checkpoint, cfg)
    model = EntityTransformer(cfg.model)
    model.load_state_dict(sd)
    device = torch.device("cpu")
    model = model.to(device).eval()

    # Load episode
    data = json.load(open(episode_path))
    info = data.get('info', {})
    agents_info = info.get('Agents', [])
    our_idx = next((i for i, a in enumerate(agents_info) if a.get('Name') == OUR_NAME), 0)
    opp_name = agents_info[1-our_idx]['Name'] if len(agents_info) > 1 else '?'

    steps = data['steps']
    ep_id = data.get('id', episode_path)
    n_total = len(steps)

    print(f"\nEpisode {ep_id} vs {opp_name} ({n_total} steps) — player idx={our_idx}")
    print(f"Checkpoint: {Path(checkpoint).name}")
    print(f"{'Step':>5} {'FireP':>7} {'TopFire':>8} {'Entropy':>8} {'Action':>12}  {'Planets us/opp':>14}  {'Diagnosis'}")
    print("─" * 85)

    for i, step in enumerate(steps[:n_steps]):
        try:
            obs_raw = step[our_idx]['observation']
            action_taken = step[our_idx]['action']
        except Exception as e:
            print(f"  step {i}: error reading obs: {e}")
            continue

        # Convert observation dict to our format
        obs = {
            "step": obs_raw.get('step', i),
            "player": our_idx,
            "planets": obs_raw.get('planets', []),
            "fleets": obs_raw.get('fleets', []),
            "angular_velocity": obs_raw.get('angular_velocity', 0.0),
            "initial_planets": obs_raw.get('initial_planets', obs_raw.get('planets', [])),
            "comet_planet_ids": obs_raw.get('comet_planet_ids', []),
        }

        try:
            features = extract_features(obs, our_idx, num_players=2)
            masks = compute_action_masks(obs, our_idx)
        except Exception as e:
            print(f"  step {i}: feature extraction error: {e}")
            continue

        with torch.no_grad():
            outputs = model(
                features["planet_features"].unsqueeze(0).to(device),
                features["fleet_features"].unsqueeze(0).to(device),
                features["global_features"].unsqueeze(0).to(device),
                features["planet_mask"].unsqueeze(0).to(device),
                features["fleet_mask"].unsqueeze(0).to(device),
                fire_mask=masks["fire_mask"].to(device),
                angle_mask=masks["angle_mask"].to(device),
                slot_valid=masks["slot_valid"].to(device),
                owned_indices=masks["owned_indices"].to(device),
                owned_count=masks["owned_count"],
                pairwise_features=features["pairwise_features"].unsqueeze(0).to(device)
                    if features.get("pairwise_features") is not None else None,
            )

        fire_logits = outputs["fire_logits"][0].cpu()  # (max_owned,)
        fire_probs = torch.sigmoid(fire_logits)
        slot_valid = masks["slot_valid"][0]  # squeeze batch dim → (max_owned,)

        # Max fire prob across owned planets
        valid_probs = fire_probs[slot_valid] if slot_valid.any() else fire_probs
        max_fire_p = float(valid_probs.max()) if len(valid_probs) > 0 else 0.0
        mean_fire_p = float(valid_probs.mean()) if len(valid_probs) > 0 else 0.0

        # Entropy of fire decision (treating as independent Bernoullis)
        entropy = float(-torch.mean(
            fire_probs * torch.log(fire_probs + 1e-8) +
            (1 - fire_probs) * torch.log(1 - fire_probs + 1e-8)
        ))

        # Planet counts
        planets = obs_raw.get('planets', [])
        our_p = sum(1 for p in planets if p[1] == our_idx)
        opp_p = sum(1 for p in planets if p[1] == (1-our_idx))

        # Action summary
        action_str = "FIRE" if (action_taken and len(str(action_taken)) > 5) else "no-op"

        # Diagnosis
        if max_fire_p > 0.5:
            diag = "firing ✓"
        elif max_fire_p > 0.2:
            diag = "hesitant"
        elif entropy < 0.1:
            diag = "CONFIDENT NO-FIRE ← paralysis"
        else:
            diag = "confused/low-conf"

        print(f"{i:>5} {max_fire_p:>7.3f} {mean_fire_p:>8.3f} {entropy:>8.4f} {action_str:>12}  {our_p:>5}/{opp_p:<8}  {diag}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--target-decode", action="store_true")
    args = parser.parse_args()
    diagnose(args.checkpoint, args.episode, args.steps, args.target_decode)
