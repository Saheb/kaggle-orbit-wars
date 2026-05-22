"""Download replays from top Kaggle agents and extract (obs, action) training data.

Usage:
    python replay_to_bc.py --episodes 20 --output bc_top_agent_data.pkl
    python replay_to_bc.py --replay-dir replays/ --output bc_top_agent_data.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from features import extract_features
from action_mask import compute_action_masks, NUM_ANGLE_BINS, ANGLE_BIN_WIDTH, SHIP_COUNTS
from model import NUM_SHIP_BINS
from model import EntityTransformer
from config import Config


def download_episodes(submission_id: int, num_episodes: int, output_dir: str):
    """Download replays for a given submission ID."""
    import subprocess
    os.makedirs(output_dir, exist_ok=True)
    
    kaggle_cmd = os.environ.get("KAGGLE_CMD", "kaggle")
    result = subprocess.run(
        [kaggle_cmd, "competitions", "episodes", str(submission_id), "-v"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error listing episodes: {result.stderr}")
        return []
    
    lines = result.stdout.strip().split("\n")
    episode_ids = []
    for line in lines[1:]:  # skip header
        parts = line.split(",")
        if len(parts) >= 1 and parts[0].strip().isdigit():
            episode_ids.append(int(parts[0].strip()))
    
    print(f"Found {len(episode_ids)} episodes for submission {submission_id}")
    
    downloaded = []
    for i, eid in enumerate(episode_ids[:num_episodes]):
        out_path = os.path.join(output_dir, f"episode-{eid}-replay.json")
        if os.path.exists(out_path):
            downloaded.append(out_path)
            continue
        print(f"  Downloading episode {eid} ({i+1}/{min(len(episode_ids), num_episodes)})...")
        subprocess.run(
            [kaggle_cmd, "competitions", "replay", str(eid), "-p", output_dir],
            capture_output=True, text=True,
        )
        if os.path.exists(out_path):
            downloaded.append(out_path)
    
    return downloaded


def _find_angle_bin(angle_rad: float) -> int:
    return int(float(angle_rad) / ANGLE_BIN_WIDTH) % NUM_ANGLE_BINS


def _find_ship_bin(ships: int) -> int:
    best_bin, best_diff = 0, float("inf")
    for b in range(NUM_SHIP_BINS):
        diff = abs(SHIP_COUNTS[b] - int(ships))
        if diff < best_diff:
            best_bin, best_diff = b, diff
    return best_bin


def extract_bc_samples_from_replay(
    replay_path: str,
    target_agent_indices: Optional[list[int]] = None,
    num_players: int = 4,
) -> list[dict]:
    """Extract BC training samples from a replay JSON file.

    For each step where a target agent takes actions, create a sample:
        - observation features (planet, fleet, global)
        - action masks
        - BC targets (fire, angle, ship bins)

    Args:
        replay_path: Path to replay JSON file
        target_agent_indices: Which agents to learn from (default: winner or top scorer)
        num_players: Expected number of players
    """
    with open(replay_path) as f:
        data = json.load(f)

    steps = data["steps"]
    rewards = data.get("rewards", [])
    if not steps or len(steps[0]) == 0:
        return []

    actual_players = len(steps[0])
    
    # Determine target agents: winners (positive reward)
    if target_agent_indices is None:
        if rewards:
            max_reward = max(rewards) if rewards else 0
            target_agent_indices = [i for i, r in enumerate(rewards) if r is not None and r == max_reward]
        else:
            target_agent_indices = list(range(actual_players))

    samples = []

    for t in range(1, len(steps)):
        step_data = steps[t]
        for agent_idx in target_agent_indices:
            if agent_idx >= len(step_data):
                continue

            agent_data = step_data[agent_idx]
            action = agent_data.get("action")
            obs_raw = agent_data.get("observation", {})
            reward = agent_data.get("reward")

            if not action and action != []:
                continue

            # Convert observation to our format
            obs = _replay_obs_to_dict(obs_raw, agent_idx)
            if obs is None:
                continue

            # Compute features and masks
            features = extract_features(obs, agent_idx, num_players=actual_players)
            masks = compute_action_masks(obs, agent_idx)

            # Convert action to BC targets
            fire_target, angle_target, ship_target = _action_to_bc_targets(
                action, obs, masks
            )

            samples.append({
                "planet_features": features["planet_features"],
                "fleet_features": features["fleet_features"],
                "global_features": features["global_features"],
                "planet_mask": features["planet_mask"],
                "fleet_mask": features["fleet_mask"],
                "fire_mask": masks["fire_mask"][0],
                "angle_mask": masks["angle_mask"][0],
                "slot_valid": masks["slot_valid"][0],
                "owned_indices": masks["owned_indices"],
                "owned_count": masks["owned_count"],
                "fire_target": fire_target,
                "angle_target": angle_target,
                "ship_target": ship_target,
                "reward": reward,
                "agent_idx": agent_idx,
                "step": t,
            })

    return samples


def _replay_obs_to_dict(obs_raw: dict, player: int) -> Optional[dict]:
    """Convert replay observation to our standard format."""
    if not obs_raw or "planets" not in obs_raw:
        return None

    planets = []
    for p in obs_raw.get("planets", []):
        planets.append([
            int(p[0]),   # id
            int(p[1]),   # owner
            float(p[2]),  # x
            float(p[3]),  # y
            float(p[4]),  # radius
            float(p[5]),  # ships
            float(p[6]),  # production
        ])

    fleets = []
    for f in obs_raw.get("fleets", []):
        fleets.append([
            int(f[0]),   # id
            int(f[1]),   # owner
            float(f[2]),  # x
            float(f[3]),  # y
            float(f[4]),  # angle
            int(f[5]),    # from_planet_id
            float(f[6]),  # ships
        ])

    initial_planets = []
    for p in obs_raw.get("initial_planets", obs_raw.get("planets", [])):
        initial_planets.append([
            int(p[0]), float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6])
        ] if len(p) >= 7 else list(p))

    return {
        "step": int(obs_raw.get("step", 0)),
        "player": player,
        "planets": planets,
        "fleets": fleets,
        "angular_velocity": float(obs_raw.get("angular_velocity", 0.0)),
        "initial_planets": obs_raw.get("initial_planets", planets),
        "comet_planet_ids": list(obs_raw.get("comet_planet_ids", [])),
    }


def _action_to_bc_targets(action, obs: dict, masks: dict) -> tuple:
    """Convert env action list to per-slot BC targets."""
    max_owned = 10
    planets = obs["planets"]
    n_owned = masks["owned_count"]
    owned_indices = masks["owned_indices"].cpu().numpy()

    fire = torch.zeros(max_owned, dtype=torch.long)
    angle = torch.zeros(max_owned, dtype=torch.long)
    ship = torch.zeros(max_owned, dtype=torch.long)

    my_planets = [(i, p) for i, p in enumerate(planets) if p[1] == obs["player"]]
    pid_to_slot = {}
    for slot in range(min(n_owned, max_owned)):
        pidx = int(owned_indices[slot])
        if pidx < len(planets):
            pid_to_slot[int(planets[pidx][0])] = slot

    max_ships = masks["max_ships"].cpu().numpy().squeeze(0)

    for move in action:
        if not isinstance(move, list) or len(move) < 3:
            continue
        from_id = int(move[0])
        angle_rad = float(move[1])
        ship_count = int(move[2])

        slot = pid_to_slot.get(from_id)
        if slot is None:
            continue

        fire[slot] = 1
        angle[slot] = _find_angle_bin(angle_rad)
        ship[slot] = _find_ship_bin(ship_count)

    return fire, angle, ship


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes to download")
    parser.add_argument("--submission-id", type=int, default=None,
                        help="Kaggle submission ID to get episodes from")
    parser.add_argument("--replay-dir", type=str, default="replays",
                        help="Directory with replay JSON files")
    parser.add_argument("--output", type=str, default="bc_top_agent_data.pkl",
                        help="Output pickle file for BC samples")
    parser.add_argument("--target-agents", type=str, default="winners",
                        help="'winners', 'all', or comma-separated indices")
    args = parser.parse_args()

    all_samples = []

    # Download episodes if submission ID provided
    if args.submission_id:
        replay_files = download_episodes(args.submission_id, args.episodes, args.replay_dir)
    else:
        replay_dir = Path(args.replay_dir)
        replay_files = sorted(replay_dir.glob("episode-*-replay.json"))
        print(f"Found {len(replay_files)} replays in {args.replay_dir}")

    for i, replay_path in enumerate(replay_files):
        rp = Path(replay_path)
        print(f"Processing {rp.name} ({i+1}/{len(replay_files)})...")
        try:
            if args.target_agents == "winners":
                samples = extract_bc_samples_from_replay(str(replay_path))
            elif args.target_agents == "all":
                with open(replay_path) as f:
                    data = json.load(f)
                n = len(data["steps"][0])
                samples = extract_bc_samples_from_replay(
                    str(replay_path), target_agent_indices=list(range(n))
                )
            else:
                indices = [int(x) for x in args.target_agents.split(",")]
                samples = extract_bc_samples_from_replay(
                    str(replay_path), target_agent_indices=indices
                )
            all_samples.extend(samples)
        except Exception as e:
            print(f"  Error processing {replay_path.name}: {e}")

    print(f"\nTotal BC samples extracted: {len(all_samples)}")

    # Save
    with open(args.output, "wb") as f:
        pickle.dump(all_samples, f)
    print(f"Saved to {args.output} ({os.path.getsize(args.output) / 1024:.1f} KB)")