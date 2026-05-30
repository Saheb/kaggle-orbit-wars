"""Replay → BC samples (modern pipeline).

Wraps bc.trajectory_to_training_sample (which produces samples with
pairwise_features and target_target — both required by the per-slot scorer
introduced 2026-05-24, see docs/bugs.md). The older replay_to_bc.py was
written before that change and produced samples missing those fields.

Filtering:
  - 2-player games only (the model is 2P)
  - Strict winner only: skip draws (max reward must be > all others)
  - Drop samples where the winner had no move (action == [])

Usage:
    python replay_bc_v2.py --replay-dir replays --output /tmp/replay_bc.pkl
    python replay_bc_v2.py --replay-dir replays --output /tmp/replay_bc_frac.pkl \
        --ship-bin-mode fraction --min-ship-bin 1
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import pickle
from pathlib import Path

from bc import trajectory_to_training_sample


def winner_index(rewards) -> int | None:
    """Return strict-winner index, or None for draws / missing rewards."""
    if not rewards or any(r is None for r in rewards):
        return None
    mx = max(rewards)
    winners = [i for i, r in enumerate(rewards) if r == mx]
    if len(winners) != 1:
        return None  # draw
    return winners[0]


def extract_winner_trajectories(replay_path: str) -> list[dict]:
    """Yield only the winning player's (obs, action) pairs from a replay.

    Returns list of dicts compatible with bc.trajectory_to_training_sample.
    Skips: 4P games, draws, replays with malformed rewards, steps with no action.
    """
    with open(replay_path) as f:
        data = json.load(f)
    steps = data.get("steps") or []
    if not steps:
        return []
    n_players = len(steps[0])
    if n_players != 2:
        return []  # 2P-only
    w = winner_index(data.get("rewards", []))
    if w is None:
        return []  # draws / missing

    trajs = []
    for t in range(1, len(steps)):
        step = steps[t]
        if w >= len(step):
            continue
        agent_data = step[w]
        action = agent_data.get("action")
        if not action:  # empty or None
            continue
        obs_raw = agent_data.get("observation")
        if not obs_raw or "planets" not in obs_raw:
            continue
        # Make sure player field is set (replays may omit it for non-acting players)
        obs = dict(obs_raw)
        obs["player"] = w
        # Defaults bc.py expects to be present
        obs.setdefault("step", t)
        obs.setdefault("angular_velocity", 0.0)
        obs.setdefault("comet_planet_ids", [])
        obs.setdefault("initial_planets", obs["planets"])
        trajs.append({"obs": obs, "action": action})
    return trajs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay-dir", required=True,
                   help="Directory containing episode-*-replay.json files")
    p.add_argument("--output", required=True, help="Output .pkl path")
    p.add_argument("--glob", default="episode-*-replay.json",
                   help="Replay filename glob (default: episode-*-replay.json)")
    p.add_argument("--ship-bin-mode", choices=["absolute", "fraction"], default="absolute",
                   help="Label ship targets as absolute 32-bin counts or fraction 10-bin sends.")
    p.add_argument("--min-ship-bin", type=int, default=0,
                   help="Only for --ship-bin-mode fraction: clamp labels >= this bin.")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.replay_dir, args.glob)))
    print(f"Found {len(paths)} replays in {args.replay_dir}")

    n_replays_used = 0
    n_replays_skipped = 0
    samples = []
    if args.ship_bin_mode == "fraction":
        from bc_frac import trajectory_to_fraction_sample

    for i, path in enumerate(paths, 1):
        if i % 20 == 0 or i == len(paths):
            print(f"  [{i}/{len(paths)}] used={n_replays_used} "
                  f"skipped={n_replays_skipped} samples={len(samples)}")
        try:
            trajs = extract_winner_trajectories(path)
        except Exception as e:
            n_replays_skipped += 1
            continue
        if not trajs:
            n_replays_skipped += 1
            continue
        n_replays_used += 1
        for traj in trajs:
            if args.ship_bin_mode == "fraction":
                s = trajectory_to_fraction_sample(traj, min_ship_bin=args.min_ship_bin)
            else:
                s = trajectory_to_training_sample(traj)
            if s is not None:
                samples.append(s)

    print()
    print(f"Replays used:     {n_replays_used}")
    print(f"Replays skipped:  {n_replays_skipped} (4P, draws, malformed)")
    print(f"Samples emitted:  {len(samples)}")
    if samples:
        s0 = samples[0]
        print(f"Sample 0 keys:    {sorted(s0.keys())}")
        has_pair = "pairwise_features" in s0
        has_tgt = "target_target" in s0
        print(f"Has pairwise_features: {has_pair}")
        print(f"Has target_target:     {has_tgt}")
        print(f"Ship bin mode:         {args.ship_bin_mode}")
        if args.ship_bin_mode == "fraction":
            print(f"Min ship bin:          {args.min_ship_bin}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(samples, f)
    size_kb = os.path.getsize(args.output) / 1024
    print(f"Saved to {args.output} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
