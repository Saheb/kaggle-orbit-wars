"""Build a curated replay-supervised BC dataset.

This is the pure-supervised branch: clone selected winning seats from strong
replays, rebalance away from idle no-op frames, and emit the standard bc.py
sample format.

Timing matters: the action recorded at steps[t] was chosen from the observation
at steps[t - 1], so every sample below pairs obs[t - 1] with action[t].
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_rl.bc import trajectory_to_training_sample  # noqa: E402


def _team_names(replay: dict) -> list[str]:
    names = replay.get("info", {}).get("TeamNames")
    if names:
        return names
    agents = replay.get("info", {}).get("Agents", [])
    return [a.get("Name", f"player_{i}") for i, a in enumerate(agents)]


def _strict_winner(rewards: list | None) -> int | None:
    if not rewards or any(r is None for r in rewards):
        return None
    best = max(rewards)
    winners = [i for i, r in enumerate(rewards) if r == best]
    return winners[0] if len(winners) == 1 else None


def _matches_any(name: str, filters: list[str]) -> bool:
    if not filters:
        return True
    lower = name.lower()
    return any(f.lower() in lower for f in filters)


def _moves(action) -> list:
    if not isinstance(action, list):
        return []
    return [m for m in action if isinstance(m, list) and len(m) >= 3]


def _normalize_obs(obs: dict, player: int, step: int) -> dict:
    out = dict(obs)
    out["player"] = player
    out.setdefault("step", step)
    out.setdefault("fleets", [])
    out.setdefault("angular_velocity", 0.0)
    out.setdefault("initial_planets", out.get("planets", []))
    out.setdefault("comet_planet_ids", [])
    return out


def _clone_sample(sample: dict) -> dict:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sample.items()}


def _add_sample_stats(stats: Counter, sample: dict) -> None:
    valid = sample["slot_valid"].bool()
    fired = sample["fire_target"].bool() & valid
    target_labels = sample["target_target"] >= 0
    stats["valid_slots"] += int(valid.sum().item())
    stats["fire_slots"] += int(fired.sum().item())
    stats["nofire_slots"] += int((valid & ~fired).sum().item())
    stats["target_labels"] += int((target_labels & fired).sum().item())


def _reinforce_label_count(sample: dict, obs: dict, player: int) -> int:
    planets = obs.get("planets", [])
    targets = sample["target_target"]
    valid = sample["slot_valid"].bool()
    fired = sample["fire_target"].bool() & valid
    count = 0
    for slot in range(min(len(targets), len(fired))):
        if not bool(fired[slot]):
            continue
        tidx = int(targets[slot].item())
        if 0 <= tidx < len(planets) and int(planets[tidx][1]) == player:
            count += 1
    return count


def build(
    replay_dirs: list[str],
    glob_pattern: str = "*.json",
    winner_name_filters: list[str] | None = None,
    steps_min: int = 1,
    steps_max: int = 0,
    noop_keep_prob: float = 0.05,
    fire_repeat: int = 1,
    reinforce_repeat: int = 1,
    seed: int = 0,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    filters = winner_name_filters or []
    samples: list[dict] = []
    stats: Counter = Counter()
    subjects: Counter = Counter()

    replay_paths: list[str] = []
    for replay_dir in replay_dirs:
        replay_paths.extend(sorted(glob.glob(os.path.join(replay_dir, glob_pattern))))
    stats["replays_found"] = len(replay_paths)

    for replay_path in replay_paths:
        try:
            replay = json.loads(Path(replay_path).read_text())
        except Exception:
            stats["replay_load_failed"] += 1
            continue

        steps = replay.get("steps") or []
        if len(steps) < 2:
            stats["replay_too_short"] += 1
            continue
        winner = _strict_winner(replay.get("rewards"))
        if winner is None:
            stats["no_strict_winner"] += 1
            continue

        names = _team_names(replay)
        winner_name = names[winner] if winner < len(names) else f"player_{winner}"
        if not _matches_any(winner_name, filters):
            stats["winner_filtered_out"] += 1
            continue

        stats["replays_selected"] += 1
        subjects[winner_name] += 1
        t_start = max(1, steps_min)
        t_end = len(steps) if steps_max <= 0 else min(len(steps), steps_max + 1)

        for t in range(t_start, t_end):
            if winner >= len(steps[t]) or winner >= len(steps[t - 1]):
                stats["missing_agent_step"] += 1
                continue
            obs = steps[t - 1][winner].get("observation")
            if not obs or "planets" not in obs:
                stats["missing_obs"] += 1
                continue
            action = steps[t][winner].get("action") or []
            moves = _moves(action)
            if moves:
                stats["decision_frames_seen"] += 1
                repeat = max(1, fire_repeat)
            else:
                stats["noop_frames_seen"] += 1
                if rng.random() >= noop_keep_prob:
                    continue
                stats["noop_frames_kept"] += 1
                repeat = 1

            sample = trajectory_to_training_sample({
                "obs": _normalize_obs(obs, winner, t - 1),
                "action": moves,
            })
            if sample is None:
                stats["sample_build_failed"] += 1
                continue
            reinforce_labels = _reinforce_label_count(sample, obs, winner)
            if reinforce_labels:
                stats["reinforce_frames_seen"] += 1
                stats["reinforce_labels_seen"] += reinforce_labels
                repeat = max(repeat, reinforce_repeat)
            for _ in range(repeat):
                samples.append(_clone_sample(sample))
                _add_sample_stats(stats, sample)
                if reinforce_labels:
                    stats["reinforce_labels"] += reinforce_labels
            if moves:
                stats["decision_samples_added"] += repeat
                if reinforce_labels:
                    stats["reinforce_samples_added"] += repeat
            else:
                stats["noop_samples_added"] += 1

    summary = {
        "config": {
            "replay_dirs": replay_dirs,
            "glob": glob_pattern,
            "winner_name_filters": filters,
            "steps_min": steps_min,
            "steps_max": steps_max,
            "noop_keep_prob": noop_keep_prob,
            "fire_repeat": fire_repeat,
            "reinforce_repeat": reinforce_repeat,
            "seed": seed,
        },
        "subjects": dict(subjects.most_common(20)),
        "samples": len(samples),
        "stats": dict(stats),
    }
    fire_slots = stats["fire_slots"]
    valid_slots = stats["valid_slots"]
    if valid_slots:
        summary["fire_slot_rate"] = fire_slots / valid_slots
    if stats["decision_frames_seen"] + stats["noop_frames_seen"]:
        summary["decision_frame_rate_seen"] = stats["decision_frames_seen"] / (
            stats["decision_frames_seen"] + stats["noop_frames_seen"]
        )
    if samples:
        summary["decision_sample_share"] = stats["decision_samples_added"] / len(samples)
    return samples, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay-dir", action="append", required=True)
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--winner-name", action="append", default=[],
                    help="Substring filter for cloned winners; repeatable.")
    ap.add_argument("--steps-min", type=int, default=1,
                    help="Minimum action step to clone; actions pair with previous obs.")
    ap.add_argument("--steps-max", type=int, default=0,
                    help="Maximum action step to clone; 0 = full game.")
    ap.add_argument("--noop-keep-prob", type=float, default=0.05,
                    help="Probability of keeping an idle no-action frame.")
    ap.add_argument("--fire-repeat", type=int, default=1,
                    help="Repeat decision-frame samples to rebalance toward actions.")
    ap.add_argument("--reinforce-repeat", type=int, default=1,
                    help="Minimum repeat count for frames with own-target labels.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples-out", required=True)
    ap.add_argument("--summary-out", default="")
    args = ap.parse_args()

    if not 0.0 <= args.noop_keep_prob <= 1.0:
        raise SystemExit("--noop-keep-prob must be in [0, 1]")

    samples, summary = build(
        replay_dirs=args.replay_dir,
        glob_pattern=args.glob,
        winner_name_filters=args.winner_name,
        steps_min=args.steps_min,
        steps_max=args.steps_max,
        noop_keep_prob=args.noop_keep_prob,
        fire_repeat=args.fire_repeat,
        reinforce_repeat=args.reinforce_repeat,
        seed=args.seed,
    )

    out_path = Path(args.samples_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(samples, f)
    summary["samples_out"] = str(out_path)

    print(json.dumps(summary, indent=2))
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
