"""Build sharded replay-supervised BC samples for one strong player.

This is the scalable pure-supervised path. It streams replay JSON files, keeps
only seats whose TeamName matches the requested player filter, optionally keeps
only wins by that player, and writes standard bc.py sample lists in shards.

The action at steps[t] is paired with the observation at steps[t - 1].
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_rl.bc import trajectory_to_training_sample  # noqa: E402
from orbit_wars_rl.build_supervised_bc import (  # noqa: E402
    _add_sample_stats,
    _clone_sample,
    _matches_any,
    _moves,
    _normalize_obs,
    _reinforce_label_count,
    _strict_winner,
    _team_names,
)


def _replay_paths(replay_dirs: list[str], glob_pattern: str) -> list[str]:
    paths: list[str] = []
    metadata_names = {
        "manifest.json",
        "fetch_best_player_summary.json",
        "fetch_best_player_scan_cache.json",
    }
    for replay_dir in replay_dirs:
        for path in sorted(glob.glob(os.path.join(replay_dir, glob_pattern))):
            name = os.path.basename(path)
            if name in metadata_names or name.endswith("_summary.json"):
                continue
            paths.append(path)
    return sorted(set(paths))


def _matching_seats(replay: dict, player_filters: list[str], require_win: bool) -> list[tuple[int, str, bool, float]]:
    names = _team_names(replay)
    if not names:
        return []
    winner = _strict_winner(replay.get("rewards"))
    rewards = replay.get("rewards") or []
    seats: list[tuple[int, str, bool, float]] = []
    for seat, name in enumerate(names):
        if not _matches_any(name, player_filters):
            continue
        won = seat == winner
        if require_win and not won:
            continue
        reward = float(rewards[seat]) if seat < len(rewards) and rewards[seat] is not None else 0.0
        seats.append((seat, name, won, reward))
    return seats


def _flush_shard(samples: list[dict], out_dir: Path, shard_idx: int) -> dict:
    path = out_dir / f"samples_{shard_idx:05d}.pkl"
    with path.open("wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {"path": str(path), "samples": len(samples)}


def build_shards(
    replay_dirs: list[str],
    out_dir: str,
    player_filters: list[str],
    glob_pattern: str = "*.json",
    require_win: bool = False,
    steps_min: int = 1,
    steps_max: int = 0,
    noop_keep_prob: float = 0.02,
    fire_repeat: int = 1,
    reinforce_repeat: int = 1,
    samples_per_shard: int = 50_000,
    max_replays: int = 0,
    max_samples: int = 0,
    seed: int = 0,
    output_format: str = "tensor",
    nonwin_keep_prob: float = 1.0,
    win_repeat: int = 1,
    nonwin_repeat: int = 1,
) -> dict:
    if not player_filters:
        raise ValueError("At least one player filter is required")
    if samples_per_shard <= 0:
        raise ValueError("samples_per_shard must be positive")
    if output_format not in {"tensor", "frame"}:
        raise ValueError("output_format must be 'tensor' or 'frame'")
    if not 0.0 <= nonwin_keep_prob <= 1.0:
        raise ValueError("nonwin_keep_prob must be in [0, 1]")
    if win_repeat <= 0 or nonwin_repeat <= 0:
        raise ValueError("win_repeat and nonwin_repeat must be positive")

    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    subjects: Counter = Counter()
    subject_samples: Counter = Counter()
    shard: list[dict] = []
    shards: list[dict] = []
    shard_idx = 0

    replay_paths = _replay_paths(replay_dirs, glob_pattern)
    stats["replays_found"] = len(replay_paths)

    def maybe_add(sample: dict, frame: dict, repeat: int, name: str,
                  reinforce_labels: int, has_moves: bool) -> bool:
        nonlocal shard, shard_idx
        for _ in range(repeat):
            if max_samples and stats["samples"] >= max_samples:
                return False
            shard.append(_clone_sample(sample) if output_format == "tensor" else frame)
            _add_sample_stats(stats, sample)
            stats["samples"] += 1
            subject_samples[name] += 1
            if reinforce_labels:
                stats["reinforce_labels"] += reinforce_labels
            if has_moves:
                stats["decision_samples_added"] += 1
                if reinforce_labels:
                    stats["reinforce_samples_added"] += 1
            else:
                stats["noop_samples_added"] += 1
            if len(shard) >= samples_per_shard:
                shards.append(_flush_shard(shard, out, shard_idx))
                shard_idx += 1
                shard = []
        return True

    for replay_path in replay_paths:
        if max_replays and stats["replays_selected"] >= max_replays:
            break
        try:
            replay = json.loads(Path(replay_path).read_text())
        except Exception:
            stats["replay_load_failed"] += 1
            continue

        steps = replay.get("steps") or []
        if len(steps) < 2:
            stats["replay_too_short"] += 1
            continue

        seats = _matching_seats(replay, player_filters, require_win=require_win)
        if not seats:
            stats["replays_without_matching_player"] += 1
            continue

        stats["replays_selected"] += 1
        stats["matching_seats"] += len(seats)
        t_start = max(1, steps_min)
        t_end = len(steps) if steps_max <= 0 else min(len(steps), steps_max + 1)

        for seat, name, won, reward in seats:
            if not won and rng.random() >= nonwin_keep_prob:
                stats["nonwin_seats_dropped"] += 1
                continue
            subjects[name] += 1
            stats["win_seats_selected" if won else "nonwin_seats_selected"] += 1
            for t in range(t_start, t_end):
                if seat >= len(steps[t]) or seat >= len(steps[t - 1]):
                    stats["missing_agent_step"] += 1
                    continue
                obs = steps[t - 1][seat].get("observation")
                if not obs or "planets" not in obs:
                    stats["missing_obs"] += 1
                    continue
                action = steps[t][seat].get("action") or []
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

                frame = {
                    "obs": _normalize_obs(obs, seat, t - 1),
                    "action": moves,
                    "player_name": name,
                    "replay_path": replay_path,
                    "action_step": t,
                    "won": won,
                    "reward": reward,
                }
                sample = trajectory_to_training_sample(frame)
                if sample is None:
                    stats["sample_build_failed"] += 1
                    continue

                reinforce_labels = _reinforce_label_count(sample, obs, seat)
                if reinforce_labels:
                    stats["reinforce_frames_seen"] += 1
                    stats["reinforce_labels_seen"] += reinforce_labels
                    repeat = max(repeat, reinforce_repeat)
                repeat *= win_repeat if won else nonwin_repeat
                if not maybe_add(sample, frame, repeat, name, reinforce_labels, has_moves=bool(moves)):
                    break
            if max_samples and stats["samples"] >= max_samples:
                break
        if max_samples and stats["samples"] >= max_samples:
            break

    if shard:
        shards.append(_flush_shard(shard, out, shard_idx))

    summary = {
        "config": {
            "replay_dirs": replay_dirs,
            "glob": glob_pattern,
            "player_filters": player_filters,
            "require_win": require_win,
            "steps_min": steps_min,
            "steps_max": steps_max,
            "noop_keep_prob": noop_keep_prob,
            "fire_repeat": fire_repeat,
            "reinforce_repeat": reinforce_repeat,
            "samples_per_shard": samples_per_shard,
            "max_replays": max_replays,
            "max_samples": max_samples,
            "seed": seed,
            "format": output_format,
            "nonwin_keep_prob": nonwin_keep_prob,
            "win_repeat": win_repeat,
            "nonwin_repeat": nonwin_repeat,
        },
        "subjects": dict(subjects.most_common(30)),
        "subject_samples": dict(subject_samples.most_common(30)),
        "stats": dict(stats),
        "samples": int(stats["samples"]),
        "shards": shards,
        "sample_paths": [s["path"] for s in shards],
    }
    valid_slots = stats["valid_slots"]
    if valid_slots:
        summary["fire_slot_rate"] = stats["fire_slots"] / valid_slots
    seen_frames = stats["decision_frames_seen"] + stats["noop_frames_seen"]
    if seen_frames:
        summary["decision_frame_rate_seen"] = stats["decision_frames_seen"] / seen_frames
    if stats["samples"]:
        summary["decision_sample_share"] = stats["decision_samples_added"] / stats["samples"]
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2))
    summary["manifest_path"] = str(manifest_path)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay-dir", action="append", required=True)
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--player-name", action="append", required=True,
                    help="Substring filter for the strong player; repeatable.")
    ap.add_argument("--require-win", action="store_true",
                    help="Keep only games where the matching player is the strict winner.")
    ap.add_argument("--steps-min", type=int, default=1)
    ap.add_argument("--steps-max", type=int, default=0,
                    help="Maximum action step to clone; 0 = full game.")
    ap.add_argument("--noop-keep-prob", type=float, default=0.02)
    ap.add_argument("--fire-repeat", type=int, default=1)
    ap.add_argument("--reinforce-repeat", type=int, default=1)
    ap.add_argument("--samples-per-shard", type=int, default=50_000)
    ap.add_argument("--max-replays", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", choices=["tensor", "frame"], default="tensor",
                    help="tensor = bc.py sample tensors; frame = compact raw obs/action records.")
    ap.add_argument("--nonwin-keep-prob", type=float, default=1.0,
                    help="When --require-win is omitted, keep this fraction of matching non-winning seats.")
    ap.add_argument("--win-repeat", type=int, default=1,
                    help="Repeat factor for samples from winning top-player seats.")
    ap.add_argument("--nonwin-repeat", type=int, default=1,
                    help="Repeat factor for samples from non-winning top-player seats.")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if not 0.0 <= args.noop_keep_prob <= 1.0:
        raise SystemExit("--noop-keep-prob must be in [0, 1]")
    if not 0.0 <= args.nonwin_keep_prob <= 1.0:
        raise SystemExit("--nonwin-keep-prob must be in [0, 1]")

    summary = build_shards(
        replay_dirs=args.replay_dir,
        out_dir=args.out_dir,
        player_filters=args.player_name,
        glob_pattern=args.glob,
        require_win=args.require_win,
        steps_min=args.steps_min,
        steps_max=args.steps_max,
        noop_keep_prob=args.noop_keep_prob,
        fire_repeat=args.fire_repeat,
        reinforce_repeat=args.reinforce_repeat,
        samples_per_shard=args.samples_per_shard,
        max_replays=args.max_replays,
        max_samples=args.max_samples,
        seed=args.seed,
        output_format=args.format,
        nonwin_keep_prob=args.nonwin_keep_prob,
        win_repeat=args.win_repeat,
        nonwin_repeat=args.nonwin_repeat,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
