"""Build producer-labeled BC samples for target-head supervision.

This emits standard ``bc.py`` sample dicts, but relabels the fired slot's
``target_target`` to the producer-style best target for that source planet.

Primary use:
- take replay slices from our failures or mixed corpora
- supervise the current target head directly toward a short-horizon economic
  ranking signal
- fine-tune only ``tgt_q`` / ``tgt_k`` / ``target_scorer`` with ``bc.py``

Example:
  source /Users/saheb/home/.venv/bin/activate
  python orbit_wars_rl/build_producer_target_bc.py \
    --replay-dir /tmp/sub53359633_eps \
    --player-name Saheb \
    --step-limit 40 \
    --mismatch-repeat 4 \
    --samples-out /tmp/producer_target_bc.pkl \
    --summary-out /tmp/producer_target_bc_summary.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_rl.producer_ranking import (
    infer_player_slot,
    producer_candidates_for_source,
)
from orbit_wars_rl.audit_submission_targets import normalize_obs, resolve_replay_paths
from orbit_wars_rl.bc import trajectory_to_training_sample


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir")
    ap.add_argument("--replay-path", action="append", default=[])
    ap.add_argument("--episode-id", action="append", default=[])
    ap.add_argument("--player-name", default="")
    ap.add_argument("--player-slot", type=int)
    ap.add_argument("--step-limit", type=int, default=40)
    ap.add_argument("--only-mismatch", action="store_true")
    ap.add_argument("--matched-repeat", type=int, default=1)
    ap.add_argument("--mismatch-repeat", type=int, default=3)
    ap.add_argument("--samples-out", required=True)
    ap.add_argument("--summary-out", default="")
    return ap.parse_args()


def _sample_for_single_move(obs_prev: dict, move: list) -> tuple[dict | None, int | None, int | None]:
    sample = trajectory_to_training_sample({"obs": obs_prev, "action": [move]})
    if sample is None:
        return None, None, None
    fired_slots = torch.nonzero(sample["fire_target"] == 1, as_tuple=False).flatten().tolist()
    if len(fired_slots) != 1:
        return None, None, None
    slot = int(fired_slots[0])
    current_target_idx = int(sample["target_target"][slot].item())
    if current_target_idx < 0:
        return None, None, None
    return sample, slot, current_target_idx


def main() -> None:
    args = parse_args()
    replay_paths = resolve_replay_paths(args.replay_dir, args.replay_path, args.episode_id)
    samples: list[dict] = []
    stats = Counter()

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

        player_slot = infer_player_slot(replay, args.player_name, args.player_slot)
        stats["replays_seen"] += 1

        for t in range(1, len(steps)):
            if args.step_limit is not None and t > args.step_limit:
                break
            if player_slot >= len(steps[t]) or player_slot >= len(steps[t - 1]):
                stats["step_player_slot_oob"] += 1
                continue

            acts = steps[t][player_slot].get("action") or []
            if not acts:
                continue
            obs_prev = normalize_obs(steps[t - 1][player_slot]["observation"], fallback_step=t - 1)
            planets = obs_prev["planets"]

            for move in acts:
                if len(move) < 3:
                    stats["move_malformed"] += 1
                    continue
                from_pid = int(move[0])
                src_idx = next((i for i, p in enumerate(planets) if int(p[0]) == from_pid), None)
                if src_idx is None:
                    stats["src_not_found"] += 1
                    continue

                producer = producer_candidates_for_source(obs_prev, src_idx)
                producer_best = producer.get("producer_best_target_idx")
                if not producer.get("source_valid", False):
                    stats["producer_source_invalid"] += 1
                    continue
                if producer_best is None:
                    stats["producer_best_missing"] += 1
                    continue

                sample, slot, current_target_idx = _sample_for_single_move(obs_prev, move)
                if sample is None or slot is None or current_target_idx is None:
                    stats["sample_build_failed"] += 1
                    continue

                mismatch = int(current_target_idx) != int(producer_best)
                if args.only_mismatch and not mismatch:
                    stats["matched_skipped"] += 1
                    continue

                relabeled = {}
                for k, v in sample.items():
                    relabeled[k] = v.clone() if torch.is_tensor(v) else v
                relabeled["target_target"][slot] = int(producer_best)

                repeat = max(1, args.mismatch_repeat if mismatch else args.matched_repeat)
                for _ in range(repeat):
                    samples.append({k: (v.clone() if torch.is_tensor(v) else v) for k, v in relabeled.items()})

                stats["launches_seen"] += 1
                stats["producer_labeled_samples"] += repeat
                if mismatch:
                    stats["mismatch_launches"] += 1
                    stats["mismatch_repeats_added"] += repeat
                else:
                    stats["matched_launches"] += 1
                    stats["matched_repeats_added"] += repeat

    out_path = Path(args.samples_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(samples, f)

    payload = {
        "replay_count": len(replay_paths),
        "sample_count": len(samples),
        "player_name": args.player_name,
        "player_slot": args.player_slot,
        "step_limit": args.step_limit,
        "only_mismatch": args.only_mismatch,
        "matched_repeat": args.matched_repeat,
        "mismatch_repeat": args.mismatch_repeat,
        "stats": dict(stats),
    }
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    print(f"samples saved -> {args.samples_out}")


if __name__ == "__main__":
    main()
