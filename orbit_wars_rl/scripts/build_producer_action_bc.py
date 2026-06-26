"""Build producer-labeled whole-action BC samples.

This upgrades the earlier target-only relabeling. For each replay state with at
least one launch, we label the single best producer-style attack action in the
state:
  - source slot
  - target planet
  - ship bin

All other slots are labeled no-fire. This gives direct supervision for the
coupled `(source, target, ships)` decision instead of `target | chosen_source`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_rl.producer_action_ranking import _enumerate_attack_candidates
from orbit_wars_rl.producer_ranking import infer_player_slot
from orbit_wars_rl.audit_submission_targets import normalize_obs, resolve_replay_paths
from orbit_wars_rl.bc import _find_ship_bin, trajectory_to_training_sample


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir")
    ap.add_argument("--replay-path", action="append", default=[])
    ap.add_argument("--episode-id", action="append", default=[])
    ap.add_argument("--player-name", default="")
    ap.add_argument("--player-slot", type=int)
    ap.add_argument("--step-limit", type=int, default=40)
    ap.add_argument("--best-repeat", type=int, default=2)
    ap.add_argument("--samples-out", required=True)
    ap.add_argument("--summary-out", default="")
    return ap.parse_args()


def _blank_sample_from_first_action(obs_prev: dict, first_move: list) -> tuple[dict | None, dict[int, int]]:
    sample = trajectory_to_training_sample({"obs": obs_prev, "action": [first_move]})
    if sample is None:
        return None, {}
    sample["fire_target"].zero_()
    sample["ship_target"].zero_()
    sample["target_target"].fill_(-1)
    pid_to_slot = {}
    planets = obs_prev["planets"]
    owned_indices = sample["owned_indices"].tolist()
    slot_valid = sample["slot_valid"].tolist()
    for slot, valid in enumerate(slot_valid):
        if not valid:
            continue
        pidx = int(owned_indices[slot])
        if 0 <= pidx < len(planets):
            pid_to_slot[int(planets[pidx][0])] = slot
    return sample, pid_to_slot


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
                continue
            acts = steps[t][player_slot].get("action") or []
            if not acts:
                continue
            obs_prev = normalize_obs(steps[t - 1][player_slot]["observation"], fallback_step=t - 1)
            cand_info = _enumerate_attack_candidates(obs_prev)
            candidates = cand_info["candidates"]
            if not candidates:
                stats["no_action_candidates"] += 1
                continue
            best = candidates[0]
            sample, pid_to_slot = _blank_sample_from_first_action(obs_prev, acts[0])
            if sample is None:
                stats["sample_build_failed"] += 1
                continue
            slot = pid_to_slot.get(int(best.source_id))
            if slot is None:
                stats["best_source_not_in_slots"] += 1
                continue
            sample["fire_target"][slot] = 1
            sample["ship_target"][slot] = _find_ship_bin(int(best.ships))
            sample["target_target"][slot] = int(best.target_idx)

            for _ in range(max(1, args.best_repeat)):
                samples.append({k: (v.clone() if torch.is_tensor(v) else v) for k, v in sample.items()})
            stats["best_action_samples"] += max(1, args.best_repeat)
            stats["states_labeled"] += 1

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
        "best_repeat": args.best_repeat,
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
