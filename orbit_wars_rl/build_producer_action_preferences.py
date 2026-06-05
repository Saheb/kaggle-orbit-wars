"""Build pairwise producer-action preference samples.

Each sample says: on this state, the producer-best action should score above a
replay action we actually took.

This avoids the failure mode of hard action BC, which forces every non-best
slot to no-fire and can wipe out useful multi-action behavior.
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

from orbit_wars_rl.analyze_producer_action_ranking import (
    _enumerate_attack_candidates,
    _score_replay_move,
    action_extra_features,
)
from orbit_wars_rl.analyze_producer_ranking import infer_player_slot
from orbit_wars_rl.audit_submission_targets import normalize_obs, resolve_replay_paths
from orbit_wars_rl.bc import _find_ship_bin, _find_target_planet_index, trajectory_to_training_sample


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir")
    ap.add_argument("--replay-path", action="append", default=[])
    ap.add_argument("--episode-id", action="append", default=[])
    ap.add_argument("--player-name", default="")
    ap.add_argument("--player-slot", type=int)
    ap.add_argument("--step-limit", type=int, default=40)
    ap.add_argument("--min-score-gap", type=float, default=0.0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--samples-out", required=True)
    ap.add_argument("--summary-out", default="")
    return ap.parse_args()


def _base_sample(obs_prev: dict, move: list) -> tuple[dict | None, dict[int, int]]:
    sample = trajectory_to_training_sample({"obs": obs_prev, "action": [move]})
    if sample is None:
        return None, {}
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
            planets = obs_prev["planets"]
            cand_info = _enumerate_attack_candidates(obs_prev)
            candidates = cand_info["candidates"]
            if not candidates:
                stats["no_candidates"] += 1
                continue
            best = candidates[0]

            for move in acts:
                if len(move) < 3:
                    continue
                from_pid = int(move[0])
                angle = float(move[1])
                ship_count = int(move[2])
                sample, pid_to_slot = _base_sample(obs_prev, move)
                if sample is None:
                    stats["sample_build_failed"] += 1
                    continue

                src_idx = next((i for i, p in enumerate(planets) if int(p[0]) == from_pid), None)
                if src_idx is None:
                    stats["src_not_found"] += 1
                    continue
                src = planets[src_idx]
                tgt_idx = _find_target_planet_index(
                    (float(src[2]), float(src[3])),
                    angle,
                    ship_count,
                    planets,
                    obs_prev.get("initial_planets", planets),
                    float(obs_prev.get("angular_velocity", 0.0)),
                    int(obs_prev.get("step", 0)),
                    max_planets=min(len(planets), 48),
                )
                if tgt_idx < 0 or tgt_idx >= len(planets):
                    stats["replay_target_decode_failed"] += 1
                    continue
                tgt_id = int(planets[tgt_idx][0])
                replay_scored = _score_replay_move(obs_prev, from_pid, tgt_id, ship_count)
                if replay_scored is None or not replay_scored.get("valid", False):
                    stats["replay_move_invalid"] += 1
                    continue

                pos_slot = pid_to_slot.get(int(best.source_id))
                neg_slot = pid_to_slot.get(int(from_pid))
                if pos_slot is None or neg_slot is None:
                    stats["slot_lookup_failed"] += 1
                    continue

                score_gap = float(best.score) - float(replay_scored["score"])
                if score_gap < float(args.min_score_gap):
                    stats["small_gap_skipped"] += 1
                    continue
                if int(best.source_id) == int(from_pid) and int(best.target_id) == int(tgt_id) and int(best.ships) == int(ship_count):
                    stats["exact_match_skipped"] += 1
                    continue

                pref = {
                    "planet_features": sample["planet_features"].clone(),
                    "fleet_features": sample["fleet_features"].clone(),
                    "global_features": sample["global_features"].clone(),
                    "planet_mask": sample["planet_mask"].clone(),
                    "fleet_mask": sample["fleet_mask"].clone(),
                    "fire_mask": sample["fire_mask"].clone(),
                    "angle_mask": sample["angle_mask"].clone(),
                    "slot_valid": sample["slot_valid"].clone(),
                    "owned_indices": sample["owned_indices"].clone(),
                    "pairwise_features": sample["pairwise_features"].clone(),
                    "pos_slot": torch.tensor(int(pos_slot), dtype=torch.long),
                    "pos_ship_bin": torch.tensor(_find_ship_bin(int(best.ships)), dtype=torch.long),
                    "pos_target_idx": torch.tensor(int(best.target_idx), dtype=torch.long),
                    "pos_action_extra": torch.tensor(action_extra_features(
                        ships=int(best.ships),
                        eta=int(best.eta),
                        valid=bool(best.valid),
                        source_ships=int(best.source_ships),
                        target_prod=int(best.target_prod),
                        floor_at_arrival=int(best.floor_at_arrival),
                        score=float(best.score),
                        target_is_mine=bool(best.target_is_mine),
                        target_is_neutral=bool(best.target_is_neutral),
                    ), dtype=torch.float32),
                    "neg_slot": torch.tensor(int(neg_slot), dtype=torch.long),
                    "neg_ship_bin": torch.tensor(_find_ship_bin(int(ship_count)), dtype=torch.long),
                    "neg_target_idx": torch.tensor(int(tgt_idx), dtype=torch.long),
                    "neg_action_extra": torch.tensor(action_extra_features(
                        ships=int(ship_count),
                        eta=int(replay_scored["eta"] or 32),
                        valid=bool(replay_scored["valid"]),
                        source_ships=int(replay_scored["source_ships"]),
                        target_prod=int(replay_scored["target_prod"]),
                        floor_at_arrival=int(replay_scored["floor_at_arrival"]),
                        score=float(replay_scored["score"]),
                        target_is_mine=bool(replay_scored["target_is_mine"]),
                        target_is_neutral=bool(replay_scored["target_is_neutral"]),
                    ), dtype=torch.float32),
                    "weight": torch.tensor(max(1.0, score_gap), dtype=torch.float32),
                }
                for _ in range(max(1, args.repeat)):
                    samples.append({k: (v.clone() if torch.is_tensor(v) else v) for k, v in pref.items()})
                stats["preference_pairs"] += 1
                stats["samples_added"] += max(1, args.repeat)

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
        "min_score_gap": args.min_score_gap,
        "repeat": args.repeat,
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
