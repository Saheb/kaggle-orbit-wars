"""Build joint-ranker preference samples from Ajay replay actions.

For each Ajay launch in a replay slice:
  - positive = the exact `(source, target, ships)` action Ajay took
  - negatives = top legal alternative candidates from the same state

This keeps the supervision on the state distribution we actually care about:
our checkpoints losing early tempo to Ajay.
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
    ActionCandidate,
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
    ap.add_argument("--player-name", default="Ajay")
    ap.add_argument("--player-slot", type=int)
    ap.add_argument("--step-limit", type=int, default=20)
    ap.add_argument("--negatives-per-positive", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--samples-out", required=True)
    ap.add_argument("--summary-out", default="")
    return ap.parse_args()


def _base_sample(obs_prev: dict, move: list) -> tuple[dict | None, dict[int, int]]:
    sample = trajectory_to_training_sample({"obs": obs_prev, "action": [move]})
    if sample is None:
        return None, {}
    pid_to_slot: dict[int, int] = {}
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


def _is_exact_action_match(
    cand: ActionCandidate,
    *,
    source_id: int,
    target_id: int,
    ships: int,
) -> bool:
    return (
        int(cand.source_id) == int(source_id)
        and int(cand.target_id) == int(target_id)
        and int(cand.ships) == int(ships)
    )


def _negative_candidates(
    candidates: list[ActionCandidate],
    *,
    source_id: int,
    target_id: int,
    ships: int,
    limit: int,
) -> list[ActionCandidate]:
    out = []
    for cand in candidates:
        if not cand.valid or int(cand.ships) <= 0:
            continue
        if _is_exact_action_match(cand, source_id=source_id, target_id=target_id, ships=ships):
            continue
        out.append(cand)
        if len(out) >= limit:
            break
    return out


def _action_extra_from_candidate(
    *,
    ships: int,
    eta: int | None,
    valid: bool,
    source_ships: int,
    target_prod: int,
    floor_at_arrival: int,
    score: float,
    target_is_mine: bool,
    target_is_neutral: bool,
) -> torch.Tensor:
    return torch.tensor(
        action_extra_features(
            ships=ships,
            eta=eta,
            valid=valid,
            source_ships=source_ships,
            target_prod=target_prod,
            floor_at_arrival=floor_at_arrival,
            score=score,
            target_is_mine=target_is_mine,
            target_is_neutral=target_is_neutral,
        ),
        dtype=torch.float32,
    )


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

                pos_slot = pid_to_slot.get(int(from_pid))
                if pos_slot is None:
                    stats["pos_slot_lookup_failed"] += 1
                    continue

                negatives = _negative_candidates(
                    candidates,
                    source_id=from_pid,
                    target_id=tgt_id,
                    ships=ship_count,
                    limit=max(1, args.negatives_per_positive),
                )
                if not negatives:
                    stats["no_negative_candidates"] += 1
                    continue

                pos_action_extra = _action_extra_from_candidate(
                    ships=int(ship_count),
                    eta=int(replay_scored["eta"] or 32),
                    valid=bool(replay_scored["valid"]),
                    source_ships=int(replay_scored["source_ships"]),
                    target_prod=int(replay_scored["target_prod"]),
                    floor_at_arrival=int(replay_scored["floor_at_arrival"]),
                    score=float(replay_scored["score"]),
                    target_is_mine=bool(replay_scored["target_is_mine"]),
                    target_is_neutral=bool(replay_scored["target_is_neutral"]),
                )

                for neg in negatives:
                    neg_slot = pid_to_slot.get(int(neg.source_id))
                    if neg_slot is None:
                        stats["neg_slot_lookup_failed"] += 1
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
                        "pos_ship_bin": torch.tensor(_find_ship_bin(int(ship_count)), dtype=torch.long),
                        "pos_target_idx": torch.tensor(int(tgt_idx), dtype=torch.long),
                        "pos_action_extra": pos_action_extra.clone(),
                        "neg_slot": torch.tensor(int(neg_slot), dtype=torch.long),
                        "neg_ship_bin": torch.tensor(_find_ship_bin(int(neg.ships)), dtype=torch.long),
                        "neg_target_idx": torch.tensor(int(neg.target_idx), dtype=torch.long),
                        "neg_action_extra": _action_extra_from_candidate(
                            ships=int(neg.ships),
                            eta=int(neg.eta),
                            valid=bool(neg.valid),
                            source_ships=int(neg.source_ships),
                            target_prod=int(neg.target_prod),
                            floor_at_arrival=int(neg.floor_at_arrival),
                            score=float(neg.score),
                            target_is_mine=bool(neg.target_is_mine),
                            target_is_neutral=bool(neg.target_is_neutral),
                        ),
                        "weight": torch.tensor(1.0, dtype=torch.float32),
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
        "negatives_per_positive": args.negatives_per_positive,
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
