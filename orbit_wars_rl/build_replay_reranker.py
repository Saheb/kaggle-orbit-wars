"""Train a candidate reranker from top-player replay actions.

This is the replay-only analogue of ``build_producer_reranker.py``. Producer
candidate enumeration is used only to define plausible alternatives and the
runtime candidate feature schema; positives are the strong player's recorded
source-target-ship launches, not Producer's preferred candidate.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_rl.analyze_producer_action_ranking import (  # noqa: E402
    ActionCandidate,
    _enumerate_attack_candidates,
    _score_replay_move,
)
from orbit_wars_rl.bc import _find_target_planet_index  # noqa: E402
from orbit_wars_rl.build_policy_teacher_bc import _selected_seats, _threatened_owned_pids  # noqa: E402
from orbit_wars_rl.build_producer_reranker import FEATURE_NAMES, _candidate_features, train_reranker  # noqa: E402
from orbit_wars_rl.build_supervised_bc import _moves, _normalize_obs, _team_names  # noqa: E402
from orbit_wars_rl.score_good_play_replays import _capture_steps_by_pid  # noqa: E402


def _replay_paths(replay_dirs: list[str], glob_pattern: str) -> list[str]:
    paths: list[str] = []
    skip_names = {"manifest.json", "fetch_best_player_summary.json", "fetch_best_player_scan_cache.json"}
    for replay_dir in replay_dirs:
        for path in glob.glob(os.path.join(replay_dir, glob_pattern)):
            name = os.path.basename(path)
            if name in skip_names or name.endswith("_summary.json"):
                continue
            paths.append(path)
    return sorted(set(paths))


def _target_owner_ok(owner: int, player: int, mode: str) -> bool:
    if mode == "any":
        return True
    if mode == "own":
        return owner == player
    if mode == "not-own":
        return owner != player
    if mode == "neutral":
        return owner < 0
    if mode == "enemy":
        return owner >= 0 and owner != player
    raise ValueError(f"unknown target owner mode: {mode}")


def _candidate_from_replay_score(score: dict) -> ActionCandidate:
    return ActionCandidate(
        source_idx=int(score["source_idx"]),
        source_id=int(score["source_id"]),
        target_idx=int(score["target_idx"]),
        target_id=int(score["target_id"]),
        ships=int(score["ships"]),
        eta=int(score["eta"] if score["eta"] is not None else 32),
        score=float(score["score"]),
        valid=bool(score["valid"]),
        target_is_mine=bool(score["target_is_mine"]),
        target_is_neutral=bool(score["target_is_neutral"]),
        source_ships=int(score["source_ships"]),
        target_prod=int(score["target_prod"]),
        floor_at_arrival=int(score["floor_at_arrival"]),
    )


def _negative_candidates(
    candidates: list[ActionCandidate],
    *,
    planets: list,
    player: int,
    source_id: int,
    target_id: int,
    ships: int,
    eta: int,
    target_owner: str,
    score_floor: float,
    score_slack: float,
    max_eta_gap: int,
    limit: int,
) -> list[ActionCandidate]:
    out: list[ActionCandidate] = []
    for candidate in candidates:
        if not candidate.valid or int(candidate.ships) <= 0:
            continue
        if int(candidate.source_id) != int(source_id):
            continue
        target_idx = int(candidate.target_idx)
        if target_idx < 0 or target_idx >= len(planets):
            continue
        if not _target_owner_ok(int(planets[target_idx][1]), player, target_owner):
            continue
        if int(candidate.target_id) == int(target_id) and int(candidate.ships) == int(ships):
            continue
        if float(candidate.score) < score_floor - score_slack:
            continue
        if max_eta_gap >= 0 and abs(int(candidate.eta) - int(eta)) > max_eta_gap:
            continue
        out.append(candidate)
        if len(out) >= limit:
            break
    return out


def _decode_replay_target(obs: dict, move: list) -> tuple[int | None, int | None]:
    planets = obs.get("planets") or []
    if len(move) < 3:
        return None, None
    from_pid, angle, ships = int(move[0]), float(move[1]), int(move[2])
    src_idx = next((i for i, p in enumerate(planets) if int(p[0]) == from_pid), None)
    if src_idx is None:
        return None, None
    src = planets[src_idx]
    tgt_idx = _find_target_planet_index(
        (float(src[2]), float(src[3])),
        angle,
        ships,
        planets,
        obs.get("initial_planets", planets),
        float(obs.get("angular_velocity", 0.0)),
        int(obs.get("step", 0)),
        max_planets=min(len(planets), 48),
    )
    if tgt_idx < 0 or tgt_idx >= len(planets):
        return src_idx, None
    return src_idx, tgt_idx


def build_records(
    replay_dirs: list[str],
    glob_pattern: str = "*.json",
    max_replays: int = 0,
    seat_mode: str = "winner",
    player_slot: int | None = None,
    winner_name_filters: list[str] | None = None,
    steps_min: int = 1,
    steps_max: int = 160,
    target_owner: str = "any",
    state_recent_capture_window: int = 0,
    candidate_recent_capture_window: int = 0,
    inbound_threat_horizon: int = 0,
    candidate_threatened_only: bool = False,
    negatives_per_positive: int = 8,
    score_slack: float = 5.0,
    max_eta_gap: int = -1,
    max_records: int = 0,
) -> tuple[list[dict], dict]:
    paths = _replay_paths(replay_dirs, glob_pattern)
    if max_replays > 0:
        paths = paths[:max_replays]

    records: list[dict] = []
    stats: Counter = Counter()
    subjects: Counter = Counter()
    state_id = 0
    winner_name_filters = winner_name_filters or []
    stats["replays_found"] = len(paths)

    for replay_path in paths:
        try:
            replay = json.loads(Path(replay_path).read_text())
        except Exception:
            stats["replay_load_failed"] += 1
            continue
        steps = replay.get("steps") or []
        if len(steps) < 2:
            stats["replay_too_short"] += 1
            continue
        seats = _selected_seats(replay, seat_mode, player_slot, winner_name_filters, stats)
        if not seats:
            continue
        names = _team_names(replay)
        t_end = len(steps) if steps_max <= 0 else min(len(steps), steps_max + 1)

        for seat in seats:
            subjects[names[seat] if seat < len(names) else f"player_{seat}"] += 1
            captures = _capture_steps_by_pid(steps, seat) if (
                state_recent_capture_window > 0 or candidate_recent_capture_window > 0
            ) else {}
            for t in range(max(1, steps_min), t_end):
                if max_records > 0 and len(records) >= max_records:
                    break
                if seat >= len(steps[t]) or seat >= len(steps[t - 1]):
                    stats["missing_agent_step"] += 1
                    continue
                moves = _moves(steps[t][seat].get("action") or [])
                if not moves:
                    stats["noop_frames_seen"] += 1
                    continue
                obs_raw = steps[t - 1][seat].get("observation")
                if not obs_raw or "planets" not in obs_raw:
                    stats["missing_obs"] += 1
                    continue
                obs = _normalize_obs(obs_raw, seat, t - 1)
                planets = obs.get("planets") or []
                step = int(obs.get("step", t - 1))
                recent_capture_pids: set[int] = set()
                recent_window = max(state_recent_capture_window, candidate_recent_capture_window)
                if recent_window > 0:
                    recent_capture_pids = {
                        pid for pid, cap_step in captures.items()
                        if 0 <= step - cap_step <= recent_window
                    }
                if state_recent_capture_window > 0 and not recent_capture_pids:
                    stats["states_skipped_no_recent_capture"] += 1
                    continue
                threatened_pids = _threatened_owned_pids(obs, seat, inbound_threat_horizon)
                if inbound_threat_horizon > 0 and not threatened_pids:
                    stats["states_skipped_no_inbound_threat"] += 1
                    continue
                try:
                    candidates = _enumerate_attack_candidates(obs)["candidates"]
                except Exception:
                    stats["enumerate_failed"] += 1
                    continue
                if not candidates:
                    stats["no_candidates"] += 1
                    continue

                for move in moves:
                    if max_records > 0 and len(records) >= max_records:
                        break
                    if len(move) < 3:
                        continue
                    from_pid, ships = int(move[0]), int(move[2])
                    _, tgt_idx = _decode_replay_target(obs, move)
                    if tgt_idx is None:
                        stats["target_decode_failed"] += 1
                        continue
                    target = planets[tgt_idx]
                    target_pid = int(target[0])
                    if not _target_owner_ok(int(target[1]), seat, target_owner):
                        stats[f"positive_skipped_target_owner_{target_owner}"] += 1
                        continue
                    if candidate_recent_capture_window > 0 and target_pid not in recent_capture_pids:
                        stats["positive_skipped_not_recent_capture"] += 1
                        continue
                    if candidate_threatened_only and target_pid not in threatened_pids:
                        stats["positive_skipped_not_threatened"] += 1
                        continue
                    replay_score = _score_replay_move(obs, from_pid, target_pid, ships)
                    if replay_score is None:
                        stats["replay_score_failed"] += 1
                        continue
                    if not bool(replay_score.get("valid")):
                        stats["positive_invalid"] += 1
                        continue
                    pos_candidate = _candidate_from_replay_score(replay_score)
                    negatives = _negative_candidates(
                        candidates,
                        planets=planets,
                        player=seat,
                        source_id=from_pid,
                        target_id=target_pid,
                        ships=ships,
                        eta=int(pos_candidate.eta),
                        target_owner=target_owner,
                        score_floor=float(pos_candidate.score),
                        score_slack=score_slack,
                        max_eta_gap=max_eta_gap,
                        limit=max(1, negatives_per_positive),
                    )
                    if not negatives:
                        stats["no_negatives"] += 1
                        continue
                    group: list[dict] = [{
                        "state_id": state_id,
                        "features": _candidate_features(obs, pos_candidate),
                        "label": 1,
                        "rank": 0,
                        "producer_score": float(pos_candidate.score),
                        "candidate": pos_candidate.to_dict(),
                    }]
                    for rank, neg in enumerate(negatives, start=1):
                        group.append({
                            "state_id": state_id,
                            "features": _candidate_features(obs, neg),
                            "label": 0,
                            "rank": rank,
                            "producer_score": float(neg.score),
                            "candidate": neg.to_dict(),
                        })
                    records.extend(group)
                    stats["states_added"] += 1
                    stats["records_added"] += len(group)
                    stats["positive_records"] += 1
                    stats["negative_records"] += len(group) - 1
                    stats[f"positive_target_owner_{int(target[1])}"] += 1
                    state_id += 1
            if max_records > 0 and len(records) >= max_records:
                break
        if max_records > 0 and len(records) >= max_records:
            break

    summary = {
        "config": {
            "replay_dirs": replay_dirs,
            "glob": glob_pattern,
            "max_replays": max_replays,
            "seat_mode": seat_mode,
            "player_slot": player_slot,
            "winner_name_filters": winner_name_filters,
            "steps_min": steps_min,
            "steps_max": steps_max,
            "target_owner": target_owner,
            "state_recent_capture_window": state_recent_capture_window,
            "candidate_recent_capture_window": candidate_recent_capture_window,
            "inbound_threat_horizon": inbound_threat_horizon,
            "candidate_threatened_only": candidate_threatened_only,
            "negatives_per_positive": negatives_per_positive,
            "score_slack": score_slack,
            "max_eta_gap": max_eta_gap,
            "max_records": max_records,
        },
        "feature_names": FEATURE_NAMES,
        "records": len(records),
        "states": state_id,
        "subjects": dict(subjects.most_common(20)),
        "stats": dict(stats),
    }
    if records:
        summary["positive_rate"] = sum(r["label"] for r in records) / len(records)
    return records, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay-dir", action="append", required=True)
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--max-replays", type=int, default=0)
    ap.add_argument("--seat-mode", choices=["winner", "all", "slot"], default="winner")
    ap.add_argument("--player-slot", type=int, default=None)
    ap.add_argument("--winner-name-filter", action="append", default=[])
    ap.add_argument("--steps-min", type=int, default=1)
    ap.add_argument("--steps-max", type=int, default=160)
    ap.add_argument("--target-owner", choices=["any", "own", "not-own", "neutral", "enemy"], default="any")
    ap.add_argument("--state-recent-capture-window", type=int, default=0)
    ap.add_argument("--candidate-recent-capture-window", type=int, default=0)
    ap.add_argument("--inbound-threat-horizon", type=int, default=0)
    ap.add_argument("--candidate-threatened-only", action="store_true")
    ap.add_argument("--negatives-per-positive", type=int, default=8)
    ap.add_argument("--score-slack", type=float, default=5.0)
    ap.add_argument("--max-eta-gap", type=int, default=-1)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--train-steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--records-out", required=True)
    ap.add_argument("--reranker-out", default="")
    ap.add_argument("--summary-out", default="")
    args = ap.parse_args()

    records, summary = build_records(
        replay_dirs=args.replay_dir,
        glob_pattern=args.glob,
        max_replays=args.max_replays,
        seat_mode=args.seat_mode,
        player_slot=args.player_slot,
        winner_name_filters=args.winner_name_filter,
        steps_min=args.steps_min,
        steps_max=args.steps_max,
        target_owner=args.target_owner,
        state_recent_capture_window=args.state_recent_capture_window,
        candidate_recent_capture_window=args.candidate_recent_capture_window,
        inbound_threat_horizon=args.inbound_threat_horizon,
        candidate_threatened_only=args.candidate_threatened_only,
        negatives_per_positive=args.negatives_per_positive,
        score_slack=args.score_slack,
        max_eta_gap=args.max_eta_gap,
        max_records=args.max_records,
    )

    records_out = Path(args.records_out)
    records_out.parent.mkdir(parents=True, exist_ok=True)
    with records_out.open("wb") as f:
        pickle.dump(records, f)
    summary["records_out"] = str(records_out)

    if records and args.reranker_out:
        reranker = train_reranker(records, args.train_steps, args.lr, args.seed)
        payload = {
            "feature_names": FEATURE_NAMES,
            "weights": reranker["weights"],
            "bias": reranker["bias"],
            "mean": reranker["mean"],
            "std": reranker["std"],
            "config": summary["config"],
            "metrics": reranker["metrics"],
        }
        reranker_out = Path(args.reranker_out)
        reranker_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, reranker_out)
        summary["reranker_out"] = str(reranker_out)
        summary["reranker_metrics"] = reranker["metrics"]

    if args.summary_out:
        summary_out = Path(args.summary_out)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
