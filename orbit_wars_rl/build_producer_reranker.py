"""Build and train a lightweight Producer-candidate reranker.

This is deliberately separate from the policy network. It distills Producer's
candidate ordering into a small supervised scorer over source-target-ship
candidates, avoiding another cross-entropy pass that overwrites the main BC
target prior.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_rl.analyze_producer_action_ranking import _enumerate_attack_candidates  # noqa: E402
from orbit_wars_rl.build_policy_teacher_bc import _selected_seats, _threatened_owned_pids  # noqa: E402
from orbit_wars_rl.build_supervised_bc import _normalize_obs, _team_names  # noqa: E402
from orbit_wars_rl.score_good_play_replays import (  # noqa: E402
    DEFAULT_STRONG_NAMES,
    QualityThresholds,
    _capture_steps_by_pid,
    score_replay,
)


FEATURE_NAMES = [
    "step",
    "owned_count",
    "enemy_count",
    "neutral_count",
    "source_ships",
    "send_ships",
    "send_frac",
    "eta",
    "target_ships",
    "target_prod",
    "floor_at_arrival",
    "surplus",
    "target_is_mine",
    "target_is_neutral",
    "target_is_enemy",
    "source_target_dist",
    "source_nearest_enemy_dist",
    "target_nearest_enemy_dist",
    "enemy_dist_gap",
    "producer_score_tanh",
]


def _nearest_enemy_dist(planet: list, planets: list, seat: int) -> float:
    enemies = [p for p in planets if int(p[1]) >= 0 and int(p[1]) != seat]
    if not enemies:
        return 100.0
    px, py = float(planet[2]), float(planet[3])
    return min(math.hypot(px - float(e[2]), py - float(e[3])) for e in enemies)


def _dist(a: list, b: list) -> float:
    return math.hypot(float(a[2]) - float(b[2]), float(a[3]) - float(b[3]))


def _candidate_features(obs: dict, candidate) -> list[float]:
    planets = obs.get("planets") or []
    seat = int(obs.get("player", 0))
    source = planets[int(candidate.source_idx)]
    target = planets[int(candidate.target_idx)]
    owned_count = sum(1 for p in planets if int(p[1]) == seat)
    enemy_count = sum(1 for p in planets if int(p[1]) >= 0 and int(p[1]) != seat)
    neutral_count = sum(1 for p in planets if int(p[1]) < 0)
    source_enemy_dist = _nearest_enemy_dist(source, planets, seat)
    target_enemy_dist = _nearest_enemy_dist(target, planets, seat)
    send_frac = float(candidate.ships) / max(float(candidate.source_ships), 1.0)
    surplus = float(candidate.ships) - float(candidate.floor_at_arrival)
    return [
        float(obs.get("step", 0)) / 200.0,
        float(owned_count) / 16.0,
        float(enemy_count) / 16.0,
        float(neutral_count) / 16.0,
        float(candidate.source_ships) / 420.0,
        float(candidate.ships) / 420.0,
        min(send_frac, 2.0),
        float(candidate.eta) / 32.0,
        float(target[5]) / 420.0,
        float(candidate.target_prod) / 6.0,
        float(candidate.floor_at_arrival) / 420.0,
        max(min(surplus / 100.0, 2.0), -2.0),
        1.0 if candidate.target_is_mine else 0.0,
        1.0 if candidate.target_is_neutral else 0.0,
        0.0 if candidate.target_is_mine or candidate.target_is_neutral else 1.0,
        _dist(source, target) / 100.0,
        source_enemy_dist / 100.0,
        target_enemy_dist / 100.0,
        (source_enemy_dist - target_enemy_dist) / 100.0,
        math.tanh(float(candidate.score) / 20.0),
    ]


def _target_owner_ok(candidate, mode: str) -> bool:
    if mode == "any":
        return True
    if mode == "own":
        return bool(candidate.target_is_mine)
    if mode == "not-own":
        return not bool(candidate.target_is_mine)
    if mode == "neutral":
        return bool(candidate.target_is_neutral)
    if mode == "enemy":
        return (not bool(candidate.target_is_mine)) and (not bool(candidate.target_is_neutral))
    raise ValueError(f"unknown target owner mode: {mode}")


def _quality_filter_active(
    min_replay_score: float,
    max_lost_cap: float,
    min_median_hold: int,
    min_cap_attack: float,
    min_planets50: int,
    require_accepted: bool,
) -> bool:
    return (
        min_replay_score > 0
        or max_lost_cap < 1.0
        or min_median_hold > 0
        or min_cap_attack > 0
        or min_planets50 > 0
        or require_accepted
    )


def _quality_winner_seat(
    replay_path: str,
    winner_name_filters: list[str],
    *,
    min_replay_score: float,
    max_lost_cap: float,
    min_median_hold: int,
    min_cap_attack: float,
    min_planets50: int,
    require_accepted: bool,
    stats: Counter,
) -> int | None:
    thresholds = QualityThresholds(
        min_cap_attack=max(min_cap_attack, 0.0),
        min_planets50=max(min_planets50, 0),
        max_lost_cap=max_lost_cap,
        min_score=max(min_replay_score, 0.0),
    )
    rows = score_replay(
        replay_path,
        thresholds,
        winner_name_filters,
        DEFAULT_STRONG_NAMES,
        require_known=False,
    )
    row = rows[0] if rows else {}
    metrics = row.get("metrics") or {}
    if not metrics:
        stats["quality_skipped_no_metrics"] += 1
        return None
    if require_accepted and not row.get("accepted", False):
        stats["quality_skipped_not_accepted"] += 1
        return None
    if float(row.get("score", 0.0)) < min_replay_score:
        stats["quality_skipped_low_score"] += 1
        return None
    if float(metrics.get("lost_cap", 1.0)) > max_lost_cap:
        stats["quality_skipped_lost_cap"] += 1
        return None
    if int(metrics.get("median_hold", 0)) < min_median_hold:
        stats["quality_skipped_median_hold"] += 1
        return None
    if float(metrics.get("cap_attack", 0.0)) < min_cap_attack:
        stats["quality_skipped_cap_attack"] += 1
        return None
    p50 = metrics.get("p50")
    if p50 is None or int(p50) < min_planets50:
        stats["quality_skipped_planets50"] += 1
        return None
    stats["quality_replays_kept"] += 1
    stats["quality_score_sum"] += float(row.get("score", 0.0))
    stats["quality_lost_cap_sum"] += float(metrics.get("lost_cap", 0.0))
    return int(row["seat"])


def build_records(
    replay_dirs: list[str],
    glob_pattern: str = "*.json",
    max_replays: int = 0,
    seat_mode: str = "all",
    player_slot: int | None = None,
    winner_name_filters: list[str] | None = None,
    min_replay_score: float = 0.0,
    max_lost_cap: float = 1.0,
    min_median_hold: int = 0,
    min_cap_attack: float = 0.0,
    min_planets50: int = 0,
    require_quality_accepted: bool = False,
    steps_min: int = 1,
    steps_max: int = 160,
    max_candidates_per_state: int = 16,
    positive_top_k: int = 1,
    score_min: float = -1e9,
    target_owner: str = "any",
    state_recent_capture_window: int = 0,
    candidate_recent_capture_window: int = 0,
    inbound_threat_horizon: int = 0,
    candidate_threatened_only: bool = False,
    max_records: int = 0,
) -> tuple[list[dict], dict]:
    replay_paths: list[str] = []
    for replay_dir in replay_dirs:
        replay_paths.extend(sorted(glob.glob(os.path.join(replay_dir, glob_pattern))))
    if max_replays > 0:
        replay_paths = replay_paths[:max_replays]

    records: list[dict] = []
    stats: Counter = Counter()
    subjects: Counter = Counter()
    stats["replays_found"] = len(replay_paths)
    winner_name_filters = winner_name_filters or []
    use_quality_filter = _quality_filter_active(
        min_replay_score,
        max_lost_cap,
        min_median_hold,
        min_cap_attack,
        min_planets50,
        require_quality_accepted,
    )
    state_id = 0

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
        seats = _selected_seats(replay, seat_mode, player_slot, winner_name_filters, stats)
        if use_quality_filter:
            quality_seat = _quality_winner_seat(
                replay_path,
                winner_name_filters,
                min_replay_score=min_replay_score,
                max_lost_cap=max_lost_cap,
                min_median_hold=min_median_hold,
                min_cap_attack=min_cap_attack,
                min_planets50=min_planets50,
                require_accepted=require_quality_accepted,
                stats=stats,
            )
            if quality_seat is None:
                continue
            seats = [seat for seat in seats if seat == quality_seat]
            if not seats:
                stats["quality_filtered_all_selected_seats"] += 1
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
                if seat >= len(steps[t - 1]):
                    stats["missing_agent_step"] += 1
                    continue
                obs = steps[t - 1][seat].get("observation")
                if not obs or "planets" not in obs:
                    stats["missing_obs"] += 1
                    continue
                obs_norm = _normalize_obs(obs, seat, t - 1)
                try:
                    candidates = _enumerate_attack_candidates(obs_norm)["candidates"]
                except Exception:
                    stats["enumerate_failed"] += 1
                    continue

                step = int(obs_norm.get("step", t - 1))
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
                threatened_pids = _threatened_owned_pids(obs_norm, seat, inbound_threat_horizon)
                if inbound_threat_horizon > 0 and not threatened_pids:
                    stats["states_skipped_no_inbound_threat"] += 1
                    continue

                valid = []
                for c in candidates:
                    if not c.valid or c.score < score_min:
                        continue
                    if not _target_owner_ok(c, target_owner):
                        stats[f"candidate_skipped_target_owner_{target_owner}"] += 1
                        continue
                    target_pid = int(c.target_id)
                    if candidate_recent_capture_window > 0 and target_pid not in recent_capture_pids:
                        stats["candidate_skipped_not_recent_capture"] += 1
                        continue
                    if candidate_threatened_only and target_pid not in threatened_pids:
                        stats["candidate_skipped_not_threatened"] += 1
                        continue
                    valid.append(c)
                if len(valid) < 2:
                    stats["states_skipped_lt2_candidates"] += 1
                    continue
                state_records = []
                for rank, candidate in enumerate(valid[:max_candidates_per_state]):
                    state_records.append({
                        "state_id": state_id,
                        "features": _candidate_features(obs_norm, candidate),
                        "label": int(rank < positive_top_k),
                        "rank": rank,
                        "producer_score": float(candidate.score),
                        "candidate": candidate.to_dict(),
                    })
                if not any(r["label"] for r in state_records):
                    continue
                if max_records > 0 and len(records) + len(state_records) > max_records:
                    state_records = state_records[: max_records - len(records)]
                records.extend(state_records)
                stats["states_added"] += 1
                stats["records_added"] += len(state_records)
                stats["positive_records"] += sum(r["label"] for r in state_records)
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
            "min_replay_score": min_replay_score,
            "max_lost_cap": max_lost_cap,
            "min_median_hold": min_median_hold,
            "min_cap_attack": min_cap_attack,
            "min_planets50": min_planets50,
            "require_quality_accepted": require_quality_accepted,
            "steps_min": steps_min,
            "steps_max": steps_max,
            "max_candidates_per_state": max_candidates_per_state,
            "positive_top_k": positive_top_k,
            "score_min": score_min,
            "target_owner": target_owner,
            "state_recent_capture_window": state_recent_capture_window,
            "candidate_recent_capture_window": candidate_recent_capture_window,
            "inbound_threat_horizon": inbound_threat_horizon,
            "candidate_threatened_only": candidate_threatened_only,
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
    if stats["quality_replays_kept"]:
        summary["quality_avg_score"] = stats["quality_score_sum"] / stats["quality_replays_kept"]
        summary["quality_avg_lost_cap"] = stats["quality_lost_cap_sum"] / stats["quality_replays_kept"]
    return records, summary


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float32)
    pos = labels > 0.5
    n_pos = int(pos.sum().item())
    n_neg = int((~pos).sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = ranks[pos].sum().item()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _group_metrics(records: list[dict], scores: torch.Tensor) -> dict:
    by_state: dict[int, list[tuple[dict, float]]] = defaultdict(list)
    for record, score in zip(records, scores.tolist()):
        by_state[int(record["state_id"])].append((record, float(score)))
    top1 = top3 = 0
    score_regret_sum = 0.0
    rank_sum = 0.0
    for group in by_state.values():
        chosen, _ = max(group, key=lambda item: item[1])
        best_score = max(float(r["producer_score"]) for r, _ in group)
        top1 += int(int(chosen["rank"]) == 0)
        top3 += int(int(chosen["rank"]) < 3)
        rank_sum += float(chosen["rank"])
        score_regret_sum += best_score - float(chosen["producer_score"])
    n = max(len(by_state), 1)
    return {
        "states": len(by_state),
        "top1": top1 / n,
        "top3": top3 / n,
        "avg_rank": rank_sum / n,
        "producer_score_regret": score_regret_sum / n,
    }


def train_reranker(records: list[dict], steps: int, lr: float, seed: int) -> dict:
    rng = random.Random(seed)
    states = sorted({int(r["state_id"]) for r in records})
    rng.shuffle(states)
    n_val = max(1, int(0.2 * len(states)))
    val_states = set(states[:n_val])
    train = [r for r in records if int(r["state_id"]) not in val_states]
    val = [r for r in records if int(r["state_id"]) in val_states]
    if not train or not val:
        raise ValueError("Need at least one train and validation state")

    x_train = torch.tensor([r["features"] for r in train], dtype=torch.float32)
    y_train = torch.tensor([r["label"] for r in train], dtype=torch.float32)
    x_val = torch.tensor([r["features"] for r in val], dtype=torch.float32)
    y_val = torch.tensor([r["label"] for r in val], dtype=torch.float32)

    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0).clamp(min=1e-6)
    x_train_n = (x_train - mean) / std
    x_val_n = (x_val - mean) / std

    w = torch.zeros(x_train.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    pos_weight = ((y_train == 0).sum().float() / (y_train == 1).sum().float().clamp(min=1.0)).clamp(min=0.1, max=20.0)

    for _ in range(steps):
        logits = x_train_n @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_scores = torch.sigmoid(x_train_n @ w + b)
        val_scores = torch.sigmoid(x_val_n @ w + b)
        val_pred = val_scores >= 0.5
        val_acc = (val_pred == (y_val > 0.5)).float().mean().item()
        val_auc = _auc(val_scores, y_val)
        val_base = max(y_val.mean().item(), 1.0 - y_val.mean().item())

    return {
        "weights": w.detach(),
        "bias": b.detach(),
        "mean": mean,
        "std": std,
        "metrics": {
            "train_records": len(train),
            "val_records": len(val),
            "train_states": len(states) - len(val_states),
            "val_states": len(val_states),
            "val_acc": val_acc,
            "val_base_acc": val_base,
            "val_auc": val_auc,
            "train_group": _group_metrics(train, train_scores),
            "val_group": _group_metrics(val, val_scores),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay-dir", action="append", required=True)
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--max-replays", type=int, default=0)
    ap.add_argument("--seat-mode", choices=["winner", "all", "slot"], default="all")
    ap.add_argument("--player-slot", type=int, default=None)
    ap.add_argument("--winner-name-filter", action="append", default=[],
                    help="With --seat-mode winner, keep only winners whose team name contains this substring.")
    ap.add_argument("--min-replay-score", type=float, default=0.0)
    ap.add_argument("--max-lost-cap", type=float, default=1.0)
    ap.add_argument("--min-median-hold", type=int, default=0)
    ap.add_argument("--min-cap-attack", type=float, default=0.0)
    ap.add_argument("--min-planets50", type=int, default=0)
    ap.add_argument("--require-quality-accepted", action="store_true",
                    help="Require the existing good-play scorer to mark the winning seat accepted.")
    ap.add_argument("--steps-min", type=int, default=1)
    ap.add_argument("--steps-max", type=int, default=160)
    ap.add_argument("--max-candidates-per-state", type=int, default=16)
    ap.add_argument("--positive-top-k", type=int, default=1)
    ap.add_argument("--score-min", type=float, default=-1e9)
    ap.add_argument("--target-owner", choices=["any", "own", "not-own", "neutral", "enemy"], default="any")
    ap.add_argument("--state-recent-capture-window", type=int, default=0,
                    help="If >0, keep only states where this seat recently captured any current owned planet.")
    ap.add_argument("--candidate-recent-capture-window", type=int, default=0,
                    help="If >0, keep only candidates targeting a recently captured owned planet.")
    ap.add_argument("--inbound-threat-horizon", type=int, default=0,
                    help="If >0, keep only states with an enemy fleet arriving at an owned planet within this horizon.")
    ap.add_argument("--candidate-threatened-only", action="store_true",
                    help="With --inbound-threat-horizon, keep only candidates targeting threatened owned planets.")
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
        min_replay_score=args.min_replay_score,
        max_lost_cap=args.max_lost_cap,
        min_median_hold=args.min_median_hold,
        min_cap_attack=args.min_cap_attack,
        min_planets50=args.min_planets50,
        require_quality_accepted=args.require_quality_accepted,
        steps_min=args.steps_min,
        steps_max=args.steps_max,
        max_candidates_per_state=args.max_candidates_per_state,
        positive_top_k=args.positive_top_k,
        score_min=args.score_min,
        target_owner=args.target_owner,
        state_recent_capture_window=args.state_recent_capture_window,
        candidate_recent_capture_window=args.candidate_recent_capture_window,
        inbound_threat_horizon=args.inbound_threat_horizon,
        candidate_threatened_only=args.candidate_threatened_only,
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
