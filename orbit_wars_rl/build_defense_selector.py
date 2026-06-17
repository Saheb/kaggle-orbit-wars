"""Build and train a supervised selector for synthetic defense overlay moves.

This is deliberately separate from the policy model. It mines candidate
rear-source support opportunities from replay states and labels each candidate
by future outcome: did the threatened owned target survive the next H steps?
The goal is to test whether a lightweight supervised selector can filter the
deterministic defense overlay before wiring it into inference.
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
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_rl.build_policy_teacher_bc import _selected_seats  # noqa: E402
from orbit_wars_rl.build_supervised_bc import _normalize_obs, _team_names  # noqa: E402
from orbit_wars_rl.score_good_play_replays import _capture_steps_by_pid, _loss_steps_by_pid  # noqa: E402
from orbit_wars_rl.action_mask import _fleet_eta_to_planet  # noqa: E402


FEATURE_NAMES = [
    "step",
    "owned_count",
    "target_age",
    "target_ships",
    "target_prod",
    "inbound_ships",
    "min_eta",
    "projected_garrison",
    "need",
    "source_ships",
    "sendable",
    "need_sendable_ratio",
    "target_enemy_dist",
    "source_enemy_dist",
    "enemy_dist_gap",
    "source_target_dist",
    "support_eta",
    "eta_margin",
    "support_arrives_before",
]


def _nearest_enemy_dist(planet: list, planets: list, seat: int) -> float:
    enemies = [p for p in planets if int(p[1]) >= 0 and int(p[1]) != seat]
    if not enemies:
        return 100.0
    px, py = float(planet[2]), float(planet[3])
    return min(math.hypot(px - float(e[2]), py - float(e[3])) for e in enemies)


def _candidate_records(
    obs: dict,
    seat: int,
    captures: dict[int, int],
    losses: dict[int, list[int]],
    hold_horizon: int,
    garrison_floor: int,
    min_need: int,
    max_target_age: int,
    stats: Counter,
) -> list[dict]:
    planets = obs.get("planets") or []
    own_planets = [p for p in planets if int(p[1]) == seat]
    if len(own_planets) < 2:
        return []

    step = int(obs.get("step", 0))
    owned_count = len(own_planets)
    out: list[dict] = []
    for target in own_planets:
        target_pid = int(target[0])
        cap_step = captures.get(target_pid)
        if cap_step is None:
            stats["skip_not_captured"] += 1
            continue
        target_age = step - cap_step
        if target_age < 0 or (max_target_age > 0 and target_age > max_target_age):
            stats["skip_capture_age"] += 1
            continue

        inbound_ships = 0
        min_eta: int | None = None
        for fleet in obs.get("fleets") or []:
            if int(fleet[1]) == seat:
                continue
            eta = _fleet_eta_to_planet(fleet, target)
            if eta is None:
                continue
            inbound_ships += int(fleet[6]) if len(fleet) > 6 else 0
            min_eta = eta if min_eta is None else min(min_eta, eta)
        if inbound_ships <= 0 or min_eta is None:
            stats["skip_no_inbound"] += 1
            continue

        projected_garrison = float(target[5]) + float(target[6]) * min_eta
        need = int(math.ceil(inbound_ships + min_need - projected_garrison))
        if need < min_need:
            stats["skip_sufficient_garrison"] += 1
            continue

        target_enemy_dist = _nearest_enemy_dist(target, planets, seat)
        candidates = []
        for src in own_planets:
            if int(src[0]) == target_pid:
                continue
            sendable = int(src[5]) - int(garrison_floor)
            if sendable < min_need:
                continue
            source_enemy_dist = _nearest_enemy_dist(src, planets, seat)
            if source_enemy_dist <= target_enemy_dist:
                continue
            source_target_dist = math.hypot(float(src[2]) - float(target[2]), float(src[3]) - float(target[3]))
            candidates.append((sendable, source_enemy_dist, source_target_dist, src))
        if not candidates:
            stats["skip_no_source"] += 1
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        sendable, source_enemy_dist, source_target_dist, src = candidates[0]
        ships = min(sendable, max(need, min_need))
        support_speed = 1.0 + (6.0 - 1.0) * (math.log(max(ships, 1)) / math.log(1000.0)) ** 1.5
        support_speed = min(support_speed, 6.0)
        support_gap = float(src[4]) + 0.1 + float(target[4])
        support_eta = max(1, int(math.ceil(max(0.0, source_target_dist - support_gap) / max(support_speed, 1e-6))))
        eta_margin = float(min_eta - support_eta)
        future_loss = any(step < loss_step <= step + hold_horizon for loss_step in losses.get(target_pid, []))
        label = 0 if future_loss else 1
        features = [
            float(step),
            float(owned_count),
            float(target_age),
            float(target[5]),
            float(target[6]),
            float(inbound_ships),
            float(min_eta),
            float(projected_garrison),
            float(need),
            float(src[5]),
            float(sendable),
            float(need) / max(float(sendable), 1.0),
            float(target_enemy_dist),
            float(source_enemy_dist),
            float(source_enemy_dist - target_enemy_dist),
            float(source_target_dist),
            float(support_eta),
            eta_margin,
            float(support_eta <= min_eta),
        ]
        out.append({
            "features": features,
            "label": label,
            "step": step,
            "target_pid": target_pid,
            "source_pid": int(src[0]),
            "future_loss": future_loss,
        })
        stats["candidates"] += 1
        stats["positive_survive"] += int(label == 1)
        stats["negative_lost"] += int(label == 0)

    return out


def build_records(
    replay_dirs: list[str],
    glob_pattern: str = "*.json",
    max_replays: int = 0,
    seat_mode: str = "all",
    player_slot: int | None = None,
    steps_min: int = 16,
    steps_max: int = 180,
    hold_horizon: int = 30,
    garrison_floor: int = 10,
    min_need: int = 5,
    max_target_age: int = 40,
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
        seats = _selected_seats(replay, seat_mode, player_slot, [], stats)
        names = _team_names(replay)
        t_end = len(steps) if steps_max <= 0 else min(len(steps), steps_max + 1)

        for seat in seats:
            subjects[names[seat] if seat < len(names) else f"player_{seat}"] += 1
            captures = _capture_steps_by_pid(steps, seat)
            losses = _loss_steps_by_pid(steps, seat)
            for t in range(max(1, steps_min), t_end):
                if seat >= len(steps[t - 1]):
                    stats["missing_agent_step"] += 1
                    continue
                obs = steps[t - 1][seat].get("observation")
                if not obs or "planets" not in obs:
                    stats["missing_obs"] += 1
                    continue
                obs_norm = _normalize_obs(obs, seat, t - 1)
                records.extend(_candidate_records(
                    obs_norm,
                    seat,
                    captures,
                    losses,
                    hold_horizon,
                    garrison_floor,
                    min_need,
                    max_target_age,
                    stats,
                ))

    summary = {
        "config": {
            "replay_dirs": replay_dirs,
            "glob": glob_pattern,
            "max_replays": max_replays,
            "seat_mode": seat_mode,
            "player_slot": player_slot,
            "steps_min": steps_min,
            "steps_max": steps_max,
            "hold_horizon": hold_horizon,
            "garrison_floor": garrison_floor,
            "min_need": min_need,
            "max_target_age": max_target_age,
        },
        "feature_names": FEATURE_NAMES,
        "records": len(records),
        "subjects": dict(subjects.most_common(20)),
        "stats": dict(stats),
    }
    if records:
        summary["positive_rate"] = sum(r["label"] for r in records) / len(records)
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


def train_selector(records: list[dict], steps: int, lr: float, seed: int) -> dict:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(0.2 * len(shuffled)))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    if not train:
        raise ValueError("No training records")

    x_train = torch.tensor([r["features"] for r in train], dtype=torch.float32)
    y_train = torch.tensor([r["label"] for r in train], dtype=torch.float32)
    x_val = torch.tensor([r["features"] for r in val], dtype=torch.float32)
    y_val = torch.tensor([r["label"] for r in val], dtype=torch.float32)

    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0).clamp(min=1e-6)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    w = torch.zeros(x_train.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    pos_weight = ((y_train == 0).sum().float() / (y_train == 1).sum().float().clamp(min=1.0)).clamp(min=0.1, max=10.0)

    for _ in range(steps):
        logits = x_train @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        val_logits = x_val @ w + b
        val_scores = torch.sigmoid(val_logits)
        pred = val_scores >= 0.5
        acc = (pred == (y_val > 0.5)).float().mean().item()
        auc = _auc(val_scores, y_val)
        base = max(y_val.mean().item(), 1.0 - y_val.mean().item())

    return {
        "weights": w.detach(),
        "bias": b.detach(),
        "mean": mean,
        "std": std,
        "metrics": {
            "val_acc": acc,
            "val_auc": auc,
            "val_base_acc": base,
            "val_positive_rate": y_val.mean().item(),
            "train_records": len(train),
            "val_records": len(val),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay-dir", action="append", required=True)
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--max-replays", type=int, default=0)
    ap.add_argument("--seat-mode", choices=["winner", "all", "slot"], default="all")
    ap.add_argument("--player-slot", type=int, default=None)
    ap.add_argument("--steps-min", type=int, default=16)
    ap.add_argument("--steps-max", type=int, default=180)
    ap.add_argument("--hold-horizon", type=int, default=30)
    ap.add_argument("--garrison-floor", type=int, default=10)
    ap.add_argument("--min-need", type=int, default=5)
    ap.add_argument("--max-target-age", type=int, default=40)
    ap.add_argument("--train-steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--records-out", required=True)
    ap.add_argument("--selector-out", default="")
    ap.add_argument("--summary-out", default="")
    args = ap.parse_args()

    records, summary = build_records(
        replay_dirs=args.replay_dir,
        glob_pattern=args.glob,
        max_replays=args.max_replays,
        seat_mode=args.seat_mode,
        player_slot=args.player_slot,
        steps_min=args.steps_min,
        steps_max=args.steps_max,
        hold_horizon=args.hold_horizon,
        garrison_floor=args.garrison_floor,
        min_need=args.min_need,
        max_target_age=args.max_target_age,
    )
    out = Path(args.records_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(records, f)
    summary["records_out"] = str(out)

    if records and args.selector_out:
        selector = train_selector(records, args.train_steps, args.lr, args.seed)
        payload = {
            "feature_names": FEATURE_NAMES,
            "weights": selector["weights"],
            "bias": selector["bias"],
            "mean": selector["mean"],
            "std": selector["std"],
            "config": summary["config"],
            "metrics": selector["metrics"],
        }
        selector_out = Path(args.selector_out)
        selector_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, selector_out)
        summary["selector_out"] = str(selector_out)
        summary["selector_metrics"] = selector["metrics"]

    if args.summary_out:
        summary_out = Path(args.summary_out)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
