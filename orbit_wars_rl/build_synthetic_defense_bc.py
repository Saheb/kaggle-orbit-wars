"""Build synthetic defense BC samples from threatened replay states.

Unlike replay cloning or policy-teacher BC, these labels are generated from a
simple defensive heuristic: when an owned planet has enemy inbound and projected
garrison is insufficient, send ships from a rear owned planet to the threatened
planet. This directly targets the supervised retention gap: source selection and
ship sizing for holding captures.
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
    _normalize_obs,
    _team_names,
)
from orbit_wars_rl.build_policy_teacher_bc import _selected_seats  # noqa: E402
from orbit_wars_rl.score_good_play_replays import _synthetic_defense_moves  # noqa: E402
from orbit_wars_rl.score_good_play_replays import _capture_steps_by_pid, _loss_steps_by_pid  # noqa: E402


def _eligible_hold_success_pids(
    obs: dict,
    seat: int,
    captures: dict[int, int],
    losses: dict[int, list[int]],
    recent_capture_window: int,
    hold_success_horizon: int,
    stats: Counter,
) -> set[int] | None:
    if recent_capture_window <= 0 and hold_success_horizon <= 0:
        return None

    step = int(obs.get("step", 0))
    eligible: set[int] = set()
    for planet in obs.get("planets") or []:
        if int(planet[1]) != seat:
            continue
        pid = int(planet[0])
        if recent_capture_window > 0:
            cap_step = captures.get(pid)
            if cap_step is None:
                stats["hold_success_skipped_not_captured"] += 1
                continue
            age = step - cap_step
            if age < 0 or age > recent_capture_window:
                stats["hold_success_skipped_capture_age"] += 1
                continue
        if hold_success_horizon > 0:
            if any(step < loss_step <= step + hold_success_horizon for loss_step in losses.get(pid, [])):
                stats["hold_success_skipped_future_loss"] += 1
                continue
        eligible.add(pid)

    if not eligible:
        stats["hold_success_no_eligible_targets"] += 1
    return eligible


def build(
    replay_dirs: list[str],
    glob_pattern: str = "*.json",
    max_replays: int = 0,
    seat_mode: str = "all",
    player_slot: int | None = None,
    winner_name_filters: list[str] | None = None,
    steps_min: int = 1,
    steps_max: int = 0,
    action_repeat: int = 1,
    garrison_floor: int = 10,
    min_need: int = 5,
    recent_capture_window: int = 0,
    hold_success_horizon: int = 0,
    max_samples: int = 0,
    seed: int = 0,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    samples: list[dict] = []
    stats: Counter = Counter()
    subjects: Counter = Counter()
    filters = winner_name_filters or []

    replay_paths: list[str] = []
    for replay_dir in replay_dirs:
        replay_paths.extend(sorted(glob.glob(os.path.join(replay_dir, glob_pattern))))
    if max_replays > 0:
        replay_paths = replay_paths[:max_replays]
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
        seats = _selected_seats(replay, seat_mode, player_slot, filters, stats)
        if not seats:
            continue
        names = _team_names(replay)
        t_end = len(steps) if steps_max <= 0 else min(len(steps), steps_max + 1)

        for seat in seats:
            subject = names[seat] if seat < len(names) else f"player_{seat}"
            subjects[subject] += 1
            stats["seat_sequences_selected"] += 1
            captures = _capture_steps_by_pid(steps, seat) if recent_capture_window > 0 else {}
            losses = _loss_steps_by_pid(steps, seat) if hold_success_horizon > 0 else {}

            for t in range(1, t_end):
                if max_samples > 0 and len(samples) >= max_samples:
                    break
                if t < max(1, steps_min):
                    stats["warmup_frames"] += 1
                    continue
                if seat >= len(steps[t - 1]):
                    stats["missing_agent_step"] += 1
                    continue
                obs = steps[t - 1][seat].get("observation")
                if not obs or "planets" not in obs:
                    stats["missing_obs"] += 1
                    continue
                obs_norm = _normalize_obs(obs, seat, t - 1)
                eligible_target_pids = _eligible_hold_success_pids(
                    obs_norm,
                    seat,
                    captures,
                    losses,
                    recent_capture_window,
                    hold_success_horizon,
                    stats,
                )
                if eligible_target_pids is not None and not eligible_target_pids:
                    continue

                synthetic_moves = _synthetic_defense_moves(
                    obs_norm,
                    moves=[],
                    seat=seat,
                    stats=stats,
                    garrison_floor=garrison_floor,
                    min_need=min_need,
                    eligible_target_pids=eligible_target_pids,
                )
                if not synthetic_moves:
                    stats["frames_without_synthetic_defense"] += 1
                    continue
                stats["synthetic_defense_frames"] += 1

                rng.shuffle(synthetic_moves)
                for move in synthetic_moves:
                    if max_samples > 0 and len(samples) >= max_samples:
                        break
                    sample = trajectory_to_training_sample({
                        "obs": obs_norm,
                        "action": [move],
                    })
                    if sample is None:
                        stats["sample_build_failed"] += 1
                        continue
                    repeat = max(1, action_repeat)
                    for _ in range(repeat):
                        samples.append(_clone_sample(sample))
                        _add_sample_stats(stats, sample)
                    stats["synthetic_defense_samples_added"] += repeat

    summary = {
        "config": {
            "replay_dirs": replay_dirs,
            "glob": glob_pattern,
            "max_replays": max_replays,
            "seat_mode": seat_mode,
            "player_slot": player_slot,
            "winner_name_filters": filters,
            "steps_min": steps_min,
            "steps_max": steps_max,
            "action_repeat": action_repeat,
            "garrison_floor": garrison_floor,
            "min_need": min_need,
            "recent_capture_window": recent_capture_window,
            "hold_success_horizon": hold_success_horizon,
            "max_samples": max_samples,
            "seed": seed,
        },
        "subjects": dict(subjects.most_common(20)),
        "samples": len(samples),
        "stats": dict(stats),
    }
    valid_slots = stats["valid_slots"]
    if valid_slots:
        summary["fire_slot_rate"] = stats["fire_slots"] / valid_slots
    if samples:
        summary["decision_sample_share"] = stats["synthetic_defense_samples_added"] / len(samples)
    return samples, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay-dir", action="append", required=True)
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--max-replays", type=int, default=0)
    ap.add_argument("--seat-mode", choices=["winner", "all", "slot"], default="all")
    ap.add_argument("--player-slot", type=int, default=None)
    ap.add_argument("--winner-name", action="append", default=[])
    ap.add_argument("--steps-min", type=int, default=1)
    ap.add_argument("--steps-max", type=int, default=0)
    ap.add_argument("--action-repeat", type=int, default=1)
    ap.add_argument("--garrison-floor", type=int, default=10)
    ap.add_argument("--min-need", type=int, default=5)
    ap.add_argument("--recent-capture-window", type=int, default=0,
                    help="If >0, only generate defense labels for owned targets "
                         "captured within this many steps.")
    ap.add_argument("--hold-success-horizon", type=int, default=0,
                    help="If >0, only generate defense labels for owned targets "
                         "that are not lost within this many future steps.")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples-out", required=True)
    ap.add_argument("--summary-out", default="")
    args = ap.parse_args()

    samples, summary = build(
        replay_dirs=args.replay_dir,
        glob_pattern=args.glob,
        max_replays=args.max_replays,
        seat_mode=args.seat_mode,
        player_slot=args.player_slot,
        winner_name_filters=args.winner_name,
        steps_min=args.steps_min,
        steps_max=args.steps_max,
        action_repeat=args.action_repeat,
        garrison_floor=args.garrison_floor,
        min_need=args.min_need,
        recent_capture_window=args.recent_capture_window,
        hold_success_horizon=args.hold_success_horizon,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    out_path = Path(args.samples_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(samples, f)
    summary["samples_out"] = str(out_path)

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
