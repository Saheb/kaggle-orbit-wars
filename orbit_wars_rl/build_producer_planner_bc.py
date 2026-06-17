"""Build BC samples from Producer-style planner candidate rankings.

This is narrower than full policy-teacher cloning. For each replay observation,
it enumerates Producer/Ajay planner candidates, keeps high-scoring valid
source-target-ship waves, and emits one ordinary BC sample per candidate. The
label source is the planner's internal action ranking, not the replay action and
not a rollout reward.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_rl.action_mask import _target_intercept_angle  # noqa: E402
from orbit_wars_rl.analyze_producer_action_ranking import _enumerate_attack_candidates  # noqa: E402
from orbit_wars_rl.bc import trajectory_to_training_sample  # noqa: E402
from orbit_wars_rl.build_policy_teacher_bc import _selected_seats  # noqa: E402
from orbit_wars_rl.build_supervised_bc import (  # noqa: E402
    _add_sample_stats,
    _clone_sample,
    _normalize_obs,
    _reinforce_label_count,
    _team_names,
)
from orbit_wars_rl.build_policy_teacher_bc import _threatened_owned_pids  # noqa: E402
from orbit_wars_rl.score_good_play_replays import _capture_steps_by_pid  # noqa: E402


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


def _candidate_to_move(obs: dict, candidate) -> list:
    planets = obs.get("planets") or []
    src = planets[int(candidate.source_idx)]
    target = planets[int(candidate.target_idx)]
    ships = int(candidate.ships)
    angle = _target_intercept_angle(src, target, ships, obs)
    return [int(candidate.source_id), float(angle), ships]


def _candidate_target_pid(candidate) -> int:
    return int(candidate.target_id)


def build(
    replay_dirs: list[str],
    glob_pattern: str = "*.json",
    max_replays: int = 0,
    seat_mode: str = "all",
    player_slot: int | None = None,
    steps_min: int = 1,
    steps_max: int = 120,
    top_k: int = 1,
    score_min: float = 1.5,
    target_owner: str = "any",
    recent_capture_window: int = 0,
    inbound_threat_horizon: int = 0,
    repeat: int = 1,
    reinforce_repeat: int = 1,
    max_samples: int = 0,
) -> tuple[list[dict], dict]:
    replay_paths: list[str] = []
    for replay_dir in replay_dirs:
        replay_paths.extend(sorted(glob.glob(os.path.join(replay_dir, glob_pattern))))
    if max_replays > 0:
        replay_paths = replay_paths[:max_replays]

    samples: list[dict] = []
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
            stats["seat_sequences_selected"] += 1
            captures = _capture_steps_by_pid(steps, seat) if recent_capture_window > 0 else {}
            for t in range(max(1, steps_min), t_end):
                if max_samples > 0 and len(samples) >= max_samples:
                    break
                if seat >= len(steps[t - 1]):
                    stats["missing_agent_step"] += 1
                    continue
                obs = steps[t - 1][seat].get("observation")
                if not obs or "planets" not in obs:
                    stats["missing_obs"] += 1
                    continue
                obs_norm = _normalize_obs(obs, seat, t - 1)
                recent_capture_pids: set[int] = set()
                if recent_capture_window > 0:
                    recent_capture_pids = {
                        pid for pid, cap_step in captures.items()
                        if 0 <= (t - 1) - cap_step <= recent_capture_window
                    }
                    if recent_capture_pids:
                        stats["recent_capture_frames_seen"] += 1
                    else:
                        stats["frames_skipped_no_recent_capture"] += 1
                        continue
                threatened_pids = _threatened_owned_pids(obs_norm, seat, inbound_threat_horizon)
                if inbound_threat_horizon > 0:
                    if threatened_pids:
                        stats["inbound_threat_frames_seen"] += 1
                        stats["inbound_threatened_planets"] += len(threatened_pids)
                    else:
                        stats["frames_skipped_no_inbound_threat"] += 1
                        continue
                try:
                    candidates = _enumerate_attack_candidates(obs_norm)["candidates"]
                except Exception:
                    stats["enumerate_failed"] += 1
                    continue

                kept = 0
                for candidate in candidates:
                    if kept >= top_k:
                        break
                    if not candidate.valid:
                        stats["candidate_invalid"] += 1
                        continue
                    if candidate.score < score_min:
                        stats["candidate_below_score"] += 1
                        continue
                    if not _target_owner_ok(candidate, target_owner):
                        stats[f"candidate_skipped_target_owner_{target_owner}"] += 1
                        continue
                    target_pid = _candidate_target_pid(candidate)
                    if recent_capture_window > 0 and target_pid not in recent_capture_pids:
                        stats["candidate_skipped_not_recent_capture"] += 1
                        continue
                    if inbound_threat_horizon > 0 and target_pid not in threatened_pids:
                        stats["candidate_skipped_not_threatened"] += 1
                        continue
                    move = _candidate_to_move(obs_norm, candidate)
                    sample = trajectory_to_training_sample({"obs": obs_norm, "action": [move]})
                    if sample is None:
                        stats["sample_build_failed"] += 1
                        continue
                    reinforce_labels = _reinforce_label_count(sample, obs_norm, seat)
                    sample_repeat = max(1, repeat)
                    if reinforce_labels:
                        sample_repeat = max(sample_repeat, reinforce_repeat)
                    for _ in range(sample_repeat):
                        samples.append(_clone_sample(sample))
                        _add_sample_stats(stats, sample)
                        if reinforce_labels:
                            stats["reinforce_labels"] += reinforce_labels
                    kept += 1
                    stats["candidate_samples_added"] += sample_repeat
                    stats["candidate_frames_with_sample"] += 1
                    stats["candidate_score_sum"] += float(candidate.score)
                    stats["candidate_valid_used"] += 1
                    if candidate.target_is_mine:
                        stats["candidate_own_target_used"] += 1
                    if candidate.target_is_neutral:
                        stats["candidate_neutral_target_used"] += 1
                if kept == 0:
                    stats["frames_no_candidate_kept"] += 1
            if max_samples > 0 and len(samples) >= max_samples:
                break
        if max_samples > 0 and len(samples) >= max_samples:
            break

    summary = {
        "config": {
            "replay_dirs": replay_dirs,
            "glob": glob_pattern,
            "max_replays": max_replays,
            "seat_mode": seat_mode,
            "player_slot": player_slot,
            "steps_min": steps_min,
            "steps_max": steps_max,
            "top_k": top_k,
            "score_min": score_min,
            "target_owner": target_owner,
            "recent_capture_window": recent_capture_window,
            "inbound_threat_horizon": inbound_threat_horizon,
            "repeat": repeat,
            "reinforce_repeat": reinforce_repeat,
            "max_samples": max_samples,
        },
        "subjects": dict(subjects.most_common(20)),
        "samples": len(samples),
        "stats": dict(stats),
    }
    if stats["valid_slots"]:
        summary["fire_slot_rate"] = stats["fire_slots"] / stats["valid_slots"]
    if samples:
        summary["avg_candidate_score"] = stats["candidate_score_sum"] / max(stats["candidate_valid_used"], 1)
    return samples, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay-dir", action="append", required=True)
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--max-replays", type=int, default=0)
    ap.add_argument("--seat-mode", choices=["winner", "all", "slot"], default="all")
    ap.add_argument("--player-slot", type=int, default=None)
    ap.add_argument("--steps-min", type=int, default=1)
    ap.add_argument("--steps-max", type=int, default=120)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--score-min", type=float, default=1.5)
    ap.add_argument("--target-owner", choices=["any", "own", "not-own", "neutral", "enemy"], default="any")
    ap.add_argument("--recent-capture-window", type=int, default=0)
    ap.add_argument("--inbound-threat-horizon", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--reinforce-repeat", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--samples-out", required=True)
    ap.add_argument("--summary-out", default="")
    args = ap.parse_args()

    samples, summary = build(
        replay_dirs=args.replay_dir,
        glob_pattern=args.glob,
        max_replays=args.max_replays,
        seat_mode=args.seat_mode,
        player_slot=args.player_slot,
        steps_min=args.steps_min,
        steps_max=args.steps_max,
        top_k=args.top_k,
        score_min=args.score_min,
        target_owner=args.target_owner,
        recent_capture_window=args.recent_capture_window,
        inbound_threat_horizon=args.inbound_threat_horizon,
        repeat=args.repeat,
        reinforce_repeat=args.reinforce_repeat,
        max_samples=args.max_samples,
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
