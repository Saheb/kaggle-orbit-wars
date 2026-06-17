"""Build BC samples by running a policy teacher on replay observations.

This is the supervised-only alternative to replay cloning: choose a teacher
agent (for example Ajay/Producer), run it on historical game states, and train
our model to imitate the teacher's action labels in the standard ``bc.py` sample
format. No rewards, PPO, or RL checkpoint are involved.

Timing convention: each sample uses the observation from steps[t - 1]. Unlike
replay cloning, the label is not steps[t]'s recorded action; it is the teacher's
fresh action on that observation.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
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


def _load_teacher_module(agent_path: str):
    spec = importlib.util.spec_from_file_location("policy_teacher_agent", agent_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import teacher agent: {agent_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "agent"):
        raise AttributeError(f"No agent(obs) function found in {agent_path}")
    return module


def _reset_teacher(module) -> None:
    runtime = getattr(module, "_RUNTIME", None)
    if runtime is not None and hasattr(runtime, "reset"):
        runtime.reset()


def _selected_seats(replay: dict, seat_mode: str, player_slot: int | None,
                    winner_name_filters: list[str], stats: Counter) -> list[int]:
    steps = replay.get("steps") or []
    n_players = len(steps[0]) if steps else 0
    if n_players <= 0:
        return []

    if seat_mode == "slot":
        if player_slot is None or player_slot < 0 or player_slot >= n_players:
            stats["slot_oob"] += 1
            return []
        return [player_slot]

    if seat_mode == "all":
        return list(range(n_players))

    winner = _strict_winner(replay.get("rewards"))
    if winner is None:
        stats["no_strict_winner"] += 1
        return []
    names = _team_names(replay)
    name = names[winner] if winner < len(names) else f"player_{winner}"
    if not _matches_any(name, winner_name_filters):
        stats["winner_filtered_out"] += 1
        return []
    return [winner]


def _target_owner_matches(sample: dict, obs: dict, player: int, mode: str) -> bool:
    if mode == "any":
        return True
    planets = obs.get("planets") or []
    fired = sample["fire_target"].bool() & sample["slot_valid"].bool()
    targets = sample["target_target"]
    matched = False
    for slot in range(min(len(fired), len(targets))):
        if not bool(fired[slot]):
            continue
        tidx = int(targets[slot].item())
        if tidx < 0 or tidx >= len(planets):
            return False
        owner = int(planets[tidx][1])
        if mode == "own":
            ok = owner == player
        elif mode == "not-own":
            ok = owner != player
        elif mode == "neutral":
            ok = owner == -1
        elif mode == "enemy":
            ok = owner >= 0 and owner != player
        else:
            raise ValueError(f"unknown target owner mode: {mode}")
        if not ok:
            return False
        matched = True
    return matched


def _fleet_speed(ships: int) -> float:
    if ships <= 0:
        return 1.0
    speed = 1.0 + (6.0 - 1.0) * (math.log(max(ships, 1)) / math.log(1000.0)) ** 1.5
    return min(speed, 6.0)


def _fleet_eta_to_planet(fleet: list, planet: list) -> int | None:
    tx, ty, tr = float(planet[2]), float(planet[3]), float(planet[4])
    fx, fy, angle = float(fleet[2]), float(fleet[3]), float(fleet[4])
    c, s = math.cos(angle), math.sin(angle)
    vx, vy = tx - fx, ty - fy
    along = vx * c + vy * s
    perp = abs(vx * s - vy * c)
    if along <= 0 or perp >= tr + 1.5:
        return None
    speed = _fleet_speed(int(fleet[6]) if len(fleet) > 6 else 1)
    return max(1, int(math.ceil(max(0.0, along - tr) / speed)))


def _threatened_owned_pids(obs: dict, player: int, horizon: int) -> set[int]:
    if horizon <= 0:
        return set()
    planets = obs.get("planets") or []
    own_planets = [p for p in planets if int(p[1]) == player]
    threatened: set[int] = set()
    for fleet in obs.get("fleets") or []:
        if int(fleet[1]) == player:
            continue
        for planet in own_planets:
            eta = _fleet_eta_to_planet(fleet, planet)
            if eta is not None and eta <= horizon:
                threatened.add(int(planet[0]))
    return threatened


def _targets_threatened_owned(sample: dict, obs: dict, threatened_pids: set[int]) -> bool:
    if not threatened_pids:
        return False
    planets = obs.get("planets") or []
    fired = sample["fire_target"].bool() & sample["slot_valid"].bool()
    targets = sample["target_target"]
    matched = False
    for slot in range(min(len(fired), len(targets))):
        if not bool(fired[slot]):
            continue
        tidx = int(targets[slot].item())
        if tidx < 0 or tidx >= len(planets):
            return False
        if int(planets[tidx][0]) not in threatened_pids:
            return False
        matched = True
    return matched


def _filter_moves_by_source_limit(
    moves: list,
    max_moves_per_source: int,
    stats: Counter,
) -> list:
    if max_moves_per_source <= 0:
        return moves
    source_counts = Counter()
    for move in moves:
        if len(move) >= 1:
            source_counts[int(move[0])] += 1
    filtered = [
        move for move in moves
        if len(move) >= 1 and source_counts[int(move[0])] <= max_moves_per_source
    ]
    skipped = len(moves) - len(filtered)
    if skipped:
        stats["frames_filtered_too_many_teacher_moves_per_source"] += 1
        stats["moves_skipped_too_many_teacher_moves_per_source"] += skipped
    return filtered


def build(
    replay_dirs: list[str],
    teacher_agent: str,
    glob_pattern: str = "*.json",
    max_replays: int = 0,
    seat_mode: str = "winner",
    player_slot: int | None = None,
    winner_name_filters: list[str] | None = None,
    steps_min: int = 1,
    steps_max: int = 0,
    noop_keep_prob: float = 0.02,
    action_repeat: int = 1,
    reinforce_repeat: int = 1,
    split_moves: bool = False,
    max_teacher_moves_per_frame: int = 0,
    max_teacher_moves_per_source: int = 0,
    target_owner: str = "any",
    inbound_threat_horizon: int = 0,
    seed: int = 0,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    filters = winner_name_filters or []
    teacher_module = _load_teacher_module(teacher_agent)
    teacher_fn = teacher_module.agent
    samples: list[dict] = []
    stats: Counter = Counter()
    subjects: Counter = Counter()

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
            _reset_teacher(teacher_module)
            subject = names[seat] if seat < len(names) else f"player_{seat}"
            subjects[subject] += 1
            stats["seat_sequences_selected"] += 1

            for t in range(1, t_end):
                if seat >= len(steps[t - 1]):
                    stats["missing_agent_step"] += 1
                    continue
                obs = steps[t - 1][seat].get("observation")
                if not obs or "planets" not in obs:
                    stats["missing_obs"] += 1
                    continue
                obs_norm = _normalize_obs(obs, seat, t - 1)

                try:
                    teacher_action = teacher_fn(obs_norm)
                except Exception:
                    stats["teacher_action_failed"] += 1
                    continue

                if t < max(1, steps_min):
                    stats["teacher_warmup_frames"] += 1
                    continue

                threatened_pids = _threatened_owned_pids(obs_norm, seat, inbound_threat_horizon)
                if inbound_threat_horizon > 0:
                    if threatened_pids:
                        stats["inbound_threat_frames_seen"] += 1
                        stats["inbound_threatened_planets"] += len(threatened_pids)
                    else:
                        stats["frames_skipped_no_inbound_threat"] += 1
                        continue

                moves = _moves(teacher_action)
                if moves:
                    if max_teacher_moves_per_frame > 0 and len(moves) > max_teacher_moves_per_frame:
                        stats["frames_skipped_too_many_teacher_moves"] += 1
                        stats["moves_skipped_too_many_teacher_moves"] += len(moves)
                        continue
                    moves = _filter_moves_by_source_limit(
                        moves,
                        max_teacher_moves_per_source,
                        stats,
                    )
                    if not moves:
                        stats["frames_skipped_no_moves_after_source_filter"] += 1
                        continue
                    stats["teacher_decision_frames_seen"] += 1
                    stats["teacher_moves_seen"] += len(moves)
                    action_batches = [[move] for move in moves] if split_moves else [moves]
                    if split_moves:
                        stats["teacher_split_move_frames"] += 1
                        stats["teacher_split_move_labels"] += len(moves)
                else:
                    stats["teacher_noop_frames_seen"] += 1
                    if rng.random() >= noop_keep_prob:
                        continue
                    stats["teacher_noop_frames_kept"] += 1
                    action_batches = [[]]

                for action_batch in action_batches:
                    repeat = max(1, action_repeat) if action_batch else 1
                    sample = trajectory_to_training_sample({
                        "obs": obs_norm,
                        "action": action_batch,
                    })
                    if sample is None:
                        stats["sample_build_failed"] += 1
                        continue
                    if not _target_owner_matches(sample, obs_norm, seat, target_owner):
                        stats[f"samples_skipped_target_owner_{target_owner}"] += 1
                        continue
                    if inbound_threat_horizon > 0:
                        if not _targets_threatened_owned(sample, obs_norm, threatened_pids):
                            stats["samples_skipped_target_not_threatened"] += 1
                            continue
                        stats["threat_target_samples_added"] += 1
                    reinforce_labels = _reinforce_label_count(sample, obs_norm, seat)
                    if reinforce_labels:
                        stats["reinforce_frames_seen"] += 1
                        stats["reinforce_labels_seen"] += reinforce_labels
                        repeat = max(repeat, reinforce_repeat)
                    for _ in range(repeat):
                        samples.append(_clone_sample(sample))
                        _add_sample_stats(stats, sample)
                        if reinforce_labels:
                            stats["reinforce_labels"] += reinforce_labels
                    if action_batch:
                        stats["teacher_decision_samples_added"] += repeat
                        if split_moves:
                            stats["teacher_split_move_samples_added"] += repeat
                        if reinforce_labels:
                            stats["reinforce_samples_added"] += repeat
                    else:
                        stats["teacher_noop_samples_added"] += 1

    summary = {
        "config": {
            "replay_dirs": replay_dirs,
            "glob": glob_pattern,
            "max_replays": max_replays,
            "teacher_agent": teacher_agent,
            "seat_mode": seat_mode,
            "player_slot": player_slot,
            "winner_name_filters": filters,
            "steps_min": steps_min,
            "steps_max": steps_max,
            "noop_keep_prob": noop_keep_prob,
            "action_repeat": action_repeat,
            "reinforce_repeat": reinforce_repeat,
            "split_moves": split_moves,
            "max_teacher_moves_per_frame": max_teacher_moves_per_frame,
            "max_teacher_moves_per_source": max_teacher_moves_per_source,
            "target_owner": target_owner,
            "inbound_threat_horizon": inbound_threat_horizon,
            "seed": seed,
        },
        "subjects": dict(subjects.most_common(20)),
        "samples": len(samples),
        "stats": dict(stats),
    }
    valid_slots = stats["valid_slots"]
    if valid_slots:
        summary["fire_slot_rate"] = stats["fire_slots"] / valid_slots
    if stats["teacher_decision_frames_seen"] + stats["teacher_noop_frames_seen"]:
        summary["decision_frame_rate_seen"] = stats["teacher_decision_frames_seen"] / (
            stats["teacher_decision_frames_seen"] + stats["teacher_noop_frames_seen"]
        )
    if samples:
        summary["decision_sample_share"] = stats["teacher_decision_samples_added"] / len(samples)
    return samples, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay-dir", action="append", required=True)
    ap.add_argument("--teacher-agent", required=True)
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--max-replays", type=int, default=0,
                    help="If >0, use only the first N sorted replay paths.")
    ap.add_argument("--seat-mode", choices=["winner", "all", "slot"], default="winner")
    ap.add_argument("--player-slot", type=int, default=None)
    ap.add_argument("--winner-name", action="append", default=[],
                    help="Substring filter for selected winners when --seat-mode=winner.")
    ap.add_argument("--steps-min", type=int, default=1)
    ap.add_argument("--steps-max", type=int, default=0)
    ap.add_argument("--noop-keep-prob", type=float, default=0.02)
    ap.add_argument("--action-repeat", type=int, default=1)
    ap.add_argument("--reinforce-repeat", type=int, default=1)
    ap.add_argument("--split-moves", action="store_true",
                    help="Emit one BC sample per teacher move instead of one "
                         "multi-action sample per frame. Useful for target-only "
                         "supervision of source/target choices.")
    ap.add_argument("--max-teacher-moves-per-frame", type=int, default=0,
                    help="If >0, skip frames where the teacher emits more than "
                         "this many valid moves.")
    ap.add_argument("--max-teacher-moves-per-source", type=int, default=0,
                    help="If >0, drop teacher moves from source planets that emit "
                         "more than this many moves in the same frame. This avoids "
                         "contradictory split-move labels for one-source/one-target "
                         "policy heads.")
    ap.add_argument("--target-owner", choices=["any", "own", "not-own", "neutral", "enemy"],
                    default="any",
                    help="Keep only samples whose fired targets all match this "
                         "owner class.")
    ap.add_argument("--inbound-threat-horizon", type=int, default=0,
                    help="If >0, keep only samples whose decoded target is an "
                         "owned planet with an enemy fleet arriving within this "
                         "many steps.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples-out", required=True)
    ap.add_argument("--summary-out", default="")
    args = ap.parse_args()

    if not 0.0 <= args.noop_keep_prob <= 1.0:
        raise SystemExit("--noop-keep-prob must be in [0, 1]")

    samples, summary = build(
        replay_dirs=args.replay_dir,
        teacher_agent=args.teacher_agent,
        glob_pattern=args.glob,
        max_replays=args.max_replays,
        seat_mode=args.seat_mode,
        player_slot=args.player_slot,
        winner_name_filters=args.winner_name,
        steps_min=args.steps_min,
        steps_max=args.steps_max,
        noop_keep_prob=args.noop_keep_prob,
        action_repeat=args.action_repeat,
        reinforce_repeat=args.reinforce_repeat,
        split_moves=args.split_moves,
        max_teacher_moves_per_frame=args.max_teacher_moves_per_frame,
        max_teacher_moves_per_source=args.max_teacher_moves_per_source,
        target_owner=args.target_owner,
        inbound_threat_horizon=args.inbound_threat_horizon,
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
