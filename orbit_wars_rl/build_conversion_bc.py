"""Build an opening-conversion-focused BC dataset.

This targets the failure mode where target ranking improves, but first captures
still arrive late because the opening spends too much ship capital before the
first productive capture.

The primary source is strong teacher replays filtered to openings with:
  - early opponent pressure
  - fast first capture
  - restrained ship spend before first capture

Optionally, the output can be mixed with the earlier target-relabel failure
dataset from build_tempo_bc.py so PPO gets both better ranking and better
opening conversion supervision.
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

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_rl.bc import trajectory_to_training_sample, _find_ship_bin  # noqa: E402


def _strict_winner_idx(data: dict) -> int | None:
    rewards = data.get("rewards") or []
    if len(rewards) != 2 or rewards[0] == rewards[1]:
        return None
    return 0 if rewards[0] > rewards[1] else 1


def _agent_name(data: dict, slot: int) -> str:
    agents = ((data.get("info") or {}).get("Agents") or [{}, {}])
    if slot < len(agents):
        return str(agents[slot].get("Name", ""))
    return ""


def _obs_from_step_agent(step_agent: dict, player_idx: int, step_idx: int) -> dict | None:
    obs = step_agent.get("observation")
    if not obs or "planets" not in obs:
        return None
    out = dict(obs)
    out["player"] = player_idx
    out.setdefault("step", step_idx)
    out.setdefault("angular_velocity", 0.0)
    out.setdefault("comet_planet_ids", [])
    out.setdefault("initial_planets", out["planets"])
    return out


def _obs_before_action(steps: list, player_idx: int, action_step_idx: int) -> dict | None:
    """Replay action at step t must be paired with observation at step t-1."""
    if action_step_idx <= 0 or action_step_idx - 1 >= len(steps):
        return None
    prev_agent = steps[action_step_idx - 1][player_idx]
    return _obs_from_step_agent(prev_agent, player_idx, action_step_idx - 1)


def _initial_planet_count(data: dict, slot: int) -> int:
    for step in data.get("steps") or []:
        agent = step[slot]
        obs = agent.get("observation")
        if not obs:
            continue
        return sum(1 for p in obs["planets"] if int(p[1]) == slot)
    return 0


def _first_capture_step(data: dict, slot: int) -> int | None:
    baseline = _initial_planet_count(data, slot)
    for step_idx, step in enumerate(data.get("steps") or []):
        obs = step[slot].get("observation")
        if not obs:
            continue
        owned = sum(1 for p in obs["planets"] if int(p[1]) == slot)
        if owned > baseline:
            return step_idx
    return None


def _first_launch_step(data: dict, slot: int) -> int | None:
    for step_idx, step in enumerate(data.get("steps") or []):
        actions = step[slot].get("action") or []
        if actions:
            return step_idx
    return None


def _launches_and_ships_before_step(data: dict, slot: int, limit_step: int | None) -> tuple[int, int]:
    launches = 0
    ships = 0
    cutoff = limit_step if limit_step is not None else len(data.get("steps") or [])
    for step_idx, step in enumerate(data.get("steps") or []):
        if step_idx >= cutoff:
            break
        actions = step[slot].get("action") or []
        launches += len(actions)
        ships += sum(int(a[2]) for a in actions if len(a) >= 3)
    return launches, ships


def _planet_idx_by_id(planets: list[list], planet_id: int) -> int | None:
    for idx, p in enumerate(planets):
        if int(p[0]) == int(planet_id):
            return idx
    return None


def _capture_cost(target: list, player_slot: int) -> int:
    owner = int(target[1])
    ships = int(target[5])
    production = int(target[6])
    if owner == -1:
        return ships + 1
    if owner != player_slot:
        return ships + production * 3 + 1
    return 0


def _max_sendable_ships(src_ships: int) -> int:
    """Absolute-mode action semantics: a launch may send the full ship count."""
    return max(0, int(src_ships))


def _slot_for_from_pid(obs: dict, player_idx: int, from_pid: int) -> int | None:
    from orbit_wars_rl.action_mask import compute_action_masks  # local import to avoid cycle

    masks = compute_action_masks(obs, player_idx)
    planets = obs["planets"]
    owned_indices = masks["owned_indices"].numpy()
    for slot in range(masks["owned_count"]):
        pidx = int(owned_indices[slot])
        if pidx < len(planets) and int(planets[pidx][0]) == int(from_pid):
            return slot
    return None


def _clone_sample(sample: dict) -> dict:
    out = {}
    for k, v in sample.items():
        out[k] = v.clone() if torch.is_tensor(v) else v
    return out


def _zero_action_targets(sample: dict) -> dict:
    out = _clone_sample(sample)
    out["fire_target"].zero_()
    out["ship_target"].zero_()
    out["target_target"].fill_(-1)
    return out


def _retarget_with_ship(
    sample: dict,
    slot: int,
    target_idx: int,
    ship_bin: int,
) -> dict:
    out = _zero_action_targets(sample)
    out["fire_target"][slot] = 1
    out["ship_target"][slot] = int(ship_bin)
    out["target_target"][slot] = int(target_idx)
    return out


def _first_fire_step(data: dict, slot: int, max_steps: int | None = None) -> int | None:
    limit = len(data.get("steps") or []) if max_steps is None else min(len(data.get("steps") or []), max_steps + 1)
    for step_idx in range(1, limit):
        actions = data["steps"][step_idx][slot].get("action") or []
        if actions:
            return step_idx
    return None


def build_teacher_conversion_samples(
    replay_paths: list[str],
    player_name_filters: list[str],
    steps_max: int,
    require_opponent_first_fire_by: int | None,
    max_first_capture_step: int | None,
    max_ships_before_first_capture: int | None,
    max_launches_before_first_capture: int | None,
    stop_at_first_capture: bool,
) -> tuple[list[dict], Counter]:
    stats = Counter()
    samples: list[dict] = []

    for replay_path in replay_paths:
        try:
            data = json.loads(Path(replay_path).read_text())
        except Exception:
            stats["replay_read_fail"] += 1
            continue

        steps = data.get("steps") or []
        if not steps or len(steps[0]) != 2:
            stats["skip_not_2p"] += 1
            continue

        winner_idx = _strict_winner_idx(data)
        if winner_idx is None:
            stats["skip_not_strict_win"] += 1
            continue
        loser_idx = 1 - winner_idx

        winner_name = _agent_name(data, winner_idx)
        if player_name_filters and not any(f.lower() in winner_name.lower() for f in player_name_filters):
            stats["skip_name_filter"] += 1
            continue

        loser_first_fire = _first_fire_step(data, loser_idx, max_steps=steps_max)
        if require_opponent_first_fire_by is not None:
            if loser_first_fire is None or loser_first_fire > require_opponent_first_fire_by:
                stats["skip_pressure_filter"] += 1
                continue

        winner_first_cap = _first_capture_step(data, winner_idx)
        winner_first_launch = _first_launch_step(data, winner_idx)
        launches_pre_cap, ships_pre_cap = _launches_and_ships_before_step(data, winner_idx, winner_first_cap)

        if max_first_capture_step is not None:
            if winner_first_cap is None or winner_first_cap > max_first_capture_step:
                stats["skip_first_capture"] += 1
                continue
        if max_ships_before_first_capture is not None:
            if ships_pre_cap > max_ships_before_first_capture:
                stats["skip_precap_ships"] += 1
                continue
        if max_launches_before_first_capture is not None:
            if launches_pre_cap > max_launches_before_first_capture:
                stats["skip_precap_launches"] += 1
                continue

        stats["teacher_replays_kept"] += 1
        stats["teacher_first_capture_sum"] += 0 if winner_first_cap is None else winner_first_cap
        stats["teacher_first_launch_sum"] += 0 if winner_first_launch is None else winner_first_launch
        stats["teacher_precap_ships_sum"] += ships_pre_cap
        stats["teacher_precap_launches_sum"] += launches_pre_cap

        max_step_idx = min(len(steps), steps_max + 1)
        if stop_at_first_capture and winner_first_cap is not None:
            max_step_idx = min(max_step_idx, winner_first_cap)

        for step_idx in range(1, max_step_idx):
            step = steps[step_idx]
            agent_data = step[winner_idx]
            action = agent_data.get("action") or []
            obs = _obs_before_action(steps, winner_idx, step_idx)
            if obs is None:
                stats["skip_missing_obs"] += 1
                continue
            sample = trajectory_to_training_sample({"obs": obs, "action": action})
            if sample is None:
                stats["skip_sample_none"] += 1
                continue
            if not action:
                samples.append(sample)
                stats["teacher_no_fire_samples_kept"] += 1
                continue
            samples.append(sample)
            stats["teacher_samples_kept"] += 1
            stats["teacher_actions_kept"] += len(action)

    return samples, stats


def build_failure_conversion_relabels(
    audit_json_paths: list[str],
    relabel_mode: str,
    min_eta_regret: int,
    max_steps: int,
) -> tuple[list[dict], Counter]:
    stats = Counter()
    samples: list[dict] = []
    for audit_path in audit_json_paths:
        rep = json.loads(Path(audit_path).read_text())
        for ep in rep.get("episodes", []):
            replay = json.loads(Path(ep["replay_path"]).read_text())
            player_slot = int(ep["player_slot"])
            first_cap = _first_capture_step(replay, player_slot)
            for a in ep.get("actions", []):
                step_idx = int(a["step"])
                if step_idx > max_steps:
                    continue
                if first_cap is not None and step_idx >= first_cap:
                    stats["skip_post_capture_action"] += 1
                    continue

                step = replay["steps"][step_idx]
                agent_data = step[player_slot]
                action = agent_data.get("action") or []
                obs = _obs_before_action(replay["steps"], player_slot, step_idx)
                if obs is None:
                    stats["skip_missing_obs"] += 1
                    continue
                sample = trajectory_to_training_sample({"obs": obs, "action": action})
                if sample is None:
                    stats["skip_sample_none"] += 1
                    continue

                from_pid = int(a["from_planet_id"])
                slot = _slot_for_from_pid(obs, player_slot, from_pid)
                if slot is None:
                    stats["skip_missing_slot"] += 1
                    continue

                from_idx = _planet_idx_by_id(obs["planets"], from_pid)
                if from_idx is None:
                    stats["skip_bad_indices"] += 1
                    continue

                src = obs["planets"][from_idx]
                max_sendable = _max_sendable_ships(src[5])
                current_target = a.get("decoded_target")
                current_sent = int(a.get("ships", 0))
                if not current_target or int(current_target["planet_idx"]) >= len(obs["planets"]):
                    stats["skip_missing_current_target"] += 1
                    continue
                current_tgt = obs["planets"][int(current_target["planet_idx"])]
                current_required = _capture_cost(current_tgt, player_slot)
                if not (current_required <= max_sendable and current_sent > current_required):
                    stats["skip_not_overspend"] += 1
                    continue

                target_key = "tempo_target" if relabel_mode == "tempo" else "nearest_target"
                better = a.get(target_key)

                chosen_target_idx = None
                required = None
                eta_gap = a.get("eta_gap_vs_nearest")
                if (
                    a.get("classification") == "target_priority"
                    and eta_gap is not None
                    and int(eta_gap) >= min_eta_regret
                    and better
                    and int(better["planet_idx"]) < len(obs["planets"])
                ):
                    better_tgt = obs["planets"][int(better["planet_idx"])]
                    better_required = _capture_cost(better_tgt, player_slot)
                    if better_required <= max_sendable:
                        chosen_target_idx = int(better["planet_idx"])
                        required = better_required
                        stats["failure_retarget_and_downsize"] += 1

                if chosen_target_idx is None:
                    chosen_target_idx = int(current_target["planet_idx"])
                    required = current_required
                    stats["failure_downsize_only"] += 1

                if chosen_target_idx is None or required is None:
                    stats["skip_no_feasible_conversion_fix"] += 1
                    continue

                ship_bin = _find_ship_bin(required)
                relabeled = _retarget_with_ship(sample, slot=slot, target_idx=chosen_target_idx, ship_bin=ship_bin)
                samples.append(relabeled)
                stats["failure_relabels"] += 1
                stats["failure_required_ships_sum"] += required
    return samples, stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-replay-dir", action="append", default=[])
    ap.add_argument("--teacher-glob", default="*.json")
    ap.add_argument("--teacher-agent", action="append", default=[])
    ap.add_argument("--teacher-steps-max", type=int, default=40)
    ap.add_argument("--teacher-require-opponent-first-fire-by", type=int, default=12)
    ap.add_argument("--teacher-max-first-capture-step", type=int, default=12)
    ap.add_argument("--teacher-max-ships-before-first-capture", type=int, default=28)
    ap.add_argument("--teacher-max-launches-before-first-capture", type=int, default=2)
    ap.add_argument("--teacher-stop-at-first-capture", action="store_true")
    ap.add_argument("--failure-audit-json", action="append", default=[])
    ap.add_argument("--failure-relabel-mode", choices=["tempo", "nearest"], default="tempo")
    ap.add_argument("--failure-min-eta-regret", type=int, default=4)
    ap.add_argument("--failure-steps-max", type=int, default=40)
    ap.add_argument("--samples-out", required=True)
    ap.add_argument("--summary-out", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    teacher_paths: list[str] = []
    for d in args.teacher_replay_dir:
        teacher_paths.extend(sorted(glob.glob(os.path.join(d, args.teacher_glob))))

    teacher_samples, teacher_stats = build_teacher_conversion_samples(
        replay_paths=teacher_paths,
        player_name_filters=args.teacher_agent,
        steps_max=args.teacher_steps_max,
        require_opponent_first_fire_by=args.teacher_require_opponent_first_fire_by,
        max_first_capture_step=args.teacher_max_first_capture_step,
        max_ships_before_first_capture=args.teacher_max_ships_before_first_capture,
        max_launches_before_first_capture=args.teacher_max_launches_before_first_capture,
        stop_at_first_capture=args.teacher_stop_at_first_capture,
    )

    failure_samples, failure_stats = build_failure_conversion_relabels(
        audit_json_paths=args.failure_audit_json,
        relabel_mode=args.failure_relabel_mode,
        min_eta_regret=args.failure_min_eta_regret,
        max_steps=args.failure_steps_max,
    )

    samples = teacher_samples + failure_samples
    os.makedirs(os.path.dirname(args.samples_out) or ".", exist_ok=True)
    with open(args.samples_out, "wb") as f:
        pickle.dump(samples, f)

    payload = {
        "teacher_replay_dirs": args.teacher_replay_dir,
        "teacher_agents": args.teacher_agent,
        "teacher_replay_count": len(teacher_paths),
        "teacher_sample_count": len(teacher_samples),
        "failure_audit_json": args.failure_audit_json,
        "failure_sample_count": len(failure_samples),
        "total_sample_count": len(samples),
        "teacher_stats": dict(teacher_stats),
        "failure_stats": dict(failure_stats),
    }
    if args.summary_out:
        os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
        Path(args.summary_out).write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    print(f"samples saved -> {args.samples_out}")


if __name__ == "__main__":
    main()
