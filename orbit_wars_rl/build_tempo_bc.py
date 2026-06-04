"""Build a tempo-focused opening BC dataset.

This combines two sources:
1. positive teacher openings from strong-agent wins where the chosen target is
   close to the local tempo choice
2. relabeled failure states from our audited losses, where the target label is
   replaced with a better local target (tempo or nearest)

The output is a standard bc.py sample .pkl so we can fine-tune an existing BC
warmstart without changing the training loop.
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

from orbit_wars_rl.audit_submission_targets import (  # noqa: E402
    audit_episode,
    choose_player_slot,
    infer_outputs,
    corrected_move_target_idx,
    best_alt_by,
    rank_targets,
)
from orbit_wars_rl.bc import trajectory_to_training_sample  # noqa: E402
from orbit_wars_rl.eval import load_checkpoint  # noqa: E402
from orbit_wars_rl.config import Config  # noqa: E402
from orbit_wars_rl.model import EntityTransformer  # noqa: E402

import torch  # noqa: E402


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


def _slot_for_from_pid(obs: dict, player_idx: int, from_pid: int) -> tuple[int | None, int | None, dict]:
    from orbit_wars_rl.action_mask import compute_action_masks  # local import to avoid cycle

    masks = compute_action_masks(obs, player_idx)
    planets = obs["planets"]
    owned_indices = masks["owned_indices"].numpy()
    pid_to_slot: dict[int, int] = {}
    pid_to_idx: dict[int, int] = {}
    for slot in range(masks["owned_count"]):
        pidx = int(owned_indices[slot])
        if pidx < len(planets):
            pid = int(planets[pidx][0])
            pid_to_slot[pid] = slot
            pid_to_idx[pid] = pidx
    return pid_to_slot.get(int(from_pid)), pid_to_idx.get(int(from_pid)), masks


def _retarget_sample(sample: dict, slot: int, target_idx: int) -> dict:
    out = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        else:
            out[k] = v
    out["fire_target"][slot] = 1
    out["target_target"][slot] = int(target_idx)
    return out


def _teacher_sample_ok(audit_action: dict, max_eta_gap: int, max_dist_gap: float, require_tempo_match: bool) -> bool:
    chosen = audit_action.get("decoded_target")
    nearest = audit_action.get("nearest_target")
    tempo = audit_action.get("tempo_target")
    if not chosen or not nearest:
        return False
    if require_tempo_match and (not tempo or int(chosen["planet_id"]) != int(tempo["planet_id"])):
        return False
    eta_gap = audit_action.get("eta_gap_vs_nearest")
    dist_gap = audit_action.get("distance_gap_vs_nearest")
    if eta_gap is None or dist_gap is None:
        return False
    return int(eta_gap) <= max_eta_gap and float(dist_gap) <= max_dist_gap


def build_teacher_samples(
    replay_paths: list[str],
    model: EntityTransformer,
    device: torch.device,
    player_name_filters: list[str],
    steps_max: int,
    opponent_first_fire_by: int | None,
    tempo_max_eta_gap: int,
    tempo_max_dist_gap: float,
    require_tempo_match: bool,
) -> tuple[list[dict], Counter]:
    stats = Counter()
    samples: list[dict] = []
    for replay_path in replay_paths:
        try:
            data = json.loads(Path(replay_path).read_text())
        except Exception:
            continue
        steps = data.get("steps") or []
        if not steps or len(steps[0]) != 2:
            continue
        rewards = data.get("rewards") or []
        if len(rewards) != 2 or rewards[0] == rewards[1]:
            continue
        winner_idx = 0 if rewards[0] > rewards[1] else 1
        loser_idx = 1 - winner_idx
        winner_name = ((data.get("info") or {}).get("Agents") or [{} , {}])[winner_idx].get("Name", "")
        if player_name_filters and not any(f.lower() in winner_name.lower() for f in player_name_filters):
            continue

        # pressure filter
        loser_first_fire = None
        for t in range(1, min(len(steps), steps_max + 1)):
            act = steps[t][loser_idx].get("action") or []
            if act:
                loser_first_fire = t
                break
        if opponent_first_fire_by is not None:
            if loser_first_fire is None or loser_first_fire > opponent_first_fire_by:
                continue

        # audit the teacher replay so we can filter to tempo-clean launches
        ep_report = audit_episode(
            Path(replay_path),
            data,
            model,
            device,
            player_slot=winner_idx,
            top_k=5,
            aim_gap_deg=15.0,
            step_limit=steps_max,
        )
        action_by_key = {
            (int(a["step"]), int(a["from_planet_id"])): a
            for a in ep_report["actions"]
        }

        for step_idx in range(1, min(len(steps), steps_max + 1)):
            step = steps[step_idx]
            agent_data = step[winner_idx]
            action = agent_data.get("action") or []
            if not action:
                continue
            obs = _obs_from_step_agent(agent_data, winner_idx, step_idx)
            if obs is None:
                continue
            sample = trajectory_to_training_sample({"obs": obs, "action": action})
            if sample is None:
                continue
            kept_any = False
            # blank all slots, then selectively restore good teacher-fired slots
            sample["fire_target"].zero_()
            sample["target_target"].fill_(-1)
            sample["ship_target"].zero_()
            for mv in action:
                if len(mv) < 3:
                    continue
                from_pid, _, ship_count = int(mv[0]), float(mv[1]), int(mv[2])
                audit_action = action_by_key.get((step_idx, from_pid))
                if audit_action is None:
                    continue
                if not _teacher_sample_ok(
                    audit_action,
                    max_eta_gap=tempo_max_eta_gap,
                    max_dist_gap=tempo_max_dist_gap,
                    require_tempo_match=require_tempo_match,
                ):
                    stats["teacher_launch_dropped"] += 1
                    continue
                slot, _, _ = _slot_for_from_pid(obs, winner_idx, from_pid)
                if slot is None:
                    continue
                target_idx = audit_action["decoded_target"]["planet_idx"]
                sample["fire_target"][slot] = 1
                sample["target_target"][slot] = int(target_idx)
                from orbit_wars_rl.bc import _find_ship_bin  # local import
                sample["ship_target"][slot] = _find_ship_bin(ship_count)
                kept_any = True
                stats["teacher_launch_kept"] += 1
            if kept_any:
                samples.append(sample)
                stats["teacher_samples_kept"] += 1
    return samples, stats


def build_failure_relabels(
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
            for a in ep.get("actions", []):
                step_idx = int(a["step"])
                if step_idx > max_steps:
                    continue
                if a.get("classification") != "target_priority":
                    continue
                eta_gap = a.get("eta_gap_vs_nearest")
                if eta_gap is None or int(eta_gap) < min_eta_regret:
                    continue
                target_key = "tempo_target" if relabel_mode == "tempo" else "nearest_target"
                better = a.get(target_key)
                if not better:
                    continue

                step = replay["steps"][step_idx]
                agent_data = step[player_slot]
                action = agent_data.get("action") or []
                obs = _obs_from_step_agent(agent_data, player_slot, step_idx)
                if obs is None:
                    continue
                sample = trajectory_to_training_sample({"obs": obs, "action": action})
                if sample is None:
                    continue

                from_pid = int(a["from_planet_id"])
                slot, _, _ = _slot_for_from_pid(obs, player_slot, from_pid)
                if slot is None:
                    continue
                relabeled = _retarget_sample(sample, slot=slot, target_idx=int(better["planet_idx"]))
                samples.append(relabeled)
                stats["failure_relabels"] += 1
    return samples, stats


def _load_model(checkpoint_path: str, device: torch.device) -> EntityTransformer:
    cfg = Config()
    sd, _ = load_checkpoint(checkpoint_path, cfg)
    model = EntityTransformer(cfg.model)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    return model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-replay-dir", action="append", default=[])
    ap.add_argument("--teacher-glob", default="*.json")
    ap.add_argument("--teacher-agent", action="append", default=[])
    ap.add_argument("--teacher-checkpoint", required=True,
                    help="Checkpoint only used to audit teacher replays consistently.")
    ap.add_argument("--teacher-steps-max", type=int, default=50)
    ap.add_argument("--teacher-require-opponent-first-fire-by", type=int, default=12)
    ap.add_argument("--teacher-tempo-max-eta-gap", type=int, default=2)
    ap.add_argument("--teacher-tempo-max-dist-gap", type=float, default=8.0)
    ap.add_argument("--teacher-require-tempo-match", action="store_true")
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

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model(args.teacher_checkpoint, device)

    teacher_samples, teacher_stats = build_teacher_samples(
        replay_paths=teacher_paths,
        model=model,
        device=device,
        player_name_filters=args.teacher_agent,
        steps_max=args.teacher_steps_max,
        opponent_first_fire_by=args.teacher_require_opponent_first_fire_by,
        tempo_max_eta_gap=args.teacher_tempo_max_eta_gap,
        tempo_max_dist_gap=args.teacher_tempo_max_dist_gap,
        require_tempo_match=args.teacher_require_tempo_match,
    )
    failure_samples, failure_stats = build_failure_relabels(
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
