"""Build whole-action BC samples from actual replay action lists.

Timing convention: action at steps[t][seat] was selected from observation at
steps[t-1][seat].

Unlike build_producer_action_bc.py, this script labels the replay player's real
projected action. It can optionally filter attack moves using the target-value
pairwise channels added for the prod-share branch.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "orbit_wars_rl"
for _path in (ROOT, PKG):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from orbit_wars_rl.action_mask import (  # noqa: E402
    _DEF_HORIZON,
    _DEF_OVERHEAD,
    _def_eta,
    _fleet_speed,
    _head_threat_maps,
    compute_action_masks,
)
from orbit_wars_rl.bc import _find_target_planet_index, trajectory_to_training_sample  # noqa: E402
from orbit_wars_rl.features import extract_features  # noqa: E402


def _iter_paths(inputs: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out.extend(p.rglob("*.json"))
        elif any(ch in item for ch in "*?[]"):
            out.extend(Path(x) for x in glob.glob(item))
        else:
            out.append(p)
    return sorted(set(out))


def _winner_seat(replay: dict[str, Any]) -> int | None:
    rewards = replay.get("rewards") or []
    if len(rewards) != 2:
        return None
    return int(max(range(len(rewards)), key=lambda i: rewards[i]))


def _copy_replay_obs(obs: dict[str, Any], seat: int, step: int) -> dict[str, Any]:
    planets = obs.get("planets") or []
    return {
        "step": int(obs.get("step", step)),
        "player": int(obs.get("player", seat)),
        "planets": [list(p) for p in planets],
        "fleets": [list(f) for f in (obs.get("fleets") or [])],
        "angular_velocity": float(obs.get("angular_velocity", 0.0)),
        "initial_planets": [list(p) for p in (obs.get("initial_planets") or planets)],
        "comet_planet_ids": list(obs.get("comet_planet_ids") or []),
        "comets": list(obs.get("comets") or []),
    }


def _context(obs: dict[str, Any], max_planets: int) -> dict[str, Any] | None:
    player = int(obs["player"])
    try:
        features = extract_features(obs, player, num_players=2, max_planets=max_planets, max_fleets=128)
        masks = compute_action_masks(obs, player)
    except Exception:
        return None

    pairwise = features["pairwise_features"]
    owned_indices = masks["owned_indices"]
    max_ships = masks["max_ships"]
    if hasattr(pairwise, "detach"):
        pairwise = pairwise.detach().cpu().numpy()
    if hasattr(owned_indices, "detach"):
        owned_indices = owned_indices.detach().cpu().numpy()
    if hasattr(max_ships, "detach"):
        max_ships = max_ships.detach().cpu().numpy()
    max_ships = max_ships.squeeze(0)

    planets = obs["planets"]
    pid_to_idx = {int(p[0]): i for i, p in enumerate(planets)}
    pid_to_slot: dict[int, int] = {}
    owned_count = int(masks["owned_count"].item() if hasattr(masks["owned_count"], "item") else masks["owned_count"])
    for slot in range(min(owned_count, len(owned_indices), pairwise.shape[0])):
        pidx = int(owned_indices[slot])
        if 0 <= pidx < len(planets) and int(planets[pidx][1]) == player:
            pid_to_slot[int(planets[pidx][0])] = slot
    return {
        "features": features,
        "masks": masks,
        "pairwise": pairwise,
        "pid_to_idx": pid_to_idx,
        "pid_to_slot": pid_to_slot,
        "owned_count": owned_count,
        "max_ships": max_ships,
    }


def _decode_move(obs: dict[str, Any], move: Any, ctx: dict[str, Any] | None,
                 max_planets: int) -> dict[str, Any] | None:
    if ctx is None or not isinstance(move, (list, tuple)) or len(move) < 3:
        return None
    planets = obs["planets"]
    player = int(obs["player"])
    try:
        src_pid = int(move[0])
        angle = float(move[1])
        ships = max(0, int(move[2]))
    except Exception:
        return None

    src_idx = ctx["pid_to_idx"].get(src_pid)
    if src_idx is None or int(planets[src_idx][1]) != player:
        return {"drop_reason": "source_not_owned"}
    slot = ctx["pid_to_slot"].get(src_pid)
    if slot is None:
        return {"drop_reason": "source_not_in_slots"}

    tgt_idx = _find_target_planet_index(
        (float(planets[src_idx][2]), float(planets[src_idx][3])),
        angle,
        ships,
        planets,
        obs.get("initial_planets", planets),
        float(obs.get("angular_velocity", 0.0)),
        int(obs.get("step", 0)),
        max_planets=max_planets,
    )
    if tgt_idx < 0 or tgt_idx >= min(len(planets), max_planets):
        return {"drop_reason": "unresolved_target"}

    row = ctx["pairwise"][slot, tgt_idx]
    target_owner = int(planets[tgt_idx][1])
    kind = "save" if target_owner == player else "attack"
    return {
        "move": [src_pid, angle, ships],
        "source_id": src_pid,
        "source_idx": int(src_idx),
        "source_slot": int(slot),
        "target_id": int(planets[tgt_idx][0]),
        "target_idx": int(tgt_idx),
        "target_owner": target_owner,
        "owned_count": int(ctx["owned_count"]),
        "max_ships": int(ctx["max_ships"][slot]),
        "step": int(obs.get("step", 0)),
        "ships": ships,
        "kind": kind,
        "obs": obs,
        "capture_value": float(row[16]) if row.shape[0] > 16 else 0.0,
        "reactive_roi": float(row[17]) if row.shape[0] > 17 else 0.0,
        "keepability": float(row[19]) if row.shape[0] > 19 else 0.0,
    }


def _reachable_enemy_mass(planets: list[list[Any]], tgt: list[Any], player: int, horizon: float) -> float:
    mass = 0.0
    tid = int(tgt[0])
    for ep in planets:
        if int(ep[0]) == tid or int(ep[1]) < 0 or int(ep[1]) == player:
            continue
        d = math.hypot(float(ep[2]) - float(tgt[2]), float(ep[3]) - float(tgt[3]))
        reach = max(_fleet_speed(int(ep[5])) * horizon, 1e-6)
        mass += float(ep[5]) * max(1.0 - d / reach, 0.0)
    return mass


def _reachable_friendly_mass(planets: list[list[Any]], fleets: list[list[Any]], tgt: list[Any],
                             player: int, threat_eta: float, reinforce_garrison_floor: float) -> float:
    enemy_in, _, _ = _head_threat_maps(planets, fleets, player)
    tid = int(tgt[0])
    available = 0.0
    for src in planets:
        sid = int(src[0])
        if sid == tid or int(src[1]) != player:
            continue
        garr = float(src[5])
        own_threat = float(enemy_in.get(sid, 0.0))
        spare = garr if own_threat >= garr else max(0.0, garr - own_threat)
        if reinforce_garrison_floor > 0.0:
            spare = min(spare, max(0.0, garr - float(reinforce_garrison_floor)))
        if spare <= 0:
            continue
        eta = _def_eta(src, tgt, int(spare))
        if (not math.isfinite(threat_eta)) or eta <= threat_eta:
            available += spare
    return available


def _reverse_blocked(label: dict[str, Any], cooldown_last: dict[tuple[int, int], int] | None,
                     cooldown: int) -> bool:
    if cooldown <= 0 or cooldown_last is None:
        return False
    step = int(label["step"])
    src = int(label["source_id"])
    tgt = int(label["target_id"])
    prev = cooldown_last.get((tgt, src))
    return prev is not None and 0 < (step - int(prev)) <= cooldown


def _passes_save_quality(label: dict[str, Any], args: argparse.Namespace, stats: Counter,
                         cooldown_last: dict[tuple[int, int], int] | None) -> bool:
    if args.drop_saves:
        stats["filtered_save_moves"] += 1
        return False
    stats["save_moves_seen"] += 1

    if args.reinforce_gate_min_planets > 0 and int(label["owned_count"]) < int(args.reinforce_gate_min_planets):
        stats["filtered_save_gate"] += 1
        return False
    if _reverse_blocked(label, cooldown_last, int(args.reverse_edge_cooldown)):
        stats["filtered_save_reverse_edge"] += 1
        return False
    if not args.save_quality_filter:
        return True

    obs = label["obs"]
    planets = obs["planets"]
    fleets = obs.get("fleets") or []
    player = int(obs["player"])
    src = planets[int(label["source_idx"])]
    tgt = planets[int(label["target_idx"])]
    tid = int(tgt[0])

    enemy_in, enemy_eta, friendly_in = _head_threat_maps(planets, fleets, player)
    inbound = float(enemy_in.get(tid, 0.0))
    if inbound <= 0:
        # No active threat on the target. With --proactive-saves we KEEP these: winners'
        # reinforcement is mostly proactive forward-staging (~75% of save moves), the skill
        # we want the BC to learn. The reactive late/hopeless checks below don't apply.
        if getattr(args, "proactive_saves", False):
            stats["kept_proactive_save_moves"] += 1
            stats["kept_quality_save_moves"] += 1
            return True
        stats["filtered_save_no_threat"] += 1
        return False

    threat_eta = float(enemy_eta.get(tid, math.inf))
    eta = _def_eta(src, tgt, int(label["ships"]))
    if math.isfinite(threat_eta) and eta > threat_eta:
        stats["filtered_save_late"] += 1
        return False

    reactive = _reachable_enemy_mass(planets, tgt, player, float(args.save_horizon))
    floor = inbound + float(args.save_beta) * reactive + float(args.save_overhead)
    cover = float(tgt[5]) + sum(
        ships for ships, f_eta in friendly_in.get(tid, [])
        if (not math.isfinite(threat_eta)) or f_eta <= threat_eta
    )
    deficit = floor - cover
    if deficit <= 0:
        # Threatened but currently covered. --proactive-saves keeps it (user: only drop
        # late + hopeless); the reactive filter drops it as redundant.
        if getattr(args, "proactive_saves", False):
            stats["kept_threatened_safe_save_moves"] += 1
            stats["kept_quality_save_moves"] += 1
            return True
        stats["filtered_save_already_safe"] += 1
        return False

    reachable = _reachable_friendly_mass(
        planets, fleets, tgt, player, threat_eta, float(args.reinforce_garrison_floor)
    )
    if reachable + 1e-6 < deficit:
        stats["filtered_save_hopeless"] += 1
        return False

    value = max(float(tgt[6]) * float(args.save_horizon), 1.0)
    ratio = deficit / value
    if ratio > float(args.max_save_cost_ratio) and not getattr(args, "proactive_saves", False):
        stats["filtered_save_expensive"] += 1
        return False
    if args.min_save_ship_fraction > 0.0 and float(label["ships"]) + 1e-6 < deficit * float(args.min_save_ship_fraction):
        stats["filtered_save_tiny"] += 1
        return False

    label["save_deficit"] = float(deficit)
    label["save_cost_ratio"] = float(ratio)
    stats["kept_quality_save_moves"] += 1
    return True


def _passes_filters(label: dict[str, Any], args: argparse.Namespace, stats: Counter,
                    cooldown_last: dict[tuple[int, int], int] | None = None) -> bool:
    if "drop_reason" in label:
        stats[label["drop_reason"]] += 1
        return False
    if label["kind"] == "save":
        return _passes_save_quality(label, args, stats, cooldown_last)

    stats["attack_moves_seen"] += 1
    if args.min_attack_value is not None and label["capture_value"] < args.min_attack_value:
        stats["filtered_low_value"] += 1
        return False
    if args.min_reactive_roi is not None and label["reactive_roi"] < args.min_reactive_roi:
        stats["filtered_low_roi"] += 1
        return False
    if args.min_keepability is not None and label["keepability"] < args.min_keepability:
        stats["filtered_low_keep"] += 1
        return False
    return True


def _dedupe_by_source(labels: list[dict[str, Any]], stats: Counter) -> list[dict[str, Any]]:
    by_src: dict[int, list[dict[str, Any]]] = {}
    for label in labels:
        by_src.setdefault(int(label["source_id"]), []).append(label)

    kept: list[dict[str, Any]] = []
    for rows in by_src.values():
        if len(rows) > 1:
            stats["same_source_groups"] += 1
            stats["same_source_dropped_moves"] += len(rows) - 1
        best = max(rows, key=lambda r: (int(r["ships"]), 1 if r["kind"] == "attack" else 0))
        stats["kept_attack_moves"] += int(best["kind"] == "attack")
        stats["kept_save_moves"] += int(best["kind"] == "save")
        kept.append(best)
    return kept


def _record_save_cooldowns(labels: list[dict[str, Any]], cooldown_last: dict[tuple[int, int], int] | None) -> None:
    if cooldown_last is None:
        return
    for label in labels:
        if label["kind"] == "save":
            cooldown_last[(int(label["source_id"]), int(label["target_id"]))] = int(label["step"])


def _clone_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sample.items()}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="Replay JSON files, directories, or globs.")
    ap.add_argument("--mode", choices=("winner", "all", "slot"), default="winner")
    ap.add_argument("--player-slot", type=int, default=None)
    ap.add_argument("--no-require-1v1", action="store_true")
    ap.add_argument("--step-limit", type=int, default=500)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = no cap.")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--max-planets", type=int, default=48)
    ap.add_argument("--drop-saves", action="store_true")
    ap.add_argument("--save-quality-filter", action=argparse.BooleanOptionalAction, default=True,
                    help="Keep only threatened, reachable, holdable, value-positive save labels.")
    ap.add_argument("--proactive-saves", action="store_true", default=False,
                    help="Keep proactive forward-staging saves (unthreatened targets) and threatened "
                         "saves; drop ONLY late and hopeless. Teaches the proactive-reinforcement skill "
                         "that the reactive filter discards (~75%% of winner saves are unthreatened).")
    ap.add_argument("--reinforce-gate-min-planets", type=int, default=2)
    ap.add_argument("--reverse-edge-cooldown", type=int, default=3)
    ap.add_argument("--reinforce-garrison-floor", type=float, default=0.0)
    ap.add_argument("--save-beta", type=float, default=2.2)
    ap.add_argument("--save-horizon", type=float, default=_DEF_HORIZON)
    ap.add_argument("--save-overhead", type=float, default=_DEF_OVERHEAD)
    ap.add_argument("--max-save-cost-ratio", type=float, default=1.0)
    ap.add_argument("--min-save-ship-fraction", type=float, default=0.0)
    ap.add_argument("--min-attack-value", type=float, default=None)
    ap.add_argument("--min-reactive-roi", type=float, default=None)
    ap.add_argument("--min-keepability", type=float, default=None)
    ap.add_argument("--samples-out", required=True)
    ap.add_argument("--summary-out", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    paths = _iter_paths(args.paths)
    samples: list[dict[str, Any]] = []
    stats = Counter({"paths": len(paths)})

    for path in paths:
        if args.max_samples and len(samples) >= args.max_samples:
            break
        try:
            replay = json.loads(path.read_text())
        except Exception:
            stats["read_fail"] += 1
            continue
        rewards = replay.get("rewards") or []
        if not args.no_require_1v1 and len(rewards) != 2:
            stats["excluded_non_1v1"] += 1
            continue
        steps = replay.get("steps") or []
        if len(steps) < 2:
            stats["short_replay"] += 1
            continue

        if args.mode == "winner":
            winner = _winner_seat(replay)
            seats = [] if winner is None else [winner]
        elif args.mode == "slot":
            if args.player_slot is None:
                raise SystemExit("--mode slot requires --player-slot")
            seats = [int(args.player_slot)]
        else:
            seats = list(range(len(steps[0])))
        if not seats:
            stats["no_selected_seat"] += 1
            continue

        stats["replays_used"] += 1
        cooldown_by_seat: dict[int, dict[tuple[int, int], int]] = {seat: {} for seat in seats}
        for seat in seats:
            cooldown_last = cooldown_by_seat[seat]
            for t in range(1, len(steps)):
                if args.max_samples and len(samples) >= args.max_samples:
                    break
                if args.step_limit is not None and t > args.step_limit:
                    break
                if seat >= len(steps[t]) or seat >= len(steps[t - 1]):
                    continue
                action = steps[t][seat].get("action") or []
                if not action:
                    continue
                obs = _copy_replay_obs(steps[t - 1][seat].get("observation") or {}, seat, t - 1)
                if not obs["planets"]:
                    stats["empty_obs"] += 1
                    continue
                stats["action_turns_seen"] += 1
                stats["raw_moves_seen"] += len(action)

                ctx = _context(obs, args.max_planets)
                passed_labels = []
                for move in action:
                    label = _decode_move(obs, move, ctx, args.max_planets)
                    if label is None:
                        stats["bad_move"] += 1
                        continue
                    if _passes_filters(label, args, stats, cooldown_last):
                        passed_labels.append(label)
                if not passed_labels:
                    stats["turns_all_moves_filtered"] += 1
                    continue

                kept_labels = _dedupe_by_source(passed_labels, stats)
                kept_action = [label["move"] for label in kept_labels]
                _record_save_cooldowns(kept_labels, cooldown_last)
                sample = trajectory_to_training_sample(
                    {"obs": obs, "action": kept_action},
                    max_planets=args.max_planets,
                )
                if sample is None:
                    stats["sample_build_failed"] += 1
                    continue
                fired_slots = int(sample["fire_target"].sum().item())
                valid_targets = int((sample["target_target"] >= 0).sum().item())
                if fired_slots <= 0:
                    stats["zero_fire_samples"] += 1
                    continue
                stats["samples"] += max(1, args.repeat)
                stats["fired_slots"] += fired_slots * max(1, args.repeat)
                stats["valid_target_slots"] += valid_targets * max(1, args.repeat)
                for _ in range(max(1, args.repeat)):
                    samples.append(_clone_sample(sample))
                    if args.max_samples and len(samples) >= args.max_samples:
                        break

    out_path = Path(args.samples_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(samples, f)

    payload = {
        "sample_count": len(samples),
        "args": {
            "paths": args.paths,
            "mode": args.mode,
            "player_slot": args.player_slot,
            "require_1v1": not args.no_require_1v1,
            "step_limit": args.step_limit,
            "max_samples": args.max_samples,
            "repeat": args.repeat,
            "drop_saves": args.drop_saves,
            "save_quality_filter": args.save_quality_filter,
            "proactive_saves": args.proactive_saves,
            "reinforce_gate_min_planets": args.reinforce_gate_min_planets,
            "reverse_edge_cooldown": args.reverse_edge_cooldown,
            "reinforce_garrison_floor": args.reinforce_garrison_floor,
            "save_beta": args.save_beta,
            "save_horizon": args.save_horizon,
            "save_overhead": args.save_overhead,
            "max_save_cost_ratio": args.max_save_cost_ratio,
            "min_save_ship_fraction": args.min_save_ship_fraction,
            "min_attack_value": args.min_attack_value,
            "min_reactive_roi": args.min_reactive_roi,
            "min_keepability": args.min_keepability,
        },
        "stats": dict(stats),
    }
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"samples saved -> {args.samples_out}")


if __name__ == "__main__":
    main()
