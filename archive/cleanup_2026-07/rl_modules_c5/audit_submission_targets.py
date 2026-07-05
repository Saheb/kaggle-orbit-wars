"""Audit target selection in submission replay episodes.

Loads a trained checkpoint, replays Kaggle episode JSONs, and aligns each
replay action at step t with the actor observation at step t-1. For every
launch by the audited player, it reports:
  - which planet the replay action actually targeted
  - whether the launch angle matches the decoded target intercept
  - the model's top target logits for that source slot
  - nearby alternative targets (nearest / cheapest / best tempo score)

Primary use: post-submission review of loss episodes where a move "looks wrong"
in the Kaggle viewer. This tells us whether the issue is:
  - open-space / aiming gap
  - invalid raw argmax target
  - target-priority under pressure

Example:
  python orbit_wars_rl/audit_submission_targets.py \
    --checkpoint gpu_run_artifacts/h100_rev31/checkpoints/torch_step_10485760_rev31_20260603_153146.pt \
    --replay-dir /tmp/sub53336058_eps \
    --player-name Saheb \
    --output-json /tmp/rev31_target_audit.json \
    --output-md /tmp/rev31_target_audit.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_rl.action_mask import compute_action_masks, _target_intercept_angle
from orbit_wars_rl.bc import _find_target_planet_index
from orbit_wars_rl.config import Config
from orbit_wars_rl.eval import load_checkpoint
from orbit_wars_rl.features import extract_features, fleet_speed
from orbit_wars_rl.model import EntityTransformer


@dataclass
class RankedTarget:
    planet_idx: int
    planet_id: int
    owner: int
    ships: int
    production: int
    distance: float
    eta: int
    capture_cost: int
    tempo_score: float
    logit: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "planet_idx": self.planet_idx,
            "planet_id": self.planet_id,
            "owner": self.owner,
            "ships": self.ships,
            "production": self.production,
            "distance": round(self.distance, 3),
            "eta": self.eta,
            "capture_cost": self.capture_cost,
            "tempo_score": round(self.tempo_score, 6),
            "logit": round(self.logit, 6),
        }


def circ_err_deg(a: float, b: float) -> float:
    d = abs(a - b)
    d = min(d, 2 * math.pi - d)
    return math.degrees(d)


def obs_player_slot(obs: dict, fallback_slot: int) -> int:
    try:
        return int(obs.get("player", fallback_slot))
    except Exception:
        return fallback_slot


def capture_cost(target: list, player: int) -> int:
    owner = int(target[1])
    ships = int(target[5])
    production = int(target[6])
    if owner == -1:
        return ships + 1
    if owner != player:
        return ships + production * 3 + 1
    return 0


def target_distance_eta(src: list, tgt: list, ships: int) -> tuple[float, int]:
    dist = math.hypot(float(tgt[2]) - float(src[2]), float(tgt[3]) - float(src[3]))
    eta = max(1, int(math.ceil(dist / max(fleet_speed(ships), 1e-6))))
    return dist, eta


def tempo_score(src: list, tgt: list, ships: int, player: int) -> float:
    dist, eta = target_distance_eta(src, tgt, ships)
    cost = max(1, capture_cost(tgt, player))
    prod = max(0, int(tgt[6]))
    # Simple review heuristic, not training logic:
    # reward production, penalize longer arrival and higher capture cost.
    return prod / (cost * max(1, eta))


def normalize_obs(obs: dict, fallback_step: int | None = None) -> dict:
    step = obs.get("step", fallback_step if fallback_step is not None else 0)
    return {
        "step": int(step if step is not None else 0),
        "player": int(obs.get("player", 0)),
        "planets": obs["planets"],
        "fleets": obs.get("fleets", []),
        "angular_velocity": float(obs.get("angular_velocity", 0.0)),
        "initial_planets": obs.get("initial_planets", obs["planets"]),
        "comet_planet_ids": list(obs.get("comet_planet_ids", [])),
    }


def resolve_replay_paths(replay_dir: str | None, replay_paths: list[str], episode_ids: list[str]) -> list[Path]:
    paths = [Path(p) for p in replay_paths]
    if replay_dir:
        root = Path(replay_dir)
        if episode_ids:
            paths.extend(root / f"{ep}.json" for ep in episode_ids)
        else:
            paths.extend(sorted(root.glob("*.json")))
    resolved = []
    seen: set[Path] = set()
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        resolved.append(rp)
    return resolved


def load_model(checkpoint_path: str, device: torch.device) -> tuple[EntityTransformer, Config, str]:
    cfg = Config()
    sd, action_decode = load_checkpoint(checkpoint_path, cfg)
    model = EntityTransformer(cfg.model)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    return model, cfg, action_decode


def infer_outputs(model: EntityTransformer, device: torch.device, obs: dict) -> tuple[dict, dict]:
    player = int(obs["player"])
    features = extract_features(obs, player, num_players=2)
    masks = compute_action_masks(obs, player)
    with torch.no_grad():
        outputs = model(
            features["planet_features"].unsqueeze(0).to(device),
            features["fleet_features"].unsqueeze(0).to(device),
            features["global_features"].unsqueeze(0).to(device),
            features["planet_mask"].unsqueeze(0).to(device),
            features["fleet_mask"].unsqueeze(0).to(device),
            fire_mask=masks["fire_mask"].to(device),
            angle_mask=masks["angle_mask"].to(device),
            slot_valid=masks["slot_valid"].to(device),
            owned_indices=masks["owned_indices"].to(device),
            owned_count=masks["owned_count"],
            pairwise_features=features["pairwise_features"].unsqueeze(0).to(device)
            if "pairwise_features" in features else None,
        )
    return (
        {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in outputs.items()},
        {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in masks.items()},
    )


def pid_to_planet_idx(planets: list[list], pid: int) -> int | None:
    for i, p in enumerate(planets):
        if int(p[0]) == int(pid):
            return i
    return None


def valid_target_indices(planets: list[list], player: int, src_idx: int) -> list[int]:
    out = []
    for i, p in enumerate(planets):
        if i == src_idx:
            continue
        if int(p[1]) == player:
            continue
        out.append(i)
    return out


def rank_targets(
    planets: list[list],
    logits_for_slot: torch.Tensor,
    player: int,
    src_idx: int,
    ships: int,
    top_k: int,
) -> tuple[list[RankedTarget], int | None, int | None]:
    valid = valid_target_indices(planets, player, src_idx)
    if not valid:
        raw_argmax = int(torch.argmax(logits_for_slot).item()) if logits_for_slot.numel() else None
        return [], raw_argmax, None

    raw_argmax = int(torch.argmax(logits_for_slot).item())
    src = planets[src_idx]
    ranked: list[RankedTarget] = []
    for tidx in valid:
        tgt = planets[tidx]
        dist, eta = target_distance_eta(src, tgt, ships)
        ranked.append(
            RankedTarget(
                planet_idx=tidx,
                planet_id=int(tgt[0]),
                owner=int(tgt[1]),
                ships=int(tgt[5]),
                production=int(tgt[6]),
                distance=dist,
                eta=eta,
                capture_cost=capture_cost(tgt, player),
                tempo_score=tempo_score(src, tgt, ships, player),
                logit=float(logits_for_slot[tidx].item()),
            )
        )
    ranked.sort(key=lambda r: r.logit, reverse=True)
    top1_valid = ranked[0].planet_idx if ranked else None
    return ranked[:top_k], raw_argmax, top1_valid


def slot_src_target_valid(planets: list[list], player: int, src_idx: int, tidx: int) -> bool:
    if src_idx >= len(planets) or tidx >= len(planets):
        return False
    src = planets[src_idx]
    tgt = planets[tidx]
    if int(tgt[1]) == player:
        return False
    if int(tgt[0]) == int(src[0]):
        return False
    return True


def corrected_move_target_idx(
    planets: list[list],
    raw_logits_for_slot: torch.Tensor,
    player: int,
    src_idx: int,
) -> int | None:
    if src_idx >= len(planets):
        return None
    masked = raw_logits_for_slot.clone()
    width = min(len(planets), masked.shape[0])
    for tidx in range(width):
        if not slot_src_target_valid(planets, player, src_idx, tidx):
            masked[tidx] = -1e9
    if not torch.isfinite(masked[:width]).any():
        return None
    return int(torch.argmax(masked).item())


def best_alt_by(ranked: list[RankedTarget], key: str) -> dict[str, Any] | None:
    if not ranked:
        return None
    if key == "nearest":
        best = min(ranked, key=lambda r: (r.distance, -r.production, r.capture_cost))
    elif key == "cheapest":
        best = min(ranked, key=lambda r: (r.capture_cost, r.distance, -r.production))
    elif key == "tempo":
        best = max(ranked, key=lambda r: (r.tempo_score, r.production, -r.distance))
    else:
        raise ValueError(f"unknown key {key}")
    return best.to_dict()


def nearby_targets(ranked: list[RankedTarget]) -> list[RankedTarget]:
    if not ranked:
        return []
    nearest = min(r.distance for r in ranked)
    radius = max(30.0, nearest + 8.0)
    return [r for r in ranked if r.distance <= radius]


def best_nearby_by(ranked: list[RankedTarget], key: str) -> dict[str, Any] | None:
    near = nearby_targets(ranked)
    if not near:
        return None
    if key == "weakest":
        best = min(near, key=lambda r: (r.ships, r.distance, -r.production))
    elif key == "highest_prod":
        best = max(near, key=lambda r: (r.production, -r.distance, -r.ships))
    else:
        raise ValueError(f"unknown key {key}")
    return best.to_dict()


def choose_player_slot(replay: dict, player_name: str | None, player_slot: int | None) -> int:
    if player_slot is not None:
        return int(player_slot)
    teams = replay.get("info", {}).get("TeamNames") or []
    if player_name:
        for i, name in enumerate(teams):
            if str(name) == player_name:
                return i
    if len(teams) == 2 and "Saheb" in teams:
        return teams.index("Saheb")
    return 0


def audit_episode(
    replay_path: Path,
    replay: dict,
    model: EntityTransformer,
    device: torch.device,
    player_slot: int,
    top_k: int,
    aim_gap_deg: float,
    step_limit: int | None,
) -> dict[str, Any]:
    raw_episode_id = replay.get("info", {}).get("EpisodeId") or replay_path.stem or replay.get("id")
    try:
        episode_id = int(raw_episode_id)
    except Exception:
        episode_id = str(raw_episode_id)
    team_names = replay.get("info", {}).get("TeamNames") or []
    player_name = team_names[player_slot] if player_slot < len(team_names) else f"slot{player_slot}"
    steps = replay["steps"]

    actions_out: list[dict[str, Any]] = []
    per_step_cache: dict[int, tuple[dict, dict]] = {}

    launches = 0
    open_space = 0
    aim_gap = 0
    policy_mismatch = 0
    invalid_raw_argmax = 0
    corrected_target_differs = 0
    dropped_slot_actions = 0
    farther_than_nearest = 0

    for t in range(1, len(steps)):
        if step_limit is not None and t > step_limit:
            break
        if player_slot >= len(steps[t]) or player_slot >= len(steps[t - 1]):
            continue
        acts = steps[t][player_slot].get("action") or []
        if not acts:
            continue
        obs_prev = normalize_obs(steps[t - 1][player_slot]["observation"], fallback_step=t - 1)
        player = obs_player_slot(obs_prev, player_slot)
        if t - 1 not in per_step_cache:
            per_step_cache[t - 1] = infer_outputs(model, device, obs_prev)
        outputs, masks = per_step_cache[t - 1]
        planets = obs_prev["planets"]
        owned_indices = masks["owned_indices"].numpy()
        pid_to_slot = {}
        for slot in range(masks["owned_count"]):
            pidx = int(owned_indices[slot])
            if pidx < len(planets):
                pid_to_slot[int(planets[pidx][0])] = slot

        for move in acts:
            if len(move) < 3:
                continue
            launches += 1
            from_pid = int(move[0])
            emitted_angle = float(move[1])
            ship_count = int(move[2])
            src_idx = pid_to_planet_idx(planets, from_pid)
            slot = pid_to_slot.get(from_pid)
            if src_idx is None or slot is None:
                continue
            src = planets[src_idx]
            tgt_idx = _find_target_planet_index(
                (float(src[2]), float(src[3])),
                emitted_angle,
                ship_count,
                planets,
                obs_prev.get("initial_planets", planets),
                float(obs_prev.get("angular_velocity", 0.0)),
                int(obs_prev.get("step", 0)),
                max_planets=min(len(planets), 48),
            )

            slot_target_logits = outputs["target_logits"][0, slot]
            ranked, raw_argmax_idx, top1_valid_idx = rank_targets(
                planets, slot_target_logits, player, src_idx, ship_count, top_k=max(top_k, 8)
            )
            corrected_tidx = corrected_move_target_idx(planets, slot_target_logits, player, src_idx)

            raw_argmax_is_invalid = False
            if raw_argmax_idx is not None and raw_argmax_idx < len(planets):
                raw_tgt = planets[raw_argmax_idx]
                raw_argmax_is_invalid = (
                    int(raw_tgt[1]) == player or int(raw_tgt[0]) == int(src[0])
                )
            if raw_argmax_is_invalid:
                invalid_raw_argmax += 1
                dropped_slot_actions += 1
            if corrected_tidx is not None and raw_argmax_idx is not None and corrected_tidx != raw_argmax_idx:
                corrected_target_differs += 1

            chosen_target = None
            angle_error_deg = None
            nearest = best_alt_by(ranked, "nearest")
            weakest_nearby = best_nearby_by(ranked, "weakest")
            highest_prod_nearby = best_nearby_by(ranked, "highest_prod")
            tempo_choice = best_alt_by(ranked, "tempo")
            if tgt_idx >= 0 and tgt_idx < len(planets):
                tgt = planets[tgt_idx]
                intercept = _target_intercept_angle(src, tgt, ship_count, obs_prev)
                angle_error_deg = circ_err_deg(emitted_angle, intercept)
                chosen_dist, chosen_eta = target_distance_eta(src, tgt, ship_count)
                chosen_target = {
                    "planet_idx": int(tgt_idx),
                    "planet_id": int(tgt[0]),
                    "owner": int(tgt[1]),
                    "ships": int(tgt[5]),
                    "production": int(tgt[6]),
                    "distance": round(chosen_dist, 3),
                    "eta": chosen_eta,
                    "capture_cost": capture_cost(tgt, player),
                    "tempo_score": round(tempo_score(src, tgt, ship_count, player), 6),
                    "angle_error_deg": round(angle_error_deg, 3),
                }
                if top1_valid_idx is not None and top1_valid_idx != tgt_idx:
                    policy_mismatch += 1
                if nearest and int(nearest["planet_id"]) != int(tgt[0]) and chosen_dist > float(nearest["distance"]) + 8.0:
                    farther_than_nearest += 1
                if angle_error_deg is not None and angle_error_deg > aim_gap_deg:
                    aim_gap += 1
            else:
                open_space += 1

            top_ranked = [r.to_dict() for r in ranked[:top_k]]
            action_row = {
                "step": t,
                "obs_step": int(obs_prev["step"]),
                "from_planet_id": from_pid,
                "from_planet_idx": src_idx,
                "ships": ship_count,
                "emitted_angle_rad": round(emitted_angle, 6),
                "slot": int(slot),
                "fire_logit": round(float(outputs["fire_logits"][0, slot].item()), 6),
                "fire_prob": round(float(torch.sigmoid(outputs["fire_logits"][0, slot]).item()), 6),
                "raw_argmax_target_idx": raw_argmax_idx,
                "raw_argmax_invalid": raw_argmax_is_invalid,
                "corrected_valid_target_idx": corrected_tidx,
                "top1_valid_target_idx": top1_valid_idx,
                "decoded_target": chosen_target,
                "top_targets": top_ranked,
                "nearest_target": nearest,
                "weakest_nearby_target": weakest_nearby,
                "highest_prod_nearby_target": highest_prod_nearby,
                "cheapest_target": best_alt_by(ranked, "cheapest"),
                "tempo_target": tempo_choice,
                "distance_gap_vs_nearest": (
                    round(chosen_target["distance"] - nearest["distance"], 3)
                    if chosen_target and nearest else None
                ),
                "eta_gap_vs_nearest": (
                    int(chosen_target["eta"] - nearest["eta"])
                    if chosen_target and nearest else None
                ),
                "classification": (
                    "open_space_or_decode_fail" if tgt_idx < 0 else
                    "aim_gap" if angle_error_deg is not None and angle_error_deg > aim_gap_deg else
                    "policy_mismatch" if top1_valid_idx is not None and top1_valid_idx != tgt_idx else
                    "target_priority"
                ),
            }
            actions_out.append(action_row)

    return {
        "episode_id": episode_id,
        "replay_path": str(replay_path),
        "player_slot": int(player_slot),
        "player_name": player_name,
        "opponent_name": team_names[1 - player_slot] if len(team_names) > 1 else None,
        "steps": len(steps),
        "summary": {
            "launches": launches,
            "open_space_or_decode_fail": open_space,
            "aim_gap": aim_gap,
            "policy_mismatch": policy_mismatch,
            "invalid_raw_argmax": invalid_raw_argmax,
            "corrected_target_differs": corrected_target_differs,
            "dropped_slot_actions": dropped_slot_actions,
            "farther_than_nearest": farther_than_nearest,
        },
        "actions": actions_out,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Submission Target Audit")
    lines.append("")
    lines.append(f"- checkpoint: `{report['checkpoint']}`")
    lines.append(f"- action_decode: `{report['action_decode']}`")
    lines.append(f"- episodes: `{len(report['episodes'])}`")
    lines.append(f"- audited player: `{report['player_name']}`")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    agg = report["aggregate"]
    for k in (
        "episodes",
        "launches",
        "open_space_or_decode_fail",
        "aim_gap",
        "policy_mismatch",
        "invalid_raw_argmax",
        "corrected_target_differs",
        "dropped_slot_actions",
        "farther_than_nearest",
    ):
        lines.append(f"- {k}: `{agg[k]}`")
    lines.append("")

    for ep in report["episodes"]:
        lines.append(f"## Episode {ep['episode_id']}")
        lines.append("")
        lines.append(
            f"- `{ep['player_name']}` vs `{ep['opponent_name']}`"
            f" | launches `{ep['summary']['launches']}`"
            f" | open-space `{ep['summary']['open_space_or_decode_fail']}`"
            f" | aim-gap `{ep['summary']['aim_gap']}`"
            f" | invalid-raw-argmax `{ep['summary']['invalid_raw_argmax']}`"
            f" | dropped-slots `{ep['summary']['dropped_slot_actions']}`"
        )
        suspicious = sorted(
            ep["actions"],
            key=lambda a: (
                0 if a["classification"] != "target_priority" else 1,
                0 if not a["decoded_target"] else a["decoded_target"]["distance"]
                - (a["nearest_target"]["distance"] if a["nearest_target"] else 0.0),
            ),
            reverse=True,
        )[:8]
        if not suspicious:
            lines.append("")
            continue
        lines.append("")
        lines.append("| step | src | chosen | nearest | weak-near | hi-prod-near | tempo | cls | notes |")
        lines.append("|---|---:|---|---|---|---|---|---|---|")
        for a in suspicious:
            chosen = a["decoded_target"]
            nearest = a["nearest_target"]
            weak = a["weakest_nearby_target"]
            hi_prod = a["highest_prod_nearby_target"]
            tempo = a["tempo_target"]
            notes = []
            if a["raw_argmax_invalid"]:
                notes.append("raw-argmax-invalid")
            if chosen and nearest and int(chosen["planet_id"]) != int(nearest["planet_id"]):
                notes.append(f"dist+{chosen['distance'] - nearest['distance']:.1f}")
                notes.append(f"eta+{chosen['eta'] - nearest['eta']}")
            if chosen and chosen.get("angle_error_deg", 0.0) > 0:
                notes.append(f"ang {chosen['angle_error_deg']:.1f}d")
            lines.append(
                f"| {a['step']} | {a['from_planet_id']} | "
                f"{format_target_md(chosen)} | {format_target_md(nearest)} | "
                f"{format_target_md(weak)} | {format_target_md(hi_prod)} | {format_target_md(tempo)} | "
                f"{a['classification']} | {', '.join(notes) if notes else '-'} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_target_md(t: dict[str, Any] | None) -> str:
    if not t:
        return "`-`"
    return (
        f"`p{t['planet_id']}`"
        f" d={t['distance']:.1f}"
        f" eta={t['eta']}"
        f" s={t['ships']}"
        f" prod={t['production']}"
    )


def build_report(
    checkpoint: str,
    action_decode: str,
    player_name: str,
    episode_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate = {
        "episodes": len(episode_reports),
        "launches": 0,
        "open_space_or_decode_fail": 0,
        "aim_gap": 0,
        "policy_mismatch": 0,
        "invalid_raw_argmax": 0,
        "corrected_target_differs": 0,
        "dropped_slot_actions": 0,
        "farther_than_nearest": 0,
    }
    for ep in episode_reports:
        summary = ep["summary"]
        for k in (
            "launches",
            "open_space_or_decode_fail",
            "aim_gap",
            "policy_mismatch",
            "invalid_raw_argmax",
            "corrected_target_differs",
            "dropped_slot_actions",
            "farther_than_nearest",
        ):
            aggregate[k] += int(summary[k])

    return {
        "checkpoint": str(checkpoint),
        "action_decode": action_decode,
        "player_name": player_name,
        "aggregate": aggregate,
        "episodes": episode_reports,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--replay-dir")
    ap.add_argument("--replay", action="append", default=[], help="Specific replay JSON path; repeatable")
    ap.add_argument("--episode-id", action="append", default=[], help="Episode id inside --replay-dir; repeatable")
    ap.add_argument("--player-name", default="Saheb")
    ap.add_argument("--player-slot", type=int)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--aim-gap-deg", type=float, default=15.0)
    ap.add_argument("--step-limit", type=int)
    ap.add_argument("--output-json")
    ap.add_argument("--output-md")
    ap.add_argument("--device", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    replay_paths = resolve_replay_paths(args.replay_dir, args.replay, args.episode_id)
    if not replay_paths:
        raise SystemExit("No replay files found.")

    device = torch.device(
        args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    )
    model, _, action_decode = load_model(args.checkpoint, device)

    episode_reports = []
    effective_player_name = args.player_name
    for replay_path in replay_paths:
        with replay_path.open() as f:
            replay = json.load(f)
        slot = choose_player_slot(replay, args.player_name, args.player_slot)
        team_names = replay.get("info", {}).get("TeamNames") or []
        if slot < len(team_names):
            effective_player_name = str(team_names[slot])
        episode_reports.append(
            audit_episode(
                replay_path,
                replay,
                model,
                device,
                player_slot=slot,
                top_k=args.top_k,
                aim_gap_deg=args.aim_gap_deg,
                step_limit=args.step_limit,
            )
        )

    report = build_report(args.checkpoint, action_decode, effective_player_name, episode_reports)

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2))
    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render_markdown(report))

    print(
        f"audit complete: episodes={report['aggregate']['episodes']} "
        f"launches={report['aggregate']['launches']} "
        f"open_space={report['aggregate']['open_space_or_decode_fail']} "
        f"aim_gap={report['aggregate']['aim_gap']} "
        f"policy_mismatch={report['aggregate']['policy_mismatch']} "
        f"invalid_raw_argmax={report['aggregate']['invalid_raw_argmax']} "
        f"dropped_slot_actions={report['aggregate']['dropped_slot_actions']}"
    )
    for ep in report["episodes"]:
        s = ep["summary"]
        print(
            f"  ep={ep['episode_id']} {ep['player_name']} vs {ep['opponent_name']}: "
            f"launches={s['launches']} open_space={s['open_space_or_decode_fail']} "
            f"aim_gap={s['aim_gap']} invalid_raw_argmax={s['invalid_raw_argmax']} "
            f"dropped_slot_actions={s['dropped_slot_actions']} "
            f"farther_than_nearest={s['farther_than_nearest']}"
        )


if __name__ == "__main__":
    main()
