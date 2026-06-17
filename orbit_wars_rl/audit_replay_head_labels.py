"""Audit policy heads against top-player replay actions.

This is a read-only diagnostic. It projects replay action lists into the current
one-action-per-source interface, forwards our model on the replay observations,
and compares head agreement for winner moves versus loser moves from the same
replay corpus.

Replay timing convention: action at steps[t][seat] was selected from the
observation at steps[t-1][seat].
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbit_wars_rl.action_mask import compute_action_masks, _def_rank, _def_ship_adequacy_rank
from orbit_wars_rl.bc import _find_target_planet_index
from orbit_wars_rl.config import Config
from orbit_wars_rl.eval import load_checkpoint
from orbit_wars_rl.features import extract_features, set_game_phase_features
from orbit_wars_rl.model import EntityTransformer


PHASES = (("all", 0, 10**9), ("open", 0, 50), ("mid", 50, 100), ("late", 100, 10**9))


def _iter_paths(inputs: Iterable[str], replay_kind: str) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out.extend(Path(x) for x in glob.glob(str(p / "*.json")))
        elif any(ch in item for ch in "*?[]"):
            out.extend(Path(x) for x in glob.glob(item))
        else:
            out.append(p)
    paths = sorted(set(out))
    if replay_kind == "1v1":
        paths = [p for p in paths if "_1v1" in p.name]
    elif replay_kind == "ffa":
        paths = [p for p in paths if "_ffa" in p.name]
    return paths


def _load_model(checkpoint: str, device: torch.device) -> tuple[EntityTransformer, Config]:
    cfg = Config()
    state_dict, _ = load_checkpoint(checkpoint, cfg)
    cfg.model.game_phase_features = bool(cfg.model.game_phase_features or cfg.model.global_feature_dim >= 15)
    set_game_phase_features(cfg.model.game_phase_features)
    model = EntityTransformer(cfg.model).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"target_head.weight", "target_head.bias"}
    bad_missing = [k for k in missing if k not in allowed_missing]
    bad_unexpected = [k for k in unexpected if not k.startswith("value_pp_")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"checkpoint load mismatch: missing={bad_missing} unexpected={bad_unexpected}")
    return model.eval(), cfg


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


def _pid_to_slot(obs: dict[str, Any], masks: dict[str, Any]) -> dict[int, int]:
    out: dict[int, int] = {}
    planets = obs["planets"]
    owned_indices = masks["owned_indices"].cpu().numpy()
    for slot in range(int(masks["owned_count"])):
        pidx = int(owned_indices[slot])
        if 0 <= pidx < len(planets):
            out[int(planets[pidx][0])] = slot
    return out


def _mask_target_logits(logits: torch.Tensor, obs: dict[str, Any], masks: dict[str, Any],
                        allow_reinforce: bool, gate_min: int) -> torch.Tensor:
    planets = obs["planets"]
    player = int(obs["player"])
    out = logits.clone()
    owned_indices = masks["owned_indices"].cpu().numpy()
    owned_count = int(masks["owned_count"])
    gate_block = allow_reinforce and gate_min > 0 and owned_count < gate_min
    for slot in range(min(owned_count, out.shape[0])):
        pidx = int(owned_indices[slot])
        if pidx >= len(planets):
            continue
        src_id = int(planets[pidx][0])
        for tidx, tgt in enumerate(planets[:out.shape[-1]]):
            is_source = int(tgt[0]) == src_id
            is_own = int(tgt[1]) == player
            illegal = is_source or (is_own and (not allow_reinforce or gate_block))
            if illegal:
                out[slot, tidx] = -1e9
    return out


def _infer_outputs(model: EntityTransformer, device: torch.device, obs: dict[str, Any]) -> tuple[dict, dict]:
    player = int(obs["player"])
    features = extract_features(obs, player, num_players=2)
    masks = compute_action_masks(obs, player)
    pairwise = features.get("pairwise_features")
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
            pairwise_features=pairwise.unsqueeze(0).to(device) if pairwise is not None else None,
        )
    return (
        {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in outputs.items()},
        {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in masks.items()},
    )


def _target_idx_for_move(obs: dict[str, Any], src_idx: int, angle: float, ships: int,
                         max_planets: int) -> int:
    src = obs["planets"][src_idx]
    return _find_target_planet_index(
        (float(src[2]), float(src[3])),
        float(angle),
        int(ships),
        obs["planets"],
        obs.get("initial_planets", obs["planets"]),
        float(obs.get("angular_velocity", 0.0)),
        int(obs.get("step", 0)),
        max_planets=min(len(obs["planets"]), int(max_planets)),
    )


def _project_action(obs: dict[str, Any], action: list, masks: dict[str, Any],
                    max_planets: int = 48) -> tuple[list[dict[str, Any]], Counter]:
    planets = obs["planets"]
    player = int(obs["player"])
    pidx_by_pid = {int(p[0]): i for i, p in enumerate(planets)}
    own_pids = {int(p[0]) for p in planets if int(p[1]) == player}
    pid_slot = _pid_to_slot(obs, masks)
    stats = Counter()
    by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    raw_moves = action if isinstance(action, list) else []

    stats["turns"] += 1
    stats["raw_moves"] += len(raw_moves)
    if len(raw_moves) > 16:
        stats["turns_gt16_moves"] += 1

    for move in raw_moves:
        if not isinstance(move, (list, tuple)) or len(move) < 3:
            stats["bad_move"] += 1
            continue
        try:
            src_pid = int(move[0])
            angle = float(move[1])
            ships = int(move[2])
        except Exception:
            stats["bad_move"] += 1
            continue
        if src_pid not in own_pids:
            stats["not_owned_source"] += 1
            continue
        stats["owned_moves"] += 1
        stats["owned_ship_mass"] += max(0, ships)
        src_idx = pidx_by_pid.get(src_pid)
        if src_idx is None:
            stats["missing_source"] += 1
            continue
        target_idx = _target_idx_for_move(obs, src_idx, angle, ships, max_planets)
        if target_idx < 0 or target_idx >= min(len(planets), int(max_planets)):
            stats["unresolved_target"] += 1
            stats["unresolved_ship_mass"] += max(0, ships)
            continue
        if src_pid not in pid_slot:
            stats["source_not_top16"] += 1
            stats["source_not_top16_ship_mass"] += max(0, ships)
            continue
        target_owner = int(planets[target_idx][1])
        row = {
            "source_id": src_pid,
            "slot": int(pid_slot[src_pid]),
            "target_idx": int(target_idx),
            "target_id": int(planets[target_idx][0]),
            "ships": max(0, ships),
            "kind": "save" if target_owner == player else "attack",
        }
        by_source[src_pid].append(row)

    for src_pid, rows in by_source.items():
        if len(rows) > 1:
            stats["same_source_sources"] += 1
            stats["same_source_moves"] += len(rows)
            stats["same_source_lost_moves"] += len(rows) - 1
            if len({r["target_id"] for r in rows}) > 1:
                stats["split_source_sources"] += 1
        best = max(rows, key=lambda r: (int(r["ships"]), -int(r["target_idx"])))
        stats["projected_ship_mass"] += int(best["ships"])
        if len(rows) > 1:
            stats["same_source_lost_ship_mass"] += sum(int(r["ships"]) for r in rows if r is not best)
        stats["projected_moves"] += 1
        if best["kind"] == "save":
            stats["projected_save"] += 1
        else:
            stats["projected_attack"] += 1

    return [max(rows, key=lambda r: (int(r["ships"]), -int(r["target_idx"]))) for rows in by_source.values()], stats


def _same_source_nearest_baseline(obs: dict[str, Any], label: dict[str, Any],
                                  max_planets: int = 48) -> dict[str, Any] | None:
    planets = obs["planets"]
    player = int(obs["player"])
    src_id = int(label["source_id"])
    actual_target = int(label["target_id"])
    src = next((p for p in planets if int(p[0]) == src_id), None)
    if src is None:
        return None
    sx, sy = float(src[2]), float(src[3])
    best = None
    best_dist = None
    for tidx, tgt in enumerate(planets[:int(max_planets)]):
        tid = int(tgt[0])
        if tid == src_id or tid == actual_target:
            continue
        # Nearest non-own target is a simple within-state target baseline.
        if int(tgt[1]) == player:
            continue
        dist = (float(tgt[2]) - sx) ** 2 + (float(tgt[3]) - sy) ** 2
        if best_dist is None or dist < best_dist:
            best = (tidx, tgt)
            best_dist = dist
    if best is None:
        return None
    tidx, tgt = best
    return {
        "source_id": src_id,
        "slot": int(label["slot"]),
        "target_idx": int(tidx),
        "target_id": int(tgt[0]),
        "ships": int(label["ships"]),
        "kind": str(label["kind"]),
    }


def _phase_names(step: int) -> list[str]:
    return [name for name, lo, hi in PHASES if lo <= step < hi]


def _add_counter(dst: Counter, src: Counter) -> None:
    dst.update(src)


def _audit_projected_moves(model: EntityTransformer, cfg: Config, device: torch.device,
                           obs: dict[str, Any], projected: list[dict[str, Any]],
                           side_stats: dict[str, Counter]) -> None:
    if not projected:
        return
    outputs, masks = _infer_outputs(model, device, obs)
    fire_probs = torch.sigmoid(outputs["fire_logits"][0]).detach().cpu()
    target_logits = _mask_target_logits(
        outputs["target_logits"][0].detach().cpu(),
        obs,
        masks,
        bool(getattr(cfg.model, "allow_reinforce", False)),
        int(getattr(cfg.model, "reinforce_gate_min_planets", 0)),
    )
    ship_logits = outputs["ship_logits"][0].detach().cpu()
    max_ships = masks["max_ships"].cpu().numpy().squeeze(0)
    step = int(obs.get("step", 0))

    for label in projected:
        for ph in _phase_names(step):
            stats = side_stats[ph]
            key = str(label["kind"])
            stats["labels"] += 1
            stats[f"{key}_labels"] += 1
            slot = int(label["slot"])
            target_idx = int(label["target_idx"])
            required = int(label["ships"])
            fire_ready = float(fire_probs[slot]) >= 0.5
            target_rank = _def_rank(target_logits[slot].tolist(), target_idx)
            ship_rank = _def_ship_adequacy_rank(
                ship_logits[slot].tolist(),
                required,
                int(max_ships[slot]),
                str(cfg.model.ship_bin_mode),
            )
            if fire_ready:
                stats["fire_ready"] += 1
                stats[f"{key}_fire_ready"] += 1
            if target_rank is not None:
                stats["target_rank_sum"] += float(target_rank)
                stats["target_rank_n"] += 1
                if target_rank <= 1:
                    stats["target_top1"] += 1
                    stats[f"{key}_target_top1"] += 1
                if target_rank <= 3:
                    stats["target_top3"] += 1
                    stats[f"{key}_target_top3"] += 1
                if target_rank <= 5:
                    stats["target_top5"] += 1
            if ship_rank is not None:
                stats["ship_rank_sum"] += float(ship_rank)
                stats["ship_rank_n"] += 1
                if ship_rank <= 1:
                    stats["ship_top1"] += 1
                if ship_rank <= 3:
                    stats["ship_top3"] += 1
            if fire_ready and target_rank is not None and ship_rank is not None:
                if target_rank <= 1 and ship_rank <= 1:
                    stats["joint_top1"] += 1
                if target_rank <= 3 and ship_rank <= 3:
                    stats["joint_top3"] += 1


def _audit_fire_source_contrast(model: EntityTransformer, device: torch.device,
                                obs: dict[str, Any], projected: list[dict[str, Any]],
                                side_stats: dict[str, Counter]) -> None:
    if not projected:
        return
    outputs, masks = _infer_outputs(model, device, obs)
    fire_probs = torch.sigmoid(outputs["fire_logits"][0]).detach().cpu()
    owned_count = min(int(masks["owned_count"]), int(fire_probs.shape[0]))
    if owned_count <= 0:
        return
    used_slots = {
        int(label["slot"])
        for label in projected
        if 0 <= int(label["slot"]) < owned_count
    }
    step = int(obs.get("step", 0))
    for ph in _phase_names(step):
        stats = side_stats[ph]
        for slot in range(owned_count):
            p = float(fire_probs[slot])
            if slot in used_slots:
                stats["used_sources"] += 1
                stats["used_fire_p_sum"] += p
                if p >= 0.5:
                    stats["used_fire_ready"] += 1
            else:
                stats["unused_sources"] += 1
                stats["unused_fire_p_sum"] += p
                if p >= 0.5:
                    stats["unused_fire_ready"] += 1


def _winner_loser_seats(replay: dict[str, Any]) -> dict[int, str]:
    rewards = replay.get("rewards") or []
    if not rewards:
        return {}
    best = max(rewards)
    worst = min(rewards)
    if best == worst:
        return {}
    out = {}
    for seat, reward in enumerate(rewards):
        if reward == best:
            out[seat] = "winner"
        elif reward == worst:
            out[seat] = "loser"
    return out


def _pct(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _summarize_projection(c: Counter) -> dict[str, Any]:
    owned = c.get("owned_moves", 0)
    projected = c.get("projected_moves", 0)
    owned_mass = c.get("owned_ship_mass", 0)
    projected_mass = c.get("projected_ship_mass", 0)
    return {
        "turns": int(c.get("turns", 0)),
        "raw_moves": int(c.get("raw_moves", 0)),
        "owned_moves": int(owned),
        "projected_moves": int(projected),
        "owned_ship_mass": int(owned_mass),
        "projected_ship_mass": int(projected_mass),
        "projected_attack": int(c.get("projected_attack", 0)),
        "projected_save": int(c.get("projected_save", 0)),
        "owned_to_projected": _pct(projected, owned),
        "owned_mass_to_projected": _pct(projected_mass, owned_mass),
        "unresolved_target": _pct(c.get("unresolved_target", 0), owned),
        "unresolved_ship_mass": _pct(c.get("unresolved_ship_mass", 0), owned_mass),
        "source_not_top16": _pct(c.get("source_not_top16", 0), owned),
        "source_not_top16_ship_mass": _pct(c.get("source_not_top16_ship_mass", 0), owned_mass),
        "same_source_lost_moves": _pct(c.get("same_source_lost_moves", 0), owned),
        "same_source_lost_ship_mass": _pct(c.get("same_source_lost_ship_mass", 0), owned_mass),
        "split_source_sources": int(c.get("split_source_sources", 0)),
        "turns_gt16_moves": _pct(c.get("turns_gt16_moves", 0), c.get("turns", 0)),
    }


def _summarize_heads(c: Counter) -> dict[str, Any]:
    n = c.get("labels", 0)
    rn = c.get("target_rank_n", 0)
    sn = c.get("ship_rank_n", 0)
    return {
        "labels": int(n),
        "attack_labels": int(c.get("attack_labels", 0)),
        "save_labels": int(c.get("save_labels", 0)),
        "fire_ready": _pct(c.get("fire_ready", 0), n),
        "target_top1": _pct(c.get("target_top1", 0), n),
        "target_top3": _pct(c.get("target_top3", 0), n),
        "target_top5": _pct(c.get("target_top5", 0), n),
        "target_rank_avg": _pct(c.get("target_rank_sum", 0.0), rn),
        "ship_top1": _pct(c.get("ship_top1", 0), n),
        "ship_top3": _pct(c.get("ship_top3", 0), n),
        "ship_rank_avg": _pct(c.get("ship_rank_sum", 0.0), sn),
        "joint_top1": _pct(c.get("joint_top1", 0), n),
        "joint_top3": _pct(c.get("joint_top3", 0), n),
        "attack_fire_ready": _pct(c.get("attack_fire_ready", 0), c.get("attack_labels", 0)),
        "attack_target_top1": _pct(c.get("attack_target_top1", 0), c.get("attack_labels", 0)),
        "attack_target_top3": _pct(c.get("attack_target_top3", 0), c.get("attack_labels", 0)),
        "save_fire_ready": _pct(c.get("save_fire_ready", 0), c.get("save_labels", 0)),
        "save_target_top1": _pct(c.get("save_target_top1", 0), c.get("save_labels", 0)),
        "save_target_top3": _pct(c.get("save_target_top3", 0), c.get("save_labels", 0)),
    }


def _summarize_fire_source(c: Counter) -> dict[str, Any]:
    used = c.get("used_sources", 0)
    unused = c.get("unused_sources", 0)
    return {
        "used_sources": int(used),
        "unused_sources": int(unused),
        "used_fire_ready": _pct(c.get("used_fire_ready", 0), used),
        "unused_fire_ready": _pct(c.get("unused_fire_ready", 0), unused),
        "used_fire_p": _pct(c.get("used_fire_p_sum", 0.0), used),
        "unused_fire_p": _pct(c.get("unused_fire_p_sum", 0.0), unused),
    }


def _summarize(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for side in ("winner", "loser"):
        out[side] = {}
        for ph in ("all", "open", "mid", "late"):
            out[side][ph] = {
                "projection": _summarize_projection(payload["projection"][side][ph]),
                "heads": _summarize_heads(payload["heads"][side][ph]),
                "same_source_baseline": _summarize_heads(payload["same_source_baseline"][side][ph]),
                "fire_source": _summarize_fire_source(payload["fire_source"][side][ph]),
            }
    return out


def _fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def render_markdown(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# Replay Head Audit",
        "",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- replay paths: {payload['paths']} ({payload['replay_kind']})",
        f"- replays used: {payload['replays_used']}",
        f"- errors: {payload['errors']}",
        "",
        "## Head Agreement",
        "",
        "| side | phase | labels | attack/save | fire>=.5 | target top1/3/5 | ship top1/3 | joint top1/3 | target rank avg |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for side in ("winner", "loser"):
        for ph in ("all", "open", "mid", "late"):
            h = s[side][ph]["heads"]
            lines.append(
                f"| {side} | {ph} | {h['labels']} | {h['attack_labels']}/{h['save_labels']} | "
                f"{_fmt_pct(h['fire_ready'])} | "
                f"{_fmt_pct(h['target_top1'])}/{_fmt_pct(h['target_top3'])}/{_fmt_pct(h['target_top5'])} | "
                f"{_fmt_pct(h['ship_top1'])}/{_fmt_pct(h['ship_top3'])} | "
                f"{_fmt_pct(h['joint_top1'])}/{_fmt_pct(h['joint_top3'])} | "
                f"{h['target_rank_avg']:.1f} |"
            )
    lines += [
        "",
        "## Same-State Target Baseline",
        "",
        "Baseline preserves each replay move's source and ship count, but replaces the target with the nearest non-own target from the same source.",
        "",
        "| side | phase | labels | replay target top1/3 | baseline target top1/3 | replay target rank | baseline target rank |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for side in ("winner", "loser"):
        for ph in ("all", "open", "mid", "late"):
            h = s[side][ph]["heads"]
            b = s[side][ph]["same_source_baseline"]
            lines.append(
                f"| {side} | {ph} | {h['labels']} | "
                f"{_fmt_pct(h['target_top1'])}/{_fmt_pct(h['target_top3'])} | "
                f"{_fmt_pct(b['target_top1'])}/{_fmt_pct(b['target_top3'])} | "
                f"{h['target_rank_avg']:.1f} | {b['target_rank_avg']:.1f} |"
            )
    lines += [
        "",
        "## Same-State Target Baseline By Move Kind",
        "",
        "For save labels, the baseline is an opportunity-cost attack target from the same source, not another legal save target.",
        "",
        "| side | phase | kind | labels | replay target top1/3 | baseline target top1/3 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for side in ("winner", "loser"):
        for ph in ("all", "open", "mid", "late"):
            h = s[side][ph]["heads"]
            b = s[side][ph]["same_source_baseline"]
            for kind in ("attack", "save"):
                labels = h[f"{kind}_labels"]
                lines.append(
                    f"| {side} | {ph} | {kind} | {labels} | "
                    f"{_fmt_pct(h[f'{kind}_target_top1'])}/{_fmt_pct(h[f'{kind}_target_top3'])} | "
                    f"{_fmt_pct(b[f'{kind}_target_top1'])}/{_fmt_pct(b[f'{kind}_target_top3'])} |"
                )
    lines += [
        "",
        "## Same-State Fire Source Contrast",
        "",
        "Compares fire readiness on replay-used source slots against unused owned source slots from the same state.",
        "",
        "| side | phase | used/unused sources | used fire>=.5 | unused fire>=.5 | used fire_p | unused fire_p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for side in ("winner", "loser"):
        for ph in ("all", "open", "mid", "late"):
            f = s[side][ph]["fire_source"]
            lines.append(
                f"| {side} | {ph} | {f['used_sources']}/{f['unused_sources']} | "
                f"{_fmt_pct(f['used_fire_ready'])} | {_fmt_pct(f['unused_fire_ready'])} | "
                f"{f['used_fire_p']:.3f} | {f['unused_fire_p']:.3f} |"
            )
    lines += [
        "",
        "## Projection Loss",
        "",
        "| side | phase | owned moves | projected | move keep | mass keep | source not top16 move/mass | same-source lost move/mass | unresolved move/mass | >16-turn |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for side in ("winner", "loser"):
        for ph in ("all", "open", "mid", "late"):
            p = s[side][ph]["projection"]
            lines.append(
                f"| {side} | {ph} | {p['owned_moves']} | {p['projected_moves']} | "
                f"{_fmt_pct(p['owned_to_projected'])} | {_fmt_pct(p['owned_mass_to_projected'])} | "
                f"{_fmt_pct(p['source_not_top16'])}/{_fmt_pct(p['source_not_top16_ship_mass'])} | "
                f"{_fmt_pct(p['same_source_lost_moves'])}/{_fmt_pct(p['same_source_lost_ship_mass'])} | "
                f"{_fmt_pct(p['unresolved_target'])}/{_fmt_pct(p['unresolved_ship_mass'])} | "
                f"{_fmt_pct(p['turns_gt16_moves'])} |"
            )
    lines += [
        "",
        "Read `owned->projected` as the fraction of replay owned-source moves that survive the current one-action-per-source/top-16 projection.",
        "",
    ]
    return "\n".join(lines)


def audit_replays(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cpu")
    model, cfg = _load_model(args.checkpoint, device)
    paths = _iter_paths(args.paths or ["leader-replays/rank1"], args.replay_kind)
    if args.max_replays > 0:
        paths = paths[: int(args.max_replays)]

    payload = {
        "checkpoint": args.checkpoint,
        "paths": len(paths),
        "replay_kind": args.replay_kind,
        "replays_used": 0,
        "errors": 0,
        "projection": {
            side: {ph: Counter() for ph, _, _ in PHASES}
            for side in ("winner", "loser")
        },
        "heads": {
            side: {ph: Counter() for ph, _, _ in PHASES}
            for side in ("winner", "loser")
        },
        "same_source_baseline": {
            side: {ph: Counter() for ph, _, _ in PHASES}
            for side in ("winner", "loser")
        },
        "fire_source": {
            side: {ph: Counter() for ph, _, _ in PHASES}
            for side in ("winner", "loser")
        },
    }
    max_planets = int(getattr(cfg.model, "max_planets", 48))

    for path in paths:
        try:
            replay = json.loads(path.read_text())
            steps = replay.get("steps") or []
            seats = _winner_loser_seats(replay)
            if len(steps) < 2 or not seats:
                continue
            payload["replays_used"] += 1
            max_t = min(len(steps), int(args.max_step) + 1 if args.max_step > 0 else len(steps))
            for t in range(1, max_t):
                for seat, side in seats.items():
                    if seat >= len(steps[t]) or seat >= len(steps[t - 1]):
                        continue
                    raw_obs = (steps[t - 1][seat] or {}).get("observation") or {}
                    action = (steps[t][seat] or {}).get("action") or []
                    obs = _copy_replay_obs(raw_obs, seat, t - 1)
                    if not obs["planets"]:
                        continue
                    masks = compute_action_masks(obs, int(obs["player"]))
                    projected, proj_stats = _project_action(obs, action, masks, max_planets=max_planets)
                    for ph in _phase_names(int(obs.get("step", t - 1))):
                        _add_counter(payload["projection"][side][ph], proj_stats)
                    _audit_projected_moves(model, cfg, device, obs, projected, payload["heads"][side])
                    _audit_fire_source_contrast(model, device, obs, projected, payload["fire_source"][side])
                    baseline = [
                        row for row in (
                            _same_source_nearest_baseline(obs, label, max_planets=max_planets)
                            for label in projected
                        )
                        if row is not None
                    ]
                    _audit_projected_moves(model, cfg, device, obs, baseline, payload["same_source_baseline"][side])
        except Exception as exc:
            payload["errors"] += 1
            if args.verbose:
                print(f"error {path}: {exc}", file=sys.stderr)

    payload["projection"] = {
        side: {ph: dict(counter) for ph, counter in phases.items()}
        for side, phases in payload["projection"].items()
    }
    payload["heads"] = {
        side: {ph: dict(counter) for ph, counter in phases.items()}
        for side, phases in payload["heads"].items()
    }
    payload["same_source_baseline"] = {
        side: {ph: dict(counter) for ph, counter in phases.items()}
        for side, phases in payload["same_source_baseline"].items()
    }
    payload["fire_source"] = {
        side: {ph: dict(counter) for ph, counter in phases.items()}
        for side, phases in payload["fire_source"].items()
    }
    payload["summary"] = _summarize({
        "projection": {
            side: {ph: Counter(counter) for ph, counter in phases.items()}
            for side, phases in payload["projection"].items()
        },
        "heads": {
            side: {ph: Counter(counter) for ph, counter in phases.items()}
            for side, phases in payload["heads"].items()
        },
        "same_source_baseline": {
            side: {ph: Counter(counter) for ph, counter in phases.items()}
            for side, phases in payload["same_source_baseline"].items()
        },
        "fire_source": {
            side: {ph: Counter(counter) for ph, counter in phases.items()}
            for side, phases in payload["fire_source"].items()
        },
    })
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="Replay JSON files, directories, or globs. Defaults to leader-replays/rank1.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--replay-kind", choices=["1v1", "ffa", "all"], default="1v1")
    ap.add_argument("--max-replays", type=int, default=0)
    ap.add_argument("--max-step", type=int, default=0, help="0 = all replay steps.")
    ap.add_argument("--output-json", default="gpu_run_artifacts/head_audit/replay_head_audit.json")
    ap.add_argument("--output-md", default="gpu_run_artifacts/head_audit/replay_head_audit.md")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit_replays(args)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(payload))
    print(json.dumps(payload["summary"]["winner"]["all"], indent=2))
    print(json.dumps(payload["summary"]["loser"]["all"], indent=2))
    print(f"replays_used={payload['replays_used']} errors={payload['errors']}")
    print(f"saved json -> {out_json}")
    print(f"saved md -> {out_md}")


if __name__ == "__main__":
    main()
