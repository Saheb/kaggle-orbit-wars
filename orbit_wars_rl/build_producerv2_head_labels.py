"""Build Producer-v2-backed fire/target head labels on our visited states.

This is the next step after the head audits: label each owned source slot with what
Producer-v2 would consider a good per-source opportunity, plus what Producer-v2 would
actually emit from that state. The output is an offline dataset for later auxiliary or
supervised fire/target experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbit_wars_rl.action_mask import compute_action_masks, _def_rank, _def_ship_adequacy_rank
from orbit_wars_rl.bc import _find_ship_bin, _find_target_planet_index
from orbit_wars_rl.eval import build_agent_fn
from orbit_wars_rl.features import extract_features, set_game_phase_features
from orbit_wars_rl.validate_head_audit_candidates import (
    _copy_obs,
    _load_model,
    _producer_by_source,
    _producer_v2_candidates,
)

from opponents import candidate_producer_v2 as producer_v2
from opponents.orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves


def _fresh_producerv2_moves(obs: dict[str, Any]) -> list[list]:
    player = int(obs["player"])
    runtime = producer_v2.ProducerLiteRuntime()
    obs_tensors = single_obs_to_tensor(obs, player_id=player)
    with torch.no_grad():
        row = runtime.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(row, obs, player_id=player) or []


def _infer_outputs(model, device: torch.device, obs: dict[str, Any]) -> tuple[dict, dict]:
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


def _pid_to_slot(obs: dict[str, Any], masks: dict[str, Any]) -> dict[int, int]:
    out: dict[int, int] = {}
    planets = obs["planets"]
    owned_indices = masks["owned_indices"].cpu().numpy()
    for slot in range(int(masks["owned_count"])):
        pidx = int(owned_indices[slot])
        if 0 <= pidx < len(planets):
            out[int(planets[pidx][0])] = slot
    return out


def _empty_label_tensors(max_owned: int):
    return {
        "kind": torch.zeros(max_owned, dtype=torch.long),          # 0 none, 1 attack, 2 save/regroup
        "target_idx": torch.full((max_owned,), -1, dtype=torch.long),
        "ship_bin": torch.full((max_owned,), -1, dtype=torch.long),
        "ship_count": torch.zeros(max_owned, dtype=torch.long),
        "score": torch.zeros(max_owned, dtype=torch.float32),
    }


def _set_label(label: dict[str, torch.Tensor], slot: int, kind: int, target_idx: int,
               ships: int, score: float) -> None:
    label["kind"][slot] = int(kind)
    label["target_idx"][slot] = int(target_idx)
    label["ship_bin"][slot] = int(_find_ship_bin(int(ships)))
    label["ship_count"][slot] = int(max(0, ships))
    label["score"][slot] = float(score)


def _build_candidate_label(obs: dict[str, Any], masks: dict[str, Any], min_score: float):
    max_owned = int(masks["slot_valid"].shape[-1])
    label = _empty_label_tensors(max_owned)
    pid_slot = _pid_to_slot(obs, masks)
    candidates = _producer_by_source(_producer_v2_candidates(obs))
    stats = {"candidate_sources": 0, "candidate_attack": 0, "candidate_save": 0}
    for sid, by_kind in candidates.items():
        slot = pid_slot.get(int(sid))
        if slot is None:
            continue
        best_attack = by_kind["attack"][0] if by_kind["attack"] else None
        best_save = by_kind["save"][0] if by_kind["save"] else None
        choices = []
        if best_attack is not None and float(best_attack["score"]) >= min_score:
            choices.append((1, best_attack))
        if best_save is not None and float(best_save["score"]) >= min_score:
            choices.append((2, best_save))
        if not choices:
            continue
        kind, best = max(choices, key=lambda x: float(x[1]["score"]))
        _set_label(label, slot, kind, int(best["target_idx"]), int(best["ships"]), float(best["score"]))
        stats["candidate_sources"] += 1
        stats["candidate_attack" if kind == 1 else "candidate_save"] += 1
    return label, stats


def _build_selected_label(obs: dict[str, Any], masks: dict[str, Any]):
    max_owned = int(masks["slot_valid"].shape[-1])
    label = _empty_label_tensors(max_owned)
    pid_slot = _pid_to_slot(obs, masks)
    planets = obs["planets"]
    initial_planets = obs.get("initial_planets", planets)
    angular_velocity = float(obs.get("angular_velocity", 0.0))
    step = int(obs.get("step", 0))
    stats = {"selected_sources": 0, "selected_attack": 0, "selected_save": 0,
             "selected_multi_source_conflict": 0, "selected_decode_failed": 0}
    best_by_slot: dict[int, tuple[int, int, int, float]] = {}
    for move in _fresh_producerv2_moves(obs):
        if len(move) < 3:
            continue
        from_pid, angle, ships = int(move[0]), float(move[1]), int(move[2])
        slot = pid_slot.get(from_pid)
        if slot is None:
            continue
        src_idx = next((i for i, p in enumerate(planets) if int(p[0]) == from_pid), None)
        if src_idx is None:
            continue
        src = planets[src_idx]
        tgt_idx = _find_target_planet_index(
            (float(src[2]), float(src[3])),
            angle,
            ships,
            planets,
            initial_planets,
            angular_velocity,
            step,
            max_planets=min(len(planets), 48),
        )
        if tgt_idx < 0 or tgt_idx >= len(planets):
            stats["selected_decode_failed"] += 1
            continue
        kind = 2 if int(planets[tgt_idx][1]) == int(obs["player"]) else 1
        # Our policy has one action per source; if Producer-v2 emits multiple moves from
        # one source, keep the largest send as the least lossy single-head label.
        if slot in best_by_slot:
            stats["selected_multi_source_conflict"] += 1
            if ships <= best_by_slot[slot][2]:
                continue
        best_by_slot[slot] = (kind, int(tgt_idx), int(ships), 1.0)
    for slot, (kind, tgt_idx, ships, score) in best_by_slot.items():
        _set_label(label, slot, kind, tgt_idx, ships, score)
        stats["selected_sources"] += 1
        stats["selected_attack" if kind == 1 else "selected_save"] += 1
    return label, stats


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


def _audit_label(prefix: str, label: dict[str, torch.Tensor], outputs: dict, obs: dict[str, Any],
                 masks: dict[str, Any], ship_bin_mode: str, allow_reinforce: bool, gate_min: int,
                 stats: dict[str, float]) -> None:
    fire_probs = torch.sigmoid(outputs["fire_logits"][0]).detach().cpu()
    target_logits = _mask_target_logits(
        outputs["target_logits"][0].detach().cpu(), obs, masks, allow_reinforce, gate_min)
    ship_logits = outputs["ship_logits"][0].detach().cpu()
    max_ships = masks["max_ships"].cpu().numpy().squeeze(0)
    for slot in range(int(masks["owned_count"])):
        kind = int(label["kind"][slot])
        if kind <= 0:
            continue
        key = "attack" if kind == 1 else "save"
        base = f"{prefix}_{key}"
        stats[f"{base}_n"] = stats.get(f"{base}_n", 0.0) + 1.0
        if float(fire_probs[slot]) >= 0.5:
            stats[f"{base}_fire_ready"] = stats.get(f"{base}_fire_ready", 0.0) + 1.0
        target_idx = int(label["target_idx"][slot])
        target_rank = _def_rank(target_logits[slot].tolist(), target_idx)
        if target_rank is not None:
            stats[f"{base}_target_rank_sum"] = stats.get(f"{base}_target_rank_sum", 0.0) + float(target_rank)
            if target_rank <= 1:
                stats[f"{base}_target_top1"] = stats.get(f"{base}_target_top1", 0.0) + 1.0
            if target_rank <= 3:
                stats[f"{base}_target_top3"] = stats.get(f"{base}_target_top3", 0.0) + 1.0
            if target_rank <= 5:
                stats[f"{base}_target_top5"] = stats.get(f"{base}_target_top5", 0.0) + 1.0
        ship_rank = _def_ship_adequacy_rank(
            ship_logits[slot].tolist(), int(label["ship_count"][slot]), int(max_ships[slot]), ship_bin_mode)
        if ship_rank is not None:
            stats[f"{base}_ship_rank_sum"] = stats.get(f"{base}_ship_rank_sum", 0.0) + float(ship_rank)
            if ship_rank <= 1:
                stats[f"{base}_ship_top1"] = stats.get(f"{base}_ship_top1", 0.0) + 1.0
            if ship_rank <= 3:
                stats[f"{base}_ship_top3"] = stats.get(f"{base}_ship_top3", 0.0) + 1.0
        if (float(fire_probs[slot]) >= 0.5 and target_rank is not None and ship_rank is not None):
            if target_rank <= 1 and ship_rank <= 1:
                stats[f"{base}_joint_top1"] = stats.get(f"{base}_joint_top1", 0.0) + 1.0
            if target_rank <= 3 and ship_rank <= 3:
                stats[f"{base}_joint_top3"] = stats.get(f"{base}_joint_top3", 0.0) + 1.0


def _collect_states(model, cfg, args) -> list[dict[str, Any]]:
    from kaggle_environments import make

    device = torch.device("cpu")
    states: list[dict[str, Any]] = []
    agent = build_agent_fn(
        model,
        device,
        fire_threshold=0.5,
        sample=False,
        ship_bin_mode=cfg.model.ship_bin_mode,
        target_decode=True,
    )

    def recording_agent(obs):
        obs_dict = _copy_obs(obs)
        step = int(obs_dict.get("step", 0))
        if (step <= int(args.max_step)
                and step % max(1, int(args.state_stride)) == 0
                and len(states) < int(args.max_states)):
            states.append(obs_dict)
        return agent(obs)

    for i in range(int(args.games)):
        if len(states) >= int(args.max_states):
            break
        env = make("orbit_wars", configuration={"seed": int(args.seed_start) + i}, debug=False)
        env.run([recording_agent, args.opponent])
    return states


def _make_sample(obs: dict[str, Any], candidate_label: dict[str, torch.Tensor],
                 selected_label: dict[str, torch.Tensor]) -> dict[str, Any]:
    player = int(obs["player"])
    features = extract_features(obs, player, num_players=2)
    masks = compute_action_masks(obs, player)
    sample = {
        "planet_features": features["planet_features"],
        "fleet_features": features["fleet_features"],
        "global_features": features["global_features"],
        "planet_mask": features["planet_mask"],
        "fleet_mask": features["fleet_mask"],
        "fire_mask": masks["fire_mask"][0],
        "angle_mask": masks["angle_mask"][0],
        "slot_valid": masks["slot_valid"][0],
        "owned_indices": masks["owned_indices"],
        "owned_count": torch.tensor(int(masks["owned_count"]), dtype=torch.long),
    }
    if "pairwise_features" in features:
        sample["pairwise_features"] = features["pairwise_features"]
    for prefix, label in (("candidate", candidate_label), ("selected", selected_label)):
        for key, value in label.items():
            sample[f"{prefix}_{key}"] = value.clone()
    return sample


def _summarize_audit(stats: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for prefix in ("candidate", "selected"):
        out[prefix] = {}
        for key in ("attack", "save"):
            base = f"{prefix}_{key}"
            n = max(stats.get(f"{base}_n", 0.0), 1.0)
            out[prefix][key] = {
                "n": int(stats.get(f"{base}_n", 0.0)),
                "fire_ready": stats.get(f"{base}_fire_ready", 0.0) / n,
                "target_rank_avg": stats.get(f"{base}_target_rank_sum", 0.0) / n,
                "target_top1": stats.get(f"{base}_target_top1", 0.0) / n,
                "target_top3": stats.get(f"{base}_target_top3", 0.0) / n,
                "target_top5": stats.get(f"{base}_target_top5", 0.0) / n,
                "ship_rank_avg": stats.get(f"{base}_ship_rank_sum", 0.0) / n,
                "ship_top1": stats.get(f"{base}_ship_top1", 0.0) / n,
                "ship_top3": stats.get(f"{base}_ship_top3", 0.0) / n,
                "joint_top1": stats.get(f"{base}_joint_top1", 0.0) / n,
                "joint_top3": stats.get(f"{base}_joint_top3", 0.0) / n,
            }
    return out


def _render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Producer-v2 Head Labels",
        "",
        f"- states: {payload['states']}",
        f"- samples: {payload['samples']}",
        f"- min candidate score: {payload['min_candidate_score']}",
        "",
        "## Label Counts",
        "",
        "```json",
        json.dumps(payload["label_stats"], indent=2),
        "```",
        "",
        "## Baseline Model Audit",
        "",
        "| label | kind | n | fire>=.5 | target top1/3/5 | ship top1/3 | joint top1/3 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for prefix in ("candidate", "selected"):
        for kind in ("attack", "save"):
            row = payload["audit"][prefix][kind]
            lines.append(
                f"| {prefix} | {kind} | {row['n']} | {row['fire_ready']:.1%} | "
                f"{row['target_top1']:.1%}/{row['target_top3']:.1%}/{row['target_top5']:.1%} | "
                f"{row['ship_top1']:.1%}/{row['ship_top3']:.1%} | "
                f"{row['joint_top1']:.1%}/{row['joint_top3']:.1%} |"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", default="opponents/candidate_ajay_1200.py")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-step", type=int, default=120)
    ap.add_argument("--state-stride", type=int, default=3)
    ap.add_argument("--max-states", type=int, default=1000)
    ap.add_argument("--min-candidate-score", type=float, default=None,
                    help="Producer-v2 candidate score threshold. Default uses Producer-v2 roi_threshold.")
    ap.add_argument("--samples-out", default="gpu_run_artifacts/head_audit/producerv2_head_labels.pkl")
    ap.add_argument("--summary-out", default="gpu_run_artifacts/head_audit/producerv2_head_labels.json")
    ap.add_argument("--md-out", default="gpu_run_artifacts/head_audit/producerv2_head_labels.md")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    model, cfg = _load_model(args.checkpoint, device)
    # Some older checkpoints infer 15 global features from weights but lack a clean
    # persisted game_phase_features flag. Keep feature extraction aligned to weights.
    cfg.model.game_phase_features = bool(cfg.model.game_phase_features or cfg.model.global_feature_dim >= 15)
    set_game_phase_features(cfg.model.game_phase_features)
    model.allow_reinforce = bool(getattr(cfg.model, "allow_reinforce", False))
    model.reinforce_gate_min_planets = int(getattr(cfg.model, "reinforce_gate_min_planets", 0))
    model.reinforce_forward_only = bool(getattr(cfg.model, "reinforce_forward_only", False))
    model.reverse_edge_cooldown = int(getattr(cfg.model, "reverse_edge_cooldown", 0))
    model.reinforce_garrison_floor = float(getattr(cfg.model, "reinforce_garrison_floor", 0.0))
    model.sufficient_commit_factor = float(getattr(cfg.model, "sufficient_commit_factor", 0.0))
    min_score = (producer_v2._config_for(2).roi_threshold
                 if args.min_candidate_score is None else float(args.min_candidate_score))
    states = _collect_states(model, cfg, args)
    samples = []
    label_stats: dict[str, float] = {}
    audit_stats: dict[str, float] = {}

    for obs in states:
        masks = compute_action_masks(obs, int(obs["player"]))
        candidate_label, candidate_stats = _build_candidate_label(obs, masks, min_score)
        selected_label, selected_stats = _build_selected_label(obs, masks)
        for d in (candidate_stats, selected_stats):
            for k, v in d.items():
                label_stats[k] = label_stats.get(k, 0.0) + float(v)
        samples.append(_make_sample(obs, candidate_label, selected_label))
        outputs, infer_masks = _infer_outputs(model, device, obs)
        _audit_label(
            "candidate", candidate_label, outputs, obs, infer_masks, cfg.model.ship_bin_mode,
            bool(getattr(model, "allow_reinforce", False)),
            int(getattr(model, "reinforce_gate_min_planets", 0)),
            audit_stats,
        )
        _audit_label(
            "selected", selected_label, outputs, obs, infer_masks, cfg.model.ship_bin_mode,
            bool(getattr(model, "allow_reinforce", False)),
            int(getattr(model, "reinforce_gate_min_planets", 0)),
            audit_stats,
        )

    out_path = Path(args.samples_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(samples, f)

    payload = {
        "checkpoint": args.checkpoint,
        "opponent": args.opponent,
        "games": int(args.games),
        "states": len(states),
        "samples": len(samples),
        "min_candidate_score": min_score,
        "label_stats": {k: int(v) for k, v in label_stats.items()},
        "audit": _summarize_audit(audit_stats),
        "samples_out": args.samples_out,
    }
    summary = Path(args.summary_out)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, indent=2))
    md = Path(args.md_out)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(_render_md(payload))
    print(json.dumps(payload, indent=2))
    print(f"samples saved -> {out_path}")
    print(f"summary saved -> {summary}")
    print(f"markdown saved -> {md}")


if __name__ == "__main__":
    main()
