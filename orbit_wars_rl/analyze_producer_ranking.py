"""Analyze replay launches against producer-style candidate scoring.

For each launch in a replay slice:
  - align replay action[t] with observation[t-1]
  - decode the actual targeted planet
  - recover the model's target-head ranking for that source slot
  - score the same source's producer-style target shortlist using the
    local horizon planner (H=18 in 2-player mode)

Primary use: check whether our failure is mostly:
  - bad planet ranking by the target head
  - bad source selection / source not shortlisted by producer logic
  - bad commitment / no producer-valid candidate for the chosen move

Example:
  source /Users/saheb/home/.venv/bin/activate
  python orbit_wars_rl/analyze_producer_ranking.py \
    --checkpoint gpu_run_artifacts/jarvis/checkpoints/torch_step_5242880_rev36_20260605_024319.pt \
    --replay-dir /tmp/ajay_seed_replays \
    --player-slot 0 \
    --step-limit 40 \
    --output-json /tmp/producer_ranking.json \
    --output-md /tmp/producer_ranking.md
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

from orbit_wars_rl.audit_submission_targets import (
    RankedTarget,
    corrected_move_target_idx,
    infer_outputs,
    normalize_obs,
    obs_player_slot,
    pid_to_planet_idx,
    rank_targets,
    resolve_replay_paths,
    target_distance_eta,
    capture_cost,
    tempo_score,
    render_markdown,
)
from orbit_wars_rl.bc import _find_target_planet_index
from orbit_wars_rl.config import Config
from orbit_wars_rl.eval import load_checkpoint
from orbit_wars_rl.model import EntityTransformer
from orbit_wars_rl.action_mask import _target_intercept_angle
from orbit_wars_rl.features import fleet_speed

from opponents.candidate_ajay_1200 import ProducerLiteConfig, _config_for, _movement_config
from opponents.orbit_lite.adapter import single_obs_to_tensor
from opponents.orbit_lite.distance_cache import build_distance_cache
from opponents.orbit_lite.movement_step import ensure_planet_movement
from opponents.orbit_lite.obs import parse_obs
from opponents.orbit_lite.planner_core import (
    build_target_shortlist,
    capture_floor,
    make_launch_set,
    reachable_mask,
    safe_drain,
    score_candidates,
)
from opponents.orbit_lite.intercept_aim import intercept_angle


@dataclass
class ProducerTarget:
    planet_idx: int
    planet_id: int
    owner: int
    ships: int
    production: int
    distance: float
    eta: int
    capture_cost: int
    producer_score: float
    valid: bool
    shortlist_rank: int
    send_ships: int

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
            "producer_score": round(self.producer_score, 6),
            "valid": self.valid,
            "shortlist_rank": self.shortlist_rank,
            "send_ships": self.send_ships,
        }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--replay-dir")
    ap.add_argument("--replay-path", action="append", default=[])
    ap.add_argument("--episode-id", action="append", default=[])
    ap.add_argument("--player-name", default="")
    ap.add_argument("--player-slot", type=int)
    ap.add_argument("--step-limit", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--output-json", default="/tmp/producer_ranking.json")
    ap.add_argument("--output-md", default="/tmp/producer_ranking.md")
    return ap.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> tuple[EntityTransformer, Config]:
    cfg = Config()
    sd, _ = load_checkpoint(checkpoint_path, cfg)
    model = EntityTransformer(cfg.model)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    return model, cfg


def infer_player_slot(replay: dict[str, Any], player_name: str, explicit_slot: int | None) -> int:
    if explicit_slot is not None:
        return int(explicit_slot)
    if player_name:
        agents = replay.get("info", {}).get("Agents") or []
        for idx, agent in enumerate(agents):
            if str(agent.get("Name", "")) == player_name:
                return idx
    return 0


def producer_candidates_for_source(obs: dict[str, Any], src_idx: int) -> dict[str, Any]:
    player = int(obs["player"])
    obs_tensors = single_obs_to_tensor(obs, player_id=player, device="cpu")
    player_count = int(obs_tensors["player_count"].item())
    config: ProducerLiteConfig = _config_for(player_count)
    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(config, player_count=player_count),
        cached_movement=None,
    )
    parsed = parse_obs(obs_tensors, player_id=player)
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]
    cache = build_distance_cache(movement, max_k=H)
    prod = movement.planet_prod

    source_mask = parsed.owned & parsed.alive & (parsed.ships >= float(config.min_ships_to_launch))
    if src_idx >= parsed.P or not bool(source_mask[src_idx].item()):
        return {
            "source_valid": False,
            "reason": "source_below_min_launch_or_not_owned",
            "producer_targets": [],
            "producer_best_target_idx": None,
        }

    target_idx, target_exists = build_target_shortlist(
        parsed,
        obs_tensors,
        status,
        cache,
        config=config,
        K_eta=max(1, min(H, max(H, 1))),
        H=H,
        prod=prod,
        source_mask=source_mask,
    )
    if not bool(target_exists.any()):
        return {
            "source_valid": True,
            "reason": "no_targets_in_shortlist",
            "producer_targets": [],
            "producer_best_target_idx": None,
        }

    valid_tidx = target_idx[target_exists]
    T = int(valid_tidx.shape[0])
    src_tensor = torch.tensor([src_idx], dtype=torch.long)
    source_ships = parsed.ships[src_tensor].to(dtype=parsed.ships.dtype)
    H_eff = torch.full((), float(H), dtype=parsed.ships.dtype)
    drain = safe_drain(
        status,
        source_idx=src_tensor,
        source_ships=source_ships,
        H_eff=H_eff,
        player_id=player,
    )  # [1]
    send_ships = int(torch.floor(drain[0]).item())
    if send_ships < 1:
        return {
            "source_valid": True,
            "reason": "safe_drain_zero",
            "producer_targets": [],
            "producer_best_target_idx": None,
        }

    sizes = torch.full((1, T), float(send_ships), dtype=parsed.ships.dtype)
    eta_cap = torch.full((T,), float(H), dtype=parsed.ships.dtype)
    floor = capture_floor(
        status,
        target_idx=valid_tidx,
        k_max=H,
        capture_overhead=1.0,
        player_id=player,
    )
    K = int(floor.shape[-1])
    active = reachable_mask(
        movement,
        source_idx=src_tensor,
        target_idx=valid_tidx,
        fleet_sizes=sizes.unsqueeze(-1),
        eta_cap=eta_cap,
    ).squeeze(-1)  # [1, T]
    aim = intercept_angle(
        movement,
        src_tensor.unsqueeze(1),          # [1,1]
        valid_tidx.unsqueeze(0),          # [1,T]
        sizes,                            # [1,T]
        active=active,
    )
    eta = aim["eta"]                      # [1,T]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))
    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor.unsqueeze(0).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones((1, T), dtype=parsed.ships.dtype)
    clears_floor = sizes >= floor_at_arr
    valid = (viable & clears_floor).reshape(T)

    cand_src = src_tensor.view(1, 1).expand(T, 1)
    cand_tgt = valid_tidx.view(T, 1)
    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt,
        ships=torch.where(valid.view(T, 1), sizes.reshape(T, 1), torch.zeros((T, 1), dtype=sizes.dtype)),
        eta=torch.where(valid.view(T, 1), eta.reshape(T, 1), torch.ones((T, 1), dtype=eta.dtype)),
        valid=valid.view(T, 1),
        player_id=player,
    )
    scores = score_candidates(
        status,
        prod=prod,
        alive_by_step=alive_by_step,
        player_count=player_count,
        launches=launches,
        player_id=player,
    )
    planets = obs["planets"]
    src = planets[src_idx]
    candidates: list[ProducerTarget] = []
    for shortlist_rank, tidx_t in enumerate(valid_tidx.tolist()):
        tgt = planets[int(tidx_t)]
        dist, eta_int = target_distance_eta(src, tgt, send_ships)
        candidates.append(
            ProducerTarget(
                planet_idx=int(tidx_t),
                planet_id=int(tgt[0]),
                owner=int(tgt[1]),
                ships=int(tgt[5]),
                production=int(tgt[6]),
                distance=dist,
                eta=eta_int,
                capture_cost=capture_cost(tgt, player),
                producer_score=float(scores[shortlist_rank].item()),
                valid=bool(valid[shortlist_rank].item()),
                shortlist_rank=shortlist_rank,
                send_ships=send_ships,
            )
        )
    candidates.sort(key=lambda c: c.producer_score, reverse=True)
    best = next((c for c in candidates if math.isfinite(c.producer_score) and c.valid), None)
    return {
        "source_valid": True,
        "reason": None,
        "producer_targets": [c.to_dict() for c in candidates],
        "producer_best_target_idx": None if best is None else int(best.planet_idx),
    }


def analyze_replay(
    replay_path: Path,
    replay: dict[str, Any],
    model: EntityTransformer,
    device: torch.device,
    player_slot: int,
    step_limit: int,
    top_k: int,
) -> dict[str, Any]:
    steps = replay.get("steps", [])
    per_step_cache: dict[int, tuple[dict, dict]] = {}
    actions_out: list[dict[str, Any]] = []
    launches = 0
    producer_match = 0
    producer_available = 0
    source_invalid = 0
    policy_producer_match = 0
    policy_available = 0

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
        pid_to_slot: dict[int, int] = {}
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
            producer_info = producer_candidates_for_source(obs_prev, src_idx)
            producer_targets = producer_info["producer_targets"]
            producer_best_idx = producer_info["producer_best_target_idx"]
            if not producer_info["source_valid"]:
                source_invalid += 1
            if producer_best_idx is not None:
                producer_available += 1
                if tgt_idx == producer_best_idx:
                    producer_match += 1
                if top1_valid_idx is not None:
                    policy_available += 1
                    if int(top1_valid_idx) == int(producer_best_idx):
                        policy_producer_match += 1

            producer_rank = None
            chosen_producer = None
            for idx, candidate in enumerate(producer_targets):
                if int(candidate["planet_idx"]) == int(tgt_idx):
                    producer_rank = idx
                    chosen_producer = candidate
                    break

            corrected_tidx = corrected_move_target_idx(planets, slot_target_logits, player, src_idx)
            intercept = None
            if 0 <= tgt_idx < len(planets):
                intercept = _target_intercept_angle(src, planets[tgt_idx], ship_count, obs_prev)

            actions_out.append(
                {
                    "step": t,
                    "obs_step": int(obs_prev["step"]),
                    "from_planet_id": from_pid,
                    "from_planet_idx": src_idx,
                    "ships": ship_count,
                    "slot": int(slot),
                    "decoded_target_idx": tgt_idx,
                    "decoded_target_id": None if tgt_idx < 0 or tgt_idx >= len(planets) else int(planets[tgt_idx][0]),
                    "raw_argmax_target_idx": raw_argmax_idx,
                    "corrected_valid_target_idx": corrected_tidx,
                    "policy_top1_valid_target_idx": top1_valid_idx,
                    "policy_top_targets": [r.to_dict() for r in ranked[:top_k]],
                    "producer_source_valid": producer_info["source_valid"],
                    "producer_reason": producer_info["reason"],
                    "producer_best_target_idx": producer_best_idx,
                    "producer_match": producer_best_idx is not None and tgt_idx == producer_best_idx,
                    "producer_chosen_rank": producer_rank,
                    "producer_chosen": chosen_producer,
                    "producer_targets": producer_targets[:top_k],
                    "emitted_angle_rad": round(emitted_angle, 6),
                    "decoded_intercept_angle_rad": None if intercept is None else round(float(intercept), 6),
                }
            )

    return {
        "episode_id": replay.get("info", {}).get("EpisodeId", replay_path.stem),
        "replay_path": str(replay_path),
        "player_slot": player_slot,
        "summary": {
            "launches": launches,
            "producer_available": producer_available,
            "producer_match": producer_match,
            "producer_match_rate": (producer_match / producer_available) if producer_available else None,
            "policy_available": policy_available,
            "policy_producer_match": policy_producer_match,
            "policy_producer_match_rate": (policy_producer_match / policy_available) if policy_available else None,
            "source_invalid": source_invalid,
        },
        "actions": actions_out,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Producer Ranking Analysis",
        "",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- episodes: `{len(report['episodes'])}`",
        f"- launches: `{report['aggregate']['launches']}`",
        f"- producer available: `{report['aggregate']['producer_available']}`",
        f"- producer match rate: `{report['aggregate']['producer_match_rate']:.3f}`" if report["aggregate"]["producer_match_rate"] is not None else "- producer match rate: `n/a`",
        f"- policy-vs-producer match rate: `{report['aggregate']['policy_producer_match_rate']:.3f}`" if report["aggregate"]["policy_producer_match_rate"] is not None else "- policy-vs-producer match rate: `n/a`",
        f"- source invalid count: `{report['aggregate']['source_invalid']}`",
        "",
    ]
    for ep in report["episodes"]:
        lines.append(f"## Episode {ep['episode_id']}")
        lines.append("")
        s = ep["summary"]
        pmr = "n/a" if s["producer_match_rate"] is None else f"{s['producer_match_rate']:.3f}"
        ppmr = "n/a" if s["policy_producer_match_rate"] is None else f"{s['policy_producer_match_rate']:.3f}"
        lines.append(
            f"- launches={s['launches']} producer_available={s['producer_available']} "
            f"producer_match={s['producer_match']} producer_match_rate={pmr} "
            f"policy_producer_match={s['policy_producer_match']} policy_producer_match_rate={ppmr} "
            f"source_invalid={s['source_invalid']}"
        )
        for a in ep["actions"][:12]:
            lines.append(
                f"  - step {a['step']} src={a['from_planet_id']} ships={a['ships']} "
                f"chosen={a['decoded_target_id']} policy_top1={a['policy_top1_valid_target_idx']} "
                f"producer_best={a['producer_best_target_idx']} producer_rank={a['producer_chosen_rank']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    replay_paths = resolve_replay_paths(args.replay_dir, args.replay_path, args.episode_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.checkpoint, device)

    episodes: list[dict[str, Any]] = []
    aggregate_launches = 0
    aggregate_avail = 0
    aggregate_match = 0
    aggregate_invalid = 0
    aggregate_policy_avail = 0
    aggregate_policy_match = 0
    for replay_path in replay_paths:
        replay = json.loads(replay_path.read_text())
        player_slot = infer_player_slot(replay, args.player_name, args.player_slot)
        episode = analyze_replay(
            replay_path,
            replay,
            model,
            device,
            player_slot=player_slot,
            step_limit=args.step_limit,
            top_k=args.top_k,
        )
        episodes.append(episode)
        aggregate_launches += int(episode["summary"]["launches"])
        aggregate_avail += int(episode["summary"]["producer_available"])
        aggregate_match += int(episode["summary"]["producer_match"])
        aggregate_invalid += int(episode["summary"]["source_invalid"])
        aggregate_policy_avail += int(episode["summary"]["policy_available"])
        aggregate_policy_match += int(episode["summary"]["policy_producer_match"])

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episodes": episodes,
        "aggregate": {
            "launches": aggregate_launches,
            "producer_available": aggregate_avail,
            "producer_match": aggregate_match,
            "producer_match_rate": (aggregate_match / aggregate_avail) if aggregate_avail else None,
            "policy_available": aggregate_policy_avail,
            "policy_producer_match": aggregate_policy_match,
            "policy_producer_match_rate": (aggregate_policy_match / aggregate_policy_avail) if aggregate_policy_avail else None,
            "source_invalid": aggregate_invalid,
        },
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2))
    Path(args.output_md).write_text(render_markdown(report))
    print(json.dumps(report["aggregate"], indent=2))
    print(f"saved json -> {args.output_json}")
    print(f"saved md -> {args.output_md}")


if __name__ == "__main__":
    main()
