"""Analyze replay moves against producer-style whole-action ranking.

Unlike target-only analysis, this compares a replay move against the producer
planner's global action candidates across all sources:
  - best candidate action in the state
  - whether replay source matches producer-best source
  - whether replay target matches producer-best target
  - score/rank of the replay move itself under the same short-horizon scorer

This is intended to answer whether our failures are really due to:
  - wrong target for a good source
  - wrong source entirely
  - wrong ship commitment for the intended source/target
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

from orbit_wars_rl.audit_submission_targets import normalize_obs, resolve_replay_paths
from orbit_wars_rl.producer_ranking import infer_player_slot
from orbit_wars_rl.features import fleet_speed

from opponents.candidate_ajay_1200 import ProducerLiteConfig, _config_for, _movement_config
from opponents.orbit_lite.adapter import single_obs_to_tensor
from opponents.orbit_lite.distance_cache import build_distance_cache
from opponents.orbit_lite.intercept_aim import intercept_angle
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


@dataclass
class ActionCandidate:
    source_idx: int
    source_id: int
    target_idx: int
    target_id: int
    ships: int
    eta: int
    score: float
    valid: bool
    target_is_mine: bool
    target_is_neutral: bool
    source_ships: int
    target_prod: int
    floor_at_arrival: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_idx": self.source_idx,
            "source_id": self.source_id,
            "target_idx": self.target_idx,
            "target_id": self.target_id,
            "ships": self.ships,
            "eta": self.eta,
            "score": round(self.score, 6),
            "valid": self.valid,
            "target_is_mine": self.target_is_mine,
            "target_is_neutral": self.target_is_neutral,
            "source_ships": self.source_ships,
            "target_prod": self.target_prod,
            "floor_at_arrival": self.floor_at_arrival,
        }


def action_extra_features(
    *,
    ships: int,
    eta: int | None,
    valid: bool,
    source_ships: int,
    target_prod: int,
    floor_at_arrival: int,
    score: float,
    target_is_mine: bool,
    target_is_neutral: bool,
) -> list[float]:
    eta_val = float(eta if eta is not None else 32)
    surplus = max(0.0, float(ships - floor_at_arrival))
    return [
        eta_val / 32.0,
        float(ships) / 420.0,
        1.0 if valid else 0.0,
        float(source_ships) / 420.0,
        float(target_prod) / 6.0,
        float(floor_at_arrival) / 420.0,
        min(surplus / 100.0, 2.0),
        math.tanh(float(score) / 20.0),
        1.0 if target_is_mine else 0.0,
        1.0 if target_is_neutral else 0.0,
    ]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir")
    ap.add_argument("--replay-path", action="append", default=[])
    ap.add_argument("--episode-id", action="append", default=[])
    ap.add_argument("--player-name", default="")
    ap.add_argument("--player-slot", type=int)
    ap.add_argument("--step-limit", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--output-json", default="/tmp/producer_action_ranking.json")
    ap.add_argument("--output-md", default="/tmp/producer_action_ranking.md")
    return ap.parse_args()


def _enumerate_attack_candidates(obs: dict) -> dict[str, Any]:
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
    P = parsed.P
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]
    cache = build_distance_cache(movement, max_k=H)
    prod = movement.planet_prod
    dtype = parsed.ships.dtype

    source_mask = parsed.owned & parsed.alive & (parsed.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return {"candidates": [], "config": config}

    source_idx = torch.nonzero(source_mask, as_tuple=False).flatten().long()
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
        return {"candidates": [], "config": config}
    target_idx = target_idx[target_exists]
    target_is_mine = parsed.owned[target_idx.clamp(0, P - 1)]

    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    source_ships = parsed.ships[source_idx].to(dtype)
    H_eff = torch.full((), float(H), dtype=dtype)
    drain = safe_drain(
        status,
        source_idx=source_idx,
        source_ships=source_ships,
        H_eff=H_eff,
        player_id=player,
    )
    sizes = drain.view(S, 1).expand(S, T).floor()
    eta_cap = torch.full((T,), float(H), dtype=dtype)
    floor = capture_floor(
        status,
        target_idx=target_idx,
        k_max=H,
        capture_overhead=1.0,
        player_id=player,
    )
    K = int(floor.shape[-1])
    active = reachable_mask(
        movement,
        source_idx=source_idx,
        target_idx=target_idx,
        fleet_sizes=sizes.unsqueeze(-1),
        eta_cap=eta_cap,
    ).squeeze(-1)
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),
        target_idx.unsqueeze(0),
        sizes,
        active=active,
    )
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))
    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype)
    clears_floor = sizes >= floor_at_arr
    valid = viable & clears_floor & (sizes >= 1.0) & (source_idx.view(S, 1) != target_idx.view(1, T))

    C = S * T
    cand_src = source_idx.view(S, 1).expand(S, T).reshape(C, 1)
    cand_tgt = target_idx.view(1, T).expand(S, T).reshape(C, 1)
    cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, 1)
    cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, 1)
    cand_valid = valid.reshape(C)
    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt,
        ships=cand_send,
        eta=cand_eta,
        valid=cand_valid.view(C, 1),
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
    out: list[ActionCandidate] = []
    for c in range(C):
        sidx = int(cand_src[c, 0].item())
        tidx = int(cand_tgt[c, 0].item())
        ships = int(cand_send[c, 0].item())
        eta_i = int(math.ceil(float(cand_eta[c, 0].item())))
        out.append(
            ActionCandidate(
                source_idx=sidx,
                source_id=int(planets[sidx][0]),
                target_idx=tidx,
                target_id=int(planets[tidx][0]),
                ships=ships,
                eta=eta_i,
                score=float(scores[c].item()),
                valid=bool(cand_valid[c].item()),
                target_is_mine=bool(target_is_mine[(c % T)].item()),
                target_is_neutral=int(planets[tidx][1]) < 0,
                source_ships=int(planets[sidx][5]),
                target_prod=int(planets[tidx][6]),
                floor_at_arrival=int(math.ceil(float(floor_at_arr[c // T, c % T].item()))),
            )
        )
    out.sort(key=lambda c: c.score, reverse=True)
    return {"candidates": out, "config": config}


def _score_replay_move(obs: dict, from_pid: int, target_pid: int | None, ship_count: int) -> dict[str, Any] | None:
    if target_pid is None:
        return None
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
    prod = movement.planet_prod
    planets = obs["planets"]
    src_idx = next((i for i, p in enumerate(planets) if int(p[0]) == int(from_pid)), None)
    tgt_idx = next((i for i, p in enumerate(planets) if int(p[0]) == int(target_pid)), None)
    if src_idx is None or tgt_idx is None:
        return None
    size = float(max(1, ship_count))
    eta_cap = torch.tensor([float(H)])
    fleet_sizes = torch.tensor([[[size]]], dtype=parsed.ships.dtype)
    active = reachable_mask(
        movement,
        source_idx=torch.tensor([src_idx]),
        target_idx=torch.tensor([tgt_idx]),
        fleet_sizes=fleet_sizes,
        eta_cap=eta_cap,
    ).squeeze(-1)
    aim = intercept_angle(
        movement,
        torch.tensor([[src_idx]]),
        torch.tensor([[tgt_idx]]),
        torch.tensor([[size]], dtype=parsed.ships.dtype),
        active=active,
    )
    eta = aim["eta"]
    viable = bool((aim["viable"] & (eta <= eta_cap.view(1, 1))).item())
    floor = capture_floor(
        status,
        target_idx=torch.tensor([tgt_idx]),
        k_max=H,
        capture_overhead=1.0,
        player_id=player,
    )
    K = int(floor.shape[-1])
    if K > 0:
        k_arr = int((eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).item())
        floor_at_arr = float(floor[0, k_arr].item())
    else:
        floor_at_arr = 1.0
    clears_floor = size >= floor_at_arr
    valid = viable and clears_floor and src_idx != tgt_idx
    launches = make_launch_set(
        source_slots=torch.tensor([[src_idx]]),
        target_slots=torch.tensor([[tgt_idx]]),
        ships=torch.tensor([[size]], dtype=parsed.ships.dtype),
        eta=eta if valid else torch.ones_like(eta),
        valid=torch.tensor([[valid]], dtype=torch.bool),
        player_id=player,
    )
    score = float(
        score_candidates(
            status,
            prod=prod,
            alive_by_step=alive_by_step,
            player_count=player_count,
            launches=launches,
            player_id=player,
        ).item()
    )
    eta_val = float(eta.item())
    eta_out = None if not math.isfinite(eta_val) else int(math.ceil(eta_val))
    return {
        "source_idx": src_idx,
        "source_id": int(from_pid),
        "target_idx": tgt_idx,
        "target_id": int(target_pid),
        "ships": int(ship_count),
        "eta": eta_out,
        "score": round(score, 6),
        "valid": bool(valid),
        "target_is_mine": int(planets[tgt_idx][1]) == player,
        "target_is_neutral": int(planets[tgt_idx][1]) < 0,
        "source_ships": int(planets[src_idx][5]),
        "target_prod": int(planets[tgt_idx][6]),
        "floor_at_arrival": int(math.ceil(floor_at_arr)),
    }


def analyze_replay(replay_path: Path, replay: dict[str, Any], player_slot: int, step_limit: int, top_k: int) -> dict[str, Any]:
    steps = replay.get("steps") or []
    actions_out: list[dict[str, Any]] = []
    exact_match = 0
    source_match = 0
    target_match = 0
    launches = 0
    scored_replay_moves = 0

    from orbit_wars_rl.bc import _find_target_planet_index

    for t in range(1, len(steps)):
        if step_limit is not None and t > step_limit:
            break
        if player_slot >= len(steps[t]) or player_slot >= len(steps[t - 1]):
            continue
        acts = steps[t][player_slot].get("action") or []
        if not acts:
            continue
        obs_prev = normalize_obs(steps[t - 1][player_slot]["observation"], fallback_step=t - 1)
        planets = obs_prev["planets"]
        cand_info = _enumerate_attack_candidates(obs_prev)
        candidates: list[ActionCandidate] = cand_info["candidates"]
        if not candidates:
            continue
        best = candidates[0]
        top = [c.to_dict() for c in candidates[:top_k]]
        by_pair = {(c.source_id, c.target_id): i for i, c in enumerate(candidates)}

        for move in acts:
            if len(move) < 3:
                continue
            launches += 1
            from_pid = int(move[0])
            angle = float(move[1])
            ship_count = int(move[2])
            src_idx = next((i for i, p in enumerate(planets) if int(p[0]) == int(from_pid)), None)
            if src_idx is None:
                continue
            src = planets[src_idx]
            tgt_idx = _find_target_planet_index(
                (float(src[2]), float(src[3])),
                angle,
                ship_count,
                planets,
                obs_prev.get("initial_planets", planets),
                float(obs_prev.get("angular_velocity", 0.0)),
                int(obs_prev.get("step", 0)),
                max_planets=min(len(planets), 48),
            )
            tgt_id = None if tgt_idx < 0 or tgt_idx >= len(planets) else int(planets[tgt_idx][0])
            exact = bool(tgt_id is not None and from_pid == best.source_id and tgt_id == best.target_id)
            src_ok = from_pid == best.source_id
            tgt_ok = tgt_id == best.target_id
            exact_match += int(exact)
            source_match += int(src_ok)
            target_match += int(tgt_ok)
            replay_scored = _score_replay_move(obs_prev, from_pid, tgt_id, ship_count) if tgt_id is not None else None
            if replay_scored is not None:
                scored_replay_moves += 1
            replay_rank = None if tgt_id is None else by_pair.get((from_pid, tgt_id))

            actions_out.append(
                {
                    "step": t,
                    "from_planet_id": from_pid,
                    "decoded_target_id": tgt_id,
                    "ships": ship_count,
                    "producer_best_action": best.to_dict(),
                    "producer_top_actions": top,
                    "replay_move_score": replay_scored,
                    "replay_action_rank": replay_rank,
                    "source_match": src_ok,
                    "target_match": tgt_ok,
                    "exact_action_match": exact,
                }
            )

    return {
        "episode_id": replay.get("info", {}).get("EpisodeId", replay_path.stem),
        "replay_path": str(replay_path),
        "player_slot": player_slot,
        "summary": {
            "launches": launches,
            "exact_action_match": exact_match,
            "exact_action_match_rate": 0.0 if launches == 0 else exact_match / launches,
            "source_match": source_match,
            "source_match_rate": 0.0 if launches == 0 else source_match / launches,
            "target_match": target_match,
            "target_match_rate": 0.0 if launches == 0 else target_match / launches,
            "scored_replay_moves": scored_replay_moves,
        },
        "actions": actions_out,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Producer Action Ranking Audit", ""]
    agg = payload["aggregate"]
    lines.append(
        f"- launches={agg['launches']} exact_action_match_rate={agg['exact_action_match_rate']:.3f} "
        f"source_match_rate={agg['source_match_rate']:.3f} target_match_rate={agg['target_match_rate']:.3f}"
    )
    lines.append("")
    for ep in payload["episodes"]:
        s = ep["summary"]
        lines.append(
            f"- ep={ep['episode_id']}: launches={s['launches']} "
            f"exact={s['exact_action_match_rate']:.3f} src={s['source_match_rate']:.3f} "
            f"tgt={s['target_match_rate']:.3f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    replay_paths = resolve_replay_paths(args.replay_dir, args.replay_path, args.episode_id)
    episodes = []
    agg = {"launches": 0, "exact_action_match": 0, "source_match": 0, "target_match": 0}

    for replay_path in replay_paths:
        try:
            replay = json.loads(Path(replay_path).read_text())
        except Exception:
            continue
        player_slot = infer_player_slot(replay, args.player_name, args.player_slot)
        ep = analyze_replay(replay_path, replay, player_slot, args.step_limit, args.top_k)
        episodes.append(ep)
        s = ep["summary"]
        agg["launches"] += s["launches"]
        agg["exact_action_match"] += s["exact_action_match"]
        agg["source_match"] += s["source_match"]
        agg["target_match"] += s["target_match"]

    launches = max(1, agg["launches"])
    aggregate = {
        **agg,
        "exact_action_match_rate": agg["exact_action_match"] / launches,
        "source_match_rate": agg["source_match"] / launches,
        "target_match_rate": agg["target_match"] / launches,
    }
    payload = {"aggregate": aggregate, "episodes": episodes}

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(payload))
    print(json.dumps(aggregate, indent=2))
    print(f"saved json -> {out_json}")
    print(f"saved md -> {out_md}")


if __name__ == "__main__":
    main()
