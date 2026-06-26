"""Validate lightweight head-audit candidates against Producer-v2 candidates.

Runs a small eval slice, records our policy's decision states, and compares the
lightweight per-source attack/save candidates used by `--natural-head-audit` against
Producer-v2's real per-source candidate ranking.

This is diagnostic only. It does not change policy actions or training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbit_wars_rl.action_mask import (
    _head_best_attack_candidate,
    _head_best_save_candidate,
    _head_threat_maps,
    compute_action_masks,
)
from orbit_wars_rl.config import Config
from orbit_wars_rl.eval import build_agent_fn, load_checkpoint
from orbit_wars_rl.model import EntityTransformer

from opponents import candidate_producer_v2 as producer_v2
from opponents.orbit_lite.adapter import single_obs_to_tensor
from opponents.orbit_lite.distance_cache import build_distance_cache
from opponents.orbit_lite.intercept_aim import intercept_angle
from opponents.orbit_lite.movement_step import ensure_planet_movement
from opponents.orbit_lite.obs import parse_obs
from opponents.orbit_lite.planner_core import (
    _candidate_indices,
    build_target_shortlist,
    capture_floor,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)


def _copy_obs(obs) -> dict[str, Any]:
    if isinstance(obs, dict):
        return {
            "step": int(obs.get("step", 0)),
            "player": int(obs.get("player", 0)),
            "planets": [list(p) for p in obs["planets"]],
            "fleets": [list(f) for f in obs.get("fleets", [])],
            "angular_velocity": float(obs.get("angular_velocity", 0.0)),
            "initial_planets": [list(p) for p in obs.get("initial_planets", obs["planets"])],
            "comet_planet_ids": list(obs.get("comet_planet_ids", [])),
        }
    planets = [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production] for p in obs.planets]
    return {
        "step": int(getattr(obs, "step", 0)),
        "player": int(getattr(obs, "player", 0)),
        "planets": planets,
        "fleets": [[f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships]
                   for f in obs.fleets],
        "angular_velocity": float(getattr(obs, "angular_velocity", 0.0)),
        "initial_planets": [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
                            for p in getattr(obs, "initial_planets", obs.planets)],
        "comet_planet_ids": list(getattr(obs, "comet_planet_ids", [])),
    }


def _load_model(checkpoint: str, device: torch.device) -> tuple[EntityTransformer, Config]:
    cfg = Config()
    sd, _ = load_checkpoint(checkpoint, cfg)
    model = EntityTransformer(cfg.model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    allowed_missing = {"target_head.weight", "target_head.bias"}
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch: missing={bad_missing} unexpected={unexpected}")
    return model.to(device).eval(), cfg


def _producer_v2_candidates(obs: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate Producer-v2's pre-greedy per-source candidates."""
    player = int(obs["player"])
    obs_tensors = single_obs_to_tensor(obs, player_id=player, device="cpu")
    player_count = int(obs_tensors["player_count"].item())
    config = producer_v2._config_for(player_count)
    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=producer_v2._movement_config(config, player_count=player_count),
        cached_movement=None,
    )
    parsed = parse_obs(obs_tensors, player_id=player)
    P = int(parsed.P)
    if P == 0:
        return []
    H = int(config.horizon)
    K_eta = max(1, min(int(config.horizon), H))
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]
    cache = build_distance_cache(movement, max_k=H)
    prod = movement.planet_prod
    dtype = parsed.ships.dtype

    source_mask = parsed.owned & parsed.alive & (parsed.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return []
    s_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(parsed.ships, source_mask, s_cap)
    target_idx, target_exists = build_target_shortlist(
        parsed,
        obs_tensors,
        status,
        cache,
        config=config,
        K_eta=K_eta,
        H=H,
        prod=prod,
        source_mask=source_mask,
    )
    if not bool(target_exists.any()):
        return []

    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = parsed.owned[target_idx.clamp(0, P - 1)]
    source_ships = parsed.ships[source_idx.clamp(0, P - 1)].to(dtype)
    h_eff = torch.full((), float(H), dtype=dtype)
    drain = safe_drain(status, source_idx=source_idx, source_ships=source_ships,
                       H_eff=h_eff, player_id=player)
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype)

    beta = float(config.reinforce_size_beta)
    enemy_mass = (
        producer_v2.cheap_enemy_pressure(parsed, cache, horizon=float(K_eta), player_id=player)
        if beta > 0.0 or bool(config.enable_regroup) else None
    )
    reinforcement = None
    if beta > 0.0:
        enemy_mass_t = enemy_mass[target_idx.clamp(0, P - 1)]
        k_arange = torch.arange(1, K_eta + 1, dtype=dtype)
        rho = reinforcement_timing_factor(
            k_arange,
            eta_free=float(config.reinforce_eta_free),
            eta_scale=float(config.reinforce_eta_scale),
        )
        reinforcement = beta * rho.view(1, K_eta) * enemy_mass_t.view(T, 1)
    floor = capture_floor(
        status,
        target_idx=target_idx,
        k_max=K_eta,
        capture_overhead=1.0,
        player_id=player,
        reinforcement=reinforcement,
    )
    K = int(floor.shape[-1])
    sizes = drain.view(S, 1).expand(S, T).floor()
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
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(
            -1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype)
    clears_floor = sizes >= floor_at_arr
    valid = (
        viable & clears_floor & (sizes >= 1.0)
        & (source_idx.view(S, 1) != target_idx.view(1, T))
        & source_exists.view(S, 1) & target_exists.view(1, T)
    )

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
    scores = torch.where(cand_valid, scores, torch.full_like(scores, float("-inf")))

    planets = obs["planets"]
    out = []
    for c in range(C):
        if not bool(cand_valid[c].item()):
            continue
        sidx = int(cand_src[c, 0].item())
        tidx = int(cand_tgt[c, 0].item())
        score = float(scores[c].item())
        if not math.isfinite(score):
            continue
        out.append({
            "source_idx": sidx,
            "source_id": int(planets[sidx][0]),
            "target_idx": tidx,
            "target_id": int(planets[tidx][0]),
            "target_is_mine": bool(target_is_mine[c % T].item()),
            "ships": int(cand_send[c, 0].item()),
            "eta": int(math.ceil(float(cand_eta[c, 0].item()))),
            "score": score,
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def _producer_by_source(candidates: list[dict[str, Any]]) -> dict[int, dict[str, list[dict[str, Any]]]]:
    by: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for cand in candidates:
        kind = "save" if cand["target_is_mine"] else "attack"
        by.setdefault(cand["source_id"], {"attack": [], "save": []})[kind].append(cand)
    return by


def _rank_in(cand: dict[str, Any] | None, ranked: list[dict[str, Any]]) -> int | None:
    if cand is None:
        return None
    tid = int(cand["target_id"])
    for i, item in enumerate(ranked, start=1):
        if int(item["target_id"]) == tid:
            return i
    return None


def _add(stats: dict[str, float], key: str, value: float = 1.0) -> None:
    stats[key] = stats.get(key, 0.0) + value


def _own_reinforce_illegal_factory(owned_count: int, gate_min: int):
    def _illegal(_src, _tgt):
        return bool(gate_min > 0 and owned_count < gate_min)
    return _illegal


def compare_state(obs: dict[str, Any], beta: float, gate_min: int) -> dict[str, float]:
    stats: dict[str, float] = {}
    planets = obs["planets"]
    player = int(obs["player"])
    masks = compute_action_masks(obs, player)
    owned_indices = masks["owned_indices"].cpu().numpy()
    max_ships = masks["max_ships"].cpu().numpy().squeeze(0)
    owned_count = int(masks["owned_count"])
    prod_candidates = _producer_by_source(_producer_v2_candidates(obs))
    threat_maps = _head_threat_maps(planets, obs.get("fleets", []), player)
    own_illegal = _own_reinforce_illegal_factory(owned_count, gate_min)

    for slot in range(min(owned_count, len(owned_indices), len(max_ships))):
        pidx = int(owned_indices[slot])
        if pidx < 0 or pidx >= len(planets) or int(planets[pidx][1]) != player:
            continue
        src = planets[pidx]
        sid = int(src[0])
        cap = int(max_ships[slot])
        source_prod = prod_candidates.get(sid, {"attack": [], "save": []})
        for kind in ("attack", "save"):
            ranked = source_prod[kind]
            if ranked:
                _add(stats, f"{kind}_producer_n")
            if kind == "attack":
                light = _head_best_attack_candidate(src, planets, player, cap)
            else:
                light = _head_best_save_candidate(
                    src, planets, obs.get("fleets", []), player, cap, beta, own_illegal, threat_maps)
            if light is not None:
                _add(stats, f"{kind}_light_n")
            if light is None and ranked:
                _add(stats, f"{kind}_missed_by_light")
                continue
            if light is not None and not ranked:
                _add(stats, f"{kind}_extra_light")
                continue
            if light is None or not ranked:
                continue
            _add(stats, f"{kind}_both_n")
            rank = _rank_in(light, ranked)
            if rank is not None:
                _add(stats, f"{kind}_rank_sum", float(rank))
                if rank <= 1:
                    _add(stats, f"{kind}_top1")
                if rank <= 3:
                    _add(stats, f"{kind}_top3")
                if rank <= 5:
                    _add(stats, f"{kind}_top5")
            else:
                _add(stats, f"{kind}_not_in_producer")
    return stats


def merge_stats(dst: dict[str, float], src: dict[str, float]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + float(v)


def summarize(stats: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in ("attack", "save"):
        prod = max(stats.get(f"{kind}_producer_n", 0.0), 1.0)
        light = max(stats.get(f"{kind}_light_n", 0.0), 1.0)
        both = max(stats.get(f"{kind}_both_n", 0.0), 1.0)
        rank_n = max(stats.get(f"{kind}_both_n", 0.0) - stats.get(f"{kind}_not_in_producer", 0.0), 1.0)
        out[kind] = {
            "producer_n": int(stats.get(f"{kind}_producer_n", 0.0)),
            "light_n": int(stats.get(f"{kind}_light_n", 0.0)),
            "both_n": int(stats.get(f"{kind}_both_n", 0.0)),
            "producer_missed_by_light": stats.get(f"{kind}_missed_by_light", 0.0) / prod,
            "light_extra_vs_producer": stats.get(f"{kind}_extra_light", 0.0) / light,
            "light_target_not_in_producer": stats.get(f"{kind}_not_in_producer", 0.0) / both,
            "producer_rank_avg_when_present": stats.get(f"{kind}_rank_sum", 0.0) / rank_n,
            "producer_top1": stats.get(f"{kind}_top1", 0.0) / both,
            "producer_top3": stats.get(f"{kind}_top3", 0.0) / both,
            "producer_top5": stats.get(f"{kind}_top5", 0.0) / both,
        }
    return out


def render_markdown(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# Head-Audit Candidate Validation",
        "",
        f"- states: {payload['states']}",
        f"- slots compared: attack producer={s['attack']['producer_n']} light={s['attack']['light_n']} both={s['attack']['both_n']}; "
        f"save producer={s['save']['producer_n']} light={s['save']['light_n']} both={s['save']['both_n']}",
        "",
        "| kind | missed by light | light extra | not in producer | rank avg | top1 | top3 | top5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind in ("attack", "save"):
        row = s[kind]
        lines.append(
            f"| {kind} | {row['producer_missed_by_light']:.1%} | "
            f"{row['light_extra_vs_producer']:.1%} | {row['light_target_not_in_producer']:.1%} | "
            f"{row['producer_rank_avg_when_present']:.1f} | {row['producer_top1']:.1%} | "
            f"{row['producer_top3']:.1%} | {row['producer_top5']:.1%} |"
        )
    lines.append("")
    lines.append("`topK` means: when both methods had a same-source candidate, the lightweight target ranked in Producer-v2's same-source candidate list.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", default="opponents/candidate_ajay_1200.py")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-step", type=int, default=120)
    ap.add_argument("--state-stride", type=int, default=3)
    ap.add_argument("--max-states", type=int, default=800)
    ap.add_argument("--beta", type=float, default=2.2)
    ap.add_argument("--output-json", default="gpu_run_artifacts/head_audit/candidate_validation.json")
    ap.add_argument("--output-md", default="gpu_run_artifacts/head_audit/candidate_validation.md")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    from kaggle_environments import make

    device = torch.device("cpu")
    model, cfg = _load_model(args.checkpoint, device)
    recorded: list[dict[str, Any]] = []
    base_agent = build_agent_fn(
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
                and len(recorded) < int(args.max_states)):
            recorded.append(obs_dict)
        return base_agent(obs)

    for i in range(int(args.games)):
        if len(recorded) >= int(args.max_states):
            break
        seed = int(args.seed_start) + i
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([recording_agent, args.opponent])

    total: dict[str, float] = {}
    errors = 0
    for obs in recorded:
        try:
            merge_stats(total, compare_state(obs, beta=float(args.beta),
                                             gate_min=int(getattr(model, "reinforce_gate_min_planets", 0))))
        except Exception:
            errors += 1
    payload = {
        "checkpoint": args.checkpoint,
        "opponent": args.opponent,
        "games": int(args.games),
        "states": len(recorded),
        "errors": errors,
        "raw": total,
        "summary": summarize(total),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(payload))
    print(json.dumps(payload["summary"], indent=2))
    print(f"states={len(recorded)} errors={errors}")
    print(f"saved json -> {out_json}")
    print(f"saved md -> {out_md}")


if __name__ == "__main__":
    main()
