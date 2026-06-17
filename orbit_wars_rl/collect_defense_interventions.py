"""Collect paired short-horizon intervention labels for defense overlay moves.

This runs the supervised BC policy against an opponent, finds synthetic-defense
overlay opportunities, then deep-copies the Kaggle env and compares two
branches:

  baseline: current policy action
  support:  current policy action + overlay support move(s)

Both branches then continue with the same model/opponent policies for a short
horizon. The label records whether the support branch owns the threatened target
when baseline does not. This is closer to the desired supervised signal:
"when is this defensive move worth firing?"
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Callable

import torch
from kaggle_environments import make

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_rl.action_mask import defensive_overlay_moves, _fleet_eta_to_planet, _fleet_speed  # noqa: E402
from orbit_wars_rl.build_defense_selector import FEATURE_NAMES  # noqa: E402
from orbit_wars_rl.config import Config  # noqa: E402
from orbit_wars_rl.eval import build_agent_fn, load_checkpoint  # noqa: E402
from orbit_wars_rl.model import EntityTransformer  # noqa: E402


def _load_agent_fn(agent_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, agent_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "agent"):
        raise AttributeError(f"No agent() in {agent_path}")
    return module.agent


def _obs_to_dict(obs, player: int) -> dict:
    def planet_row(p):
        if hasattr(p, "id"):
            return [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
        return list(p)

    def fleet_row(f):
        if hasattr(f, "id"):
            return [f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships]
        return list(f)

    return {
        "step": int(getattr(obs, "step", 0)),
        "player": player,
        "planets": [planet_row(p) for p in obs.planets],
        "fleets": [fleet_row(f) for f in obs.fleets],
        "angular_velocity": float(getattr(obs, "angular_velocity", 0.0)),
        "initial_planets": [planet_row(p) for p in getattr(obs, "initial_planets", obs.planets)],
        "comet_planet_ids": list(getattr(obs, "comet_planet_ids", [])),
    }


def _nearest_enemy_dist(planet: list, planets: list, player: int) -> float:
    enemies = [p for p in planets if int(p[1]) >= 0 and int(p[1]) != player]
    if not enemies:
        return 100.0
    px, py = float(planet[2]), float(planet[3])
    return min(math.hypot(px - float(e[2]), py - float(e[3])) for e in enemies)


def _candidate_features(obs: dict, player: int, source_pid: int, target_pid: int,
                        target_age: int, garrison_floor: int, min_need: int) -> list[float] | None:
    planets = obs.get("planets") or []
    byid = {int(p[0]): p for p in planets}
    src = byid.get(int(source_pid))
    target = byid.get(int(target_pid))
    if src is None or target is None:
        return None

    inbound_ships = 0
    min_eta = None
    for fleet in obs.get("fleets") or []:
        if int(fleet[1]) == player:
            continue
        eta = _fleet_eta_to_planet(fleet, target)
        if eta is None:
            continue
        inbound_ships += int(fleet[6]) if len(fleet) > 6 else 0
        min_eta = eta if min_eta is None else min(min_eta, eta)
    if inbound_ships <= 0 or min_eta is None:
        return None

    projected_garrison = float(target[5]) + float(target[6]) * min_eta
    need = int(math.ceil(inbound_ships + min_need - projected_garrison))
    sendable = int(src[5]) - int(garrison_floor)
    target_enemy_dist = _nearest_enemy_dist(target, planets, player)
    source_enemy_dist = _nearest_enemy_dist(src, planets, player)
    source_target_dist = math.hypot(float(src[2]) - float(target[2]), float(src[3]) - float(target[3]))
    ships = min(sendable, max(need, min_need))
    support_gap = float(src[4]) + 0.1 + float(target[4])
    support_eta = max(1, int(math.ceil(max(0.0, source_target_dist - support_gap) / max(_fleet_speed(ships), 1e-6))))
    eta_margin = float(min_eta - support_eta)
    return [
        float(obs.get("step", 0)),
        float(sum(1 for p in planets if int(p[1]) == player)),
        float(target_age),
        float(target[5]),
        float(target[6]),
        float(inbound_ships),
        float(min_eta),
        float(projected_garrison),
        float(need),
        float(src[5]),
        float(sendable),
        float(need) / max(float(sendable), 1.0),
        float(target_enemy_dist),
        float(source_enemy_dist),
        float(source_enemy_dist - target_enemy_dist),
        float(source_target_dist),
        float(support_eta),
        eta_margin,
        float(support_eta <= min_eta),
    ]


def _owner_of(env, seat: int, pid: int) -> int | None:
    obs = env.steps[-1][seat].observation
    for p in obs.planets:
        row = [p.id, p.owner] if hasattr(p, "id") else p
        if int(row[0]) == int(pid):
            return int(row[1])
    return None


def _done(env) -> bool:
    return any(s.status == "DONE" for s in env.steps[-1])


def _rollout_owner_trace(env, model_agent, opponent_agent, horizon: int, target_pid: int) -> list[int | None]:
    owners = [_owner_of(env, 0, target_pid)]
    for _ in range(horizon):
        if _done(env):
            break
        obs0 = env.steps[-1][0].observation
        obs1 = env.steps[-1][1].observation
        env.step([model_agent(obs0), opponent_agent(obs1)])
        owners.append(_owner_of(env, 0, target_pid))
    return owners


def _first_not_owner(trace: list[int | None], player: int = 0) -> int:
    for i, owner in enumerate(trace):
        if owner != player:
            return i
    return len(trace)


def _load_model_agent(checkpoint: str, device: torch.device, reinforce_target_bias: float):
    cfg = Config()
    cfg.device = str(device)
    state_dict, action_decode = load_checkpoint(checkpoint, cfg)
    target_decode = action_decode == "target"
    model = EntityTransformer(cfg.model).to(device)
    model.allow_reinforce = bool(getattr(cfg.model, "allow_reinforce", False))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in missing if k not in {"target_head.weight", "target_head.bias"}]
    bad_unexpected = [k for k in unexpected if not k.startswith("value_pp_")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch: missing={bad_missing}, unexpected={bad_unexpected}")
    model.eval()
    return build_agent_fn(
        model,
        device,
        target_decode=target_decode,
        reinforce_target_bias=reinforce_target_bias,
    )


def _build_payload(config: dict, records: list[dict], stats: dict, records_out: Path, complete: bool) -> dict:
    return {
        "config": config,
        "records": len(records),
        "stats": stats,
        "records_out": str(records_out),
        "complete": bool(complete),
    }


def _atomic_replace_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_outputs(
    records: list[dict],
    stats: dict,
    records_out: str | Path,
    summary_out: str | Path | None,
    config: dict,
    complete: bool,
) -> dict:
    out = Path(records_out)
    _atomic_replace_bytes(out, pickle.dumps(records))
    payload = _build_payload(config, records, stats, out, complete)
    if summary_out:
        summary = Path(summary_out)
        _atomic_replace_bytes(summary, json.dumps(payload, indent=2).encode("utf-8"))
    return payload


def collect(
    checkpoint: str,
    opponent: str,
    games: int,
    seed_start: int,
    horizon: int,
    recent_capture_window: int,
    garrison_floor: int,
    min_need: int,
    max_records: int,
    reinforce_target_bias: float,
    support_max_moves: int,
    multi_source_per_target: bool,
    flush_every: int = 0,
    flush_callback: Callable[[list[dict], dict, bool], None] | None = None,
) -> tuple[list[dict], dict]:
    device = torch.device("cpu")
    model_agent = _load_model_agent(checkpoint, device, reinforce_target_bias)
    opponent_agent = _load_agent_fn(opponent, "intervention_opponent_main")

    records: list[dict] = []
    stats = {
        "games": games,
        "candidate_steps": 0,
        "branch_records": 0,
        "helped": 0,
        "hurt": 0,
        "same": 0,
    }

    for seed in range(seed_start, seed_start + games):
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.reset(2)
        prev_owners: dict[int, int] = {}
        capture_steps: dict[int, int] = {}
        while not _done(env):
            obs0_raw = env.steps[-1][0].observation
            obs1_raw = env.steps[-1][1].observation
            obs0 = _obs_to_dict(obs0_raw, 0)
            step = int(obs0["step"])
            for p in obs0["planets"]:
                pid = int(p[0])
                owner = int(p[1])
                was = prev_owners.get(pid)
                if was is not None and was != 0 and owner == 0:
                    capture_steps[pid] = step
                prev_owners[pid] = owner

            base_action = model_agent(obs0_raw)
            eligible = {
                pid for pid, cap_step in capture_steps.items()
                if 0 <= step - cap_step <= recent_capture_window
            }
            target_ages = {pid: step - cap_step for pid, cap_step in capture_steps.items()}
            overlay = defensive_overlay_moves(
                obs0,
                0,
                base_action,
                garrison_floor=garrison_floor,
                min_need=min_need,
                max_moves=support_max_moves,
                eligible_target_pids=eligible,
                multi_source_per_target=multi_source_per_target,
            )

            if overlay and (max_records <= 0 or len(records) < max_records):
                stats["candidate_steps"] += 1
                support_move = overlay[0]
                source_pid = int(support_move[0])
                # The only target is the recent owned planet the support angle is aimed at;
                # resolve by choosing the eligible target closest to the launched angle.
                target_pid = None
                best_err = 0.6
                byid = {int(p[0]): p for p in obs0["planets"]}
                src = byid.get(source_pid)
                if src is not None:
                    for pid in eligible:
                        tgt = byid.get(pid)
                        if tgt is None:
                            continue
                        ang = math.atan2(float(tgt[3]) - float(src[3]), float(tgt[2]) - float(src[2]))
                        err = abs((ang - float(support_move[1]) + math.pi) % (2 * math.pi) - math.pi)
                        if err < best_err:
                            best_err = err
                            target_pid = pid

                if target_pid is not None:
                    features = _candidate_features(
                        obs0, 0, source_pid, target_pid,
                        target_ages.get(target_pid, 0),
                        garrison_floor, min_need,
                    )
                    if features is not None:
                        base_env = copy.deepcopy(env)
                        support_env = copy.deepcopy(env)
                        opp0 = opponent_agent(obs1_raw)
                        base_env.step([base_action, opp0])
                        support_env.step([base_action + overlay, opp0])
                        base_trace = _rollout_owner_trace(
                            base_env,
                            model_agent,
                            _load_agent_fn(opponent, f"intervention_opp_b_{seed}_{step}"),
                            horizon,
                            int(target_pid),
                        )
                        support_trace = _rollout_owner_trace(
                            support_env,
                            model_agent,
                            _load_agent_fn(opponent, f"intervention_opp_s_{seed}_{step}"),
                            horizon,
                            int(target_pid),
                        )
                        base_owner = _owner_of(base_env, 0, target_pid)
                        support_owner = _owner_of(support_env, 0, target_pid)
                        base_owned_steps = sum(1 for owner in base_trace if owner == 0)
                        support_owned_steps = sum(1 for owner in support_trace if owner == 0)
                        base_first_loss = _first_not_owner(base_trace, 0)
                        support_first_loss = _first_not_owner(support_trace, 0)
                        hold_delta = support_owned_steps - base_owned_steps
                        loss_delay = support_first_loss - base_first_loss
                        helped = int(base_owner != 0 and support_owner == 0)
                        hurt = int(base_owner == 0 and support_owner != 0)
                        hold_advantage = int(hold_delta > 0)
                        stats["helped"] += helped
                        stats["hurt"] += hurt
                        stats["hold_advantage"] = stats.get("hold_advantage", 0) + hold_advantage
                        stats["same"] += int(helped == 0 and hurt == 0)
                        stats["branch_records"] += 1
                        records.append({
                            "features": features,
                            "feature_names": FEATURE_NAMES,
                            "seed": seed,
                            "step": step,
                            "source_pid": source_pid,
                            "target_pid": int(target_pid),
                            "target_age": int(target_ages.get(target_pid, 0)),
                            "support_ships": int(support_move[2]),
                            "support_move_count": int(len(overlay)),
                            "support_source_pids": [int(m[0]) for m in overlay],
                            "support_total_ships": int(sum(int(m[2]) for m in overlay)),
                            "base_owner_h": base_owner,
                            "support_owner_h": support_owner,
                            "base_owner_trace": base_trace,
                            "support_owner_trace": support_trace,
                            "base_owned_steps": int(base_owned_steps),
                            "support_owned_steps": int(support_owned_steps),
                            "hold_delta": int(hold_delta),
                            "base_first_loss": int(base_first_loss),
                            "support_first_loss": int(support_first_loss),
                            "loss_delay": int(loss_delay),
                            "helped": helped,
                            "hurt": hurt,
                            "hold_advantage": hold_advantage,
                        })
                        if (
                            flush_callback is not None
                            and flush_every > 0
                            and len(records) % flush_every == 0
                        ):
                            flush_callback(records, stats, False)

            if max_records > 0 and len(records) >= max_records:
                break
            env.step([base_action, opponent_agent(obs1_raw)])
        if max_records > 0 and len(records) >= max_records:
            break

    if stats["branch_records"]:
        stats["help_rate"] = stats["helped"] / stats["branch_records"]
        stats["hurt_rate"] = stats["hurt"] / stats["branch_records"]
        stats["hold_advantage_rate"] = stats.get("hold_advantage", 0) / stats["branch_records"]
    return records, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", required=True)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--recent-capture-window", type=int, default=40)
    ap.add_argument("--garrison-floor", type=int, default=10)
    ap.add_argument("--min-need", type=int, default=5)
    ap.add_argument("--max-records", type=int, default=100)
    ap.add_argument("--reinforce-target-bias", type=float, default=-1.0)
    ap.add_argument("--support-max-moves", type=int, default=1)
    ap.add_argument("--multi-source-per-target", action="store_true")
    ap.add_argument("--flush-every", type=int, default=0,
                    help="Write partial records/summary every N collected records.")
    ap.add_argument("--records-out", required=True)
    ap.add_argument("--summary-out", default="")
    args = ap.parse_args()

    config = vars(args)
    records_out = Path(args.records_out)
    summary_out = Path(args.summary_out) if args.summary_out else None

    def flush(records: list[dict], stats: dict, complete: bool) -> dict:
        payload = write_outputs(records, stats, records_out, summary_out, config, complete)
        print(
            json.dumps({
                "records": payload["records"],
                "complete": payload["complete"],
                "stats": payload["stats"],
            }),
            file=sys.stderr,
            flush=True,
        )
        return payload

    records, stats = collect(
        checkpoint=args.checkpoint,
        opponent=args.opponent,
        games=args.games,
        seed_start=args.seed_start,
        horizon=args.horizon,
        recent_capture_window=args.recent_capture_window,
        garrison_floor=args.garrison_floor,
        min_need=args.min_need,
        max_records=args.max_records,
        reinforce_target_bias=args.reinforce_target_bias,
        support_max_moves=args.support_max_moves,
        multi_source_per_target=args.multi_source_per_target,
        flush_every=args.flush_every,
        flush_callback=flush,
    )
    payload = flush(records, stats, True)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
