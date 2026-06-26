"""Evaluate a base checkpoint with an opening-only joint-action override.

This is a narrow prototype for the new joint head:
  - for the opening window, score producer-style action candidates with the
    shadow joint ranker
  - take the top few joint actions with unique sources
  - then append base-policy actions from unused sources
  - after the opening window, fall back to the base policy entirely

The goal is to test whether the joint head can improve live play without
rewriting the full inference path yet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from orbit_wars_rl.action_mask import (
    _target_intercept_angle,
    actions_from_policy,
    actions_from_target_policy,
    compute_action_masks,
)
from orbit_wars_rl.scripts.analyze_joint_action_ranker import _action_extra, _find_ship_bin, _load_joint
from orbit_wars_rl.producer_action_ranking import _enumerate_attack_candidates, _score_replay_move
from orbit_wars_rl.config import Config
from orbit_wars_rl.eval import build_agent_fn, load_checkpoint
from orbit_wars_rl.features import extract_features
from orbit_wars_rl.model import EntityTransformer
from orbit_wars_rl.bc import _find_target_planet_index


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--joint-checkpoint", required=True)
    ap.add_argument("--opponent", required=True)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed", action="append", type=int, default=[])
    ap.add_argument("--panel", action="store_true")
    ap.add_argument("--target-decode", action="store_true")
    ap.add_argument("--opening-steps", type=int, default=20)
    ap.add_argument("--joint-max-moves", type=int, default=2)
    ap.add_argument("--hold-until-first-capture", action="store_true")
    ap.add_argument("--fire-threshold", type=float, default=0.5)
    ap.add_argument("--save-replays-dir", default="")
    return ap.parse_args()


def _build_base_model(checkpoint: str, device: torch.device) -> tuple[EntityTransformer, bool]:
    cfg = Config()
    sd, ckpt_action_decode = load_checkpoint(checkpoint, cfg)
    model = EntityTransformer(cfg.model).to(device)
    model.load_state_dict(sd)
    model.eval()
    target_decode = ckpt_action_decode == "target"
    return model, target_decode


def _batch_from_obs(obs: dict, player: int, device: torch.device) -> tuple[dict, dict]:
    features = extract_features(obs, player, num_players=2)
    masks = compute_action_masks(obs, player)
    fire_mask = masks["fire_mask"] if masks["fire_mask"].dim() == 2 else masks["fire_mask"].unsqueeze(0)
    angle_mask = masks["angle_mask"] if masks["angle_mask"].dim() == 3 else masks["angle_mask"].unsqueeze(0)
    slot_valid = masks["slot_valid"] if masks["slot_valid"].dim() == 2 else masks["slot_valid"].unsqueeze(0)
    owned_indices = masks["owned_indices"] if masks["owned_indices"].dim() == 2 else masks["owned_indices"].unsqueeze(0)
    batch = {
        "planet_features": features["planet_features"].unsqueeze(0).to(device),
        "fleet_features": features["fleet_features"].unsqueeze(0).to(device),
        "global_features": features["global_features"].unsqueeze(0).to(device),
        "planet_mask": features["planet_mask"].unsqueeze(0).to(device),
        "fleet_mask": features["fleet_mask"].unsqueeze(0).to(device),
        "fire_mask": fire_mask.to(device),
        "angle_mask": angle_mask.to(device),
        "slot_valid": slot_valid.to(device),
        "owned_indices": owned_indices.to(device),
        "pairwise_features": features["pairwise_features"].unsqueeze(0).to(device),
    }
    return batch, masks


def _score_candidate(joint_model, batch: dict, slot: int, cand) -> float:
    score = joint_model.score_actions(
        batch,
        torch.tensor([slot], device=batch["planet_features"].device),
        torch.tensor([_find_ship_bin(int(cand["ships"]))], device=batch["planet_features"].device),
        torch.tensor([int(cand["target_idx"])], device=batch["planet_features"].device),
        _action_extra(
            ships=int(cand["ships"]),
            eta=int(cand["eta"]),
            valid=bool(cand["valid"]),
            source_ships=int(cand["source_ships"]),
            target_prod=int(cand["target_prod"]),
            floor_at_arrival=int(cand["floor_at_arrival"]),
            score=float(cand["score"]),
            target_is_mine=bool(cand["target_is_mine"]),
            target_is_neutral=bool(cand["target_is_neutral"]),
        ).unsqueeze(0).to(batch["planet_features"].device),
    )
    return float(score.item())


def _slot_map_from_batch(batch: dict, planets: list[list[float]]) -> dict[int, int]:
    slot_map: dict[int, int] = {}
    owned_indices = batch["owned_indices"][0].detach().cpu().tolist()
    slot_valid = batch["slot_valid"][0].detach().cpu().tolist()
    for slot, valid in enumerate(slot_valid):
        if not valid:
            continue
        pidx = int(owned_indices[slot])
        if 0 <= pidx < len(planets):
            slot_map[int(planets[pidx][0])] = slot
    return slot_map


def _candidate_from_move(obs: dict, move: list[float]) -> dict | None:
    planets = obs["planets"]
    from_pid = int(move[0]); angle = float(move[1]); ship_count = int(move[2])
    src_idx = next((i for i, p in enumerate(planets) if int(p[0]) == from_pid), None)
    if src_idx is None:
        return None
    src = planets[src_idx]
    tgt_idx = _find_target_planet_index(
        (float(src[2]), float(src[3])),
        angle,
        ship_count,
        planets,
        obs.get("initial_planets", planets),
        float(obs.get("angular_velocity", 0.0)),
        int(obs.get("step", 0)),
        max_planets=min(len(planets), 48),
    )
    if tgt_idx < 0 or tgt_idx >= len(planets):
        return None
    tgt_id = int(planets[tgt_idx][0])
    scored = _score_replay_move(obs, from_pid, tgt_id, ship_count)
    if scored is None:
        return None
    return scored


def _top_joint_move(joint_model, obs: dict, batch: dict) -> list[float] | None:
    planets = obs["planets"]
    candidates = _enumerate_attack_candidates(obs)["candidates"]
    if not candidates:
        return None
    slot_map = _slot_map_from_batch(batch, planets)
    best_score = None
    best_move = None
    for cand in candidates:
        if not cand.valid or int(cand.ships) <= 0:
            continue
        slot = slot_map.get(int(cand.source_id))
        if slot is None:
            continue
        cand_dict = {
            "ships": int(cand.ships),
            "target_idx": int(cand.target_idx),
            "eta": int(cand.eta),
            "valid": bool(cand.valid),
            "source_ships": int(cand.source_ships),
            "target_prod": int(cand.target_prod),
            "floor_at_arrival": int(cand.floor_at_arrival),
            "score": float(cand.score),
            "target_is_mine": bool(cand.target_is_mine),
            "target_is_neutral": bool(cand.target_is_neutral),
        }
        score = _score_candidate(joint_model, batch, slot, cand_dict)
        if best_score is None or score > best_score:
            src_planet = planets[int(cand.source_idx)]
            tgt_planet = planets[int(cand.target_idx)]
            ships = int(cand.ships)
            if int(src_planet[5]) < ships:
                continue
            angle = _target_intercept_angle(src_planet, tgt_planet, ships, obs)
            best_move = [int(cand.source_id), float(angle), ships]
            best_score = score
    return best_move


def _reranked_opening_moves(
    joint_model,
    obs: dict,
    player: int,
    batch: dict,
    base_moves: list[list[float]],
) -> list[list[float]]:
    planets = obs["planets"]
    candidates = _enumerate_attack_candidates(obs)["candidates"]
    slot_map = _slot_map_from_batch(batch, planets)

    moves: list[list[float]] = []
    by_source: dict[int, list] = {}
    for cand in candidates:
        if cand.valid and int(cand.ships) > 0:
            by_source.setdefault(int(cand.source_id), []).append(cand)

    for move in base_moves:
        from_pid = int(move[0])
        slot = slot_map.get(from_pid)
        if slot is None:
            moves.append(move)
            continue
        replay_cand = _candidate_from_move(obs, move)
        base_score = None
        if replay_cand is not None:
            base_score = _score_candidate(joint_model, batch, slot, replay_cand)
        best_move = move
        best_score = base_score
        for cand in by_source.get(from_pid, []):
            cand_dict = {
                "ships": int(cand.ships),
                "target_idx": int(cand.target_idx),
                "eta": int(cand.eta),
                "valid": bool(cand.valid),
                "source_ships": int(cand.source_ships),
                "target_prod": int(cand.target_prod),
                "floor_at_arrival": int(cand.floor_at_arrival),
                "score": float(cand.score),
                "target_is_mine": bool(cand.target_is_mine),
                "target_is_neutral": bool(cand.target_is_neutral),
            }
            cand_score = _score_candidate(joint_model, batch, slot, cand_dict)
            if best_score is None or cand_score > best_score:
                src_planet = planets[int(cand.source_idx)]
                tgt_planet = planets[int(cand.target_idx)]
                ships = int(cand.ships)
                if int(src_planet[5]) < ships:
                    continue
                angle = _target_intercept_angle(src_planet, tgt_planet, ships, obs)
                best_move = [int(cand.source_id), float(angle), ships]
                best_score = cand_score
        moves.append(best_move)
    return moves


def build_joint_opening_agent(
    base_model: EntityTransformer,
    joint_model,
    device: torch.device,
    *,
    fire_threshold: float,
    target_decode: bool,
    opening_steps: int,
    hold_until_first_capture: bool,
):
    base_model.eval()
    joint_model.eval()
    state = {
        "baseline_owned": None,
        "launched_precap": False,
    }

    def agent_fn(obs):
        if not isinstance(obs, dict):
            obs = {
                "step": int(getattr(obs, "step", 0)),
                "player": int(getattr(obs, "player", 0)),
                "planets": [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production] for p in obs.planets],
                "fleets": [[f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships] for f in obs.fleets],
                "angular_velocity": float(getattr(obs, "angular_velocity", 0.0)),
                "initial_planets": [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
                                    for p in getattr(obs, "initial_planets", obs.planets)],
                "comet_planet_ids": list(getattr(obs, "comet_planet_ids", [])),
            }

        player = int(obs["player"])
        if int(obs.get("step", 0)) == 0:
            state["baseline_owned"] = sum(1 for p in obs["planets"] if int(p[1]) == player)
            state["launched_precap"] = False
        owned_now = sum(1 for p in obs["planets"] if int(p[1]) == player)
        if (
            hold_until_first_capture
            and state["baseline_owned"] is not None
            and state["launched_precap"]
            and owned_now <= int(state["baseline_owned"])
        ):
            return []

        batch, masks = _batch_from_obs(obs, player, device)
        with torch.no_grad():
            outputs = base_model(
                batch["planet_features"],
                batch["fleet_features"],
                batch["global_features"],
                batch["planet_mask"],
                batch["fleet_mask"],
                fire_mask=masks["fire_mask"].to(device),
                angle_mask=masks["angle_mask"].to(device),
                slot_valid=masks["slot_valid"].to(device),
                owned_indices=masks["owned_indices"].to(device),
                owned_count=masks["owned_count"],
                pairwise_features=batch["pairwise_features"],
            )

        base_action_fn = actions_from_target_policy if target_decode else actions_from_policy
        base_moves = base_action_fn(
            outputs["fire_logits"].cpu(),
            (outputs["target_logits"] if target_decode else outputs["angle_logits"]).cpu(),
            outputs["ship_logits"].cpu(),
            {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()},
            obs,
            player,
            fire_threshold=fire_threshold,
            sample=False,
            ship_bin_mode="absolute",
        )
        if int(obs.get("step", 0)) > opening_steps:
            return base_moves

        if not base_moves:
            fallback = _top_joint_move(joint_model, obs, batch)
            if fallback is not None and hold_until_first_capture:
                state["launched_precap"] = True
            return [fallback] if fallback is not None else base_moves
        moves = _reranked_opening_moves(joint_model, obs, player, batch, base_moves)
        if moves and hold_until_first_capture:
            state["launched_precap"] = True
        return moves

    return agent_fn


def evaluate(agent_fn, opponent: str, games: int, seed_start: int, explicit_seeds: list[int], save_replays_dir: str) -> dict:
    from kaggle_environments import make

    wins = 0
    total_games = 0
    replay_dir = Path(save_replays_dir) if save_replays_dir else None
    if replay_dir is not None:
        replay_dir.mkdir(parents=True, exist_ok=True)

    seeds = explicit_seeds or list(range(seed_start, seed_start + games))
    for seed in seeds:
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([agent_fn, opponent])
        final = env.steps[-1]
        rewards = [s.reward for s in final]
        my_reward = rewards[0] if rewards[0] is not None else 0.0
        opp_reward = rewards[1] if rewards[1] is not None else 0.0
        wins += int(my_reward > opp_reward)
        total_games += 1
        if replay_dir is not None:
            replay = env.toJSON()
            replay.setdefault("info", {})
            replay["info"]["EpisodeId"] = seed
            replay["info"]["TeamNames"] = ["JointOpening", Path(opponent).stem]
            replay["info"]["Agents"] = [{"Name": "JointOpening"}, {"Name": Path(opponent).stem}]
            (replay_dir / f"{seed}.json").write_text(json.dumps(replay))
    return {"wins": wins, "total_games": total_games, "win_rate": wins / max(1, total_games)}


def evaluate_panel(agent_fn, opponent: str, save_replays_dir: str) -> dict:
    from kaggle_environments import make
    from orbit_wars_rl.eval_panel import BY_ARCHETYPE

    wins = 0
    total_games = 0
    replay_dir = Path(save_replays_dir) if save_replays_dir else None
    if replay_dir is not None:
        replay_dir.mkdir(parents=True, exist_ok=True)

    for archetype, seeds in BY_ARCHETYPE.items():
        for seed in seeds:
            for seat in (0, 1):
                agents = [agent_fn, opponent] if seat == 0 else [opponent, agent_fn]
                env = make("orbit_wars", configuration={"seed": seed}, debug=False)
                env.run(agents)
                final = env.steps[-1]
                my_idx = seat
                opp_idx = 1 - seat
                my_reward = final[my_idx].reward if final[my_idx].reward is not None else 0.0
                opp_reward = final[opp_idx].reward if final[opp_idx].reward is not None else 0.0
                wins += int(my_reward > opp_reward)
                total_games += 1
                if replay_dir is not None:
                    replay = env.toJSON()
                    replay.setdefault("info", {})
                    replay["info"]["EpisodeId"] = seed
                    replay["info"]["TeamNames"] = (
                        ["JointOpening", Path(opponent).stem] if seat == 0
                        else [Path(opponent).stem, "JointOpening"]
                    )
                    replay["info"]["Agents"] = (
                        [{"Name": "JointOpening"}, {"Name": Path(opponent).stem}] if seat == 0
                        else [{"Name": Path(opponent).stem}, {"Name": "JointOpening"}]
                    )
                    replay["info"]["Archetype"] = archetype
                    replay["info"]["Seat"] = seat
                    out_name = f"{archetype}_{seed}_seat{seat}.json"
                    (replay_dir / out_name).write_text(json.dumps(replay))
    return {"wins": wins, "total_games": total_games, "win_rate": wins / max(1, total_games)}


def main() -> None:
    args = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    base_model, ckpt_target_decode = _build_base_model(args.checkpoint, device)
    target_decode = bool(args.target_decode or ckpt_target_decode)
    joint_model = _load_joint(args.joint_checkpoint, device)
    agent_fn = build_joint_opening_agent(
        base_model,
        joint_model,
        device,
        fire_threshold=args.fire_threshold,
        target_decode=target_decode,
        opening_steps=args.opening_steps,
        hold_until_first_capture=args.hold_until_first_capture,
    )
    if args.panel:
        result = evaluate_panel(agent_fn, args.opponent, args.save_replays_dir)
    else:
        result = evaluate(agent_fn, args.opponent, args.games, args.seed_start, args.seed, args.save_replays_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
