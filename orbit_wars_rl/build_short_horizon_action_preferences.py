"""Build short-horizon action preferences from replay states.

For each replay launch:
  - keep all other replay actions fixed
  - replace only that source's action with a same-source alternative
  - roll the game forward for a short horizon using VecTorchEnv
  - prefer the action with the better short-horizon outcome

This is outcome-defined supervision rather than plain imitation of the replay.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_rl.analyze_producer_action_ranking import (
    ActionCandidate,
    _enumerate_attack_candidates,
    _score_replay_move,
    action_extra_features,
)
from orbit_wars_rl.analyze_producer_ranking import infer_player_slot
from orbit_wars_rl.audit_submission_targets import normalize_obs, resolve_replay_paths
from orbit_wars_rl.action_mask import _target_intercept_angle
from orbit_wars_rl.bc import _find_ship_bin, _find_target_planet_index, trajectory_to_training_sample
from orbit_wars_rl.torch_env import MAX_FLEETS, MAX_OWNED, MAX_PLANETS, VecTorchEnv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir")
    ap.add_argument("--replay-path", action="append", default=[])
    ap.add_argument("--episode-id", action="append", default=[])
    ap.add_argument("--player-name", default="Ajay")
    ap.add_argument("--player-slot", type=int)
    ap.add_argument("--step-limit", type=int, default=20)
    ap.add_argument("--rollout-horizon", type=int, default=8)
    ap.add_argument("--confirm-horizon", type=int, default=4)
    ap.add_argument("--alts-per-action", type=int, default=3)
    ap.add_argument("--min-score-gap", type=float, default=0.25)
    ap.add_argument("--samples-out", required=True)
    ap.add_argument("--summary-out", default="")
    return ap.parse_args()


def _base_sample(obs_prev: dict, move: list) -> tuple[dict | None, dict[int, int]]:
    sample = trajectory_to_training_sample({"obs": obs_prev, "action": [move]})
    if sample is None:
        return None, {}
    pid_to_slot: dict[int, int] = {}
    planets = obs_prev["planets"]
    owned_indices = sample["owned_indices"].tolist()
    slot_valid = sample["slot_valid"].tolist()
    for slot, valid in enumerate(slot_valid):
        if not valid:
            continue
        pidx = int(owned_indices[slot])
        if 0 <= pidx < len(planets):
            pid_to_slot[int(planets[pidx][0])] = slot
    return sample, pid_to_slot


def _action_extra_tensor(
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
) -> torch.Tensor:
    return torch.tensor(
        action_extra_features(
            ships=ships,
            eta=eta,
            valid=valid,
            source_ships=source_ships,
            target_prod=target_prod,
            floor_at_arrival=floor_at_arrival,
            score=score,
            target_is_mine=target_is_mine,
            target_is_neutral=target_is_neutral,
        ),
        dtype=torch.float32,
    )


def _candidate_to_move(obs: dict, cand: ActionCandidate) -> list[float]:
    planets = obs["planets"]
    src_planet = planets[int(cand.source_idx)]
    tgt_planet = planets[int(cand.target_idx)]
    angle = _target_intercept_angle(src_planet, tgt_planet, int(cand.ships), obs)
    return [int(cand.source_id), float(angle), int(cand.ships)]


def _same_source_alternatives(
    candidates: list[ActionCandidate],
    *,
    source_id: int,
    target_id: int,
    ships: int,
    limit: int,
) -> list[ActionCandidate]:
    out = []
    for cand in candidates:
        if not cand.valid or int(cand.ships) <= 0:
            continue
        if int(cand.source_id) != int(source_id):
            continue
        if int(cand.target_id) == int(target_id):
            continue
        out.append(cand)
        if len(out) >= limit:
            break
    return out


def _same_source_candidates(
    candidates: list[ActionCandidate],
    *,
    source_id: int,
) -> list[ActionCandidate]:
    out = []
    for cand in candidates:
        if not cand.valid or int(cand.ships) <= 0:
            continue
        if int(cand.source_id) != int(source_id):
            continue
        out.append(cand)
    return out


def _source_target_candidate(
    candidates: list[ActionCandidate],
    *,
    source_id: int,
    target_id: int,
) -> ActionCandidate | None:
    for cand in candidates:
        if int(cand.source_id) == int(source_id) and int(cand.target_id) == int(target_id):
            return cand
    return None


def _is_sane_positive(
    candidates: list[ActionCandidate],
    *,
    source_id: int,
    target_id: int,
) -> bool:
    source_cands = _same_source_candidates(candidates, source_id=source_id)
    if not source_cands:
        return False
    pos = _source_target_candidate(source_cands, source_id=source_id, target_id=target_id)
    if pos is None:
        return False
    best = source_cands[0]
    if float(pos.score) < float(best.score) - 3.0:
        return False
    if int(pos.eta) > int(best.eta) + 4:
        return False
    return True


def _plausible_target_negatives(
    candidates: list[ActionCandidate],
    *,
    source_id: int,
    target_id: int,
    limit: int,
) -> list[ActionCandidate]:
    source_cands = _same_source_candidates(candidates, source_id=source_id)
    pos = _source_target_candidate(source_cands, source_id=source_id, target_id=target_id)
    if pos is None:
        return []
    out = []
    for cand in source_cands:
        if int(cand.target_id) == int(target_id):
            continue
        if float(cand.score) < float(pos.score) - 3.0:
            continue
        if abs(int(cand.eta) - int(pos.eta)) > 4:
            continue
        out.append(cand)
        if len(out) >= limit:
            break
    return out


def _counterfactual_alternatives(
    candidates: list[ActionCandidate],
    *,
    source_id: int,
    target_id: int,
    ships: int,
    limit: int,
) -> list[ActionCandidate]:
    seen: set[tuple[int, int, int]] = set()
    out: list[ActionCandidate] = []

    def add(cand: ActionCandidate) -> None:
        key = (int(cand.source_id), int(cand.target_id), int(cand.ships))
        if key in seen:
            return
        seen.add(key)
        out.append(cand)

    for cand in _plausible_target_negatives(
        candidates,
        source_id=source_id,
        target_id=target_id,
        limit=max(1, int(limit)),
    ):
        add(cand)
    return out


def _init_env_from_obs(obs: dict) -> VecTorchEnv:
    env = VecTorchEnv(num_envs=1, num_players=2, device="cpu", action_decode="target")
    planets = torch.zeros((1, MAX_PLANETS, 7), dtype=torch.float32)
    init_planets = torch.zeros((1, MAX_PLANETS, 7), dtype=torch.float32)
    planet_alive = torch.zeros((1, MAX_PLANETS), dtype=torch.bool)
    fleets = torch.zeros((1, MAX_FLEETS, 7), dtype=torch.float32)
    fleet_alive = torch.zeros((1, MAX_FLEETS), dtype=torch.bool)

    live_planets = obs["planets"]
    init_live = obs.get("initial_planets", live_planets)
    for i, p in enumerate(live_planets[:MAX_PLANETS]):
        planets[0, i] = torch.tensor(p, dtype=torch.float32)
        planet_alive[0, i] = True
    for i, p in enumerate(init_live[:MAX_PLANETS]):
        init_planets[0, i] = torch.tensor(p, dtype=torch.float32)
    for i, f in enumerate((obs.get("fleets") or [])[:MAX_FLEETS]):
        fleets[0, i] = torch.tensor(f, dtype=torch.float32)
        fleet_alive[0, i] = True

    env.planets = planets
    env.init_planets = init_planets
    env.planet_alive = planet_alive
    env.fleets = fleets
    env.fleet_alive = fleet_alive
    env.step_count = torch.tensor([int(obs.get("step", 0))], dtype=torch.long)
    env.angular_velocity = torch.tensor([float(obs.get("angular_velocity", 0.0))], dtype=torch.float32)
    env.next_fleet_id = torch.tensor([int(obs.get("next_fleet_id", len(obs.get("fleets") or [])))], dtype=torch.long)
    env.done = torch.zeros(1, dtype=torch.bool)
    env.rewards = torch.zeros((1, 2), dtype=torch.float32)
    env.seeds = [0]
    env._precompute_orbital_params()
    env.prev_material = env._compute_material()
    env.prev_production = env._compute_production()
    owner_p = env.planets[:, :, 1].long()
    env.prev_owned = torch.zeros((1, 2), dtype=torch.float32)
    for pl in range(2):
        env.prev_owned[:, pl] = ((owner_p == pl) & env.planet_alive).float().sum(dim=1)
    return env


def _moves_to_action_tensor(obs: dict, player_id: int, moves: list[list[float]]) -> torch.Tensor:
    env = _init_env_from_obs(obs)
    owned_idx, slot_valid = env.owned_indices_for(player_id)
    action = torch.zeros((1, MAX_OWNED, 4), dtype=torch.long)
    action[:, :, 3] = -1

    pid_to_slot: dict[int, int] = {}
    planets = obs["planets"]
    for slot, valid in enumerate(slot_valid[0].tolist()):
        if not valid:
            continue
        pidx = int(owned_idx[0, slot].item())
        pid_to_slot[int(planets[pidx][0])] = slot

    for move in moves:
        if len(move) < 3:
            continue
        from_pid = int(move[0])
        angle = float(move[1])
        ship_count = int(move[2])
        slot = pid_to_slot.get(from_pid)
        if slot is None:
            continue
        src_idx = next((i for i, p in enumerate(planets) if int(p[0]) == from_pid), None)
        if src_idx is None:
            continue
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
            continue
        action[0, slot, 0] = 1
        action[0, slot, 1] = 0
        action[0, slot, 2] = _find_ship_bin(ship_count)
        action[0, slot, 3] = int(tgt_idx)
    return action


def _score_state(env: VecTorchEnv, player_id: int) -> float:
    mat = env._compute_material()[0]
    opp = 1 - int(player_id)
    return float(mat[player_id] - mat[opp])


def _rollout_score(
    obs_prev: dict,
    replay: dict,
    player_slot: int,
    step_t: int,
    player_move_set: list[list[float]],
    horizon: int,
) -> float:
    env = _init_env_from_obs(obs_prev)
    player_id = int(obs_prev["player"])
    opp_id = 1 - player_id

    steps = replay.get("steps") or []
    for offset in range(horizon):
        replay_step_idx = step_t + offset
        if replay_step_idx >= len(steps):
            break
        obs_for_actions = normalize_obs(steps[replay_step_idx - 1][player_slot]["observation"], fallback_step=replay_step_idx - 1)
        if offset == 0:
            my_moves = player_move_set
        else:
            my_moves = steps[replay_step_idx][player_slot].get("action") or []
        opp_moves = steps[replay_step_idx][opp_id].get("action") or []
        actions = {
            player_id: _moves_to_action_tensor(obs_for_actions, player_id, my_moves),
            opp_id: _moves_to_action_tensor(obs_for_actions, opp_id, opp_moves),
        }
        env.step(actions=actions)

    return _score_state(env, player_id)


def _pair_preference(
    *,
    primary_pos_score: float,
    primary_neg_score: float,
    confirm_pos_score: float,
    confirm_neg_score: float,
    min_gap: float,
) -> tuple[bool, float] | None:
    primary_gap = float(primary_pos_score - primary_neg_score)
    confirm_gap = float(confirm_pos_score - confirm_neg_score)
    if abs(primary_gap) < float(min_gap):
        return None
    if primary_gap == 0.0 or confirm_gap == 0.0:
        return None
    if (primary_gap > 0.0) != (confirm_gap > 0.0):
        return None
    return (primary_gap > 0.0), abs(primary_gap)


def main() -> None:
    args = parse_args()
    replay_paths = resolve_replay_paths(args.replay_dir, args.replay_path, args.episode_id)
    samples: list[dict] = []
    stats = Counter()

    for replay_path in replay_paths:
        try:
            replay = json.loads(Path(replay_path).read_text())
        except Exception:
            stats["replay_load_failed"] += 1
            continue
        steps = replay.get("steps") or []
        if len(steps) < 2:
            stats["replay_too_short"] += 1
            continue
        player_slot = infer_player_slot(replay, args.player_name, args.player_slot)
        stats["replays_seen"] += 1

        for t in range(1, len(steps)):
            if args.step_limit is not None and t > args.step_limit:
                break
            if player_slot >= len(steps[t]) or player_slot >= len(steps[t - 1]):
                continue
            acts = steps[t][player_slot].get("action") or []
            if not acts:
                continue
            obs_prev = normalize_obs(steps[t - 1][player_slot]["observation"], fallback_step=t - 1)
            planets = obs_prev["planets"]
            cand_info = _enumerate_attack_candidates(obs_prev)
            candidates = cand_info["candidates"]
            if not candidates:
                stats["no_candidates"] += 1
                continue

            for move in acts:
                if len(move) < 3:
                    continue
                from_pid = int(move[0])
                angle = float(move[1])
                ship_count = int(move[2])
                sample, pid_to_slot = _base_sample(obs_prev, move)
                if sample is None:
                    stats["sample_build_failed"] += 1
                    continue

                src_idx = next((i for i, p in enumerate(planets) if int(p[0]) == from_pid), None)
                if src_idx is None:
                    stats["src_not_found"] += 1
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
                if tgt_idx < 0 or tgt_idx >= len(planets):
                    stats["replay_target_decode_failed"] += 1
                    continue
                tgt_id = int(planets[tgt_idx][0])
                replay_scored = _score_replay_move(obs_prev, from_pid, tgt_id, ship_count)
                if replay_scored is None or not replay_scored.get("valid", False):
                    stats["replay_move_invalid"] += 1
                    continue
                if not _is_sane_positive(candidates, source_id=from_pid, target_id=tgt_id):
                    stats["insane_positive_skipped"] += 1
                    continue

                pos_slot = pid_to_slot.get(int(from_pid))
                if pos_slot is None:
                    stats["pos_slot_lookup_failed"] += 1
                    continue

                alts = _counterfactual_alternatives(
                    candidates,
                    source_id=from_pid,
                    target_id=tgt_id,
                    ships=ship_count,
                    limit=max(1, args.alts_per_action),
                )
                if not alts:
                    stats["no_counterfactual_alternatives"] += 1
                    continue

                replay_actions = list(acts)
                actual_primary = _rollout_score(
                    obs_prev,
                    replay,
                    player_slot=player_slot,
                    step_t=t,
                    player_move_set=replay_actions,
                    horizon=args.rollout_horizon,
                )
                actual_confirm = _rollout_score(
                    obs_prev,
                    replay,
                    player_slot=player_slot,
                    step_t=t,
                    player_move_set=replay_actions,
                    horizon=args.confirm_horizon,
                )

                for alt in alts:
                    alt_move = _candidate_to_move(obs_prev, alt)
                    alt_actions = []
                    replaced = False
                    for act in replay_actions:
                        if int(act[0]) == from_pid and not replaced:
                            alt_actions.append(alt_move)
                            replaced = True
                        else:
                            alt_actions.append(act)
                    if not replaced:
                        stats["replace_source_failed"] += 1
                        continue

                    alt_primary = _rollout_score(
                        obs_prev,
                        replay,
                        player_slot=player_slot,
                        step_t=t,
                        player_move_set=alt_actions,
                        horizon=args.rollout_horizon,
                    )
                    alt_confirm = _rollout_score(
                        obs_prev,
                        replay,
                        player_slot=player_slot,
                        step_t=t,
                        player_move_set=alt_actions,
                        horizon=args.confirm_horizon,
                    )
                    pref = _pair_preference(
                        primary_pos_score=actual_primary,
                        primary_neg_score=alt_primary,
                        confirm_pos_score=actual_confirm,
                        confirm_neg_score=alt_confirm,
                        min_gap=float(args.min_score_gap),
                    )
                    if pref is None:
                        if abs(float(actual_primary - alt_primary)) < float(args.min_score_gap):
                            stats["small_gap_skipped"] += 1
                        else:
                            stats["horizon_disagreement_skipped"] += 1
                        continue
                    replay_is_better, gap = pref

                    if replay_is_better:
                        pos_target_idx = int(tgt_idx)
                        pos_ship_bin = _find_ship_bin(int(ship_count))
                        pos_extra = _action_extra_tensor(
                            ships=int(ship_count),
                            eta=int(replay_scored["eta"] or 32),
                            valid=bool(replay_scored["valid"]),
                            source_ships=int(replay_scored["source_ships"]),
                            target_prod=int(replay_scored["target_prod"]),
                            floor_at_arrival=int(replay_scored["floor_at_arrival"]),
                            score=float(replay_scored["score"]),
                            target_is_mine=bool(replay_scored["target_is_mine"]),
                            target_is_neutral=bool(replay_scored["target_is_neutral"]),
                        )
                        neg_target_idx = int(alt.target_idx)
                        neg_ship_bin = _find_ship_bin(int(alt.ships))
                        neg_extra = _action_extra_tensor(
                            ships=int(alt.ships),
                            eta=int(alt.eta),
                            valid=bool(alt.valid),
                            source_ships=int(alt.source_ships),
                            target_prod=int(alt.target_prod),
                            floor_at_arrival=int(alt.floor_at_arrival),
                            score=float(alt.score),
                            target_is_mine=bool(alt.target_is_mine),
                            target_is_neutral=bool(alt.target_is_neutral),
                        )
                    else:
                        pos_target_idx = int(alt.target_idx)
                        pos_ship_bin = _find_ship_bin(int(alt.ships))
                        pos_extra = _action_extra_tensor(
                            ships=int(alt.ships),
                            eta=int(alt.eta),
                            valid=bool(alt.valid),
                            source_ships=int(alt.source_ships),
                            target_prod=int(alt.target_prod),
                            floor_at_arrival=int(alt.floor_at_arrival),
                            score=float(alt.score),
                            target_is_mine=bool(alt.target_is_mine),
                            target_is_neutral=bool(alt.target_is_neutral),
                        )
                        neg_target_idx = int(tgt_idx)
                        neg_ship_bin = _find_ship_bin(int(ship_count))
                        neg_extra = _action_extra_tensor(
                            ships=int(ship_count),
                            eta=int(replay_scored["eta"] or 32),
                            valid=bool(replay_scored["valid"]),
                            source_ships=int(replay_scored["source_ships"]),
                            target_prod=int(replay_scored["target_prod"]),
                            floor_at_arrival=int(replay_scored["floor_at_arrival"]),
                            score=float(replay_scored["score"]),
                            target_is_mine=bool(replay_scored["target_is_mine"]),
                            target_is_neutral=bool(replay_scored["target_is_neutral"]),
                        )

                    pref = {
                        "planet_features": sample["planet_features"].clone(),
                        "fleet_features": sample["fleet_features"].clone(),
                        "global_features": sample["global_features"].clone(),
                        "planet_mask": sample["planet_mask"].clone(),
                        "fleet_mask": sample["fleet_mask"].clone(),
                        "fire_mask": sample["fire_mask"].clone(),
                        "angle_mask": sample["angle_mask"].clone(),
                        "slot_valid": sample["slot_valid"].clone(),
                        "owned_indices": sample["owned_indices"].clone(),
                        "pairwise_features": sample["pairwise_features"].clone(),
                        "pos_slot": torch.tensor(int(pos_slot), dtype=torch.long),
                        "pos_ship_bin": torch.tensor(int(pos_ship_bin), dtype=torch.long),
                        "pos_target_idx": torch.tensor(int(pos_target_idx), dtype=torch.long),
                        "pos_action_extra": pos_extra.clone(),
                        "neg_slot": torch.tensor(int(pos_slot), dtype=torch.long),
                        "neg_ship_bin": torch.tensor(int(neg_ship_bin), dtype=torch.long),
                        "neg_target_idx": torch.tensor(int(neg_target_idx), dtype=torch.long),
                        "neg_action_extra": neg_extra.clone(),
                        "weight": torch.tensor(max(1.0, gap), dtype=torch.float32),
                    }
                    samples.append(pref)
                    stats["preference_pairs"] += 1
                    stats["samples_added"] += 1

    out_path = Path(args.samples_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(samples, f)

    payload = {
        "replay_count": len(replay_paths),
        "sample_count": len(samples),
        "player_name": args.player_name,
        "player_slot": args.player_slot,
        "step_limit": args.step_limit,
        "rollout_horizon": args.rollout_horizon,
        "confirm_horizon": args.confirm_horizon,
        "alts_per_action": args.alts_per_action,
        "min_score_gap": args.min_score_gap,
        "stats": dict(stats),
    }
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"samples saved -> {args.samples_out}")


if __name__ == "__main__":
    main()
