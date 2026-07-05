"""Compare joint action ranker vs producer-best and replay actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_rl.action_mask import compute_action_masks
from orbit_wars_rl.producer_action_ranking import (
    _enumerate_attack_candidates,
    _score_replay_move,
    action_extra_features,
)
from orbit_wars_rl.producer_ranking import infer_player_slot
from orbit_wars_rl.audit_submission_targets import normalize_obs, resolve_replay_paths
from orbit_wars_rl.config import Config
from orbit_wars_rl.features import extract_features
from orbit_wars_rl.scripts.joint_action_ranker import JointActionRanker
from orbit_wars_rl.model import EntityTransformer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-checkpoint", required=True)
    ap.add_argument("--replay-dir")
    ap.add_argument("--replay-path", action="append", default=[])
    ap.add_argument("--episode-id", action="append", default=[])
    ap.add_argument("--player-name", default="")
    ap.add_argument("--player-slot", type=int)
    ap.add_argument("--step-limit", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--output-json", default="/tmp/joint_action_ranker_audit.json")
    ap.add_argument("--output-md", default="/tmp/joint_action_ranker_audit.md")
    return ap.parse_args()


def _load_joint(path: str, device: torch.device) -> JointActionRanker:
    ckpt = torch.load(path, map_location="cpu")
    init_checkpoint = ckpt["backbone_checkpoint"]
    cfg = Config()
    from orbit_wars_rl.eval import load_checkpoint
    sd, _ = load_checkpoint(init_checkpoint, cfg)
    backbone = EntityTransformer(cfg.model)
    backbone.load_state_dict(sd)
    model = JointActionRanker(backbone)
    model.load_state_dict(ckpt["joint_ranker"])
    return model.to(device).eval()


def _batch_from_obs(obs: dict, device: torch.device) -> dict:
    player = int(obs["player"])
    feats = extract_features(obs, player, num_players=2)
    masks = compute_action_masks(obs, player)
    fire_mask = masks["fire_mask"] if masks["fire_mask"].dim() == 2 else masks["fire_mask"].unsqueeze(0)
    angle_mask = masks["angle_mask"] if masks["angle_mask"].dim() == 3 else masks["angle_mask"].unsqueeze(0)
    slot_valid = masks["slot_valid"] if masks["slot_valid"].dim() == 2 else masks["slot_valid"].unsqueeze(0)
    owned_indices = masks["owned_indices"] if masks["owned_indices"].dim() == 2 else masks["owned_indices"].unsqueeze(0)
    return {
        "planet_features": feats["planet_features"].unsqueeze(0).to(device),
        "fleet_features": feats["fleet_features"].unsqueeze(0).to(device),
        "global_features": feats["global_features"].unsqueeze(0).to(device),
        "planet_mask": feats["planet_mask"].unsqueeze(0).to(device),
        "fleet_mask": feats["fleet_mask"].unsqueeze(0).to(device),
        "fire_mask": fire_mask.to(device),
        "angle_mask": angle_mask.to(device),
        "slot_valid": slot_valid.to(device),
        "owned_indices": owned_indices.to(device),
        "pairwise_features": feats["pairwise_features"].unsqueeze(0).to(device),
    }


def _find_ship_bin(ships: int) -> int:
    from orbit_wars_rl.bc import _find_ship_bin as _f
    return _f(ships)


def _action_extra(
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


def analyze(args: argparse.Namespace) -> dict:
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_joint(args.joint_checkpoint, device)
    replay_paths = resolve_replay_paths(args.replay_dir, args.replay_path, args.episode_id)

    aggregate = {
        "launches": 0,
        "joint_vs_replay_source_match": 0,
        "joint_vs_replay_target_match": 0,
        "joint_vs_replay_exact_match": 0,
        "joint_vs_producer_source_match": 0,
        "joint_vs_producer_target_match": 0,
        "joint_vs_producer_exact_match": 0,
    }
    episodes = []

    from orbit_wars_rl.bc import _find_target_planet_index

    for replay_path in replay_paths:
        try:
            replay = json.loads(Path(replay_path).read_text())
        except Exception:
            continue
        player_slot = infer_player_slot(replay, args.player_name, args.player_slot)
        steps = replay.get("steps") or []
        ep_actions = []
        ep_agg = {
            "launches": 0,
            "joint_vs_replay_source_match": 0,
            "joint_vs_replay_target_match": 0,
            "joint_vs_replay_exact_match": 0,
            "joint_vs_producer_source_match": 0,
            "joint_vs_producer_target_match": 0,
            "joint_vs_producer_exact_match": 0,
        }

        for t in range(1, len(steps)):
            if args.step_limit is not None and t > args.step_limit:
                break
            if player_slot >= len(steps[t]) or player_slot >= len(steps[t - 1]):
                continue
            acts = steps[t][player_slot].get("action") or []
            if not acts:
                continue
            obs_prev = normalize_obs(steps[t - 1][player_slot]["observation"], fallback_step=t - 1)
            candidates = _enumerate_attack_candidates(obs_prev)["candidates"]
            if not candidates:
                continue
            batch = _batch_from_obs(obs_prev, device)

            # Score all producer-generated candidates with the joint ranker.
            slot_map = {}
            planets = obs_prev["planets"]
            owned_indices = batch["owned_indices"][0].cpu().tolist()
            slot_valid = batch["slot_valid"][0].cpu().tolist()
            for slot, valid in enumerate(slot_valid):
                if not valid:
                    continue
                pidx = int(owned_indices[slot])
                if 0 <= pidx < len(planets):
                    slot_map[int(planets[pidx][0])] = slot
            joint_candidates = []
            for cand in candidates:
                if not cand.valid or int(cand.ships) <= 0:
                    continue
                slot = slot_map.get(int(cand.source_id))
                if slot is None:
                    continue
                score = model.score_actions(
                    batch,
                    torch.tensor([slot], device=device),
                    torch.tensor([_find_ship_bin(int(cand.ships))], device=device),
                    torch.tensor([int(cand.target_idx)], device=device),
                    _action_extra(
                        ships=int(cand.ships),
                        eta=int(cand.eta),
                        valid=bool(cand.valid),
                        source_ships=int(cand.source_ships),
                        target_prod=int(cand.target_prod),
                        floor_at_arrival=int(cand.floor_at_arrival),
                        score=float(cand.score),
                        target_is_mine=bool(cand.target_is_mine),
                        target_is_neutral=bool(cand.target_is_neutral),
                    ).unsqueeze(0).to(device),
                )
                joint_candidates.append((float(score.item()), cand, slot))
            joint_candidates.sort(key=lambda x: x[0], reverse=True)
            if not joint_candidates:
                continue
            joint_best_score, joint_best, _ = joint_candidates[0]
            producer_best = candidates[0]

            for move in acts:
                if len(move) < 3:
                    continue
                ep_agg["launches"] += 1
                aggregate["launches"] += 1
                from_pid = int(move[0]); angle = float(move[1]); ship_count = int(move[2])
                src_idx = next((i for i, p in enumerate(planets) if int(p[0]) == from_pid), None)
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

                src_match = int(from_pid == joint_best.source_id)
                tgt_match = int(tgt_id == joint_best.target_id)
                exact = int(src_match and tgt_match)
                prod_src_match = int(joint_best.source_id == producer_best.source_id)
                prod_tgt_match = int(joint_best.target_id == producer_best.target_id)
                prod_exact = int(prod_src_match and prod_tgt_match)
                ep_agg["joint_vs_replay_source_match"] += src_match
                ep_agg["joint_vs_replay_target_match"] += tgt_match
                ep_agg["joint_vs_replay_exact_match"] += exact
                ep_agg["joint_vs_producer_source_match"] += prod_src_match
                ep_agg["joint_vs_producer_target_match"] += prod_tgt_match
                ep_agg["joint_vs_producer_exact_match"] += prod_exact
                aggregate["joint_vs_replay_source_match"] += src_match
                aggregate["joint_vs_replay_target_match"] += tgt_match
                aggregate["joint_vs_replay_exact_match"] += exact
                aggregate["joint_vs_producer_source_match"] += prod_src_match
                aggregate["joint_vs_producer_target_match"] += prod_tgt_match
                aggregate["joint_vs_producer_exact_match"] += prod_exact

                ep_actions.append({
                    "step": t,
                    "replay_source_id": from_pid,
                    "replay_target_id": tgt_id,
                    "replay_ships": ship_count,
                    "producer_best": producer_best.to_dict(),
                    "joint_best": {**joint_best.to_dict(), "joint_score": round(joint_best_score, 6)},
                    "joint_vs_replay_source_match": bool(src_match),
                    "joint_vs_replay_target_match": bool(tgt_match),
                    "joint_vs_replay_exact_match": bool(exact),
                    "joint_vs_producer_source_match": bool(prod_src_match),
                    "joint_vs_producer_target_match": bool(prod_tgt_match),
                    "joint_vs_producer_exact_match": bool(prod_exact),
                    "replay_move_score": _score_replay_move(obs_prev, from_pid, tgt_id, ship_count) if tgt_id is not None else None,
                })

        launches = max(1, ep_agg["launches"])
        episodes.append({
            "episode_id": replay.get("info", {}).get("EpisodeId", replay_path.stem),
            "replay_path": str(replay_path),
            "player_slot": player_slot,
            "summary": {
                **ep_agg,
                "joint_vs_replay_source_match_rate": ep_agg["joint_vs_replay_source_match"] / launches,
                "joint_vs_replay_target_match_rate": ep_agg["joint_vs_replay_target_match"] / launches,
                "joint_vs_replay_exact_match_rate": ep_agg["joint_vs_replay_exact_match"] / launches,
                "joint_vs_producer_source_match_rate": ep_agg["joint_vs_producer_source_match"] / launches,
                "joint_vs_producer_target_match_rate": ep_agg["joint_vs_producer_target_match"] / launches,
                "joint_vs_producer_exact_match_rate": ep_agg["joint_vs_producer_exact_match"] / launches,
            },
            "actions": ep_actions,
        })

    launches = max(1, aggregate["launches"])
    return {
        "aggregate": {
            **aggregate,
            "joint_vs_replay_source_match_rate": aggregate["joint_vs_replay_source_match"] / launches,
            "joint_vs_replay_target_match_rate": aggregate["joint_vs_replay_target_match"] / launches,
            "joint_vs_replay_exact_match_rate": aggregate["joint_vs_replay_exact_match"] / launches,
            "joint_vs_producer_source_match_rate": aggregate["joint_vs_producer_source_match"] / launches,
            "joint_vs_producer_target_match_rate": aggregate["joint_vs_producer_target_match"] / launches,
            "joint_vs_producer_exact_match_rate": aggregate["joint_vs_producer_exact_match"] / launches,
        },
        "episodes": episodes,
    }


def render_markdown(payload: dict) -> str:
    agg = payload["aggregate"]
    lines = [
        "# Joint Action Ranker Audit",
        "",
        f"- launches={agg['launches']} "
        f"joint_vs_replay_exact={agg['joint_vs_replay_exact_match_rate']:.3f} "
        f"joint_vs_producer_exact={agg['joint_vs_producer_exact_match_rate']:.3f}",
        "",
    ]
    for ep in payload["episodes"]:
        s = ep["summary"]
        lines.append(
            f"- ep={ep['episode_id']}: launches={s['launches']} "
            f"replay_exact={s['joint_vs_replay_exact_match_rate']:.3f} "
            f"producer_exact={s['joint_vs_producer_exact_match_rate']:.3f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    payload = analyze(args)
    out_json = Path(args.output_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    out_md = Path(args.output_md); out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(payload))
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"saved json -> {out_json}")
    print(f"saved md -> {out_md}")


if __name__ == "__main__":
    main()
