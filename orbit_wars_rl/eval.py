"""Evaluation: pit trained PyTorch policy against baselines."""

from __future__ import annotations

import argparse
import os
from statistics import mean

import torch
import numpy as np

from config import Config
from model import EntityTransformer, NUM_ANGLE_BINS, NUM_SHIP_BINS, ANGLE_BIN_WIDTH
from features import extract_features
from action_mask import compute_action_masks, actions_from_policy, actions_from_target_policy


def load_checkpoint(path: str, cfg: Config) -> tuple[dict, str]:
    """Load a checkpoint and patch cfg.model dims to match the saved weights.

    Returns (state_dict, action_decode).  Modifies cfg.model in-place so that
    EntityTransformer(cfg.model) builds the correct architecture for this
    checkpoint — regardless of what config.py currently says.  This lets old
    (pre-Phase-1) checkpoints be evaluated after the config has been bumped.
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # --- head / bin dims from saved config or weight shapes ---
    if "num_ship_bins" in ckpt_cfg:
        cfg.model.num_ship_bins = int(ckpt_cfg["num_ship_bins"])
    elif "ship_head.weight" in sd:
        cfg.model.num_ship_bins = int(sd["ship_head.weight"].shape[0])

    if "angle_head.weight" in sd:
        n = int(sd["angle_head.weight"].shape[0])
        if n != cfg.model.num_angle_bins:
            cfg.model.num_angle_bins = n

    if "min_ship_bin" in ckpt_cfg:
        cfg.model.min_ship_bin = int(ckpt_cfg["min_ship_bin"])
    if "ship_bin_mode" in ckpt_cfg:
        cfg.model.ship_bin_mode = str(ckpt_cfg["ship_bin_mode"])

    # --- feature projection dims: always infer from weight shapes ---
    if "planet_proj.weight" in sd:
        cfg.model.planet_feature_dim = int(sd["planet_proj.weight"].shape[1])
    if "fleet_proj.weight" in sd:
        cfg.model.fleet_feature_dim = int(sd["fleet_proj.weight"].shape[1])
    if "global_proj.weight" in sd:
        cfg.model.global_feature_dim = int(sd["global_proj.weight"].shape[1])
    if "pair_kv.weight" in sd:
        D = int(sd["planet_proj.weight"].shape[0])
        cfg.model.pairwise_feature_dim = int(sd["pair_kv.weight"].shape[1]) - D
    else:
        cfg.model.pairwise_feature_dim = 0

    # Detect value head version from fc1 input width (old=D, new=2D).
    if "value_fc1.weight" in sd:
        cfg.model.value_head_in = int(sd["value_fc1.weight"].shape[1])

    action_decode = str(ckpt_cfg.get("action_decode", "angle"))
    return sd, action_decode


def build_agent_fn(model: EntityTransformer, device: torch.device,
                   fire_threshold: float = 0.5, sample: bool = False,
                   ship_bin_mode: str = "absolute",
                   target_decode: bool = False,
                   target_sanity_penalty: float = 0.0):
    """Return a kaggle_environments-compatible agent function wrapping the model.

    sample=True uses Bernoulli/Categorical sampling instead of threshold/argmax —
    helps when the training-time distribution is multi-modal but the mode is
    degenerate (e.g. 1-ship-fleet trap).
    """
    model.eval()

    def agent_fn(obs):
        # obs may be a dict or an Observation namedtuple depending on caller
        if not isinstance(obs, dict):
            obs = {
                "step": int(getattr(obs, "step", 0)),
                "player": int(getattr(obs, "player", 0)),
                "planets": [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
                            for p in obs.planets],
                "fleets": [[f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships]
                           for f in obs.fleets],
                "angular_velocity": float(getattr(obs, "angular_velocity", 0.0)),
                "initial_planets": [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
                                    for p in getattr(obs, "initial_planets", obs.planets)],
                "comet_planet_ids": list(getattr(obs, "comet_planet_ids", [])),
            }

        player = obs["player"]
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

        action_fn = actions_from_target_policy if target_decode else actions_from_policy
        if target_decode:
            return action_fn(
                outputs["fire_logits"].cpu(),
                outputs["target_logits"].cpu(),
                outputs["ship_logits"].cpu(),
                {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()},
                obs, player,
                fire_threshold=fire_threshold,
                sample=sample,
                ship_bin_mode=ship_bin_mode,
                target_sanity_penalty=target_sanity_penalty,
            )

        raise NotImplementedError(
            "angle-decode path removed (angle head deleted); Phase 1 checkpoints "
            "use target-decode. Pass target_decode=True (--target-decode)."
        )

    return agent_fn


def evaluate_against_baseline(
    model: EntityTransformer,
    device: torch.device,
    num_games: int = 32,
    seed_start: int = 0,
    opponent: str = "random",
    num_players: int = 2,
    fire_threshold: float = 0.5,
    sample: bool = False,
    ship_bin_mode: str = "absolute",
    target_decode: bool = False,
    target_sanity_penalty: float = 0.0,
) -> dict:
    """Evaluate trained policy against a baseline using kaggle_environments.

    Args:
        opponent: "random" or path to a Python agent file (e.g. "main.py")
        num_players: 2 or 4
    """
    from kaggle_environments import make

    agent_fn = build_agent_fn(model, device, fire_threshold=fire_threshold, sample=sample,
                              ship_bin_mode=ship_bin_mode, target_decode=target_decode,
                              target_sanity_penalty=target_sanity_penalty)
    opponents = [opponent] * (num_players - 1)
    agents = [agent_fn] + opponents

    wins = 0
    total_material = 0
    results = []

    for seed in range(seed_start, seed_start + num_games):
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run(agents)
        final = env.steps[-1]
        rewards = [s.reward for s in final]

        obs = final[0].observation
        material = sum(p[5] for p in obs.planets if p[1] == 0)
        material += sum(f[6] for f in obs.fleets if f[1] == 0)

        # Rank by reward; player 0 wins if their reward is strictly highest
        my_reward = rewards[0] if rewards[0] is not None else 0.0
        best_opp = max((r for r in rewards[1:] if r is not None), default=0.0)
        is_win = my_reward > best_opp

        wins += int(is_win)
        total_material += material
        results.append({
            "seed": seed,
            "win": is_win,
            "material": material,
            "rewards": rewards,
        })

    return {
        "wins": wins,
        "total_games": num_games,
        "win_rate": wins / num_games,
        "avg_material": total_material / num_games,
        "results": results,
    }


def evaluate_panel(
    model: EntityTransformer,
    device: torch.device,
    opponent: str,
    fire_threshold: float = 0.5,
    sample: bool = False,
    ship_bin_mode: str = "absolute",
    target_decode: bool = False,
    target_sanity_penalty: float = 0.0,
) -> dict:
    """Stratified eval over the 128-seed community panel, playing both seats.

    256 games per opponent (128 seeds × 2 seats). Aggregates wins per
    archetype (8 games per cell = 4 seeds × 2 seats) and per seat, so a
    +5pp overall regression hidden by an asymmetric or board-shape-specific
    weakness is visible.
    """
    from kaggle_environments import make
    from eval_panel import BY_ARCHETYPE

    agent_fn = build_agent_fn(model, device, fire_threshold=fire_threshold, sample=sample,
                              ship_bin_mode=ship_bin_mode, target_decode=target_decode,
                              target_sanity_penalty=target_sanity_penalty)

    per_arch: dict[str, dict] = {arch: {"wins": 0, "total": 0,
                                        "wins_seat0": 0, "wins_seat1": 0,
                                        "total_seat0": 0, "total_seat1": 0,
                                        "material_sum": 0}
                                 for arch in BY_ARCHETYPE}
    overall = {"wins": 0, "total": 0, "wins_seat0": 0, "wins_seat1": 0,
               "total_seat0": 0, "total_seat1": 0}
    game_idx = 0
    total_games = sum(len(seeds) for seeds in BY_ARCHETYPE.values()) * 2

    for archetype, seeds in BY_ARCHETYPE.items():
        for seed in seeds:
            for my_seat in (0, 1):
                agents = [agent_fn, opponent] if my_seat == 0 else [opponent, agent_fn]
                env = make("orbit_wars", configuration={"seed": seed}, debug=False)
                env.run(agents)
                final = env.steps[-1]
                rewards = [s.reward if s.reward is not None else 0.0 for s in final]
                my_reward = rewards[my_seat]
                opp_reward = rewards[1 - my_seat]
                is_win = my_reward > opp_reward
                # Material on the model's side
                obs = final[0].observation
                material = sum(p[5] for p in obs.planets if p[1] == my_seat)
                material += sum(f[6] for f in obs.fleets if f[1] == my_seat)

                c = per_arch[archetype]
                c["wins"] += int(is_win); c["total"] += 1
                c[f"wins_seat{my_seat}"] += int(is_win)
                c[f"total_seat{my_seat}"] += 1
                c["material_sum"] += material
                overall["wins"] += int(is_win); overall["total"] += 1
                overall[f"wins_seat{my_seat}"] += int(is_win)
                overall[f"total_seat{my_seat}"] += 1
                game_idx += 1
                if game_idx % 16 == 0 or game_idx == total_games:
                    print(f"  panel progress: {game_idx}/{total_games}  "
                          f"overall {overall['wins']}/{overall['total']} "
                          f"({100*overall['wins']/max(overall['total'],1):.1f}%)",
                          flush=True)

    return {"overall": overall, "per_archetype": per_arch}


def print_panel_report(result: dict, opponent: str) -> None:
    """Pretty-print panel results."""
    o = result["overall"]
    print()
    print("=" * 78)
    print(f"Panel eval vs {opponent}")
    print("=" * 78)
    print(f"Overall:   {o['wins']}/{o['total']}  ({100*o['wins']/max(o['total'],1):.1f}%)")
    s0 = 100 * o['wins_seat0'] / max(o['total_seat0'], 1)
    s1 = 100 * o['wins_seat1'] / max(o['total_seat1'], 1)
    print(f"  seat 0:  {o['wins_seat0']}/{o['total_seat0']}  ({s0:.1f}%)")
    print(f"  seat 1:  {o['wins_seat1']}/{o['total_seat1']}  ({s1:.1f}%)")
    asym = s0 - s1
    print(f"  asymmetry (seat0 − seat1): {asym:+.1f}pp")
    print()
    print("Per archetype  (8 games each = 4 seeds × 2 seats):")
    print(f"  {'archetype':<48s}  {'WR':>6s}  {'s0/s1':>10s}  {'mat':>8s}")
    rows = []
    for arch, c in result["per_archetype"].items():
        wr = 100 * c["wins"] / max(c["total"], 1)
        s0 = 100 * c["wins_seat0"] / max(c["total_seat0"], 1)
        s1 = 100 * c["wins_seat1"] / max(c["total_seat1"], 1)
        mat = c["material_sum"] / max(c["total"], 1)
        rows.append((wr, arch, c, s0, s1, mat))
    # sort by winrate descending so worst cells stand out at the bottom
    rows.sort(key=lambda r: -r[0])
    for wr, arch, c, s0, s1, mat in rows:
        print(f"  {arch:<48s}  {wr:>5.1f}%  {s0:>4.0f}/{s1:>3.0f}  {mat:>8.0f}")
    # quick diagnostic
    worst = min(rows, key=lambda r: r[0])
    best = max(rows, key=lambda r: r[0])
    print()
    print(f"Best:  {best[1]}  ({best[0]:.1f}%)")
    print(f"Worst: {worst[1]}  ({worst[0]:.1f}%)")
    print(f"Spread: {best[0] - worst[0]:.1f}pp")


def evaluate_checkpoint(params_path: str, cfg: Config, num_games: int = 32,
                        seed_start: int = 0,
                        opponent: str = "random", fire_threshold: float = 0.5,
                        panel: bool = False, sample: bool = False,
                        target_decode: bool = False,
                        target_sanity_penalty: float = 0.0):
    """Load a checkpoint and evaluate it."""
    device = torch.device(cfg.device)

    state_dict, ckpt_action_decode = load_checkpoint(params_path, cfg)
    if cfg.model.ship_bin_mode != "absolute":
        print(f"Checkpoint ship_bin_mode={cfg.model.ship_bin_mode}")
    # Auto-detect action_decode from checkpoint config; CLI --target-decode overrides.
    if not target_decode and ckpt_action_decode == "target":
        target_decode = True
        print("Checkpoint action_decode=target  →  enabling target_decode automatically")

    model = EntityTransformer(cfg.model).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"target_head.weight", "target_head.bias"}
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing or unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch: missing={bad_missing}, unexpected={unexpected}")
    model.eval()

    if panel:
        results = evaluate_panel(model, device, opponent=opponent,
                                 fire_threshold=fire_threshold, sample=sample,
                                 ship_bin_mode=cfg.model.ship_bin_mode,
                                 target_decode=target_decode,
                                 target_sanity_penalty=target_sanity_penalty)
        print_panel_report(results, opponent)
        return results

    results = evaluate_against_baseline(
        model, device,
        ship_bin_mode=cfg.model.ship_bin_mode,
        target_decode=target_decode,
        num_games=num_games,
        seed_start=seed_start,
        opponent=opponent,
        num_players=cfg.env.num_players,
        fire_threshold=fire_threshold,
        sample=sample,
        target_sanity_penalty=target_sanity_penalty,
    )

    print(f"Win rate vs {opponent}: {results['win_rate']:.2%}  "
          f"({results['wins']}/{results['total_games']})")
    print(f"Fire threshold: {fire_threshold}")
    print(f"Target decode: {target_decode}")
    print(f"Target sanity penalty: {target_sanity_penalty}")
    print(f"Avg material: {results['avg_material']:.1f}")
    for r in results["results"][:5]:
        print(f"  seed={r['seed']} win={r['win']} "
              f"material={r['material']} rewards={r['rewards']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=0,
                        help="First seed for non-panel eval. Ignored by --panel, which uses the fixed archetype panel.")
    parser.add_argument("--opponent", default="random",
                        help="'random' or path to agent .py file")
    parser.add_argument("--num-players", type=int, choices=[2, 4], default=2)
    parser.add_argument("--fire-threshold", type=float, default=0.5)
    parser.add_argument("--panel", action="store_true",
                        help="Use 128-seed community panel with both-seat eval "
                             "(256 games, per-archetype breakdown).")
    parser.add_argument("--sample", action="store_true",
                        help="Sample from policy distribution instead of argmax. "
                             "Use when the mode is degenerate but distribution mass "
                             "is on competent bins (1-ship-fleet trap).")
    parser.add_argument("--target-decode", action="store_true",
                        help="Aim with target_logits plus orbital intercept instead "
                             "of directly using the angle head.")
    parser.add_argument("--target-sanity-penalty", type=float, default=0.0,
                        help="Subtract this from dominated same-source target logits "
                             "before target decode.")
    args = parser.parse_args()

    cfg = Config()
    cfg.env.num_players = args.num_players
    evaluate_checkpoint(
        args.checkpoint,
        cfg,
        num_games=args.games,
        seed_start=args.seed_start,
        opponent=args.opponent,
        fire_threshold=args.fire_threshold,
        panel=args.panel,
        sample=args.sample,
        target_decode=args.target_decode,
        target_sanity_penalty=args.target_sanity_penalty,
    )
