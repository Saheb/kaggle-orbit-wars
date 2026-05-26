"""Quick validation: 8 games vs random / sniper / zach / hellburner / suneet.

Usage:
    python quick_val.py --checkpoint checkpoints/<name>.pt
    python quick_val.py --checkpoint checkpoints/<name>.pt --games 16

Prints a compact table. Takes ~60s at 8 games per opponent.
Run this before any full panel eval — see eval_runbook.md §1.
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import torch
from config import Config
from model import EntityTransformer
from eval import build_agent_fn, evaluate_against_baseline

REPO_ROOT = Path(__file__).resolve().parents[1]

OPPONENTS = [
    ("random",     "random"),
    ("sniper",     "candidate_sniper.py"),
    ("zach",       "candidate_zach_public.py"),
    ("hellburner", "candidate_hellburner.py"),
    ("suneet",     "candidate_suneet_lb1200.py"),
]

KONBU_OPPONENT = (
    "konbu_v4_ml",
    "external/kaggle_outputs/konbu17_train_submit_v4_ml_validator_topk2_tutorial/main.py",
)


def resolve_opponent(path: str) -> str:
    if path == "random":
        return path
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return str(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=0,
                        help="First seed for the eval range. Default 0 preserves the standard quick panel.")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--include-konbu", action="store_true",
                        help="Also eval against the downloaded konbu17 v4+ML agent. Slow; disabled for watcher comparability.")
    parser.add_argument("--action-decode", choices=["auto", "angle", "target"], default="auto",
                        help="Decode mode for eval. auto uses checkpoint config.")
    args = parser.parse_args()

    # Load checkpoint — mirrors evaluate_checkpoint logic
    try:
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_cfg = state_dict.get("config", {}) if isinstance(state_dict, dict) else {}
    if "model" in state_dict:
        state_dict = state_dict["model"]

    cfg = Config()
    if "num_ship_bins" in ckpt_cfg:
        cfg.model.num_ship_bins = int(ckpt_cfg["num_ship_bins"])
    elif "ship_head.weight" in state_dict:
        cfg.model.num_ship_bins = int(state_dict["ship_head.weight"].shape[0])
    if "min_ship_bin" in ckpt_cfg:
        cfg.model.min_ship_bin = int(ckpt_cfg["min_ship_bin"])
    if "ship_bin_mode" in ckpt_cfg:
        cfg.model.ship_bin_mode = str(ckpt_cfg["ship_bin_mode"])
    action_decode = str(ckpt_cfg.get("action_decode", "angle"))
    if args.action_decode != "auto":
        action_decode = args.action_decode

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  ship_bin_mode={cfg.model.ship_bin_mode}  "
          f"num_ship_bins={cfg.model.num_ship_bins}  "
          f"action_decode={action_decode}")
    print()

    device = torch.device(cfg.device)
    model = EntityTransformer(cfg.model).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # --- move-count probe on the first opponent (random) ---
    # Catches a broken eval stack before wasting time on the full table.
    print("Probing move count vs random (1 game)...")
    from kaggle_environments import make
    agent_fn = build_agent_fn(model, device,
                              ship_bin_mode=cfg.model.ship_bin_mode,
                              target_decode=(action_decode == "target"),
                              sample=args.sample)
    move_log = []
    def logging_agent(obs):
        moves = agent_fn(obs)
        move_log.append(moves)
        return moves
    env = make("orbit_wars", configuration={"seed": args.seed_start}, debug=False)
    env.run([logging_agent, "random"])
    fires = sum(1 for m in move_log if m)
    print(f"  Steps with moves: {fires}/{len(move_log)}", end="")
    if fires == 0:
        print("  ← BROKEN — agent silent, stop and debug (eval_runbook §3)")
        return
    print("  ✓")
    print()

    # --- run N games vs each opponent ---
    print(f"{'opponent':<12}  {'W/G':>5}  {'WR%':>6}  {'avg_mat':>8}  time")
    print("-" * 46)
    opponents = OPPONENTS + ([KONBU_OPPONENT] if args.include_konbu else [])
    for label, path in opponents:
        t0 = time.perf_counter()
        r = evaluate_against_baseline(
            model, device,
            num_games=args.games,
            seed_start=args.seed_start,
            opponent=resolve_opponent(path),
            ship_bin_mode=cfg.model.ship_bin_mode,
            target_decode=(action_decode == "target"),
            sample=args.sample,
        )
        elapsed = time.perf_counter() - t0
        print(f"{label:<12}  {r['wins']:>2}/{r['total_games']:<2}  "
              f"{100*r['win_rate']:>5.1f}%  "
              f"{r['avg_material']:>8.0f}  "
              f"{elapsed:>4.0f}s")

    print()
    print("Next: if moves produced > 0 and results look sane, run --panel.")
    print("  python eval.py --checkpoint <ckpt> --opponent ../candidate_zach_public.py --panel")


if __name__ == "__main__":
    main()
