"""Evaluate one CHECKPOINT in a 4-player FFA mixed field, for training progress.

Loads the checkpoint as a 4p agent (num_players=4 → mode_4p=1, matching training)
and plays it seat-rotated against a fixed opponent field (default: producer_v2,
roman_v4, hellburner — the same field as the baseline panel, so WR is comparable
across checkpoints). Reports win-rate (1st-place share = the FFA LB metric), mean
placement (from final material), and mean game length (a long-games signal). Appends
one row to a CSV so a watcher can track WR over training.

Usage:
  python3 orbit_wars_rl/eval_ffa_checkpoint.py <checkpoint.pt> \
    --seeds 16 --csv gpu_run_artifacts/ffa4p/eval_ffa.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from orbit_wars_rl.config import Config
from orbit_wars_rl.model import EntityTransformer
import orbit_wars_rl.eval as E


DEFAULT_FIELD = [
    "opponents/candidate_producer_v2.py",
    "opponents/romantamrazov/roman_agent.py",
    "opponents/candidate_hellburner.py",
]


def load_ffa_agent(ckpt_path: str, num_players: int = 4, device: str = "cpu"):
    """Checkpoint -> kaggle agent fn, with the reinforce discipline + decode read
    from the checkpoint config (so eval matches training)."""
    cfg = Config()
    state_dict, action_decode = E.load_checkpoint(ckpt_path, cfg)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck = raw.get("config", {}) if isinstance(raw, dict) else {}
    dev = torch.device(device)
    model = EntityTransformer(cfg.model).to(dev)
    model.allow_reinforce = bool(ck.get("allow_reinforce", False))
    model.reinforce_gate_min_planets = int(ck.get("reinforce_gate_min_planets", 0))
    model.reinforce_forward_only = bool(ck.get("reinforce_forward_only", False))
    model.reinforce_garrison_floor = float(ck.get("reinforce_garrison_floor", 0.0))
    model.sufficient_commit_factor = float(ck.get("sufficient_commit_factor", 0.0))
    model.reverse_edge_cooldown = int(ck.get("reverse_edge_cooldown", 0))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return E.build_agent_fn(
        model, dev, target_decode=(action_decode == "target"),
        ship_bin_mode=cfg.model.ship_bin_mode, num_players=num_players,
        allow_reinforce=model.allow_reinforce,
    )


def _final_scores(last_step, n):
    try:
        obs = last_step[0]["observation"]
        planets, fleets = obs.get("planets") or [], obs.get("fleets") or []
    except Exception:
        return None
    s = [0.0] * n
    for p in planets:
        o = int(p[1])
        if 0 <= o < n:
            s[o] += float(p[5])
    for f in fleets:
        o = int(f[1])
        if 0 <= o < n:
            s[o] += float(f[6])
    return s


def _placement_of(seat, scores):
    return 1 + sum(1 for x in scores if x > scores[seat])


def _parse_step(path: str) -> int:
    m = re.search(r"step_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    import kaggle_environments as ke
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--opponents", default=",".join(DEFAULT_FIELD),
                    help="comma-separated 3 opponent .py paths (the other 3 seats)")
    ap.add_argument("--seeds", type=int, default=16, help="seeds; 4 games each (seat rotation)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--csv", default=None, help="append a result row here")
    args = ap.parse_args()

    opps = [o if os.path.isabs(o) else os.path.join(_REPO, o)
            for o in args.opponents.split(",")]
    assert len(opps) == 3, "need exactly 3 opponents for a 4p field"

    agent_fn = load_ffa_agent(args.checkpoint, num_players=4, device=args.device)
    env = ke.make("orbit_wars")
    wins = place_sum = glen_sum = games = 0
    t0 = time.time()
    for seed in range(args.seeds):
        for shift in range(4):
            # our agent sits in seat `shift`; opponents fill the rest in order
            lineup = [None] * 4
            lineup[shift] = agent_fn
            oi = 0
            for s in range(4):
                if s != shift:
                    lineup[s] = opps[oi]; oi += 1
            env.reset()
            res = env.run(lineup)
            rewards = [(-1e9 if st["reward"] is None else st["reward"]) for st in res[-1]]
            scores = _final_scores(res[-1], 4) or rewards
            place_sum += _placement_of(shift, scores)
            if rewards[shift] == max(rewards):
                wins += 1
            glen_sum += len(res)
            games += 1
        print(f"  ffa-eval {games}/{args.seeds*4}  WR {100*wins/games:.1f}%  "
              f"place {place_sum/games:.2f}", flush=True)

    step = _parse_step(args.checkpoint)
    wr = 100 * wins / games
    mean_place = place_sum / games
    mean_glen = glen_sum / games
    print(f"\nckpt step={step}  FFA-WR={wins}/{games} ({wr:.1f}%)  "
          f"mean_place={mean_place:.2f}  mean_game_len={mean_glen:.0f}  "
          f"({time.time()-t0:.0f}s)")

    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["step", "games", "wr_pct", "mean_place", "mean_game_len",
                            "field", "checkpoint"])
            w.writerow([step, games, f"{wr:.1f}", f"{mean_place:.2f}", f"{mean_glen:.0f}",
                        ";".join(os.path.basename(o) for o in opps),
                        os.path.basename(args.checkpoint)])
        print(f"appended → {args.csv}")


if __name__ == "__main__":
    main()
