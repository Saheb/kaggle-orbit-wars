"""Build (state, action, MC-return) samples from RAW replay episodes — the PURE-REPLAY Q-head gate.

The self-play gate is confounded: the only winning data was generated INSIDE the under-aggregation
Nash (idle→win), so a Q-head regressed on it prices firing the idle spare ≤0. This script tests the
escape hatch: train the Q-head on EXPERT REPLAY data (strong players, mixed win/loss outcomes, never
self-play). If firing→win is present in real expert games, the replay-trained Q-head should price
q_fire−q_idle on idle spares > 0 — the signal the self-play Q-head lacked.

Reuses build_replay_action_bc's replay→sample extraction; the only addition is a per-sample
discounted TERMINAL return  ret_t = gamma^(T_last − t) · reward[seat]  (reward ∈ {−1,+1}; replays
carry only terminal reward). Keeps ALL turns (incl. whole-idle) for a representative action mix.

Output is a torch_env-dump-format batch dict, so q_head_opportunity_gate.py consumes it directly:
  ORBIT_GAME_PHASE_FEATURES=1 <venv>/bin/python orbit_wars_rl/build_replay_returns.py \
      'gpu_run_artifacts/ar_stage1/replays/*.json' --out /tmp/replay_returns.pt --max-games 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from collections import Counter

from orbit_wars_rl.scripts.build_replay_action_bc import (
    _iter_paths, _winner_seat, _copy_replay_obs, _context, _decode_move,
    _dedupe_by_source, trajectory_to_training_sample,
)

_FEAT_KEYS = ["planet_features", "fleet_features", "global_features", "planet_mask",
              "fleet_mask", "slot_valid", "owned_indices", "pairwise_features"]


def build(paths, max_planets, gamma, max_games, max_samples, step_limit):
    files = _iter_paths(paths)
    cols = {k: [] for k in _FEAT_KEYS}
    fire_a, ship_a, target_a, rets = [], [], [], []
    n_games = n_won = 0
    for path in files:
        if max_games and n_games >= max_games:
            break
        try:
            replay = json.loads(Path(path).read_text())
        except Exception:
            continue
        rewards = replay.get("rewards") or []
        steps = replay.get("steps") or []
        if len(rewards) != 2 or len(steps) < 2:
            continue
        n_games += 1
        T_last = len(steps) - 1
        for seat in range(len(steps[0])):
            r_seat = 1.0 if float(rewards[seat]) > 0 else -1.0
            if seat == 0 and r_seat > 0:
                n_won += 1
            for t in range(1, len(steps)):
                if step_limit and t > step_limit:
                    break
                if seat >= len(steps[t]) or seat >= len(steps[t - 1]):
                    continue
                action = steps[t][seat].get("action") or []
                obs = _copy_replay_obs(steps[t - 1][seat].get("observation") or {}, seat, t - 1)
                if not obs["planets"]:
                    continue
                ctx = _context(obs, max_planets)
                labels = []
                for move in action:
                    lab = _decode_move(obs, move, ctx, max_planets)
                    if lab is not None and "move" in lab:   # skip {"drop_reason": ...} non-moves
                        labels.append(lab)
                kept = [lab["move"] for lab in _dedupe_by_source(labels, Counter())]
                try:
                    sample = trajectory_to_training_sample({"obs": obs, "action": kept},
                                                           max_planets=max_planets)
                except Exception:
                    sample = None
                if sample is None:
                    continue
                for k in _FEAT_KEYS:
                    cols[k].append(sample[k])
                fire_a.append(sample["fire_target"].long())
                ship_a.append(sample["ship_target"].long().clamp(min=0))
                target_a.append(sample["target_target"].long().clamp(min=0))
                rets.append((gamma ** (T_last - t)) * r_seat)
                if max_samples and len(rets) >= max_samples:
                    break
            if max_samples and len(rets) >= max_samples:
                break
        if max_samples and len(rets) >= max_samples:
            break

    batch = {k: torch.stack(v) for k, v in cols.items()}
    batch["actions"] = {"fire": torch.stack(fire_a), "ship": torch.stack(ship_a),
                        "target": torch.stack(target_a)}
    batch["returns"] = torch.tensor(rets, dtype=torch.float32)
    print(f"games={n_games} (seat0 won {n_won}) | samples={len(rets)} | "
          f"global_dim={batch['global_features'].shape[-1]} | "
          f"returns mean {batch['returns'].mean():+.3f} std {batch['returns'].std():.3f} "
          f"[min {batch['returns'].min():+.2f} max {batch['returns'].max():+.2f}] | "
          f"fire-rate {batch['actions']['fire'].float().mean():.3f}", flush=True)
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-planets", type=int, default=48)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--step-limit", type=int, default=0)
    args = ap.parse_args()
    batch = build(args.paths, args.max_planets, args.gamma, args.max_games,
                  args.max_samples, args.step_limit or None)
    torch.save(batch, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
