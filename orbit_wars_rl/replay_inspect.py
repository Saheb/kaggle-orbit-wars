"""Replay inspector: run our checkpoint vs an opponent and dump per-turn moves.

Pure evidence-gathering tool. No model changes, no training. Two outputs per
game:

  - <out>/seed_<n>.json   — structured per-turn data (machine-readable, complete)
  - <out>/seed_<n>.txt    — compact human view (turn-by-turn what each side did,
                             planet ownership flips, fleet launches, terminal state)

Usage:
    python replay_inspect.py --checkpoint <pt> --opponent <py> \
        --seed 42 [--seed 43 ...] --out replays_inspect/

Reading the .txt is meant to feel like watching the game tick by tick: each
turn shows what each side decided, which planets switched hands, and which
fleets are inbound. The .json is the source of truth for any later analysis.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

from config import Config
from eval import build_agent_fn
from model import EntityTransformer


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _planet_owners(planets):
    """Returns {planet_id: owner} dict from an obs.planets list."""
    return {int(p[0]): int(p[1]) for p in planets}


def _planet_ships(planets):
    return {int(p[0]): float(p[5]) for p in planets}


def _summarize_planet_state(planets, viewer_player=0):
    """Return short string like 'mine: 3 (45 ships) | enemy: 4 (62) | neutral: 2'."""
    counts = defaultdict(lambda: [0, 0.0])
    for p in planets:
        owner, ships = int(p[1]), float(p[5])
        if owner == viewer_player: tag = "mine"
        elif owner == -1:           tag = "neutral"
        else:                       tag = "enemy"
        counts[tag][0] += 1
        counts[tag][1] += ships
    parts = []
    for tag in ("mine", "enemy", "neutral"):
        n, s = counts[tag]
        parts.append(f"{tag}: {n} ({s:.0f})")
    return "  |  ".join(parts)


def _format_move(mv):
    """Render [from_pid, angle_rad, ships] as a short string."""
    if not isinstance(mv, (list, tuple)) or len(mv) < 3:
        return f"<malformed:{mv!r}>"
    import math
    pid = int(mv[0]); ang = float(mv[1]); ships = int(mv[2])
    deg = (ang * 180.0 / math.pi) % 360
    return f"p{pid}→{deg:5.1f}°  ×{ships}"


def _ownership_diff(prev_planets, curr_planets):
    """Return list of (planet_id, prev_owner, curr_owner) for changed planets."""
    prev = _planet_owners(prev_planets)
    curr = _planet_owners(curr_planets)
    diffs = []
    for pid in set(prev) | set(curr):
        po, co = prev.get(pid, -2), curr.get(pid, -2)
        if po != co:
            diffs.append((pid, po, co))
    return diffs


# ----------------------------------------------------------------------------
# Main per-game inspection
# ----------------------------------------------------------------------------

def inspect_game(model_agent_fn, opponent_path: str, seed: int):
    """Run one game and return a structured record + a human-readable text view."""
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([model_agent_fn, opponent_path])

    turns = []          # one entry per step
    text_lines = []
    n_steps = len(env.steps)

    text_lines.append(f"=== seed={seed}  episode_len={n_steps} ===")

    for t in range(n_steps):
        # In kaggle envs, env.steps[t][p] is player p's view AT step t. The
        # 'action' field is what player p submitted to PRODUCE this step (i.e.
        # the action that landed us in observation step t). So actions taken
        # at turn (t-1) appear in env.steps[t][p]['action'].
        step = env.steps[t]
        obs0 = step[0]["observation"]
        # Actions: what each player did in the previous turn to get here
        if t == 0:
            actions = [[], []]
        else:
            actions = [env.steps[t][0].get("action") or [],
                       env.steps[t][1].get("action") or []]

        planets = obs0.get("planets", [])
        fleets = obs0.get("fleets", [])
        # Ownership flips since previous turn
        flips = []
        if t > 0:
            prev_planets = env.steps[t-1][0]["observation"].get("planets", [])
            flips = _ownership_diff(prev_planets, planets)

        turn_record = {
            "step": t,
            "planets": planets,
            "fleets": fleets,
            "action_p0": actions[0],
            "action_p1": actions[1],
            "ownership_flips": flips,
        }
        turns.append(turn_record)

        # Compact human view: only print turns where SOMETHING happened
        has_event = bool(actions[0] or actions[1] or flips)
        if has_event or t == 0 or t == n_steps - 1:
            text_lines.append(f"\n--- turn {t:3d} ---")
            text_lines.append(f"  state:     {_summarize_planet_state(planets, 0)}  | fleets_inflight={len(fleets)}")
            if flips:
                flip_strs = [f"p{pid}: {po}→{co}" for pid, po, co in flips]
                text_lines.append(f"  flips:     {', '.join(flip_strs)}")
            if actions[0]:
                text_lines.append(f"  MODEL (p0): {len(actions[0])} moves")
                for mv in actions[0]:
                    text_lines.append(f"      {_format_move(mv)}")
            if actions[1]:
                text_lines.append(f"  OPP   (p1): {len(actions[1])} moves")
                for mv in actions[1]:
                    text_lines.append(f"      {_format_move(mv)}")

    # Terminal verdict
    final = env.steps[-1]
    rewards = [s.get("reward") if s.get("reward") is not None else 0.0 for s in final]
    p0_won = rewards[0] > rewards[1]
    text_lines.append(f"\n=== terminal ===")
    text_lines.append(f"  rewards: p0={rewards[0]:+.1f}  p1={rewards[1]:+.1f}  → MODEL {'WON' if p0_won else 'LOST'}")
    final_planets = final[0]["observation"].get("planets", [])
    text_lines.append(f"  final state: {_summarize_planet_state(final_planets, 0)}")

    record = {
        "seed": seed,
        "opponent": opponent_path,
        "n_steps": n_steps,
        "model_won": p0_won,
        "final_rewards": rewards,
        "turns": turns,
    }
    return record, "\n".join(text_lines)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--opponent", required=True, help="Path to .py heuristic")
    p.add_argument("--seed", type=int, action="append", required=True,
                   help="Seed to play. Repeat for multiple games: --seed 42 --seed 43")
    p.add_argument("--out", default="replays_inspect")
    p.add_argument("--device", default="cpu")
    p.add_argument("--fire-threshold", type=float, default=0.5)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    device = torch.device(args.device)
    model = EntityTransformer(cfg.model).to(device)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" in sd: sd = sd["model"]
    model.load_state_dict(sd)
    model.eval()
    agent_fn = build_agent_fn(model, device, fire_threshold=args.fire_threshold)

    for seed in args.seed:
        record, text = inspect_game(agent_fn, args.opponent, seed)
        (out / f"seed_{seed}.json").write_text(json.dumps(record, indent=2, default=float))
        (out / f"seed_{seed}.txt").write_text(text)
        verdict = "WON" if record["model_won"] else "LOST"
        print(f"  seed={seed}  steps={record['n_steps']}  → MODEL {verdict}  "
              f"(wrote {out}/seed_{seed}.{{json,txt}})")


if __name__ == "__main__":
    main()
