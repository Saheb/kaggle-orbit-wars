"""4-player FFA mixed-field panel for standalone agent .py files.

Runs a fixed field of 4 agents in the kaggle orbit_wars 4p env. For each seed the
lineup is cyclically rotated through all 4 seats (4 games/seed), so every agent
occupies every seat equally — controlling for seat/position bias while keeping the
matchup composition fixed. Reports each agent's win-rate (1st place = the +1 reward;
the LB-relevant FFA metric — you only gain rating by winning) and mean placement.

Usage:
  python3 scripts/run_ffa_panel.py --seeds 32 \
    neural=submission_2p4p/neural_agent.py \
    producer_v2=opponents/candidate_producer_v2.py \
    roman_v4=opponents/romantamrazov/roman_agent.py \
    hellburner=opponents/candidate_hellburner.py

Each arg is `label=path/to/agent.py` (exactly 4). Agent files are run by the kaggle
env directly (they must be self-contained — producer/roman need their orbit_lite
sibling, already in place).
"""
import argparse
import json
import os
import sys

import kaggle_environments as ke

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)  # scripts/ -> repo root
sys.path.insert(0, os.path.join(_REPO, "orbit_wars_rl"))


def _resolve(path):
    """A `.py` path is run by the kaggle env directly; a `.pt` checkpoint is loaded
    as a 4-seat neural agent (num_players=4, matching how it was trained/evaluated)."""
    if path.endswith(".pt"):
        from eval_ffa_checkpoint import load_ffa_agent
        return load_ffa_agent(path, num_players=4)
    return path


def _final_scores(last_step, n_seats):
    """Per-seat material = ships on owned planets + ships in owned fleets, from the
    final observation. Distinguishes a close 2nd from an early elimination (the binary
    +1/-1 reward can't). Returns None if the board can't be read."""
    try:
        obs = last_step[0]["observation"]
        planets = obs.get("planets") or []
        fleets = obs.get("fleets") or []
    except Exception:
        return None
    scores = [0.0] * n_seats
    for p in planets:            # [id, owner, x, y, radius, ships, production]
        o = int(p[1])
        if 0 <= o < n_seats:
            scores[o] += float(p[5])
    for f in fleets:             # [id, owner, x, y, angle, from, ships]
        o = int(f[1])
        if 0 <= o < n_seats:
            scores[o] += float(f[6])
    return scores


def _placements(scores):
    """Map per-seat scores → placement 1..4 (1 = best). Ties share the better rank."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    place = [0] * len(scores)
    for rank, seat in enumerate(order):
        if rank > 0 and scores[seat] == scores[order[rank - 1]]:
            place[seat] = place[order[rank - 1]]
        else:
            place[seat] = rank + 1
    return place


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agents", nargs=4, help="4× label=path.py")
    ap.add_argument("--seeds", type=int, default=32, help="seeds; 4 games each (seat rotation)")
    ap.add_argument("--seed-start", type=int, default=0, help="first seed (for sharding disjoint blocks)")
    ap.add_argument("--dump", default="", help="if set, write JSON {labels,wins,place_sum,games} for merging")
    args = ap.parse_args()

    labels, paths = [], []
    for spec in args.agents:
        if "=" not in spec:
            ap.error(f"agent spec must be label=path.py, got: {spec}")
        lbl, p = spec.split("=", 1)
        labels.append(lbl.strip())
        paths.append(p.strip())

    resolved = [_resolve(p) for p in paths]   # build neural agents once, reuse across games

    wins = {l: 0 for l in labels}
    place_sum = {l: 0 for l in labels}
    games = 0
    env = ke.make("orbit_wars")

    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for shift in range(4):
            # cyclic rotation: agent i sits in seat (i+shift) % 4
            lineup_idx = [(i - shift) % 4 for i in range(4)]   # seat -> agent index
            agents = [resolved[a] for a in lineup_idx]
            env.reset()
            result = env.run(agents)
            rewards = [s["reward"] for s in result[-1]]
            rewards = [(-1e9 if r is None else r) for r in rewards]
            # Win = the env's +1 (winner = max material per the env's own _check_done).
            # Placement uses final material so losers are ranked 2/3/4, not all tied.
            scores = _final_scores(result[-1], 4)
            place = _placements(scores if scores is not None else rewards)
            best = max(rewards)
            for seat, a in enumerate(lineup_idx):
                lbl = labels[a]
                place_sum[lbl] += place[seat]
                if rewards[seat] == best:
                    wins[lbl] += 1
            games += 1
            if games % 4 == 0:
                line = "  ".join(f"{l} {100*wins[l]/ (games):.0f}%" for l in labels)
                print(f"  panel {games}/{args.seeds*4}  WR: {line}", flush=True)

    print("\n" + "=" * 60)
    print(f"FFA mixed-field panel — {games} games ({args.seeds} seeds × 4 seat rotations)")
    print("=" * 60)
    print(f"{'agent':<16}{'win-rate':>12}{'mean place':>14}")
    for l in sorted(labels, key=lambda x: -wins[x]):
        print(f"{l:<16}{wins[l]}/{games} ({100*wins[l]/games:>4.1f}%){place_sum[l]/games:>10.2f}")
    print("(win-rate = 1st-place share, the FFA LB metric; mean place 1=best 4=worst)")

    if args.dump:
        with open(args.dump, "w") as f:
            json.dump({"labels": labels, "wins": wins, "place_sum": place_sum, "games": games}, f)


if __name__ == "__main__":
    main()
