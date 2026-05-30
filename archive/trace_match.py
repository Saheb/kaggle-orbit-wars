"""Trace early moves/material for local Orbit Wars agent matchups."""

from __future__ import annotations

import argparse
from pathlib import Path

from kaggle_environments import make

from evaluate_agents import load_agent
from evaluate_ablation import final_material


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slot", type=int, default=0, choices=range(4))
    parser.add_argument("--until", type=int, default=80)
    args = parser.parse_args()

    agents = [load_agent(ROOT / args.baseline) for _ in range(4)]
    agents[args.slot] = load_agent(ROOT / args.candidate)
    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.run(agents)

    previous = None
    for idx, step in enumerate(env.steps):
        obs = step[0].observation
        turn = int(obs.get("step", idx))
        if turn > args.until:
            break
        material = []
        production = []
        planets = []
        for player in range(4):
            material.append(sum(int(p[5]) for p in obs["planets"] if p[1] == player) + sum(int(f[6]) for f in obs["fleets"] if f[1] == player))
            production.append(sum(int(p[6]) for p in obs["planets"] if p[1] == player))
            planets.append(sum(1 for p in obs["planets"] if p[1] == player))
        actions = [state.action for state in step]
        changed = actions != previous
        if changed or turn % 10 == 0:
            print(
                f"turn={turn:03d} material={material} prod={production} planets={planets} actions={actions}",
                flush=True,
            )
        previous = actions
    print(f"final steps={len(env.steps)} material={final_material(env, 4)} rewards={[s.reward for s in env.steps[-1]]}")


if __name__ == "__main__":
    main()
