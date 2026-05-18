"""Pit one Orbit Wars heuristic variant against another on identical boards."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from statistics import mean

from kaggle_environments import make

import main
from evaluate_ablation import VARIANTS, final_material


@contextmanager
def variant_flags(settings):
    old = {name: getattr(main, name) for name in settings}
    try:
        for name, value in settings.items():
            setattr(main, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(main, name, value)


def variant_agent(name):
    settings = VARIANTS[name]

    def agent(obs):
        with variant_flags(settings):
            return main.agent(obs)

    agent.__name__ = f"{name}_agent"
    return agent


def run_match(seed, candidate, baseline, candidate_slot):
    agents = [variant_agent(baseline) for _ in range(4)]
    agents[candidate_slot] = variant_agent(candidate)
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.run(agents)
    rewards = [state.reward for state in env.steps[-1]]
    winner = max(range(4), key=lambda i: rewards[i])
    return {
        "winner": winner,
        "candidate_won": winner == candidate_slot,
        "candidate_reward": rewards[candidate_slot],
        "rewards": rewards,
        "material": final_material(env, 4),
        "steps": len(env.steps),
    }


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--baseline", default="legacy", choices=sorted(VARIANTS))
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        for slot in range(4):
            row = run_match(seed, args.candidate, args.baseline, slot)
            rows.append(row)
            print(
                f"seed={seed:03d} slot={slot} winner={row['winner']} "
                f"candidate_won={row['candidate_won']} "
                f"candidate_reward={row['candidate_reward']} "
                f"steps={row['steps']} material={row['material']}"
            )

    wins = sum(row["candidate_won"] for row in rows)
    rewards = [row["candidate_reward"] for row in rows]
    print(
        f"\n{args.candidate} vs {args.baseline}: "
        f"wins={wins}/{len(rows)} avg_reward={mean(rewards):.3f}"
    )


if __name__ == "__main__":
    main_cli()
