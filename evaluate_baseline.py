"""Run small local Orbit Wars comparisons over deterministic seeds."""

from __future__ import annotations

import argparse
from collections import Counter

from kaggle_environments import make


def final_scores(env):
    final = env.steps[-1]
    return [state.reward for state in final], [state.status for state in final]


def final_material(env, players: int):
    observation = env.steps[-1][0].observation
    planets = observation["planets"]
    fleets = observation["fleets"]
    totals = [0 for _ in range(players)]
    for planet in planets:
        owner = planet[1]
        if 0 <= owner < players:
            totals[owner] += planet[5]
    for fleet in fleets:
        owner = fleet[1]
        if 0 <= owner < players:
            totals[owner] += fleet[6]
    return totals


def run_match(seed: int, opponents: list[str]):
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.run(["main.py", *opponents])
    rewards, statuses = final_scores(env)
    material = final_material(env, len(opponents) + 1)
    winner = max(range(len(rewards)), key=lambda i: rewards[i])
    return rewards, statuses, material, winner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument(
        "--opponent",
        choices=["random", "starter"],
        default="random",
        help="Opponent type for every non-player-0 slot.",
    )
    parser.add_argument(
        "--players",
        type=int,
        choices=[2, 4],
        default=4,
        help="Run a 2-player or 4-player local game.",
    )
    args = parser.parse_args()

    opponent_agent = "random" if args.opponent == "random" else "starter_nearest.py"
    opponents = [opponent_agent] * (args.players - 1)
    wins = Counter()
    material_sum = [0 for _ in range(args.players)]

    for seed in range(args.seeds):
        rewards, statuses, material, winner = run_match(seed, opponents)
        wins[winner] += 1
        for i, score in enumerate(material):
            material_sum[i] += score
        print(
            f"seed={seed:03d} winner={winner} "
            f"p0_reward={rewards[0]} rewards={rewards} "
            f"material={material} statuses={statuses}"
        )

    material_avg = [round(total / args.seeds, 1) for total in material_sum]
    print(f"wins={dict(wins)} p0_wins={wins[0]}/{args.seeds} avg_material={material_avg}")


if __name__ == "__main__":
    main()
