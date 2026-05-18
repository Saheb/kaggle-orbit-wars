"""Run local Orbit Wars games and report checkpoint tempo metrics."""

from __future__ import annotations

import argparse
from statistics import mean, median

from kaggle_environments import make


CHECKPOINTS = (25, 50, 75, 100, 150, 200)


def nearest_step(env, checkpoint: int):
    return min(env.steps, key=lambda s: abs(s[0].observation.get("step", 0) - checkpoint))


def player_metrics(env, player: int, checkpoint: int):
    step = nearest_step(env, checkpoint)
    obs = step[0].observation
    material = 0
    fleet_ships = 0
    production = 0
    planets = 0

    for planet in obs["planets"]:
        owner = planet[1]
        if owner == player:
            material += int(planet[5])
            production += int(planet[6])
            planets += 1

    for fleet in obs["fleets"]:
        owner = fleet[1]
        if owner == player:
            ships = int(fleet[6])
            material += ships
            fleet_ships += ships

    launched = 0
    for env_step in env.steps:
        step_no = env_step[0].observation.get("step", 0)
        if step_no > checkpoint:
            continue
        action = env_step[player].action or []
        launched += sum(int(move[2]) for move in action if len(move) >= 3)

    fleet_ratio = 0.0 if material <= 0 else fleet_ships / material
    return {
        "material": material,
        "production": production,
        "planets": planets,
        "fleet_ratio": fleet_ratio,
        "launched": launched,
    }


def final_material(env, players: int):
    obs = env.steps[-1][0].observation
    totals = [0 for _ in range(players)]
    for planet in obs["planets"]:
        owner = planet[1]
        if 0 <= owner < players:
            totals[owner] += int(planet[5])
    for fleet in obs["fleets"]:
        owner = fleet[1]
        if 0 <= owner < players:
            totals[owner] += int(fleet[6])
    return totals


def run(seed: int, players: int, opponent: str):
    opponent_agent = "random" if opponent == "random" else "starter_nearest.py"
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.run(["main.py", *([opponent_agent] * (players - 1))])
    rewards = [state.reward for state in env.steps[-1]]
    winner = max(range(players), key=lambda i: rewards[i])
    checkpoints = {cp: player_metrics(env, 0, cp) for cp in CHECKPOINTS}
    return winner, rewards, final_material(env, players), checkpoints


def summarize(rows):
    wins = sum(1 for row in rows if row[0] == 0)
    print(f"p0_wins={wins}/{len(rows)}")
    for cp in CHECKPOINTS:
        prod = [row[3][cp]["production"] for row in rows]
        planets = [row[3][cp]["planets"] for row in rows]
        material = [row[3][cp]["material"] for row in rows]
        fleet_ratio = [row[3][cp]["fleet_ratio"] for row in rows]
        launched = [row[3][cp]["launched"] for row in rows]
        print(
            f"t={cp:3d} "
            f"prod avg/med={mean(prod):5.1f}/{median(prod):4.1f} "
            f"planets avg/med={mean(planets):4.1f}/{median(planets):4.1f} "
            f"mat avg/med={mean(material):6.1f}/{median(material):5.1f} "
            f"fleet_ratio avg={mean(fleet_ratio):.2f} "
            f"cum_launch avg/med={mean(launched):6.1f}/{median(launched):5.1f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--players", type=int, choices=[2, 4], default=4)
    parser.add_argument("--opponent", choices=["random", "starter"], default="starter")
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        winner, rewards, material, checkpoints = run(seed, args.players, args.opponent)
        rows.append((winner, rewards, material, checkpoints))
        print(f"seed={seed:03d} winner={winner} rewards={rewards} material={material}")
    summarize(rows)


if __name__ == "__main__":
    main()
