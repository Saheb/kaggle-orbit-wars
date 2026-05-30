"""Diagnose Orbit Wars episodes with public-state time-series metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from kaggle_environments import make


@dataclass
class Metrics:
    step: int
    material: list[int]
    planet_ships: list[int]
    fleet_ships: list[int]
    production: list[int]
    planets: list[int]
    launches: list[int]
    launched_ships: list[int]


def public_observation(env_step):
    return env_step[0].observation


def compute_metrics(env_step, players: int) -> Metrics:
    obs = public_observation(env_step)
    step = int(obs.get("step", 0))
    material = [0 for _ in range(players)]
    planet_ships = [0 for _ in range(players)]
    fleet_ships = [0 for _ in range(players)]
    production = [0 for _ in range(players)]
    planets = [0 for _ in range(players)]
    launches = [0 for _ in range(players)]
    launched_ships = [0 for _ in range(players)]

    for planet in obs["planets"]:
        owner = planet[1]
        if 0 <= owner < players:
            ships = int(planet[5])
            prod = int(planet[6])
            planet_ships[owner] += ships
            production[owner] += prod
            planets[owner] += 1
            material[owner] += ships

    for fleet in obs["fleets"]:
        owner = fleet[1]
        if 0 <= owner < players:
            ships = int(fleet[6])
            fleet_ships[owner] += ships
            material[owner] += ships

    for player, state in enumerate(env_step[:players]):
        action = state.action or []
        launches[player] = len(action)
        launched_ships[player] = sum(int(move[2]) for move in action if len(move) >= 3)

    return Metrics(step, material, planet_ships, fleet_ships, production, planets, launches, launched_ships)


def deltas(series: list[Metrics], player: int, leader: int):
    rows = []
    for prev, cur in zip(series, series[1:]):
        p_delta = cur.material[player] - prev.material[player]
        l_delta = cur.material[leader] - prev.material[leader]
        rows.append((cur.step, p_delta - l_delta, p_delta, l_delta, cur))
    return rows


def print_snapshot(label: str, metric: Metrics, player: int, leader: int):
    print(
        f"{label} step={metric.step} "
        f"p{player}: material={metric.material[player]} "
        f"prod={metric.production[player]} planets={metric.planets[player]} "
        f"fleet={metric.fleet_ships[player]} launched={metric.launched_ships[player]} | "
        f"p{leader}: material={metric.material[leader]} "
        f"prod={metric.production[leader]} planets={metric.planets[leader]} "
        f"fleet={metric.fleet_ships[leader]} launched={metric.launched_ships[leader]}"
    )


def summarize(seed: int, players: int, opponent: str):
    opponent_agent = "random" if opponent == "random" else "starter_nearest.py"
    agents = ["main.py", *([opponent_agent] * (players - 1))]
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.run(agents)

    series = [compute_metrics(step, players) for step in env.steps]
    rewards = [state.reward for state in env.steps[-1]]
    statuses = [state.status for state in env.steps[-1]]
    final = series[-1]
    leader = max(range(players), key=lambda i: final.material[i])

    print(f"seed={seed} players={players} opponent={opponent}")
    print(f"rewards={rewards} statuses={statuses} material={final.material} leader={leader}")
    print_snapshot("start", series[0], 0, leader)

    for checkpoint in (50, 100, 150, 250, 350, 499):
        metric = min(series, key=lambda m: abs(m.step - checkpoint))
        print_snapshot(f"near_{checkpoint}", metric, 0, leader)

    print_snapshot("final", final, 0, leader)

    if leader != 0:
        print("\nLargest p0 launch turns:")
        for metric in sorted(series, key=lambda m: m.launched_ships[0], reverse=True)[:8]:
            print_snapshot("launch", metric, 0, leader)

        print("\nWorst relative material swings for p0:")
        for step, rel_delta, p_delta, l_delta, metric in sorted(deltas(series, 0, leader), key=lambda r: r[1])[:8]:
            print(
                f"step={step:03d} rel_delta={rel_delta:+5d} "
                f"p0_delta={p_delta:+5d} p{leader}_delta={l_delta:+5d} "
                f"p0_mat={metric.material[0]:5d} p{leader}_mat={metric.material[leader]:5d} "
                f"p0_prod={metric.production[0]:2d} p{leader}_prod={metric.production[leader]:2d}"
            )

    print("\nFinal by player:")
    for p in range(players):
        print(
            f"p{p}: material={final.material[p]} planet_ships={final.planet_ships[p]} "
            f"fleet_ships={final.fleet_ships[p]} production={final.production[p]} "
            f"planets={final.planets[p]}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--players", type=int, choices=[2, 4], default=4)
    parser.add_argument("--opponent", choices=["random", "starter"], default="starter")
    args = parser.parse_args()
    summarize(args.seed, args.players, args.opponent)


if __name__ == "__main__":
    main()
