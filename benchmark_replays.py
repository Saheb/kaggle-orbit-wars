"""Benchmark Orbit Wars replay openings across agents/datasets.

This is for strategy diagnosis, not imitation. It reads replay JSON files and
reports opening curves for winners versus selected agents.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


CHECKPOINTS = (25, 50, 75, 100)


@dataclass
class PlayerCurve:
    replay: str
    player: int
    name: str
    won: bool
    final_step: int
    material: dict[int, int]
    production: dict[int, int]
    planets: dict[int, int]
    fleet_ratio: dict[int, float]
    launched: dict[int, int]
    cumulative_launched: dict[int, int]
    captures: dict[int, int]
    captured_production: dict[int, int]
    first_captures: list[tuple[int, int, int, int]]


def load_replay(path: Path):
    with path.open() as f:
        return json.load(f)


def obs_at_step(data, target_step: int):
    steps = data.get("steps", [])
    if not steps:
        return None, None
    idx = min(range(len(steps)), key=lambda i: abs((steps[i][0]["observation"].get("step", i)) - target_step))
    return steps[idx], steps[idx][0]["observation"].get("step", idx)


def metrics(env_step, players: int):
    obs = env_step[0]["observation"]
    material = [0] * players
    planet_ships = [0] * players
    fleet_ships = [0] * players
    production = [0] * players
    planets = [0] * players
    launched = [0] * players

    for planet in obs["planets"]:
        owner = planet[1]
        if 0 <= owner < players:
            ships = int(planet[5])
            material[owner] += ships
            planet_ships[owner] += ships
            production[owner] += int(planet[6])
            planets[owner] += 1

    for fleet in obs["fleets"]:
        owner = fleet[1]
        if 0 <= owner < players:
            ships = int(fleet[6])
            material[owner] += ships
            fleet_ships[owner] += ships

    for i, state in enumerate(env_step[:players]):
        action = state.get("action") or []
        launched[i] = sum(int(move[2]) for move in action if len(move) >= 3)

    fleet_ratio = []
    for m, f in zip(material, fleet_ships):
        fleet_ratio.append(0.0 if m <= 0 else f / m)

    return {
        "material": material,
        "production": production,
        "planets": planets,
        "fleet_ratio": fleet_ratio,
        "launched": launched,
    }


def capture_events(data, players: int):
    events = [[] for _ in range(players)]
    prev_owner = {}
    prev_seen = set()
    for env_step in data.get("steps", []):
        obs = env_step[0]["observation"]
        step = int(obs.get("step", 0))
        for planet in obs["planets"]:
            pid, owner, ships, prod = int(planet[0]), int(planet[1]), int(planet[5]), int(planet[6])
            if pid in prev_seen and prev_owner.get(pid) != owner and 0 <= owner < players:
                events[owner].append((step, pid, prod, ships))
            prev_owner[pid] = owner
            prev_seen.add(pid)
    return events


def summarize_player(path: Path, data, player: int) -> PlayerCurve:
    players = len(data["steps"][0])
    names = data.get("info", {}).get("TeamNames") or [f"p{i}" for i in range(players)]
    rewards = data.get("rewards", [None] * players)
    won = rewards[player] == max(rewards)
    final_step = int(data["steps"][-1][0]["observation"].get("step", len(data["steps"]) - 1))

    material = {}
    production = {}
    planets = {}
    fleet_ratio = {}
    launched = {}
    cumulative_launched = {}
    for checkpoint in CHECKPOINTS:
        env_step, actual_step = obs_at_step(data, checkpoint)
        if env_step is None:
            continue
        m = metrics(env_step, players)
        material[checkpoint] = m["material"][player]
        production[checkpoint] = m["production"][player]
        planets[checkpoint] = m["planets"][player]
        fleet_ratio[checkpoint] = m["fleet_ratio"][player]
        launched[checkpoint] = m["launched"][player]

    events = capture_events(data, players)[player][:8]
    all_events = capture_events(data, players)[player]
    captures = {}
    captured_production = {}
    for checkpoint in CHECKPOINTS:
        captures[checkpoint] = sum(1 for step, _, _, _ in all_events if step <= checkpoint)
        captured_production[checkpoint] = sum(prod for step, _, prod, _ in all_events if step <= checkpoint)

    running_launches = {checkpoint: 0 for checkpoint in CHECKPOINTS}
    for env_step in data.get("steps", []):
        step = int(env_step[0]["observation"].get("step", 0))
        action = env_step[player].get("action") or []
        launched_ships = sum(int(move[2]) for move in action if len(move) >= 3)
        for checkpoint in CHECKPOINTS:
            if step <= checkpoint:
                running_launches[checkpoint] += launched_ships

    return PlayerCurve(
        replay=path.name,
        player=player,
        name=names[player],
        won=won,
        final_step=final_step,
        material=material,
        production=production,
        planets=planets,
        fleet_ratio=fleet_ratio,
        launched=launched,
        cumulative_launched=running_launches,
        captures=captures,
        captured_production=captured_production,
        first_captures=events,
    )


def replay_paths(inputs: list[str], limit: int | None, filename_contains: str | None):
    paths = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            paths.append(path)
    if limit is not None:
        paths = paths[:limit]
    if filename_contains:
        paths = [path for path in paths if filename_contains in path.name]
    return paths


def selected_curves(paths: list[Path], mode: str, player_name: str | None):
    curves = []
    for path in paths:
        data = load_replay(path)
        if not data.get("steps"):
            continue
        players = len(data["steps"][0])
        names = data.get("info", {}).get("TeamNames") or [f"p{i}" for i in range(players)]
        rewards = data.get("rewards", [None] * players)
        if mode == "winners":
            best = max(rewards)
            selected = [i for i, reward in enumerate(rewards) if reward == best]
        elif mode == "all":
            selected = list(range(players))
        elif mode == "name":
            selected = [i for i, name in enumerate(names) if name == player_name]
        else:
            raise ValueError(mode)
        for player in selected:
            curves.append(summarize_player(path, data, player))
    return curves


def aggregate(label: str, curves: list[PlayerCurve]):
    print(f"\n== {label} ==")
    print(f"players={len(curves)} wins={sum(c.won for c in curves)}")
    if not curves:
        return
    for checkpoint in CHECKPOINTS:
        prod = [c.production.get(checkpoint, 0) for c in curves]
        planets = [c.planets.get(checkpoint, 0) for c in curves]
        mat = [c.material.get(checkpoint, 0) for c in curves]
        fr = [c.fleet_ratio.get(checkpoint, 0.0) for c in curves]
        launched = [c.launched.get(checkpoint, 0) for c in curves]
        cum_launch = [c.cumulative_launched.get(checkpoint, 0) for c in curves]
        captures = [c.captures.get(checkpoint, 0) for c in curves]
        cap_prod = [c.captured_production.get(checkpoint, 0) for c in curves]
        print(
            f"t={checkpoint:3d} "
            f"prod avg/med={mean(prod):5.1f}/{median(prod):4.1f} "
            f"planets avg/med={mean(planets):4.1f}/{median(planets):4.1f} "
            f"mat avg/med={mean(mat):6.1f}/{median(mat):5.1f} "
            f"fleet_ratio avg={mean(fr):.2f} "
            f"launch_turn/cum={mean(launched):5.1f}/{mean(cum_launch):6.1f} "
            f"caps/prod={mean(captures):4.1f}/{mean(cap_prod):5.1f}"
        )


def print_examples(label: str, curves: list[PlayerCurve], count: int):
    print(f"\n-- {label} examples --")
    for c in curves[:count]:
        caps = ", ".join(f"t{s}:p{pid}/prod{prod}/ships{ships}" for s, pid, prod, ships in c.first_captures[:5])
        print(
            f"{c.replay} p{c.player} {c.name!r} won={c.won} final_step={c.final_step} "
            f"prod@50={c.production.get(50, 0)} planets@50={c.planets.get(50, 0)} "
            f"fleet_ratio@50={c.fleet_ratio.get(50, 0.0):.2f} "
            f"cum_launch@50={c.cumulative_launched.get(50, 0)} "
            f"cap_prod@50={c.captured_production.get(50, 0)} first_caps=[{caps}]"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Replay JSON files or directories")
    parser.add_argument("--mode", choices=["winners", "all", "name"], default="winners")
    parser.add_argument("--player-name", help="Required when --mode name")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--filename-contains")
    parser.add_argument("--label", default="replays")
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    if args.mode == "name" and not args.player_name:
        parser.error("--player-name is required with --mode name")

    paths = replay_paths(args.paths, args.limit, args.filename_contains)
    curves = selected_curves(paths, args.mode, args.player_name)
    aggregate(args.label, curves)
    print_examples(args.label, curves, args.examples)


if __name__ == "__main__":
    main()
