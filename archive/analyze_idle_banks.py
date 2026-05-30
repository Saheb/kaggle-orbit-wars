"""Find high-ship owned planets that are not launching in Orbit Wars replays.

This diagnoses whether an agent is failing to use captured planets as sources.
It is intentionally replay-only: it reports observable launch behavior and
nearby non-owned target pressure without assuming our current heuristic.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0
MAX_SPEED = 6.0


def fleet_speed(ships: int) -> float:
    ships = max(1, int(ships))
    scale = (math.log(ships) / math.log(1000.0)) ** 1.5
    return 1.0 + (MAX_SPEED - 1.0) * scale


def point_segment_distance(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = min(1.0, max(0.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def crosses_sun(src, dst):
    return point_segment_distance(CENTER[0], CENTER[1], src[2], src[3], dst[2], dst[3]) <= SUN_RADIUS + 0.25


def distance(a, b):
    return math.hypot(a[2] - b[2], a[3] - b[3])


def incoming_to_target(target, fleets, owner):
    pressure = 0
    for fleet in fleets:
        if int(fleet[1]) != owner:
            continue
        ux = math.cos(float(fleet[4]))
        uy = math.sin(float(fleet[4]))
        vx = target[2] - fleet[2]
        vy = target[3] - fleet[3]
        along = vx * ux + vy * uy
        if along <= 0:
            continue
        perp = abs(vx * uy - vy * ux)
        if perp <= target[4] + 2.5:
            pressure += int(fleet[6])
    return pressure


def viable_targets(src, planets, fleets, player, limit):
    targets = []
    for target in planets:
        if int(target[1]) == player:
            continue
        if crosses_sun(src, target):
            continue
        dist = distance(src, target)
        if dist > limit:
            continue
        ships = int(target[5]) + 1
        incoming = incoming_to_target(target, fleets, player)
        unsaturated = max(0, ships - incoming)
        targets.append(
            {
                "id": int(target[0]),
                "owner": int(target[1]),
                "production": int(target[6]),
                "ships": int(target[5]),
                "distance": dist,
                "incoming": incoming,
                "unsaturated": unsaturated,
                "payback": ships / max(1, int(target[6])),
            }
        )
    targets.sort(key=lambda t: (-t["production"], t["unsaturated"] <= 0, t["distance"]))
    return targets[:5]


def load(path: Path):
    with path.open() as f:
        return json.load(f)


def player_index(data, name_or_index):
    if name_or_index is None:
        return None
    try:
        return int(name_or_index)
    except ValueError:
        pass
    names = data.get("info", {}).get("TeamNames") or []
    for idx, name in enumerate(names):
        if name == name_or_index:
            return idx
    raise SystemExit(f"could not find player {name_or_index!r}; names={names}")


def analyze(path: Path, player_arg: str | None, min_ships: int, lookback: int, target_limit: float):
    data = load(path)
    if not data.get("steps"):
        return
    players = len(data["steps"][0])
    selected = player_index(data, player_arg)
    if selected is None:
        selected = list(range(players))
    else:
        selected = [selected]

    launch_history = defaultdict(list)
    rows = []
    for env_step in data["steps"]:
        obs = env_step[0]["observation"]
        step = int(obs.get("step", 0))
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        for player in selected:
            for move in env_step[player].get("action") or []:
                if len(move) >= 3:
                    launch_history[(player, int(move[0]))].append((step, int(move[2])))

            owned = [p for p in planets if int(p[1]) == player]
            for planet in owned:
                pid = int(planet[0])
                ships = int(planet[5])
                if ships < min_ships:
                    continue
                recent_launches = [
                    ships_sent
                    for launch_step, ships_sent in launch_history[(player, pid)]
                    if step - lookback < launch_step <= step
                ]
                if recent_launches:
                    continue
                targets = viable_targets(planet, planets, fleets, player, target_limit)
                useful = [t for t in targets if t["production"] >= 2 and t["unsaturated"] > 0]
                if useful:
                    rows.append((step, player, pid, ships, int(planet[6]), useful[:3]))

    print(f"{path.name}: idle_bank_events={len(rows)} min_ships={min_ships} lookback={lookback}")
    for step, player, pid, ships, prod, targets in rows[:30]:
        target_text = "; ".join(
            f"p{t['id']} owner={t['owner']} prod={t['production']} ships={t['ships']} "
            f"in={t['incoming']} need_left={t['unsaturated']} dist={t['distance']:.1f}"
            for t in targets
        )
        print(f"t={step:3d} player={player} planet={pid} prod={prod} ships={ships}: {target_text}")
    if len(rows) > 30:
        print(f"... {len(rows) - 30} more")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--player", help="Player index or exact team name")
    parser.add_argument("--min-ships", type=int, default=150)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--target-limit", type=float, default=45.0)
    args = parser.parse_args()

    for path in args.replays:
        analyze(path, args.player, args.min_ships, args.lookback, args.target_limit)


if __name__ == "__main__":
    main()
