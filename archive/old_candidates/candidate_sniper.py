"""Simple nearest-planet sniper baseline for eval panels."""

from __future__ import annotations

import math


def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets

    my_planets = [p for p in raw_planets if int(p[1]) == player]
    targets = [p for p in raw_planets if int(p[1]) != player]
    if not my_planets or not targets:
        return []

    moves = []
    for mine in my_planets:
        source_ships = int(mine[5])
        if source_ships <= 1:
            continue

        nearest = min(
            targets,
            key=lambda t: math.hypot(float(mine[2]) - float(t[2]), float(mine[3]) - float(t[3])),
        )
        ships = min(source_ships - 1, int(nearest[5]) + 1)
        if ships <= 0:
            continue

        angle = math.atan2(float(nearest[3]) - float(mine[3]), float(nearest[2]) - float(mine[2]))
        moves.append([int(mine[0]), angle, ships])

    return moves
