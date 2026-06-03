"""
Early Aggressor — fires from all owned planets toward nearest non-owned target
every step from step 0. Dumb but consistently aggressive opening.
Used to test/train against the opening paralysis vulnerability.
"""
import math


def agent(obs, cfg=None):
    if not isinstance(obs, dict):
        obs = {
            "step": int(getattr(obs, "step", 0)),
            "player": int(getattr(obs, "player", 0)),
            "planets": [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
                        for p in obs.planets],
            "fleets": [[f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships]
                       for f in obs.fleets],
        }

    player = obs["player"]
    planets = obs["planets"]  # [id, owner, x, y, radius, ships, production]

    owned = [p for p in planets if p[1] == player]
    targets = [p for p in planets if p[1] != player]

    if not owned or not targets:
        return []

    fleets = []
    for planet in owned:
        ships = planet[5]
        if ships < 8:
            continue  # keep minimum garrison

        # Find nearest non-owned planet
        px, py = planet[2], planet[3]
        nearest = min(targets, key=lambda t: (t[2]-px)**2 + (t[3]-py)**2)

        # Fire half the fleet — enough to capture but keep some reserve
        launch = max(ships // 2, min(ships - 5, ships))
        if launch > 0:
            # Action format: [source_planet_id, target_planet_id, num_ships]
            # (angle-decode format — will be converted by env)
            angle = math.atan2(nearest[3] - py, nearest[2] - px)
            fleets.append([planet[0], angle % (2 * math.pi), launch])

    return fleets
