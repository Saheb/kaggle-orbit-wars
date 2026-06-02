"""Land-grab benchmark opponent — an aggressive opening EXPANDER.

Purpose: a measurement tool, not a leaderboard agent. Our RL agent's weakness
(found via LB replay 78328277) is that it loses the opening expansion race and
gets snowballed. Zach/HB don't punish this. This bot does exactly what beat us:
every turn, from every owned planet, capture the nearest affordable neutral/weak
planet — grab the board fast, let production compound.

Strategy (deliberately simple, but effective at out-expanding a passive agent):
  - Each step, each owned planet sends a fleet to capture the cheapest reachable
    uncontested neutral (or weak enemy) planet it can afford.
  - Force sent = defender ships at arrival (ships + production*travel) + buffer,
    capped so the source keeps a small garrison.
  - Skip targets whose straight-line path crosses the sun.
  - Don't double-send to a planet already targeted by one of our in-flight fleets.

Returns FleetOrders: list of [source_planet_id, angle_radians, ships].
"""
import math
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    CENTER, SUN_RADIUS, distance, point_to_segment_distance,
)

SHIP_SPEED_MAX = 6.0
GARRISON = 1          # ships to leave behind on a source planet (commit hard to win the race)
CAPTURE_BUFFER = 2    # extra ships over the estimated defender garrison (thin = more planets)
MAX_DISTANCE = 55     # don't bother with very distant targets


def _fleet_speed(ships):
    ships = max(1, ships)
    return min(SHIP_SPEED_MAX, 1.0 + (SHIP_SPEED_MAX - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5)


_SUN = (CENTER, CENTER)  # sun is at board center (50, 50)


def _crosses_sun(sx, sy, tx, ty):
    # True if the segment source->target passes within the sun's radius of center.
    return point_to_segment_distance(_SUN, (sx, sy), (tx, ty)) < (SUN_RADIUS + 1.0)


def agent(obs, cfg=None):
    try:
        me = obs['player']
        comet_ids = set(obs.get('comet_planet_ids', []))
        # planet = [id, owner, x, y, radius, ships, production]
        planets = [p for p in obs['planets'] if p[0] not in comet_ids]
        fleets = obs.get('fleets', [])

        owned = [p for p in planets if p[1] == me]
        targets = [p for p in planets if p[1] != me]   # neutral (-1) or enemy
        if not owned or not targets:
            return []

        # Planets already targeted by one of our in-flight fleets (fleet[1]=owner, fleet[5]=dest? )
        # Fleet layout: [id, owner, x, y, angle, ?, ships] — no reliable dest id, so we
        # de-dupe by claiming targets within THIS turn's plan instead.
        claimed = set()
        orders = []

        for src in sorted(owned, key=lambda p: -p[5]):   # richest planets act first
            sid, _, sx, sy, _, s_ships, _ = src
            avail = s_ships - GARRISON
            if avail <= 1:
                continue
            # candidate targets this source can afford, cheapest-effective first
            best = None
            best_key = None
            for t in targets:
                tid, towner, tx, ty, _, t_ships, t_prod = t
                if tid in claimed:
                    continue
                d = distance((sx, sy), (tx, ty))
                if d > MAX_DISTANCE or _crosses_sun(sx, sy, tx, ty):
                    continue
                travel = d / _fleet_speed(avail)
                need = int(t_ships + t_prod * travel) + CAPTURE_BUFFER
                if need > avail:
                    continue
                # prefer: cheap to take, then close, then productive
                key = (need, d, -t_prod)
                if best_key is None or key < best_key:
                    best_key, best = key, (t, need)
            if best is None:
                continue
            t, need = best
            tid, _, tx, ty, _, _, _ = t
            angle = math.atan2(ty - sy, tx - sx)
            orders.append([sid, angle, int(need)])
            claimed.add(tid)

        return orders
    except Exception:
        return []
