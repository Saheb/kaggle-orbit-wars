
import math
def agent(obs, config=None):
    me = obs["player"] if isinstance(obs, dict) else obs.player
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    fleets = obs["fleets"] if isinstance(obs, dict) else obs.fleets
    mine = [p for p in planets if int(p[1]) == int(me)]
    if not mine: return []
    # Identify which of my planets has an incoming enemy fleet within 40 units
    threats = set()
    for f in fleets:
        if int(f[1]) == int(me): continue
        for p in mine:
            if math.hypot(float(f[2])-float(p[2]), float(f[3])-float(p[3])) < 40:
                threats.add(int(p[0]))
                break
    moves = []
    for src in mine:
        sid, _, sx, sy, sr, ships, _ = src
        if int(ships) < 8: continue
        if int(sid) in threats:
            # Reinforce: send to nearest threatened ally
            best = None
            for p in mine:
                if int(p[0]) == int(sid) or int(p[0]) not in threats: continue
                d = math.hypot(float(p[2])-float(sx), float(p[3])-float(sy))
                if best is None or d < best[0]:
                    best = (d, float(p[2]), float(p[3]), int(p[0]))
            if best is not None:
                _, tx, ty, tid = best
                ang = math.atan2(ty-float(sy), tx-float(sx))
                send = max(int(ships)//2, 1)
                moves.append([int(sid), float(ang), int(send)])
            continue
        # Otherwise attack nearest non-ally
        best = None
        for tgt in planets:
            tid, towner, tx, ty, tr, tships, _ = tgt
            if int(tid) == int(sid) or int(towner) == int(me): continue
            d = math.hypot(float(tx)-float(sx), float(ty)-float(sy))
            if best is None or d < best[0]:
                best = (d, float(tx), float(ty), int(tid), int(tships))
        if best is None: continue
        _, tx, ty, tid, tships = best
        ang = math.atan2(ty-float(sy), tx-float(sx))
        send = min(int(ships)-1, tships+2)
        if send > 0: moves.append([int(sid), float(ang), int(send)])
    return moves
