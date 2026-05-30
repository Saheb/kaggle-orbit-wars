
import math
def agent(obs, config=None):
    me = obs["player"] if isinstance(obs, dict) else obs.player
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    moves = []
    for src in planets:
        sid, owner, sx, sy, sr, ships, _ = src
        if int(owner) != int(me) or int(ships) < 5: continue
        best = None
        for tgt in planets:
            tid, towner, tx, ty, tr, tships, prod = tgt
            if int(tid) == int(sid) or int(towner) == int(me): continue
            score = float(prod) - 0.05 * int(tships)
            if best is None or score > best[0]:
                best = (score, float(tx), float(ty), int(tid), int(tships))
        if best is None: continue
        _, tx, ty, tid, tships = best
        ang = math.atan2(ty-float(sy), tx-float(sx))
        send = min(int(ships)-1, tships+3)
        if send > 0: moves.append([int(sid), float(ang), int(send)])
    return moves
