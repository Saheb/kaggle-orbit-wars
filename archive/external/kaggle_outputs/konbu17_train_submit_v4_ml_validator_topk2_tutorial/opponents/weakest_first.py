
import math
def agent(obs, config=None):
    me = obs["player"] if isinstance(obs, dict) else obs.player
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    moves = []
    for src in planets:
        sid, owner, sx, sy, sr, ships, _ = src
        if int(owner) != int(me) or int(ships) < 5: continue
        candidates = []
        for tgt in planets:
            tid, towner, tx, ty, tr, tships, _ = tgt
            if int(tid) == int(sid) or int(towner) == int(me): continue
            candidates.append((int(tships), float(tx), float(ty), int(tid)))
        if not candidates: continue
        candidates.sort()
        ts, tx, ty, tid = candidates[0]
        ang = math.atan2(ty-float(sy), tx-float(sx))
        send = min(int(ships)-1, ts+2)
        if send > 0: moves.append([int(sid), float(ang), int(send)])
    return moves
