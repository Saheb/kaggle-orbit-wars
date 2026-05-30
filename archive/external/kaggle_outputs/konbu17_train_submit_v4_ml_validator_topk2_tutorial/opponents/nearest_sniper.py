
import math
def agent(obs, config=None):
    me = obs["player"] if isinstance(obs, dict) else obs.player
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    moves = []
    for src in planets:
        sid, owner, sx, sy, sr, ships, _ = src
        if int(owner) != int(me) or int(ships) < 2: continue
        best_pid, best_d, best_pxy = -1, 1e9, None
        for tgt in planets:
            tid, towner, tx, ty, tr, tships, _ = tgt
            if int(tid) == int(sid): continue
            if int(towner) == int(me): continue
            d = math.hypot(float(tx)-float(sx), float(ty)-float(sy))
            if d < best_d:
                best_d, best_pid, best_pxy = d, int(tid), (float(tx), float(ty), int(tships))
        if best_pid < 0 or best_pxy is None: continue
        ang = math.atan2(best_pxy[1]-float(sy), best_pxy[0]-float(sx))
        send = min(int(ships)-1, best_pxy[2]+1)
        if send > 0: moves.append([int(sid), float(ang), int(send)])
    return moves
