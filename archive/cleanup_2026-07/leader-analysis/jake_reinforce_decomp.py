"""One-off: reinforce forward/backward decomposition for the winner of a single replay (81327125).

Reinforce = an owned->owned launch (target planet owned by the same seat at decision time).
Each launch's target is resolved with the SAME lead-aware collision resolver as eval.py
(_resolve_launch_target / _lead_collision_target, validated 98.4% vs true swept-collision),
copied here verbatim so the script runs without torch. Reinforces are split by DIRECTION using
the two definitions already in eval.py:

  (A) vs enemy CENTROID (eval reinf_fwd/reinf_rear, +/-3 deadband):
        forward  = target closer to enemy centroid than source  (dT < dS - 3)
        backward = target farther from enemy centroid than source (dT > dS + 3)
        lateral  = within the +/-3 deadband
  (B) vs NEAREST enemy planet (eval reinf_fwde):
        forward  = target nearer to its nearest enemy planet than source is (deT < deS)

Timing: action at steps[t] was decided on observation steps[t-1] (reference_replay_obs_action_timing).
Schema: planet [id,owner,x,y,radius,ships,prod]; action [from_id, angle, num_ships]; owner -1=neutral.
"""
import json
import math
import os

# --- engine/eval constants (BOARD_SIZE=100, ROTATION_RADIUS_LIMIT=50, MAX_SHIP_SPEED=6) ---
CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
_DM_MAX_SPEED = 6.0
_CONV_ANGVEL = 0.0   # set per-game from obs angular_velocity (orbital lead over each flight)

REPLAY = os.path.join(os.path.dirname(__file__), "81327125.json")


# ----- resolver, copied verbatim from eval.py -----
def _ship_speed_py(ships):
    s = max(float(ships), 1.0)
    base = (math.log(s) / math.log(1000.0)) ** 1.5
    return min(1.0 + (_DM_MAX_SPEED - 1.0) * base, _DM_MAX_SPEED)


def _planet_pos_at(p, t):
    dx, dy = p[2] - CENTER, p[3] - CENTER
    orb = math.hypot(dx, dy)
    if orb + p[4] >= ROTATION_RADIUS_LIMIT:
        return p[2], p[3]
    ph = math.atan2(dy, dx) + _CONV_ANGVEL * t
    return CENTER + orb * math.cos(ph), CENTER + orb * math.sin(ph)


def _lead_collision_target(planets, x, y, angle, ships, skip_pid=None):
    c, sn = math.cos(angle), math.sin(angle)
    speed = max(_ship_speed_py(ships), 1e-6)
    best, best_eta = None, None
    for p in planets:
        if skip_pid is not None and p[0] == skip_pid:
            continue
        pr = p[4]
        eta = max(0.0, (math.hypot(p[2] - x, p[3] - y) - pr) / speed)
        for _ in range(4):
            lx, ly = _planet_pos_at(p, eta)
            eta = max(0.0, (math.hypot(lx - x, ly - y) - pr) / speed)
        lx, ly = _planet_pos_at(p, eta)
        vx, vy = lx - x, ly - y
        along = vx * c + vy * sn
        if along <= 0:
            continue
        perp = abs(vx * sn - vy * c)
        if perp < pr + 0.5 and (best_eta is None or eta < best_eta):
            best_eta, best = eta, p
    return best


def _resolve_launch_target(planets, src, angle, ships):
    sx = src[2] + math.cos(angle) * (src[4] + 0.1)
    sy = src[3] + math.sin(angle) * (src[4] + 0.1)
    return _lead_collision_target(planets, sx, sy, angle, ships, skip_pid=src[0])


def decomp(path, seat):
    global _CONV_ANGVEL
    d = json.load(open(path))
    steps = d["steps"]
    names = d["info"]["TeamNames"]
    rewards = d["rewards"]
    T = len(steps)

    # orbit rate (constant per game) from the first obs that carries it
    for _s in steps:
        if seat < len(_s):
            av = _s[seat]["observation"].get("angular_velocity")
            if av is not None:
                _CONV_ANGVEL = float(av)
                break

    reinf = atk = void = 0
    fwd = rear = lat = 0
    fwde = bwde = 0
    reinf_ships = atk_ships = 0
    fwd_ships = rear_ships = lat_ships = 0

    for t in range(1, T):
        obs = steps[t - 1][seat]["observation"]
        acts = steps[t][seat]["action"] or []
        p0 = obs.get("planets") or []
        if not p0:
            continue
        byid = {p[0]: p for p in p0}
        enemy = [p for p in p0 if int(p[1]) != seat and int(p[1]) >= 0]
        ecx = sum(p[2] for p in enemy) / len(enemy) if enemy else None
        ecy = sum(p[3] for p in enemy) / len(enemy) if enemy else None

        for mv in acts:
            if not mv or len(mv) < 3:
                continue
            src = byid.get(int(mv[0]))
            if src is None:
                continue
            sent, ssh = int(mv[2]), float(src[5])
            if not (ssh > 0 and sent <= ssh):
                continue
            tgt = _resolve_launch_target(p0, src, float(mv[1]), sent)
            if tgt is None:
                void += 1
                continue
            if int(tgt[1]) == seat:                 # ---- reinforce ----
                reinf += 1
                reinf_ships += sent
                if ecx is not None:
                    dS = math.hypot(src[2] - ecx, src[3] - ecy)
                    dT = math.hypot(tgt[2] - ecx, tgt[3] - ecy)
                    if dT < dS - 3:
                        fwd += 1; fwd_ships += sent
                    elif dT > dS + 3:
                        rear += 1; rear_ships += sent
                    else:
                        lat += 1; lat_ships += sent
                if enemy:
                    deS = min(math.hypot(src[2] - e[2], src[3] - e[3]) for e in enemy)
                    deT = min(math.hypot(tgt[2] - e[2], tgt[3] - e[3]) for e in enemy)
                    if deT < deS:
                        fwde += 1
                    else:
                        bwde += 1
            else:                                   # ---- attack/capture ----
                atk += 1
                atk_ships += sent

    pct = lambda n, dn: (100.0 * n / dn) if dn else 0.0
    print(f"Replay {os.path.basename(path)}  episode {d['info']['EpisodeId']}  (angvel={_CONV_ANGVEL:.4f})")
    print(f"Seat {seat} = {names[seat]}  (reward {rewards[seat]:+d} -> "
          f"{'WIN' if rewards[seat] > 0 else 'LOSS'}),  opponent = {names[1 - seat]}")
    print()
    tot = reinf + atk
    print(f"Total resolved launches: {tot}   (+{void} flew-to-void, skipped)")
    print(f"  attack/capture: {atk} ({pct(atk, tot):.0f}%)   reinforce: {reinf} ({pct(reinf, tot):.0f}%)")
    print(f"  ships -- attack: {atk_ships}   reinforce: {reinf_ships}")
    print()
    print(f"REINFORCE decomposition  (n={reinf} launches, {reinf_ships} ships)")
    print("  (A) direction vs ENEMY CENTROID  (+/-3 deadband):")
    print(f"      forward  (toward enemy): {fwd:3d} launches ({pct(fwd, reinf):3.0f}%)   "
          f"{fwd_ships:4d} ships ({pct(fwd_ships, reinf_ships):3.0f}%)")
    print(f"      backward (toward rear):  {rear:3d} launches ({pct(rear, reinf):3.0f}%)   "
          f"{rear_ships:4d} ships ({pct(rear_ships, reinf_ships):3.0f}%)")
    print(f"      lateral  (|d|<=3):       {lat:3d} launches ({pct(lat, reinf):3.0f}%)   "
          f"{lat_ships:4d} ships ({pct(lat_ships, reinf_ships):3.0f}%)")
    print("  (B) direction vs NEAREST ENEMY PLANET:")
    print(f"      forward  (nearer enemy planet): {fwde:3d} ({pct(fwde, reinf):3.0f}%)")
    print(f"      backward (farther):             {bwde:3d} ({pct(bwde, reinf):3.0f}%)")


if __name__ == "__main__":
    decomp(REPLAY, seat=0)   # seat 0 = Jake Will (winner)
