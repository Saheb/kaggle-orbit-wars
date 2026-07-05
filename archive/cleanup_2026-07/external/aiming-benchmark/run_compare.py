"""Compare OUR inference aimer (action_mask._target_intercept_angle) vs the
benchmark's reference aimer, on a subset of the aim benchmark. Engine-scored.

Usage: python run_compare.py [N]   (N = number of samples, default 1500)
"""
import sys, os, io, contextlib, logging, math, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "orbit_wars_rl"))

logging.disable(logging.WARNING)
with contextlib.redirect_stdout(io.StringIO()):
    import kaggle_environments  # noqa
logging.disable(logging.NOTSET)

import aim_benchmark as ab

# --- OUR aimer (the exact function the exported agent / eval uses) ---
from action_mask import _target_intercept_angle

def our_aim(obs, source, target, fleet_size):
    planets = obs["planets"]
    src = next((p for p in planets if int(p[0]) == source), None)
    tgt = next((p for p in planets if int(p[0]) == target), None)
    if src is None or tgt is None:
        return None
    return _target_intercept_angle(src, tgt, fleet_size, obs)

# --- reference aimer (copied from the benchmark notebook) ---
CENTER = 50.0; ROTATION_RADIUS_LIMIT = 50.0; DEFAULT_MAX_SHIP_SPEED = 6.0
LAUNCH_OFFSET = 0.1; SUN_RADIUS = 10.0; _LEAD_ITERS = 8

def _fleet_speed_ref(ships, max_speed):
    ships = max(1.0, float(ships))
    speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5
    return min(speed, max_speed)

def _seg_circle_blocked(ax, ay, bx, by, cx, cy, r):
    dx, dy = bx - ax, by - ay
    L2 = dx*dx + dy*dy
    if L2 <= 1e-12:
        return math.hypot(cx-ax, cy-ay) < r
    t = max(0.0, min(1.0, ((cx-ax)*dx + (cy-ay)*dy) / L2))
    return math.hypot(cx-(ax+t*dx), cy-(ay+t*dy)) < r

def _planet(obs, pid):
    for row in obs["planets"]:
        if int(row[0]) == int(pid):
            return row
    return None

def ref_aim(obs, source, target, fleet_size, avoid=True):
    src = _planet(obs, source); tgt = _planet(obs, target)
    if src is None or tgt is None:
        return None if avoid else 0.0
    sx, sy, s_r = float(src[2]), float(src[3]), float(src[4])
    tx0, ty0, t_r = float(tgt[2]), float(tgt[3]), float(tgt[4])
    omega = float(obs.get("angular_velocity", 0.0))
    max_speed = float(obs.get("ship_speed", DEFAULT_MAX_SHIP_SPEED))
    speed = _fleet_speed_ref(fleet_size, max_speed)
    dx0, dy0 = tx0 - CENTER, ty0 - CENTER
    orbit_r = math.hypot(dx0, dy0)
    static = (orbit_r + t_r) >= ROTATION_RADIUS_LIMIT
    phase0 = math.atan2(dy0, dx0)
    def target_at(t):
        if static: return tx0, ty0
        a = phase0 + omega * t
        return CENTER + orbit_r*math.cos(a), CENTER + orbit_r*math.sin(a)
    gap = s_r + LAUNCH_OFFSET + t_r
    t = max(0.0, (math.hypot(tx0-sx, ty0-sy) - gap) / speed)
    for _ in range(_LEAD_ITERS):
        px, py = target_at(t)
        t = max(0.0, (math.hypot(px-sx, py-sy) - gap) / speed)
    px, py = target_at(t)
    angle = math.atan2(py-sy, px-sx)
    if not avoid:
        return angle
    lx = sx + math.cos(angle)*(s_r+LAUNCH_OFFSET); ly = sy + math.sin(angle)*(s_r+LAUNCH_OFFSET)
    if _seg_circle_blocked(lx, ly, px, py, CENTER, CENTER, SUN_RADIUS):
        return None
    return angle

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
samples = list(ab.iter_samples())[:N]
reachable = sum(1 for s in samples if s.meta.get("reachable", True))
print(f"samples: {len(samples)}  (reachable={reachable}, impossible={len(samples)-reachable})")

for name, fn in [("OURS (_target_intercept_angle)", our_aim), ("REFERENCE (8-iter lead+sun-decline)", ref_aim)]:
    t0 = time.time()
    angles = [fn(s.obs, s.source, s.target, s.fleet_size) for s in samples]
    res = [ab._validate_one(s, a) for s, a in zip(samples, angles)]
    # split accuracy by reachable vs impossible
    rok = sum(r for r, s in zip(res, samples) if s.meta.get("reachable", True))
    iok = sum(r for r, s in zip(res, samples) if not s.meta.get("reachable", True))
    ndecl = sum(1 for a in angles if a is None)
    print(f"{name}: {sum(res)}/{len(res)} = {sum(res)/len(res):.1%}  "
          f"| reachable {rok}/{reachable}={rok/max(reachable,1):.1%}  "
          f"impossible {iok}/{len(samples)-reachable}={iok/max(len(samples)-reachable,1):.1%}  "
          f"| declined={ndecl}  ({time.time()-t0:.0f}s)")
