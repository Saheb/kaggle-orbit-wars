"""Measure Ajay's fire-spare rate on the EXACT opportunity set the COMA Q-head needs.

The off-policy-Q plan (docs/q-head.md, post-gate) bets that Ajay rollouts would
supervise the counterfactual Q is missing: "firing the idle spare at a contested
neutral helps." That bet is only live if Ajay ACTUALLY fires spares at those
opportunities — if Ajay also leaves them idle (and wins on retention/timing
instead), his rollouts won't supply the missing supervision and the plan is moot.

This script scans existing Ajay-vs-us replays (no new games) and measures, on the
SAME spare-fire opportunity set `value_spare_diagnostic._spare_sources_for_neutral`
identifies for the Q-head gate:

  - the idle-spare rate (the 80%-idle number from multi_source_why, but for Ajay)
  - fired-spare rate, split by bucket (af1 = solo-takeable, agg = needs pooling)
  - won/lost split (do Ajay's wins look like idle-spare, the way ours do?)

Replays must carry TeamNames so we can locate Ajay's seat. The replays_2097152/
dir (probe_aggregation.py output) tags Ajay as "Ajay" and us as "Saheb".

Run from repo root:
  python3 orbit_wars_rl/ajay_fire_spare.py gpu_run_artifacts/r32_stage_hlr/replays_2097152/
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# --- pure-math inlines of eval.py helpers (torch-free so this scans replays
#     without pulling torch_env / torch). Constants mirror torch_env / features.
_CENTER = 50.0
_ROTATION_RADIUS_LIMIT = 50.0
_CONV_ANGVEL = 0.0               # eval._CONV_ANGVEL — orbits static in projection
_DM_MAX_SPEED = 6.0
_ETA_PROBE_SPEED = 1.0 + (_DM_MAX_SPEED - 1.0) * (math.log(20) / math.log(1000.0)) ** 1.5
_MAX_ETA = 25                    # value_spare_diagnostic._MAX_ETA
STEP_LO, STEP_HI = 10, 50        # value_spare_diagnostic STEP window


def _ship_speed_py(ships):
    s = max(float(ships), 1.0)
    base = (math.log(s) / math.log(1000.0)) ** 1.5
    return min(1.0 + (_DM_MAX_SPEED - 1.0) * base, _DM_MAX_SPEED)


def _planet_pos_at(p, t):
    dx, dy = p[2] - _CENTER, p[3] - _CENTER
    orb = math.hypot(dx, dy)
    if orb + p[4] >= _ROTATION_RADIUS_LIMIT:
        return p[2], p[3]
    ph = math.atan2(dy, dx) + _CONV_ANGVEL * t
    return _CENTER + orb * math.cos(ph), _CENTER + orb * math.sin(ph)


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


def _dm_fleet_target(planets, f):
    return _lead_collision_target(planets, f[2], f[3], f[4], f[6])


def _enemy_threat(planets, fleets, tgt, seat):
    ein = 0.0
    eta = math.inf
    for f in (fleets or []):
        o = int(f[1])
        if o < 0 or o == seat:
            continue
        r = _dm_fleet_target(planets, f)
        if r is None or r[0] != tgt[0]:
            continue
        ein += f[6]
        d = math.hypot(tgt[2] - f[2], tgt[3] - f[3])
        eta = min(eta, d / max(_ship_speed_py(f[6]), 1e-6))
    return ein, eta


def _cap_cost_at_arrival(src, tgt, seat):
    owner = int(tgt[1])
    if owner == seat:
        return 0.0
    dist = math.hypot(tgt[2] - src[2], tgt[3] - src[3])
    eta = max(1.0, math.ceil(dist / _ETA_PROBE_SPEED))
    ships_at_arrival = min(tgt[5] + tgt[6] * eta, 500.0)
    return ships_at_arrival + (1.0 if owner == -1 else tgt[6] * 3 + 1.0)


def _spare_sources_for_neutral(neutral, planets, fleets, seat):
    """Owned planets that (i) can reach `neutral` within _MAX_ETA and (ii) hold
    spare garrison (ships - own enemy-inbound threat > 0). Returns list of
    (planet_idx_in_obs, spare_ships, cost_to_capture). Mirrors
    value_spare_diagnostic._spare_sources_for_neutral exactly."""
    out = []
    nx, ny = float(neutral[2]), float(neutral[3])
    for pidx, p in enumerate(planets):
        if int(p[1]) != seat:
            continue
        dist = math.hypot(nx - float(p[2]), ny - float(p[3]))
        eta = max(1.0, math.ceil(dist / _ETA_PROBE_SPEED))
        if eta > _MAX_ETA:
            continue
        threat, _ = _enemy_threat(planets, fleets, p, seat)
        spare = max(0.0, float(p[5]) - threat)
        if spare <= 0.0:
            continue
        cost = _cap_cost_at_arrival(p, neutral, seat)
        out.append((pidx, spare, cost))
    return out


def _ajay_seat(replay):
    names = replay.get("info", {}).get("TeamNames", [])
    if "Ajay" in names:
        return names.index("Ajay")
    return None


def _outcome(replay, ajay_seat):
    rewards = replay.get("rewards") or [s.reward for s in replay["steps"][-1]]
    if rewards is None:
        return "unknown"
    r_a = rewards[ajay_seat] or 0.0
    r_o = rewards[1 - ajay_seat] or 0.0
    return "won" if r_a > r_o else ("lost" if r_a < r_o else "draw")


def _scan_replay(replay, ajay_seat):
    """For each step in [STEP_LO, STEP_HI], enumerate contested neutrals with
    spare sources (Ajay's), and record whether each spare source fired at that
    neutral. Returns rows: (bucket, fired_at_neutral, fired_anywhere, outcome)."""
    rows = []
    outcome = _outcome(replay, ajay_seat)
    steps = replay["steps"]
    for t in range(len(steps)):
        agent_state = steps[t][ajay_seat]
        obs = agent_state.get("observation")
        if obs is None:
            continue
        step = int(obs.get("step", t))
        if step < STEP_LO or step > STEP_HI:
            continue
        planets = obs["planets"]
        fleets = obs["fleets"]
        action = agent_state.get("action") or []
        # action entries: [from_planet_id, angle, ships]
        fired_from_ids = {int(a[0]) for a in action}
        # map from_planet_id -> target planet (via angle + collision geometry is
        # expensive; the Q-gate question is whether the spare fired *at all* vs
        # was idle. We also want "fired at THIS neutral" for the per-neutral
        # read. Decode target via _lead_collision_target like the action decoder.
        fired_targets = {}  # from_planet_id -> target planet_id
        for a in action:
            fid = int(a[0])
            src = _find_planet_by_id(planets, fid)
            if src is None:
                continue
            angle = a[1]
            ships = int(a[2])
            # mirror eval._lead_collision_target to find the target planet
            tgt_id = _decode_target_id(planets, src, angle, ships)
            fired_targets[fid] = tgt_id

        for nidx, neutral in enumerate(planets):
            if int(neutral[1]) >= 0:
                continue  # only neutrals
            sources = _spare_sources_for_neutral(neutral, planets, fleets, ajay_seat)
            if not sources:
                continue
            total_spare = sum(s for _, s, _ in sources)
            cheapest = min(c for _, _, c in sources)
            solo = any(s >= c for _, s, c in sources)
            bucket = "af1" if solo else ("agg" if total_spare >= cheapest else "cap")
            if bucket == "cap":
                continue
            neutral_id = int(neutral[0])
            for pidx, spare, cost in sources:
                src_id = int(planets[pidx][0])
                fired_any = src_id in fired_from_ids
                fired_at = (fired_targets.get(src_id) == neutral_id)
                rows.append({
                    "bucket": bucket,
                    "fired_at_neutral": fired_at,
                    "fired_anywhere": fired_any,
                    "outcome": outcome,
                    "step": step,
                })
    return rows


def _find_planet_by_id(planets, pid):
    for p in planets:
        if int(p[0]) == pid:
            return p
    return None


def _decode_target_id(planets, src, angle, ships):
    """Mirror eval._lead_collision_target to find which planet a launch hits."""
    sx = src[2] + math.cos(angle) * (src[4] + 0.1)
    sy = src[3] + math.sin(angle) * (src[4] + 0.1)
    tgt = _lead_collision_target(planets, sx, sy, angle, ships, skip_pid=src[0])
    return int(tgt[0]) if tgt is not None else None


def _report(rows, label):
    print("\n" + "=" * 88)
    print(f"AJAY FIRE-SPARE  ({label})")
    print("=" * 88)
    if not rows:
        print("  no spare-fire opportunities found in range")
        return
    n = len(rows)
    idle = sum(1 for r in rows if not r["fired_anywhere"])
    fired_any = n - idle
    fired_at = sum(1 for r in rows if r["fired_at_neutral"])
    print(f"opportunities (spare-source slots): {n}")
    print(f"  IDLE (fired nothing):    {idle} ({100*idle/n:.1f}%)")
    print(f"  FIRED (at anything):     {fired_any} ({100*fired_any/n:.1f}%)")
    print(f"  FIRED AT THIS NEUTRAL:   {fired_at} ({100*fired_at/n:.1f}%)  <-- the fire-spare rate")
    print()
    print(f"{'bucket':>8} | {'n':>5} | {'idle%':>6} | {'fired-anywhere%':>15} | {'fired-at-neutral%':>17}")
    print("-" * 66)
    for b in ["af1", "agg"]:
        sel = [r for r in rows if r["bucket"] == b]
        if not sel:
            continue
        nb = len(sel)
        idb = sum(1 for r in sel if not r["fired_anywhere"])
        fanb = sum(1 for r in sel if r["fired_anywhere"])
        fatb = sum(1 for r in sel if r["fired_at_neutral"])
        print(f"{b:>8} | {nb:>5} | {100*idb/nb:>5.1f} | {100*fanb/nb:>14.1f} | {100*fatb/nb:>16.1f}")
    print()
    print(f"{'outcome':>8} | {'n':>5} | {'idle%':>6} | {'fired-anywhere%':>15} | {'fired-at-neutral%':>17}")
    print("-" * 66)
    for o in ["won", "lost", "draw", "unknown"]:
        sel = [r for r in rows if r["outcome"] == o]
        if not sel:
            continue
        no = len(sel)
        ido = sum(1 for r in sel if not r["fired_anywhere"])
        fano = sum(1 for r in sel if r["fired_anywhere"])
        fato = sum(1 for r in sel if r["fired_at_neutral"])
        print(f"{o:>8} | {no:>5} | {100*ido/no:>5.1f} | {100*fano/no:>14.1f} | {100*fato/no:>16.1f}")
    print()
    print("Reference (our agent, hlr 2M, from multi_source_why / current_problem.md):")
    print("  idle-spare ~80%  (the wall the Q-head was built to break)")
    print("  If Ajay idle-spare is ALSO ~80% -> Ajay rollouts won't supply the missing")
    print("  counterfactual supervision -> off-policy-Q from Ajay is moot for aggregation.")
    print("  If Ajay idle-spare is materially LOWER -> Ajay fires spares we leave idle ->")
    print("  his rollouts DO supervise the counterfactual -> off-policy-Q plan holds.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="replay dirs (won/ and lost/ or flat)")
    ap.add_argument("--ajay-name", default="Ajay")
    ap.add_argument("--min-steps", type=int, default=10,
                    help="min steps in replay to scan (skip truncated games)")
    args = ap.parse_args()

    files = []
    for d in args.dirs:
        if os.path.isdir(d):
            for root, _, fs in os.walk(d):
                for f in fs:
                    if f.endswith(".json"):
                        files.append(os.path.join(root, f))
        elif d.endswith(".json"):
            files.append(d)
    files.sort()
    print(f"scanning {len(files)} replay files", flush=True)

    all_rows = []
    n_with_ajay = 0
    n_scanned = 0
    outcomes = Counter()
    for path in files:
        with open(path) as f:
            replay = json.load(f)
        seat = _ajay_seat(replay)
        if seat is None:
            continue
        n_with_ajay += 1
        outcomes[_outcome(replay, seat)] += 1
        if len(replay["steps"]) < args.min_steps:
            continue
        n_scanned += 1
        all_rows.extend(_scan_replay(replay, seat))
        if n_scanned % 20 == 0:
            print(f"  scanned {n_scanned} replays, {len(all_rows)} opp slots so far", flush=True)

    print(f"\nreplays with Ajay: {n_with_ajay}  scanned (len>={args.min_steps}): {n_scanned}")
    print(f"Ajay outcomes: {dict(outcomes)}")
    _report(all_rows, f"{n_scanned} replays, steps {STEP_LO}-{STEP_HI}")


if __name__ == "__main__":
    main()
