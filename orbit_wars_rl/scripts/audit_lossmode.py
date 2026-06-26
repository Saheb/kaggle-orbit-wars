"""Loss-mode attribution (Phase 5 review): decompose corrpack3e's lost captures vs Ajay into
economy / position / concentration, to test whether synchronized concentration (Phase 5) is the
actual bottleneck or whether out-massing is an economy/positioning problem waves can't fix.

Builds on hold_autopsy's loss detection (ABANDONED / OUT-MASSED / TOO-LATE). For each OUT-MASSED
loss (the dominant 95%+ bucket), measured on the decision-time obs (t-1, same geometry eval uses):

  deficit         = enemy_inbound - our_garrison           (extra mass needed to hold)
  reachable       = drainable owned garrison on OTHER planets that can reach the target within the
                    enemy's arrival window  (eval._reachable_drainable_attack_mass) — i.e. mass we
                    COULD have concentrated but didn't
  global_ratio    = our total ships (garrison+fleets) / enemy total, at the loss step

Classification of each OUT-MASSED loss:
  ECONOMY        global_ratio < 0.8                     -> globally out-produced; concentration can't save it
  CONCENTRATION  reachable >= deficit (and not economy) -> mass existed & was reachable, uncommitted (Phase 5 target)
  POSITION       else                                   -> global mass ok but unreachable in time (routing/placement)
"""
import argparse
import math
import statistics
from collections import Counter

from kaggle_environments import make

from orbit_wars_rl.eval import _friendly_inbound, _reachable_drainable_attack_mass, _dm_fleet_target, _ship_speed_py
from orbit_wars_rl.scripts.expansion_probe import load_ckpt_agent


def _enemy_eta_to(planets, fleets, tgt, enemy):
    """Min ETA (steps) of any enemy fleet currently headed at tgt; inf if none."""
    best = math.inf
    for f in (fleets or []):
        if int(f[1]) != enemy:
            continue
        r = _dm_fleet_target(planets, f)
        if r is not None and int(r[0]) == int(tgt[0]):
            dist = math.hypot(tgt[2] - f[2], tgt[3] - f[3])
            best = min(best, dist / max(_ship_speed_py(f[6]), 1e-6))
    return best


def _totals(planets, fleets, seat):
    g = sum(float(p[5]) for p in planets if int(p[1]) == seat)
    fl = sum(float(f[6]) for f in (fleets or []) if int(f[1]) == seat)
    return g + fl


def autopsy_game(steps, seat, enemy):
    prev, cap_step = {}, {}
    out_massed = []   # dicts for OUT-MASSED losses
    modes = Counter()
    total_caps = 0
    for t in range(1, len(steps)):
        if seat >= len(steps[t]) or seat >= len(steps[t - 1]):
            continue
        p0 = steps[t - 1][seat].observation.get("planets")
        p1 = steps[t][seat].observation.get("planets")
        if not p1:
            continue
        byid0 = {p[0]: p for p in (p0 or [])}
        f0 = steps[t - 1][seat].observation.get("fleets") or []
        for p in p1:
            pid, own = p[0], int(p[1])
            was = prev.get(pid)
            if was is not None and was != seat and own == seat:
                total_caps += 1
                cap_step[pid] = t
            elif was == seat and own != seat and pid in cap_step:
                tgt0 = byid0.get(pid)
                garr_loss = float(tgt0[5]) if tgt0 else 0.0
                enemy_in = _friendly_inbound(f0, tgt0, enemy) if tgt0 else 0.0
                our_reinf = _friendly_inbound(f0, tgt0, seat) if tgt0 else 0.0
                if garr_loss <= 2:
                    modes["ABANDONED"] += 1
                elif enemy_in > garr_loss:
                    modes["OUT-MASSED"] += 1
                    eta = _enemy_eta_to(p0, f0, tgt0, enemy)
                    window = max(eta if math.isfinite(eta) else 8.0, 4.0)
                    reach, nsrc = _reachable_drainable_attack_mass(p0, f0, tgt0, seat, window)
                    deficit = enemy_in - garr_loss
                    our_tot = _totals(p0, f0, seat)
                    enemy_tot = _totals(p0, f0, enemy)
                    gratio = our_tot / max(enemy_tot, 1.0)
                    out_massed.append(dict(deficit=deficit, reach=reach, nsrc=nsrc,
                                           gratio=gratio, enemy_in=enemy_in, garr=garr_loss))
                elif our_reinf > 0:
                    modes["TOO-LATE"] += 1
                else:
                    modes["OTHER"] += 1
                del cap_step[pid]
            prev[pid] = own
    return modes, out_massed, total_caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", default="opponents/candidate_ajay_1200.py")
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--gate", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    agent = load_ckpt_agent(args.checkpoint, args.gate)
    modes = Counter()
    om = []
    total_caps = 0
    game_losses = 0
    elim_losses = 0
    TRK = [25, 50, 75, 100, 150]
    traj = {"win": {t: [] for t in TRK}, "loss": {t: [] for t in TRK}}     # mass ratio
    pl = {"win": {t: [] for t in TRK}, "loss": {t: [] for t in TRK}}       # our planet count
    for s in range(args.seed0, args.seed0 + args.games):
        env = make("orbit_wars", configuration={"seed": s}, debug=False)
        env.run([agent, args.opponent])
        steps = env.steps
        last = steps[-1][0].observation.get("planets") or []
        us = sum(1 for p in last if int(p[1]) == 0)
        opp = sum(1 for p in last if int(p[1]) == 1)
        fleets = steps[-1][0].observation.get("fleets") or []
        our_mat = _totals(last, fleets, 0)
        enemy_mat = _totals(last, fleets, 1)
        is_loss = our_mat <= enemy_mat
        if is_loss:
            game_losses += 1
            if our_mat <= 0 or us == 0:
                elim_losses += 1
        # mass-ratio + planet-count trajectory at fixed steps, split by eventual outcome
        bucket = "loss" if is_loss else "win"
        for tk in TRK:
            if tk < len(steps) and 0 < len(steps[tk]):
                ob = steps[tk][0].observation
                P, Fl = ob.get("planets") or [], ob.get("fleets") or []
                our = _totals(P, Fl, 0); en = _totals(P, Fl, 1)
                traj[bucket][tk].append(our / max(en, 1.0))
                pl[bucket][tk].append(sum(1 for p in P if int(p[1]) == 0))
        m, o, caps = autopsy_game(steps, seat=0, enemy=1)
        modes += m
        om += o
        total_caps += caps

    n = sum(modes.values())
    import os
    print(f"\n=== LOSS-MODE ATTRIBUTION: {os.path.basename(args.checkpoint)} vs {args.opponent}")
    print(f"    {args.games} games, gate{args.gate}  |  captures {total_caps}  lost {n}  peel-rate {n/max(total_caps,1):.2f}")
    print(f"    GAME losses: {game_losses}/{args.games}  (of which eliminations / material->0: {elim_losses})")
    print("\n    lost-capture loss modes:")
    for mname in ("ABANDONED", "OUT-MASSED", "TOO-LATE", "OTHER"):
        c = modes.get(mname, 0)
        print(f"      {mname:11s} {c:4d}  ({100*c/max(n,1):4.1f}%)")

    if not om:
        print("\n    (no OUT-MASSED losses to decompose)")
        return
    econ = [e for e in om if e["gratio"] < 0.8]
    rest = [e for e in om if e["gratio"] >= 0.8]
    conc = [e for e in rest if e["reach"] >= e["deficit"]]
    posn = [e for e in rest if e["reach"] < e["deficit"]]
    N = len(om)
    print(f"\n    OUT-MASSED decomposition (n={N}):")
    print(f"      ECONOMY        {len(econ):4d}  ({100*len(econ)/N:4.1f}%)   global mass ratio < 0.8")
    print(f"      CONCENTRATION  {len(conc):4d}  ({100*len(conc)/N:4.1f}%)   reachable owned mass >= deficit, uncommitted")
    print(f"      POSITION       {len(posn):4d}  ({100*len(posn)/N:4.1f}%)   global ok but unreachable in time")
    med = lambda xs, k: (statistics.median([e[k] for e in xs]) if xs else 0)
    print(f"\n    medians over ALL OUT-MASSED: deficit {med(om,'deficit'):.0f}  reachable {med(om,'reach'):.0f}  "
          f"nsrc {med(om,'nsrc'):.0f}  global_ratio {med(om,'gratio'):.2f}")
    print(f"    medians CONCENTRATION bucket: deficit {med(conc,'deficit'):.0f}  reachable {med(conc,'reach'):.0f}  nsrc {med(conc,'nsrc'):.0f}")
    print(f"    medians ECONOMY bucket:       global_ratio {med(econ,'gratio'):.2f}  deficit {med(econ,'deficit'):.0f}")

    print("\n    MASS-RATIO trajectory (our total / enemy total), mean by eventual outcome:")
    print(f"      {'step':>6} | {'WIN ratio':>10} {'(planets)':>9} | {'LOSS ratio':>10} {'(planets)':>9}")
    for tk in TRK:
        wr = traj["win"][tk]; lr = traj["loss"][tk]
        wp = pl["win"][tk]; lp = pl["loss"][tk]
        wmean = statistics.mean(wr) if wr else float("nan")
        lmean = statistics.mean(lr) if lr else float("nan")
        wpm = statistics.mean(wp) if wp else float("nan")
        lpm = statistics.mean(lp) if lp else float("nan")
        print(f"      {tk:>6} | {wmean:>10.2f} {wpm:>9.1f} | {lmean:>10.2f} {lpm:>9.1f}")


if __name__ == "__main__":
    main()
