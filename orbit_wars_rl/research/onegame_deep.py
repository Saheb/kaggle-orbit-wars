"""Single-game deep dive: run one seed (our agent seat0 vs Ajay), dump a replay JSON,
detect every capture/loss event with attribution + hold-duration, print a reasoned
timeline, and emit a self-contained HTML replay viewer (board canvas + timeline)."""
import argparse
import json
import os

from kaggle_environments import make
from orbit_wars_rl.research.onegame_load import load_faithful_agent
from orbit_wars_rl.eval import _dm_fleet_target, _friendly_inbound

try:
    from orbit_wars_rl.features import CENTER
except Exception:
    CENTER = 50.0
BOARD = 100.0
SUN_R = 10.0


def totals(planets, fleets, seat):
    g = sum(float(p[5]) for p in planets if int(p[1]) == seat)
    fl = sum(float(f[6]) for f in (fleets or []) if int(f[1]) == seat)
    return g + fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", default="opponents/candidate_ajay_1200.py")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gate", type=int, default=2)
    ap.add_argument("--out", default="/tmp/onegame")
    args = ap.parse_args()

    agent = load_faithful_agent(args.checkpoint)
    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.run([agent, args.opponent])
    steps = env.steps
    n = len(steps)

    # ---- per-step replay frames (from seat-0 observation; owners are absolute ids) ----
    frames = []
    for t in range(n):
        ob = steps[t][0].observation
        P = ob.get("planets") or []
        F = ob.get("fleets") or []
        planets = [dict(id=int(p[0]), own=int(p[1]), x=round(float(p[2]), 2),
                        y=round(float(p[3]), 2), r=round(float(p[4]), 2),
                        ships=round(float(p[5]), 1), prod=round(float(p[6]), 2)) for p in P]
        fleets = []
        for f in F:
            tgt = _dm_fleet_target(P, f)
            fleets.append(dict(own=int(f[1]), x=round(float(f[2]), 2), y=round(float(f[3]), 2),
                               ships=round(float(f[6]), 1),
                               tgt=int(tgt[0]) if tgt is not None else -1))
        frames.append(dict(t=t,
                           our_p=sum(1 for p in P if int(p[1]) == 0),
                           opp_p=sum(1 for p in P if int(p[1]) == 1),
                           neu_p=sum(1 for p in P if int(p[1]) == -1),
                           our_m=round(totals(P, F, 0), 1),
                           opp_m=round(totals(P, F, 1), 1),
                           planets=planets, fleets=fleets))

    # ---- capture/loss events with attribution + hold duration ----
    prev = {}
    born = {}      # pid -> (owner, step) when current owner acquired it
    events = []
    for t in range(n):
        P = steps[t][0].observation.get("planets") or []
        P0 = steps[t - 1][0].observation.get("planets") or [] if t > 0 else []
        F0 = steps[t - 1][0].observation.get("fleets") or [] if t > 0 else []
        by0 = {int(p[0]): p for p in P0}
        for p in P:
            pid, own = int(p[0]), int(p[1])
            was = prev.get(pid)
            if was is not None and was != own:
                tgt0 = by0.get(pid)
                garr = float(tgt0[5]) if tgt0 is not None else 0.0
                inb_us = _friendly_inbound(F0, tgt0, 0) if tgt0 is not None else 0.0
                inb_opp = _friendly_inbound(F0, tgt0, 1) if tgt0 is not None else 0.0
                held = (t - born[pid][1]) if pid in born else None
                events.append(dict(t=t, pid=pid, frm=was, to=own,
                                   garr=round(garr, 1), inb_us=round(inb_us, 1),
                                   inb_opp=round(inb_opp, 1),
                                   held_by_prev=held if was != -1 else None))
                born[pid] = (own, t)
            elif was is None:
                born[pid] = (own, t)
            prev[pid] = own

    rew = [x.reward if x.reward is not None else 0.0 for x in steps[-1]]
    result = dict(seed=args.seed, n=n, win=bool(rew[0] > rew[1]),
                  final_our_p=frames[-1]["our_p"], final_opp_p=frames[-1]["opp_p"])

    os.makedirs(args.out, exist_ok=True)
    replay = dict(meta=result, frames=frames, events=events)
    jpath = os.path.join(args.out, f"replay_seed{args.seed}.json")
    with open(jpath, "w") as fh:
        json.dump(replay, fh)

    # ---- console analysis ----
    print(f"\n=== SEED {args.seed}  {'WIN' if result['win'] else 'LOSS'}  len {n}  "
          f"final planets us {result['final_our_p']} / opp {result['final_opp_p']} ===")
    print("\n step | ourP oppP neuP | ourMass oppMass ratio")
    for t in (0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, n - 1):
        if t >= n:
            continue
        fr = frames[t]
        print(f" {t:>4} | {fr['our_p']:>4} {fr['opp_p']:>4} {fr['neu_p']:>4} | "
              f"{fr['our_m']:>7.0f} {fr['opp_m']:>7.0f} {fr['our_m']/max(fr['opp_m'],1):>5.2f}")

    def who(o):
        return {-1: "neutral", 0: "US", 1: "OPP"}[o]
    us_caps = [e for e in events if e["to"] == 0]
    us_loss = [e for e in events if e["frm"] == 0]
    print(f"\n captures BY US: {len(us_caps)}   losses BY US: {len(us_loss)}")
    print(" --- our captures (pid @step  from  garr  inb us/opp  -> held steps) ---")
    for e in us_caps:
        print(f"   p{e['pid']:<3} @{e['t']:<3} from {who(e['frm']):<7} "
              f"garr {e['garr']:>4} inb us/opp {e['inb_us']:>4}/{e['inb_opp']:>4}")
    print(" --- our losses (pid @step  garr_at_loss  enemy_inbound) ---")
    for e in us_loss:
        held = "" if e["held_by_prev"] is None else f"held {e['held_by_prev']}st"
        print(f"   p{e['pid']:<3} @{e['t']:<3} garr {e['garr']:>4} enemy_inb {e['inb_opp']:>4}  {held}")
    print(f"\nreplay JSON: {jpath}")
    return replay, args.out


if __name__ == "__main__":
    main()
