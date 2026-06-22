"""Why does the policy send 1-4 ship launches (ship-head bins 0-3 = the
`--min-ship-bin 4` target)? From LOGGED replay actions: is the source garrison
small (forced — all it has) or large (deliberate undersend)? Split by behind/ahead
and won/lost.

  python orbit_wars_rl/ship0_why.py <replay_dir...> --our-name Saheb

obs[t-1] is the decision-time state for action[t]. Pairs with
ship0_counterfactual.py (model re-run: where the bin mass goes when 0-3 masked).
Run from repo root. Replays may come from probe_aggregation.py (won/ lost/ dirs,
name "OURS") or the held-out watcher (name "Saheb"); won/lost is read from rewards.
"""
import argparse
import glob
import json
import statistics
from collections import Counter

BUCKETS = ["g<=4 (forced)", "g5-10", "g11-25", "g>25 (big)"]


def gbucket(g):
    if g <= 4:
        return "g<=4 (forced)"
    if g <= 10:
        return "g5-10"
    if g <= 25:
        return "g11-25"
    return "g>25 (big)"


def material(obs, seat):
    """Our total ships (garrisons + fleets) vs the enemy's — for behind/ahead."""
    ours = sum(float(p[5]) for p in obs.get("planets", []) if int(p[1]) == seat)
    ours += sum(float(f[6]) for f in obs.get("fleets", []) if int(f[1]) == seat)
    enemy = sum(float(p[5]) for p in obs.get("planets", []) if int(p[1]) >= 0 and int(p[1]) != seat)
    enemy += sum(float(f[6]) for f in obs.get("fleets", []) if int(f[1]) >= 0 and int(f[1]) != seat)
    return ours, enemy


def scan(path, our_name, acc):
    d = json.load(open(path))
    names = d["info"]["TeamNames"]
    if our_name not in names:
        return
    seat = names.index(our_name)
    steps = d["steps"]
    res = "WON" if d["rewards"][seat] > d["rewards"][1 - seat] else "LOST"
    a = acc[res]
    for t in range(1, len(steps)):
        obs = steps[t - 1][seat].get("observation") or {}
        planets = obs.get("planets") or []
        if not planets:
            continue
        action = steps[t][seat].get("action") or []
        if not action:
            continue
        ours, enemy = material(obs, seat)
        behind = ours < enemy
        by_id = {int(p[0]): p for p in planets}
        for mv in action:
            if len(mv) < 3:
                continue
            src, ships = int(mv[0]), int(mv[2])
            sp = by_id.get(src)
            if sp is None or int(sp[1]) != seat:
                continue
            g = float(sp[5])
            b = gbucket(g)
            a["tot"] += 1
            a["by_g"][b] += 1
            a["tot_behind"] += int(behind)
            a["tot_ahead"] += int(not behind)
            if ships <= 4:                       # ship0 = bins 0-3 (1-4 ships)
                a["ship0"] += 1
                a["s0_by_g"][b] += 1
                a["s0_behind"] += int(behind)
                a["s0_ahead"] += int(not behind)
                if g > 10:
                    a["big"].append(g)


def report(res, a):
    tot, s0 = a["tot"], a["ship0"]
    if not tot:
        print(f"\n===== {res} =====  (no launches)")
        return
    print(f"\n===== {res} =====  launches={tot}  ship0(1-4 ships)={s0} ({100 * s0 / tot:.0f}%)")
    print("  ship0 RATE by source garrison bucket:")
    for b in BUCKETS:
        n, s = a["by_g"][b], a["s0_by_g"][b]
        print(f"    {b:16}: {s:5}/{n:<5} = {100 * s / max(n, 1):3.0f}% ship0  "
              f"({100 * n / tot:2.0f}% of launches)")
    big = a["big"]
    extra = f"  (median garrison {statistics.median(big):.0f} -> deliberate undersend)" if big else ""
    print(f"  ship0 from a BIG (>10) garrison: {len(big)}/{s0} = {100 * len(big) / max(s0, 1):.0f}%{extra}")
    print(f"  ship0 when BEHIND {100 * a['s0_behind'] / max(a['tot_behind'], 1):.0f}%  |  "
          f"AHEAD {100 * a['s0_ahead'] / max(a['tot_ahead'], 1):.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--our-name", default="OURS", help="team name of our seat (e.g. Saheb)")
    args = ap.parse_args()
    paths = []
    for dd in args.dirs:
        paths += glob.glob(dd + "/*.json") if not dd.endswith(".json") else [dd]
    acc = {r: {"tot": 0, "ship0": 0, "by_g": Counter(), "s0_by_g": Counter(),
               "tot_behind": 0, "tot_ahead": 0, "s0_behind": 0, "s0_ahead": 0, "big": []}
           for r in ("WON", "LOST")}
    for p in sorted(paths):
        scan(p, args.our_name, acc)
    for r in ("WON", "LOST"):
        report(r, acc[r])


if __name__ == "__main__":
    main()
