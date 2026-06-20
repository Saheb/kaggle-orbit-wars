"""Find comeback episodes: 1v1 games where the eventual WINNER was behind mid-game
(real enemy pressure → forced to defend/reinforce) and recovered to win.

Winner-only BC over top-by-score games is dominated by snowball blowouts, so the
winner is rarely under threat and reactive-defense reinforcement is nearly absent
(~75% of winner save moves go to unthreatened planets). Comeback episodes are where
the reactive-defense skill actually shows up in the winner's actions.

Material is read from the GLOBAL planet owner ids in one seat's observation:
  planet [id, owner, x, y, radius, ships, production].

Usage:
  python orbit_wars_rl/find_comeback_replays.py <replay_dir> [<replay_dir> ...] \
      [--opening-cutoff 25] [--out-list comebacks.txt] \
      [--min-planet-deficit 1] [--min-ship-share-deficit 0.10]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orbit_wars_rl.build_replay_action_bc import _winner_seat  # noqa: E402


def _material(planets, seat):
    ships = sum(float(p[5]) for p in planets if int(p[1]) == seat)
    count = sum(1 for p in planets if int(p[1]) == seat)
    return ships, count


def analyze(path: Path, opening_cutoff: int):
    """Return per-episode comeback stats, or None if not a usable 1v1 replay."""
    try:
        d = json.load(open(path))
    except Exception:
        return None
    win = _winner_seat(d)
    if win is None:
        return None
    lose = 1 - win
    steps = d.get("steps") or []
    if len(steps) < opening_cutoff + 5:
        return None

    worst_ship_share = 1.0    # winner ship fraction at its lowest (after cutoff)
    worst_planet_def = 0      # max (loser_planets - winner_planets) after cutoff
    worst_step = -1
    behind_steps = 0
    seen = 0
    for k in range(opening_cutoff, len(steps)):
        seat_cell = steps[k][win] if win < len(steps[k]) else None
        obs = (seat_cell or {}).get("observation") or {}
        planets = obs.get("planets")
        if not planets:
            continue
        wships, wcount = _material(planets, win)
        lships, lcount = _material(planets, lose)
        seen += 1
        total = wships + lships
        share = wships / total if total > 0 else 1.0
        pdef = lcount - wcount
        if share < 0.5 or pdef > 0:
            behind_steps += 1
        if share < worst_ship_share:
            worst_ship_share = share
        if pdef > worst_planet_def:
            worst_planet_def = pdef
            worst_step = k
    if seen == 0:
        return None
    return {
        "path": str(path),
        "len": len(steps),
        "worst_ship_share": worst_ship_share,      # < 0.5 means winner was out-shipped
        "ship_share_deficit": max(0.0, 0.5 - worst_ship_share),
        "worst_planet_deficit": worst_planet_def,   # >0 means winner held fewer planets
        "worst_step": worst_step,
        "behind_frac": behind_steps / seen,         # fraction of post-opening steps behind
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--opening-cutoff", type=int, default=25)
    ap.add_argument("--min-planet-deficit", type=int, default=1)
    ap.add_argument("--min-ship-share-deficit", type=float, default=0.10)
    ap.add_argument("--out-list", default=None, help="write paths of qualifying comebacks here")
    args = ap.parse_args()

    paths = []
    for d in args.dirs:
        paths.extend(sorted(Path(d).rglob("*.json")))

    rows = []
    for p in paths:
        r = analyze(p, args.opening_cutoff)
        if r is not None:
            rows.append(r)

    n = len(rows)
    print(f"usable 1v1 replays: {n} / {len(paths)} files  (opening-cutoff t>={args.opening_cutoff})")
    if n == 0:
        return

    # Distribution of how far behind the eventual winner ever got.
    pd = Counter(r["worst_planet_deficit"] for r in rows)
    print("\nmax planet-deficit the winner overcame (loser_planets - winner_planets):")
    for k in sorted(pd):
        print(f"  behind by {k:>2} planets : {pd[k]:4d}  ({100*pd[k]/n:4.1f}%)")

    for thr in (0.0, 0.05, 0.10, 0.17, 0.25):
        c = sum(1 for r in rows if r["ship_share_deficit"] >= thr)
        print(f"  winner out-shipped by >= {thr:.2f} share at worst : {c:4d}  ({100*c/n:4.1f}%)")

    # Combined comeback definition.
    comeback = [
        r for r in rows
        if r["worst_planet_deficit"] >= args.min_planet_deficit
        or r["ship_share_deficit"] >= args.min_ship_share_deficit
    ]
    print(f"\nCOMEBACKS (>= {args.min_planet_deficit} planet-deficit OR "
          f">= {args.min_ship_share_deficit:.2f} ship-share-deficit): "
          f"{len(comeback)} / {n}  ({100*len(comeback)/n:.1f}%)")
    if comeback:
        avg_behind = sum(r["behind_frac"] for r in comeback) / len(comeback)
        print(f"  mean fraction of post-opening steps spent behind: {avg_behind:.2f}")

    if args.out_list:
        Path(args.out_list).write_text("\n".join(r["path"] for r in comeback) + "\n")
        print(f"  wrote {len(comeback)} comeback paths -> {args.out_list}")


if __name__ == "__main__":
    main()
