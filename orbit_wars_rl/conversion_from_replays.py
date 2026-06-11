"""Top-player CONVERSION baseline from downloaded replays — same `game_conversion`
the eval uses, so the numbers are directly comparable (eval == replay). Groups games
by TeamName and prints the conversion summary (incl. the new `churn` + `redundant-launch`
metrics) per player.

    python conversion_from_replays.py /tmp/fresh_validate /tmp/snowball --player "Isaiah @ Tufa Labs"
    python conversion_from_replays.py /tmp/fresh_validate            # all players, ranked by n

Replay steps are plain dicts; game_conversion uses attribute access (it normally runs on
kaggle env.steps Structs), so each per-seat step is wrapped in a SimpleNamespace.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import game_conversion, new_conversion_acc, add_conversion, _fmt_conversion


def _wrap(steps):
    """kaggle replay dict-steps → attribute-accessible (.observation/.action) for game_conversion."""
    return [[SimpleNamespace(observation=(ps.get("observation") or {}),
                             action=ps.get("action")) for ps in step] for step in steps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="replay dirs of *.json")
    ap.add_argument("--player", default=None, help="only this TeamName (else every player)")
    ap.add_argument("--min-games", type=int, default=20, help="skip players with fewer games")
    args = ap.parse_args()

    accs = defaultdict(new_conversion_acc)
    for d in args.dirs:
        for f in glob.glob(os.path.join(d, "*.json")):
            try:
                data = json.load(open(f))
            except Exception:
                continue
            steps = data.get("steps") or []
            names = data.get("info", {}).get("TeamNames", [])
            if len(steps) < 5 or not names:
                continue
            wrapped = _wrap(steps)
            for seat, name in enumerate(names):
                if args.player and name != args.player:
                    continue
                if seat >= len(steps[0]):
                    continue
                conv = game_conversion(wrapped, seat)
                add_conversion(accs[name], conv)

    rows = sorted(accs.items(), key=lambda kv: -kv[1]["games"])
    for name, acc in rows:
        if acc["games"] < args.min_games:
            continue
        print(f"\n=== {name}  (n={acc['games']}) ===")
        print(_fmt_conversion(acc))


if __name__ == "__main__":
    main()
