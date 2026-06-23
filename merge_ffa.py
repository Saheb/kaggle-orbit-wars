"""Merge per-shard JSON dumps from run_ffa_panel.py --dump into one ranking.
Usage: merge_ffa.py shard0.json shard1.json ..."""
import json
import sys

wins, place_sum, games = {}, {}, 0
labels = None
for path in sys.argv[1:]:
    with open(path) as f:
        d = json.load(f)
    labels = d["labels"]
    for l in labels:
        wins[l] = wins.get(l, 0) + d["wins"][l]
        place_sum[l] = place_sum.get(l, 0) + d["place_sum"][l]
    games += d["games"]

print("=" * 60)
print(f"FFA mixed-field panel (merged) — {games} games")
print("=" * 60)
print(f"{'agent':<16}{'win-rate':>14}{'mean place':>14}")
for l in sorted(labels, key=lambda x: -wins[x]):
    print(f"{l:<16}{wins[l]}/{games} ({100*wins[l]/games:>5.1f}%){place_sum[l]/games:>10.2f}")
print("(win-rate = 1st-place share, the FFA LB metric; mean place 1=best 4=worst)")
