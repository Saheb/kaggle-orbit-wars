#!/usr/bin/env python3
"""Merge sharded panel record pickles into one full-panel report.

Each shard is produced by:
    eval.py --panel --panel-shards K --panel-shard-idx i --shard-out shard_i.pkl
which runs a deterministic 1/K partition of the fixed 256-game panel and pickles its
per-game records. This script replays ALL records from ALL shards through the SAME
accumulation a full panel uses (eval._accumulate_panel_records → add_conversion), so
the printed report is byte-for-byte what a single-process `--panel` would print — just
produced in parallel across cores.

Usage:
    python orbit_wars_rl/merge_panel_shards.py <shard_dir_or_pkls...> [--opponent NAME]
"""
import argparse
import glob
import io
import logging
import pickle
import sys

# Importing eval pulls in kaggle_environments, which prints a wall of OpenSpiel INFO
# lines to stdout on import — suppress so the merged report (tee'd to report.txt) is clean.
logging.disable(logging.CRITICAL)
_real_stdout = sys.stdout
sys.stdout = io.StringIO()
from orbit_wars_rl.eval import _accumulate_panel_records, print_panel_report  # noqa: E402
sys.stdout = _real_stdout
logging.disable(logging.NOTSET)


def _sum_flat(dicts):
    """Sum a list of flat {str: number} stat dicts (defensive_reinforce / natural_head_audit)."""
    out: dict = {}
    for d in dicts:
        for k, v in (d or {}).items():
            out[k] = out.get(k, 0) + v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+",
                    help="shard pickle files, globs, or a directory containing shard_*.pkl")
    ap.add_argument("--opponent", default="ajay", help="label for the report header")
    args = ap.parse_args()

    paths: list[str] = []
    for s in args.shards:
        import os
        if os.path.isdir(s):
            paths.extend(sorted(glob.glob(os.path.join(s, "shard_*.pkl"))))
        else:
            paths.extend(sorted(glob.glob(s)) or [s])
    if not paths:
        sys.exit(f"No shard pickles found in {args.shards}")

    records: list = []
    defr, nat = [], []
    for p in paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        records.extend(d["records"])
        defr.append(d.get("defensive_reinforce", {}))
        nat.append(d.get("natural_head_audit", {}))

    print(f"Merged {len(paths)} shards → {len(records)} games", file=sys.stderr)
    result = _accumulate_panel_records(records)
    result["defensive_reinforce"] = _sum_flat(defr)
    result["natural_head_audit"] = _sum_flat(nat)
    print_panel_report(result, args.opponent)


if __name__ == "__main__":
    main()
