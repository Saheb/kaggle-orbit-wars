"""Re-derive panel metrics from a saved --panel-out pickle — no game re-run.

`eval.py --panel` throws away the raw per-game records after printing, so a metric
added later (e.g. dm overkill/med) can't be recovered from an old run's text. Run
the panel once with `--panel-out FILE` (it still prints normally) and this script
re-prints the full conversion + tiered summary from the saved records — including
any metric added to _fmt_conversion / _fmt_tier_summary since the run.

    python orbit_wars_rl/recompute_panel.py <panel.pkl>
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval as ev  # noqa: E402


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python orbit_wars_rl/recompute_panel.py <panel.pkl>")
    with open(sys.argv[1], "rb") as f:
        blob = pickle.load(f)
    records = blob["records"]
    opp = blob.get("opponent", "?")
    res = ev._accumulate_panel_records(records)      # re-aggregate with the CURRENT metric code
    ov = res["overall"]
    conv = res["conversion"]
    print(f"Recomputed from {len(records)} games vs {opp}  "
          f"(win-rate {ov['wins']}/{ov['total']} = {100*ov['wins']/max(ov['total'],1):.1f}%)")
    print(ev._fmt_conversion(conv))
    print(ev._fmt_tier_summary(conv))


if __name__ == "__main__":
    main()
