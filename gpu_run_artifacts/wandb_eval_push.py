"""Push held-out eval panel rows to wandb from the eval watcher.

The training process owns its own wandb run on the GPU instance; the held-out panel runs
LOCALLY in run_watchers.sh _eval and only wrote CSVs — this bridges the gap. Rows land in a
companion run (id "<run>_eval_<opp>", same project) with eval/step (env steps) as the
x-axis via define_metric, so panels overlay cleanly on the training run's charts.

Usage:
  one row:  wandb_eval_push.py --run tl100m --opp ajay_1200 --step 10158080 --wr 14.5 [...]
  backfill: wandb_eval_push.py --run tl100m --opp ajay_1200 --backfill path/to/eval_ajay_1200.csv

Never raise: the watcher must not die on wandb hiccups (exit 0 always, errors to stdout).
"""
from __future__ import annotations

import argparse
import csv
import sys

FIELDS = {  # CSV column / CLI flag -> wandb key (numeric only; NA/ERR/blank skipped)
    "win_rate": "eval/win_rate",
    "seat0_wr": "eval/seat0_wr",
    "seat1_wr": "eval/seat1_wr",
    "outmassed_pct": "eval/outmassed_pct",
    "open_capatk_WON": "eval/open_capatk_won",
    "mid_capatk_WON": "eval/mid_capatk_won",
    "peelrate_WON": "eval/peelrate_won",
    "planets100_WON": "eval/planets100_won",
    # Launch-discipline diagnostics parsed from the eval log (not in the CSV — the CSV
    # schema is stable/consumed elsewhere; these ride wandb-only). The tl100m tripwire set.
    "caps_game": "eval/caps_per_game",
    "atk_launch_game": "eval/atk_launch_per_game",
    "cap_atk": "eval/cap_per_atk",
    "cap_atk_open": "eval/cap_per_atk_open50",
    "cap_atk_mid": "eval/cap_per_atk_mid",
    "ships_cap": "eval/ships_per_cap",
    "reinf_share": "eval/reinf_share",
    "launch_rate": "eval/launch_rate",
    "fire_frac": "eval/fire_frac",
    "ship0_pct": "eval/ship0_pct",
}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--opp", required=True)
    p.add_argument("--project", default="orbit-wars")
    p.add_argument("--backfill", default=None, help="CSV path: push every row, then exit")
    p.add_argument("--step", type=int)
    for col in FIELDS:
        p.add_argument(f"--{col}", default=None)
    args = p.parse_args()

    rows = []
    if args.backfill:
        with open(args.backfill) as f:
            for r in csv.DictReader(f):
                step = _num(r.get("step"))
                if step is None:
                    continue
                rows.append((int(step), {k: _num(r.get(c)) for c, k in FIELDS.items()}))
        rows.sort()
    else:
        if args.step is None:
            print("[wandb-push] no --step; skipping")
            return
        rows = [(args.step, {k: _num(getattr(args, c)) for c, k in FIELDS.items()})]

    import wandb
    run = wandb.init(
        project=args.project,
        id=f"{args.run}_eval_{args.opp}",
        name=f"{args.run}_eval_{args.opp}",
        resume="allow",
        settings=wandb.Settings(silent=True, init_timeout=60),
    )
    run.define_metric("eval/step")
    run.define_metric("eval/*", step_metric="eval/step")
    n = 0
    for step, vals in rows:
        payload = {"eval/step": step, **{k: v for k, v in vals.items() if v is not None}}
        if len(payload) > 1:
            run.log(payload)
            n += 1
    run.finish()
    print(f"[wandb-push] logged {n} row(s) to {args.run}_eval_{args.opp}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # never kill the watcher over telemetry
        print(f"[wandb-push] FAILED (non-fatal): {e}")
    sys.exit(0)
