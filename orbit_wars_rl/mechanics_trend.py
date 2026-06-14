#!/usr/bin/env python3
"""Extract the winner-referenced conversion mechanics from a set of per-checkpoint
eval logs and print them as a trend table — so we can see whether a run is IMPROVING
on the metrics that matter (planets@50, opening conversion, peel-rate, hoard, reinf<50),
not just on win-rate.

Usage:  mechanics_trend.py <eval_log_glob...>
  e.g.  mechanics_trend.py gpu_run_artifacts/stageb2r/eval_logs/eval_torch_step_*.log
Winner refs (Isaiah/top players): planets@50 9 · open<50 cap/atk 0.51 · peel 0.43 ·
  garr_frac@50 0.54 · ships/pl@50 22 · reinf<50 0.29 · launch_rate 0.036
"""
import glob
import re
import sys

R = {
    "wr":     re.compile(r"^Overall:\s+\d+/\d+\s+\(([\d.]+)%\)", re.M),
    "won":    re.compile(r"WON\((\d+)g\)\s+(\d+)/(\d+)/(\d+)/(\d+)\s+cap/atk open<50 ([\d.]+) mid50-100 ([\d.]+)"),
    "peel":   re.compile(r"WON\(\d+g\)\s+peel-rate ([\d.]+) hold"),
    "hoard":  re.compile(r"garr_frac@ [\d.]+/[\d.]+/([\d.]+)/[\d.]+\s+ships/planet@ \d+/\d+/(\d+)/\d+"),
    "reinfe": re.compile(r"reinf by step\s+<50:([\d.]+)"),
    "lr":     re.compile(r"WON\(\d+g\)\s+lr ([\d.]+) ff"),
}
STEP = re.compile(r"torch_step_(\d+)_")


def grab(rx, t, *g):
    m = rx.search(t)
    return tuple(m.group(i) for i in g) if m else tuple("—" for _ in g)


rows = []
for f in sorted(set(sum((glob.glob(a) for a in sys.argv[1:]), []))):
    sm = STEP.search(f)
    if not sm:
        continue
    t = open(f).read()
    if "Conversion:" not in t:
        continue
    step = int(sm.group(1))
    (wr,) = grab(R["wr"], t, 1)
    won = R["won"].search(t)
    p50 = won.group(4) if won else "—"            # planets@50 (WON)
    opn = won.group(6) if won else "—"            # open<50 cap/atk (WON)
    (peel,) = grab(R["peel"], t, 1)
    gf50, spp50 = grab(R["hoard"], t, 1, 2)       # garr_frac@50, ships/planet@50
    (rfe,) = grab(R["reinfe"], t, 1)
    (lr,) = grab(R["lr"], t, 1)
    rows.append((step, wr, p50, opn, peel, gf50, spp50, rfe, lr))

rows.sort(key=lambda r: r[0])
hdr = ("step", "WR%", "pl@50", "open<50", "peel", "garrf50", "shipspp50", "reinf<50", "launchR")
ref = ("winner→", "", "9", "0.51", "0.43", "0.54", "22", "0.29", "0.036")
w = [max(len(str(x)) for x in col) for col in zip(hdr, ref, *[tuple(map(str, r)) for r in rows])]
fmt = "  ".join("{:>%d}" % wi for wi in w)
print(fmt.format(*hdr))
print(fmt.format(*ref))
print("-" * (sum(w) + 2 * len(w)))
for r in rows:
    print(fmt.format(*[f"{r[0]/1e6:.2f}M" if i == 0 else str(v) for i, v in enumerate(r)]))
