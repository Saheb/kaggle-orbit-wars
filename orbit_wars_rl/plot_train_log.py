#!/usr/bin/env python3
"""Parse a train_torch.py text log into smoothed trend plots — the offline twin
of the W&B smoothing slider. Reads the per-iter `iter` lines + every-5th `diag`
lines (and optionally the held-out eval CSV) and renders raw (faint) + EMA (bold)
curves so the SIGNAL (trend) is separated from single-rollout NOISE.

Usage:
  plot_train_log.py <train.log> [--eval eval_zach_public.csv] [--ema 0.85] [-o out.png]
"""
import argparse
import csv
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- regexes for the two console line types -------------------------------
ITER_RE = re.compile(
    r"^iter\s+\d+\s+\|\s+steps\s+([\d,]+).*?EV\s+([-\d.]+).*?KL\s+([-\d.]+).*?"
    r"clip\s+([-\d.]+).*?H_fire\s+([-\d.]+)(?:.*?il_kl\s+([-\d.]+))?")
DIAG_RE = re.compile(
    r"diag .*?fire_frac\s+([-\d.]+)\s+owned\s+([-\d.]+)\s+ship0\s+([-\d.]+)\s+"
    r"meanshipbin\s+([-\d.]+).*?pl@16/32/50/100\s+(\d+)/(\d+)/(\d+)/(\d+)\s+"
    r"garrfrac@50\s+([-\d.]+)\s+shipspp@50\s+([-\d.]+)")
# reinforce-by-step (only present with --allow-reinforce): early<50 is the back-loaded signal
# (winner ramp 0.29/0.41/0.31; we run too low early). Parsed separately off the same diag line.
REINF_RE = re.compile(r"step<50/50-100/>100\s+([\d.]+)/")


def _f(x):
    return float(x.replace(",", "")) if x else None


def parse(path):
    iters = {k: [] for k in ("step", "ev", "kl", "clip", "hfire", "ilkl")}
    diags = {k: [] for k in ("step", "fire_frac", "owned", "ship0", "meanshipbin",
                             "p16", "p32", "p50", "p100", "garr50", "spp50", "reinf_e")}
    last_step = 0.0
    for line in open(path):
        m = ITER_RE.search(line)
        if m:
            last_step = _f(m.group(1))
            for key, g in zip(("step", "ev", "kl", "clip", "hfire", "ilkl"),
                              (last_step, *(_f(m.group(i)) for i in range(2, 7)))):
                iters[key].append(g)
            continue
        d = DIAG_RE.search(line)
        if d:
            vals = [last_step] + [_f(d.group(i)) for i in range(1, 11)]
            for key, v in zip(diags, vals):    # covers step..spp50 (11 keys)
                diags[key].append(v)
            r = REINF_RE.search(line)
            diags["reinf_e"].append(_f(r.group(1)) if r else None)
    return iters, diags


def ema(xs, alpha):
    out, acc = [], None
    for x in xs:
        if x is None:
            out.append(acc)
            continue
        acc = x if acc is None else alpha * acc + (1 - alpha) * x
        out.append(acc)
    return out


def panel(ax, x, y, label, alpha, color, ylim=None):
    if not any(v is not None for v in y):
        return
    ax.plot(x, y, color=color, alpha=0.18, lw=0.9)             # raw = noise
    ax.plot(x, ema(y, alpha), color=color, lw=2.0, label=label)  # ema = signal
    ax.set_title(label, fontsize=9)
    ax.grid(alpha=0.2)
    if ylim:
        ax.set_ylim(*ylim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--eval", help="held-out eval CSV (step,win_rate)")
    ap.add_argument("--ema", type=float, default=0.85, help="EMA factor (0=none, .9=heavy)")
    ap.add_argument("-o", "--out", default="train_trends.png")
    a = ap.parse_args()

    it, dg = parse(a.log)
    n_eval = 0
    a_ = a.ema
    fig, axes = plt.subplots(4, 4, figsize=(18, 13))
    fig.suptitle(f"{a.log}  (raw=faint, EMA{a_}=bold)  iters={len(it['step'])}", fontsize=11)
    ax = axes.ravel()

    panel(ax[0], it["step"], it["ev"], "EV (critic health)", a_, "tab:green", (0, 1))
    panel(ax[1], it["step"], it["kl"], "approx KL", a_, "tab:blue")
    panel(ax[2], it["step"], it["clip"], "clip_frac", a_, "tab:red", (0, 0.35))
    ax[2].axhline(0.25, color="k", ls="--", lw=0.7)
    panel(ax[3], it["step"], it["hfire"], "H_fire (entropy)", a_, "tab:purple")
    panel(ax[4], it["step"], it["ilkl"], "il_kl (teacher anchor)", a_, "tab:brown")
    panel(ax[5], dg["step"], dg["ship0"], "ship0 (1-ship probe)", a_, "tab:red", (0, 0.5))
    panel(ax[6], dg["step"], dg["meanshipbin"], "mean ship bin", a_, "tab:orange")
    panel(ax[7], dg["step"], dg["fire_frac"], "fire_frac", a_, "tab:cyan")
    # expansion trajectory — all four milestones on one axis
    for key, c, lb in (("p16", "#cce", "@16"), ("p32", "#88c", "@32"),
                       ("p50", "#44a", "@50"), ("p100", "#008", "@100")):
        panel(ax[8], dg["step"], dg[key], "planets " + lb, a_, c)
    ax[8].set_title("planets@16/32/50/100", fontsize=9)
    ax[8].legend(fontsize=6)
    panel(ax[9], dg["step"], dg["garr50"], "garr_frac@50 (hoard)", a_, "tab:olive", (0, 1))
    panel(ax[10], dg["step"], dg["owned"], "owned planets", a_, "tab:gray")
    # reinforce-by-step early<50 — back-loaded signal (winner ramp 0.29 early); watch it climb
    panel(ax[12], dg["step"], dg["reinf_e"], "reinf step<50 (winner 0.29)", a_, "tab:pink", (0, 0.5))
    ax[12].axhline(0.29, color="k", ls="--", lw=0.7)
    for j in (13, 14, 15):
        ax[j].set_visible(False)

    # arbiter: held-out WR (per-checkpoint, the real signal)
    if a.eval:
        steps, wr = [], []
        for row in csv.DictReader(open(a.eval)):
            try:
                steps.append(float(row["step"])); wr.append(float(row["win_rate"]))
            except (KeyError, ValueError):
                continue
        n_eval = len(wr)
        ax[11].plot(steps, wr, "o-", color="black", lw=2)
        ax[11].set_title(f"held-out WR (ARBITER, n={n_eval})", fontsize=9)
        ax[11].grid(alpha=0.2)
    else:
        ax[11].set_visible(False)

    for x in ax:
        x.tick_params(labelsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}  ({len(it['step'])} iter rows, {len(dg['step'])} diag rows, {n_eval} eval rows)")


if __name__ == "__main__":
    main()
