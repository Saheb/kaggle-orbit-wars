#!/usr/bin/env python3
"""Parse a train_torch.py text log into smoothed trend plots — the offline twin of the W&B
smoothing slider. Reads the per-iter `iter` lines, the every-5th `diag` line, and the `dm`
(decisive-mass concentration) line, and renders raw (faint) + EMA (bold) curves so the SIGNAL
(trend) is separated from single-rollout NOISE.

Staging-run focus (project_undermass_by_choice): the panels that matter for the PBRS staging
experiment are surfaced first — fire_frac (does firing rise off the ~4% prior), fire-clip (the
PPO clip bottleneck on the fire head), tgt-neutral (is targeting shifting toward neutrals =
staging), and dm cross/gap (is inflight reaching the capture floor = concentration).

Usage:
  plot_train_log.py <train.log> [--ema 0.85] [-o out.png] [--eval eval_ajay_1200.csv]
"""
import argparse
import csv
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# iter | steps 1,179,648 | SPS 822 | EV 0.869 | KL 1.93 | clip 0.084(fire 0.113) | H_fire 0.123
#      | V_loss 0.0496 | r_p0 -0.089 r_p1 +0.050 | LR .. | estop 1
ITER_RE = re.compile(
    r"^iter\s+\d+\s+\|\s+steps\s+([\d,]+)\s+\|\s+SPS\s+([\d,]+)\s+\|\s+EV\s+([-\d.]+)\s+\|\s+"
    r"KL\s+([-\d.]+)\s+\|\s+clip\s+([-\d.]+)\(fire\s+([-\d.]+)\)\s+\|\s+H_fire\s+([-\d.]+)\s+\|\s+"
    r"V_loss\s+([-\d.]+)\s+\|\s+r_p0\s+([-\d.]+).*?estop\s+(\d+)")
IL_RE = re.compile(r"il_kl\s+([-\d.]+)")
# diag | .. fire_frac 0.00 owned 0.0 ship0 0.00 meanshipbin 0.0 | pl@16/32/50/100 2/4/7/0
#      garrfrac@50 0.52 shipspp@50 27 | .. reinf 0.43 step<50/50-100/>100 0.28/0.57/0.53 ..
#      tgt n/e 0.29/0.27 | ..
DIAG_RE = re.compile(
    r"diag .*?fire_frac\s+([-\d.]+)\s+owned\s+([-\d.]+)\s+ship0\s+([-\d.]+)\s+"
    r"meanshipbin\s+([-\d.]+).*?pl@16/32/50/100\s+(\d+)/(\d+)/(\d+)/(\d+)\s+"
    r"garrfrac@50\s+([-\d.]+)\s+shipspp@50\s+([-\d.]+)")
REINF_RE = re.compile(r"step<50/50-100/>100\s+([\d.]+)/([\d.]+)/([\d.]+)")
TGT_RE = re.compile(r"tgt n/e\s+([\d.]+)/([\d.]+)")
# dm | gap <50/50-100/>100 0.58/0.43/0.41 | cross 0.15/0.27/0.37 | ratio 1.57 ..
DM_RE = re.compile(
    r"^\s*dm\s+\|\s+gap <50/50-100/>100\s+([\d.]+)/([\d.]+)/([\d.]+)\s+\|\s+"
    r"cross\s+([\d.]+)/([\d.]+)/([\d.]+)")


def _f(x):
    return float(x.replace(",", "")) if x else None


def parse(path):
    it = {k: [] for k in ("step", "sps", "ev", "kl", "clip", "fclip", "hfire",
                          "vloss", "rp0", "estop", "ilkl")}
    dg = {k: [] for k in ("step", "fire_frac", "owned", "ship0", "meanshipbin",
                          "p16", "p32", "p50", "p100", "garr50", "spp50",
                          "reinf_e", "reinf_m", "tgt_n", "tgt_e")}
    dm = {k: [] for k in ("step", "gap_e", "gap_m", "gap_l", "cross_e", "cross_m", "cross_l")}
    last_step = 0.0
    for line in open(path):
        m = ITER_RE.search(line)
        if m:
            last_step = _f(m.group(1))
            il = IL_RE.search(line)
            for key, v in zip(it,
                              (last_step, *(_f(m.group(i)) for i in range(2, 11)),
                               _f(il.group(1)) if il else None)):
                it[key].append(v)
            continue
        d = DIAG_RE.search(line)
        if d:
            r = REINF_RE.search(line)
            t = TGT_RE.search(line)
            vals = [last_step] + [_f(d.group(i)) for i in range(1, 11)]      # step..spp50
            vals += [_f(r.group(1)) if r else None, _f(r.group(2)) if r else None,
                     _f(t.group(1)) if t else None, _f(t.group(2)) if t else None]
            for key, v in zip(dg, vals):
                dg[key].append(v)
            continue
        q = DM_RE.search(line)
        if q:
            for key, v in zip(dm, (last_step, *(_f(q.group(i)) for i in range(1, 7)))):
                dm[key].append(v)
    return it, dg, dm


def ema(xs, alpha):
    out, acc = [], None
    for x in xs:
        if x is None:
            out.append(acc); continue
        acc = x if acc is None else alpha * acc + (1 - alpha) * x
        out.append(acc)
    return out


def panel(ax, x, y, label, alpha, color, ylim=None, ref=None):
    if not any(v is not None for v in y):
        ax.set_title(label + " (no data)", fontsize=8); ax.grid(alpha=0.2); return
    ax.plot(x, y, color=color, alpha=0.16, lw=0.9)
    ax.plot(x, ema(y, alpha), color=color, lw=2.0)
    ax.set_title(label, fontsize=9)
    ax.grid(alpha=0.2)
    if ref is not None:
        ax.axhline(ref, color="k", ls="--", lw=0.7)
    if ylim:
        ax.set_ylim(*ylim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--ema", type=float, default=0.85, help="EMA factor (0=none, .9=heavy)")
    ap.add_argument("-o", "--out", default="train_trends.png")
    ap.add_argument("--eval", help="optional held-out eval CSV (step,win_rate) for the last panel")
    a = ap.parse_args()

    it, dg, dm = parse(a.log)
    e = a.ema
    fig, axes = plt.subplots(5, 4, figsize=(19, 15))
    fig.suptitle(f"{a.log}   raw=faint EMA{e}=bold   iters={len(it['step'])} diag={len(dg['step'])} dm={len(dm['step'])}",
                 fontsize=11)
    ax = axes.ravel()

    # --- row 1: STAGING SIGNAL (the experiment's thesis) ---
    panel(ax[0], dg["step"], dg["fire_frac"], "fire_frac  * (rise off ~0.04 = staging)", e, "tab:cyan")
    panel(ax[1], it["step"], it["fclip"], "fire clip_frac  * (clip bottleneck)", e, "tab:red", (0, 0.5))
    panel(ax[2], dg["step"], dg["tgt_n"], "tgt-neutral  * (staging direction)", e, "tab:green")
    panel(ax[3], dm["step"], dm["cross_m"], "dm cross 50-100  * (reaches floor)", e, "tab:blue", (0, 1))

    # --- row 2: WALL / concentration ---
    panel(ax[4], dm["step"], dm["gap_m"], "dm gap 50-100 (shortfall, down=good)", e, "tab:purple")
    for key, c in (("p16", "#cce"), ("p32", "#88c"), ("p50", "#44a"), ("p100", "#008")):
        if any(v is not None for v in dg[key]):
            ax[5].plot(dg["step"], ema(dg[key], e), color=c, lw=1.8, label=key[1:])
    ax[5].set_title("planets@16/32/50/100", fontsize=9); ax[5].grid(alpha=0.2); ax[5].legend(fontsize=6)
    panel(ax[6], dg["step"], dg["garr50"], "garr_frac@50 (hoard)", e, "tab:olive", (0, 1))
    panel(ax[7], dg["step"], dg["spp50"], "ships/planet@50", e, "tab:orange")

    # --- row 3: PPO HEALTH ---
    panel(ax[8], it["step"], it["ev"], "EV (critic health)", e, "tab:green", (0, 1))
    panel(ax[9], it["step"], it["kl"], "approx KL", e, "tab:blue")
    panel(ax[10], it["step"], it["clip"], "clip_frac (total)", e, "tab:red", (0, 0.4), ref=0.25)
    panel(ax[11], it["step"], it["hfire"], "H_fire (entropy; spike=0.05)", e, "tab:purple")

    # --- row 4: HEALTH cont + reward + throughput ---
    panel(ax[12], it["step"], it["estop"], "estop (KL early-stop)", e, "tab:brown", (0, 1.05))
    panel(ax[13], it["step"], it["vloss"], "V_loss", e, "tab:gray")
    panel(ax[14], it["step"], it["rp0"], "reward r_p0", e, "tab:pink")
    panel(ax[15], it["step"], it["sps"], "SPS (throughput)", e, "tab:olive")

    # --- row 5: conversion / misc ---
    panel(ax[16], dg["step"], dg["ship0"], "ship0 (1-ship probe)", e, "tab:red", (0, 0.5))
    panel(ax[17], dg["step"], dg["owned"], "owned planets", e, "tab:gray")
    panel(ax[18], dg["step"], dg["reinf_e"], "reinf step<50 (winner 0.29)", e, "tab:pink", (0, 0.6), ref=0.29)

    # last panel: held-out WR if --eval, else il_kl (anchored runs), else tgt-enemy.
    if a.eval:
        steps, wr = [], []
        for row in csv.DictReader(open(a.eval)):
            try:
                steps.append(float(row["step"])); wr.append(float(row["win_rate"]))
            except (KeyError, ValueError):
                continue
        if wr:
            ax[19].plot(steps, wr, "o-", color="black", lw=2)
        ax[19].set_title(f"held-out WR (ARBITER, n={len(wr)})", fontsize=9); ax[19].grid(alpha=0.2)
    elif any(v is not None for v in it["ilkl"]):
        panel(ax[19], it["step"], it["ilkl"], "il_kl (anchor)", e, "tab:brown")
    else:
        panel(ax[19], dg["step"], dg["tgt_e"], "tgt-enemy", e, "tab:olive")

    for x in ax:
        x.tick_params(labelsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}  ({len(it['step'])} iter, {len(dg['step'])} diag, {len(dm['step'])} dm rows)")


if __name__ == "__main__":
    main()
