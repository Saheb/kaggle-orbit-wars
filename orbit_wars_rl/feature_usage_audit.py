"""Diagnostic: are the Phase 5 feature channels LIVE (vary in data) and USED (model weights them)?

A channel contributes to the output only if it BOTH varies across states AND the input projection
puts weight on it. usage = std(channel over valid pairs) * weight_norm(channel). Near-zero std =>
constant => unusable (wiring/compute bug). Near-zero weight on a live channel => model ignores it.

Pairwise layout (41): 0-11 phase1 | 12 roi20 13 roi50 14 contest | 15-20 reach-bins | 21-40 wave.
Globals (15): 0-10 base | 11-14 game-phase.  Pairwise feeds pair_kv AND tgt/fire/ship scorers.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "gpu_run_artifacts/jake_phase5/checkpoints/bc_jake_phase5_proactive_gp_pw7.0_20260620.pt"
PKL = sys.argv[2] if len(sys.argv) > 2 else \
    "gpu_run_artifacts/jake_decisive/phase5_jake_proactive.pkl"   # pairwise is game-phase-independent
N_SAMPLE = 3000

PAIR_GROUPS = [("phase1", 0, 12), ("roi/contest", 12, 15), ("reach-bins★", 15, 21), ("wave★", 21, 41)]
GLOBAL_GROUPS = [("base", 0, 11), ("game-phase★", 11, 15)]


def per_channel_std_pairwise(samples):
    """std of each pairwise channel over VALID (slot_valid source × real-planet target) pairs only."""
    sums = None; cnt = 0; sq = None
    for s in samples:
        pw = np.asarray(s["pairwise_features"])            # (MO, P, F)
        sv = np.asarray(s["slot_valid"]).astype(bool)      # (MO,)
        pm = np.asarray(s["planet_mask"]).astype(bool)     # (P,)
        if not sv.any() or not pm.any():
            continue
        v = pw[sv][:, pm, :].reshape(-1, pw.shape[-1])     # (valid_pairs, F)
        if sums is None:
            sums = v.sum(0); sq = (v * v).sum(0)
        else:
            sums += v.sum(0); sq += (v * v).sum(0)
        cnt += v.shape[0]
    mean = sums / cnt
    return np.sqrt(np.maximum(sq / cnt - mean * mean, 0.0)), cnt


def per_channel_std_global(samples):
    g = np.stack([np.asarray(s["global_features"]) for s in samples])  # (N, G)
    return g.std(0)


def col_norm(w, lo, hi):
    """L2 norm of each input column (dim 0 = out) for columns [lo:hi)."""
    return w[:, lo:hi].norm(dim=0).cpu().numpy()


def main():
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    D = sd["global_proj.weight"].shape[0]
    G = sd["global_proj.weight"].shape[1]
    Fp = sd["pair_kv.weight"].shape[1] - D
    print(f"D={D}  global_dim={G}  F_pair={Fp}\n")

    samples = pickle.load(open(PKL, "rb"))[:N_SAMPLE]
    pw_std, npairs = per_channel_std_pairwise(samples)
    g_std = per_channel_std_global(samples) if G == len(np.asarray(samples[0]["global_features"])) else None

    # ---- pairwise: weight norms across the 4 consumers (pair_kv last Fp cols; scorers last Fp cols) ----
    kv = col_norm(sd["pair_kv.weight"], D, D + Fp)
    tgt = col_norm(sd["target_scorer.0.weight"], 2 * D, 2 * D + Fp)
    fire = col_norm(sd["fire_scorer.0.weight"], 2 * D, 2 * D + Fp)
    ship = col_norm(sd["ship_scorer.0.weight"], 2 * D, 2 * D + Fp)
    pw_w = kv + tgt + fire + ship                              # total input weight per channel

    print(f"PAIRWISE channels (std over {npairs:,} valid src×tgt pairs ; weight = Σ norm over pair_kv+tgt+fire+ship)")
    print(f"  {'ch':>3} {'group':>12}  {'std':>9} {'weight':>8} {'usage=std*w':>12}  flag")
    for name, lo, hi in PAIR_GROUPS:
        for c in range(lo, hi):
            usage = pw_std[c] * pw_w[c]
            flag = "DEAD(const)" if pw_std[c] < 1e-4 else ("ignored?" if pw_w[c] < 0.5 * np.median(pw_w) else "")
            print(f"  {c:>3} {name:>12}  {pw_std[c]:>9.4f} {pw_w[c]:>8.3f} {usage:>12.4f}  {flag}")
    # group rollups
    print("\n  GROUP ROLLUP (mean usage = mean std*weight):")
    for name, lo, hi in PAIR_GROUPS:
        u = (pw_std[lo:hi] * pw_w[lo:hi]).mean()
        ndead = int((pw_std[lo:hi] < 1e-4).sum())
        print(f"    {name:>12}  mean_std {pw_std[lo:hi].mean():.4f}  mean_w {pw_w[lo:hi].mean():.3f}  mean_usage {u:.4f}  dead {ndead}/{hi-lo}")

    # ---- globals ----
    gw = col_norm(sd["global_proj.weight"], 0, G) + col_norm(sd["mode_proj.weight"], 0, G)
    print(f"\nGLOBAL channels (weight = global_proj+mode_proj):")
    for name, lo, hi in GLOBAL_GROUPS:
        for c in range(lo, hi):
            std_s = f"{g_std[c]:.4f}" if g_std is not None else "  n/a"   # n/a if local pkl is 11-global
            print(f"  {c:>3} {name:>12}  std {std_s}  weight {gw[c]:>7.3f}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
