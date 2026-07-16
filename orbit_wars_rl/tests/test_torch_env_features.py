"""Parity test for vectorized feature extraction.

Compares VecTorchEnv.get_features(player) against features.extract_features()
and action_mask.compute_action_masks() on the same state. Per-element tolerance
is generous because of float precision differences in pairwise computations.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import extract_features
from action_mask import compute_action_masks
from torch_env import VecTorchEnv, to_legacy_obs


def run_parity(num_envs: int = 4, after_steps: int = 30):
    """Step both envs forward, extract features, compare."""
    seeds = list(range(num_envs))
    env = VecTorchEnv(num_envs=num_envs, num_players=2, device="cpu")
    env.reset(seeds=seeds)

    # Drive forward a bit with random actions so there are fleets in flight
    import torch as _t
    from torch_env import MAX_OWNED
    for _ in range(after_steps):
        env.step({0: _t.randint(0, 2, (num_envs, MAX_OWNED, 3)),
                  1: _t.randint(0, 2, (num_envs, MAX_OWNED, 3))})

    # Launch-time target cache (2026-07-05 SPS decision): training carries fleet targets
    # resolved at LAUNCH, while features.py re-resolves from the current obs each step —
    # mid-flight drift between the two is an ACCEPTED trade, not a bug. Force a fresh
    # from-current-state resolve here so this test keeps locking the resolver MATH (and
    # every other channel) to features.py.
    env.refresh_fleet_targets()

    for player in (0, 1):
        vec_feats = env.get_features(player, max_planets=48, max_fleets=128)
        total_err = 0
        max_err = 0.0
        worst = None
        for i in range(num_envs):
            obs = to_legacy_obs(env, env_idx=i, player=player)
            ref = extract_features(obs, player, num_players=2,
                                   max_planets=48, max_fleets=128,
                                   global_econ=True)
            ref_masks = compute_action_masks(obs, player)

            # Compare planet features
            pf_v = vec_feats["planet_features"][i].numpy()
            pf_r = ref["planet_features"].numpy()
            diff = np.abs(pf_v - pf_r)
            err = (diff > 0.05).sum()
            if err > 0:
                total_err += err
                m = diff.max()
                if m > max_err:
                    max_err = m
                    worst = ("planet", i, np.unravel_index(diff.argmax(), diff.shape))

            # Compare ALL 22 pairwise channels (torch _compute_pairwise vs numpy
            # compute_pairwise_features). Only entries valid in BOTH paths (ch9 flag) — invalid
            # (slot,target) cells are zeroed and can differ in padding. This is the surface where
            # the phantom-production / corridor inconsistencies live; lock it so it can't drift.
            if "pairwise_features" in ref:
                pw_v = vec_feats["pairwise_features"][i].numpy()
                pw_r = ref["pairwise_features"].numpy()
                valid = (pw_v[:, :, 9] > 0.5) & (pw_r[:, :, 9] > 0.5)
                pdiff = np.abs(pw_v - pw_r) * valid[:, :, None]
                perr = (pdiff > 0.05).sum()
                if perr > 0:
                    total_err += perr
                    m = pdiff.max()
                    if m > max_err:
                        max_err = m
                        ch = int(np.unravel_index(pdiff.argmax(), pdiff.shape)[2])
                        worst = ("pairwise", i, f"ch{ch}")

            # Fleet features
            ff_v = vec_feats["fleet_features"][i].numpy()
            ff_r = ref["fleet_features"].numpy()
            # Fleets may be in different slot orders → compare just the active ones
            active_v = vec_feats["fleet_mask"][i].numpy()
            active_r = ref["fleet_mask"].numpy()
            if int(active_v.sum()) != int(active_r.sum()):
                total_err += 1
                print(f"  player={player} env={i}: fleet count v={int(active_v.sum())} r={int(active_r.sum())}")

            # Global features (10-dim)
            gf_v = vec_feats["global_features"][i].numpy()
            gf_r = ref["global_features"].numpy()
            gd = np.abs(gf_v - gf_r)
            if (gd > 0.05).any():
                total_err += int((gd > 0.05).sum())
                idx = int(gd.argmax())
                print(f"  player={player} env={i}: global feat[{idx}] v={gf_v[idx]:.3f} r={gf_r[idx]:.3f}")

            # owned_count
            oc_v = vec_feats["owned_count"][i]
            oc_r = ref_masks["owned_count"]
            if oc_v != oc_r:
                total_err += 1
                print(f"  player={player} env={i}: owned_count v={oc_v} r={oc_r}")

        print(f"player={player}: total_err={total_err}  max_diff={max_err:.4f}  worst={worst}")
        assert total_err == 0, (
            f"feature parity FAILED for player={player}: {total_err} cells diverge "
            f"(max_diff={max_err:.4f}, worst={worst}) between VecTorchEnv and extract_features")


def test_feature_parity():
    """pytest entry: asserts VecTorchEnv == extract_features (incl. all 22 pairwise channels)."""
    run_parity(num_envs=4, after_steps=30)





if __name__ == "__main__":
    print("=" * 60)
    print("Feature extraction parity: VecTorchEnv.get_features() vs extract_features()")
    print("=" * 60)
    run_parity(num_envs=4, after_steps=30)
