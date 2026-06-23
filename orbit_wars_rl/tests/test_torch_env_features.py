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

from features import (extract_features, set_roi_enemy_deflate, set_zero_roi_channels)
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

    for player in (0, 1):
        vec_feats = env.get_features(player, max_planets=48, max_fleets=128)
        total_err = 0
        max_err = 0.0
        worst = None
        for i in range(num_envs):
            obs = to_legacy_obs(env, env_idx=i, player=player)
            ref = extract_features(obs, player, num_players=2,
                                   max_planets=48, max_fleets=128)
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

        print(f"player={player}: total_err={total_err}  max_planet_diff={max_err:.4f}  worst={worst}")


def run_roi_flag_parity(num_envs: int = 4, after_steps: int = 30):
    """ROI-channel discipline (ch12/13): torch path (VecTorchEnv) must equal numpy path
    (extract_features) under --roi-enemy-deflate and --zero-roi-channels, and zeroing must
    actually zero both. Catches train/eval feature divergence on these flags."""
    from torch_env import MAX_OWNED
    import torch as _t

    for name, env_kw, on, off in (
        ("roi_enemy_deflate", dict(roi_enemy_deflate=True), set_roi_enemy_deflate, set_zero_roi_channels),
        ("zero_roi_channels", dict(zero_roi_channels=True), set_zero_roi_channels, set_roi_enemy_deflate),
    ):
        on(True); off(False)
        env = VecTorchEnv(num_envs=num_envs, num_players=2, device="cpu", **env_kw)
        env.reset(seeds=list(range(num_envs)))
        for _ in range(after_steps):
            env.step({0: _t.randint(0, 2, (num_envs, MAX_OWNED, 3)),
                      1: _t.randint(0, 2, (num_envs, MAX_OWNED, 3))})
        worst = 0.0
        zero_ok = True
        for player in (0, 1):
            vec = env.get_features(player, max_planets=48, max_fleets=128)["pairwise_features"]
            for i in range(num_envs):
                obs = to_legacy_obs(env, env_idx=i, player=player)
                ref = extract_features(obs, player, num_players=2,
                                       max_planets=48, max_fleets=128)["pairwise_features"]
                v = vec[i, :, :, 12:14].numpy(); r = ref[:, :, 12:14].numpy()
                worst = max(worst, float(np.abs(v - r).max()))
                if name == "zero_roi_channels" and (np.abs(v).max() > 0 or np.abs(r).max() > 0):
                    zero_ok = False
        on(False)  # reset global so later tests/imports are unaffected
        tag = "ch12/13 zeroed both paths" if name == "zero_roi_channels" else "torch==numpy"
        status = "OK" if (worst < 0.02 and zero_ok) else "FAIL"
        print(f"  {name}: max_ch12/13_diff={worst:.4f}  zero_ok={zero_ok}  [{tag}] {status}")


if __name__ == "__main__":
    print("=" * 60)
    print("Feature extraction parity: VecTorchEnv.get_features() vs extract_features()")
    print("=" * 60)
    run_parity(num_envs=4, after_steps=30)
    print("-" * 60)
    print("ROI-channel discipline parity (--roi-enemy-deflate / --zero-roi-channels)")
    run_roi_flag_parity(num_envs=4, after_steps=30)
