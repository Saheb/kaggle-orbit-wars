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

from features import (extract_features, set_roi_enemy_deflate, set_zero_roi_channels,
                      set_pressure_precise_resolver, set_threat_eta_surface)
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
        ok = (worst < 0.02 and zero_ok)
        print(f"  {name}: max_ch12/13_diff={worst:.4f}  zero_ok={zero_ok}  [{tag}] {'OK' if ok else 'FAIL'}")
        assert ok, f"roi flag parity FAILED for {name}: max_ch12/13_diff={worst:.4f} zero_ok={zero_ok}"


def run_pressure_resolver_parity(num_envs: int = 6, after_steps: int = 35):
    """--pressure-precise-resolver: the numpy resolver (features._resolve_fleet_targets) must match
    the torch resolver (torch_env._fleet_target_idx) so the pressure channels (ch14/19/20/21 + the
    friendly-contest deflation of ch12/13) stay in train/eval parity. Compares ALL 22 pairwise
    channels AND the planet features (the resolver now also drives planet ch12/13 friendly/enemy
    pressure) with the flag ON; any one fleet resolving differently flips enemy_contest by the fleet
    mass and fails here. This is the parity gate for the swept-collision feature path."""
    from torch_env import MAX_OWNED
    import torch as _t

    set_pressure_precise_resolver(True)
    env = VecTorchEnv(num_envs=num_envs, num_players=2, device="cpu", pressure_precise_resolver=True)
    env.reset(seeds=list(range(num_envs)))
    for _ in range(after_steps):
        env.step({0: _t.randint(0, 2, (num_envs, MAX_OWNED, 3)),
                  1: _t.randint(0, 2, (num_envs, MAX_OWNED, 3))})
    worst = 0.0
    worst_ch = None
    worst_planet = 0.0
    for player in (0, 1):
        vfeats = env.get_features(player, max_planets=48, max_fleets=128)
        vec = vfeats["pairwise_features"]
        vec_pf = vfeats["planet_features"]
        for i in range(num_envs):
            obs = to_legacy_obs(env, env_idx=i, player=player)
            rfeats = extract_features(obs, player, num_players=2, max_planets=48, max_fleets=128)
            ref = rfeats["pairwise_features"].numpy()
            v = vec[i].numpy()
            valid = (v[:, :, 9] > 0.5) & (ref[:, :, 9] > 0.5)
            d = np.abs(v - ref) * valid[:, :, None]
            if d.max() > worst:
                worst = float(d.max())
                worst_ch = int(np.unravel_index(d.argmax(), d.shape)[2])
            # Planet features: ch12/13 (friendly/enemy pressure) now resolver-driven. Assert the
            # full planet vector so the planet path can't silently drift from the pairwise one.
            pd = np.abs(vec_pf[i].numpy() - rfeats["planet_features"].numpy())
            worst_planet = max(worst_planet, float(pd.max()))
    set_pressure_precise_resolver(False)  # reset global for later tests
    print(f"  pressure_precise_resolver: max_pairwise_diff={worst:.4f} worst_ch={worst_ch} "
          f"max_planet_diff={worst_planet:.4f} "
          f"[torch _fleet_target_idx == numpy _resolve_fleet_targets] "
          f"{'OK' if max(worst, worst_planet) < 0.05 else 'FAIL'}")
    assert worst < 0.05, (
        f"pressure-resolver parity FAILED: max_pairwise_diff={worst:.4f} on ch{worst_ch} — "
        f"torch and numpy resolvers disagree on at least one fleet")
    assert worst_planet < 0.05, (
        f"pressure-resolver PLANET parity FAILED: max_planet_diff={worst_planet:.4f} — "
        f"planet ch12/13 (friendly/enemy pressure) diverge between torch and numpy under the resolver")


def run_threat_eta_surface_parity(num_envs: int = 6, after_steps: int = 35):
    """--threat-eta-surface: ch20 (enemy_mass_soon) / ch21 (threat_imminence) measure fleet arrival
    to the planet SURFACE (dist−radius) instead of the center. The numpy (features.py) and torch
    (torch_env.py) paths must apply the −radius identically, or ch20/21 drift between train and eval.
    Asserts all 22 pairwise channels match with the flag ON, and that the flag actually MOVES ch21
    vs the center version (so a no-op can't pass silently)."""
    from torch_env import MAX_OWNED
    import torch as _t

    def _pairwise(surface):
        set_threat_eta_surface(surface)
        env = VecTorchEnv(num_envs=num_envs, num_players=2, device="cpu", threat_eta_surface=surface)
        env.reset(seeds=list(range(num_envs)))
        for _ in range(after_steps):
            env.step({0: _t.randint(0, 2, (num_envs, MAX_OWNED, 3)),
                      1: _t.randint(0, 2, (num_envs, MAX_OWNED, 3))})
        worst = 0.0; worst_ch = None; ch21_sum = 0.0
        for player in (0, 1):
            vec = env.get_features(player, max_planets=48, max_fleets=128)["pairwise_features"]
            for i in range(num_envs):
                obs = to_legacy_obs(env, env_idx=i, player=player)
                ref = extract_features(obs, player, num_players=2, max_planets=48,
                                       max_fleets=128)["pairwise_features"].numpy()
                v = vec[i].numpy()
                valid = (v[:, :, 9] > 0.5) & (ref[:, :, 9] > 0.5)
                d = np.abs(v - ref) * valid[:, :, None]
                if d.max() > worst:
                    worst = float(d.max()); worst_ch = int(np.unravel_index(d.argmax(), d.shape)[2])
                ch21_sum += float((v[:, :, 21] * valid).sum())
        return worst, worst_ch, ch21_sum

    worst_on, ch_on, ch21_on = _pairwise(True)
    _, _, ch21_off = _pairwise(False)
    set_threat_eta_surface(False)  # reset global for later tests
    moved = abs(ch21_on - ch21_off) > 1e-4
    print(f"  threat_eta_surface: max_pairwise_diff={worst_on:.4f} worst_ch={ch_on} "
          f"ch21_sum surface={ch21_on:.3f} center={ch21_off:.3f} moved={moved} "
          f"{'OK' if worst_on < 0.05 and moved else 'FAIL'}")
    assert worst_on < 0.05, (
        f"threat-eta-surface parity FAILED: max_pairwise_diff={worst_on:.4f} on ch{ch_on} — "
        f"torch and numpy disagree on the surface (dist−radius) threat ETA")
    assert moved, ("threat-eta-surface is a NO-OP: ch21 identical surface vs center — "
                   "the −radius term is not being applied")


def test_feature_parity():
    """pytest entry: asserts VecTorchEnv == extract_features (incl. all 22 pairwise channels)."""
    run_parity(num_envs=4, after_steps=30)


def test_roi_flag_parity():
    """pytest entry: asserts roi-deflate / zero-roi flags match across torch + numpy paths."""
    run_roi_flag_parity(num_envs=4, after_steps=30)


def test_pressure_resolver_parity():
    """pytest entry: asserts the precise-resolver pressure channels match across torch + numpy."""
    run_pressure_resolver_parity(num_envs=6, after_steps=35)


def test_threat_eta_surface_parity():
    """pytest entry: asserts the surface threat-ETA (ch20/21) matches across torch + numpy."""
    run_threat_eta_surface_parity(num_envs=6, after_steps=35)


if __name__ == "__main__":
    print("=" * 60)
    print("Feature extraction parity: VecTorchEnv.get_features() vs extract_features()")
    print("=" * 60)
    run_parity(num_envs=4, after_steps=30)
    print("-" * 60)
    print("ROI-channel discipline parity (--roi-enemy-deflate / --zero-roi-channels)")
    run_roi_flag_parity(num_envs=4, after_steps=30)
    print("-" * 60)
    print("Pressure-resolver parity (--pressure-precise-resolver)")
    run_pressure_resolver_parity(num_envs=6, after_steps=35)
    print("-" * 60)
    print("Threat-ETA surface parity (--threat-eta-surface)")
    run_threat_eta_surface_parity(num_envs=6, after_steps=35)
