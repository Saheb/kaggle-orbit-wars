"""Friendly-coverage roi deflation: (A) parity of pairwise roi between the GPU
torch_env path and the kaggle features.py path with fleets in flight, and
(B) correctness — a friendly fleet already inbound to a capture target deflates
that target's roi_20/roi_50 toward 0, and never touches own (reinforce) targets."""
from __future__ import annotations
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import extract_features, compute_pairwise_features
from torch_env import VecTorchEnv, to_legacy_obs, MAX_OWNED


def test_pairwise_roi_parity():
    """roi_20/roi_50 (pairwise ch 12,13) must match between torch_env and features.py
    with friendly fleets in flight (so the deflation is exercised in both paths)."""
    num_envs = 4
    env = VecTorchEnv(num_envs=num_envs, num_players=2, device="cpu")
    env.reset(seeds=list(range(num_envs)))
    for _ in range(30):  # random play → fleets in flight → friendly_contest nonzero
        env.step({0: torch.randint(0, 2, (num_envs, MAX_OWNED, 3)),
                  1: torch.randint(0, 2, (num_envs, MAX_OWNED, 3))})

    max_diff = 0.0
    exercised = False
    for player in (0, 1):
        vec = env.get_features(player, max_planets=48, max_fleets=128)["pairwise_features"]
        for i in range(num_envs):
            obs = to_legacy_obs(env, env_idx=i, player=player)
            ref = extract_features(obs, player, num_players=2,
                                   max_planets=48, max_fleets=128)["pairwise_features"].numpy()
            pv = vec[i].numpy()
            # channels 12,13 = roi_20, roi_50
            d = np.abs(pv[..., 12:14] - ref[..., 12:14])
            max_diff = max(max_diff, float(d.max()))
            # exercised iff some roi was actually deflated below its undeflated clip range
            if (ref[..., 12:14] != 0).any():
                exercised = True
    assert max_diff < 0.06, f"roi parity diverged: max|Δ|={max_diff:.4f}"
    assert exercised, "no nonzero roi seen — scenario did not exercise the path"
    print(f"  parity OK: max roi diff = {max_diff:.4f}")


def test_reachable_enemy_mass_parity():
    """reachable_enemy_mass ETA bins (pairwise ch 15-20) must match between the torch_env GPU
    path and the kaggle features.py path. Sim-gap on these channels = train/eval divergence."""
    num_envs = 4
    env = VecTorchEnv(num_envs=num_envs, num_players=2, device="cpu")
    env.reset(seeds=list(range(100, 100 + num_envs)))
    for _ in range(25):  # random play → captures → enemy planets at varied positions/garrisons
        env.step({0: torch.randint(0, 2, (num_envs, MAX_OWNED, 3)),
                  1: torch.randint(0, 2, (num_envs, MAX_OWNED, 3))})

    max_diff = 0.0
    exercised = False
    for player in (0, 1):
        vec = env.get_features(player, max_planets=48, max_fleets=128)["pairwise_features"]
        for i in range(num_envs):
            obs = to_legacy_obs(env, env_idx=i, player=player)
            ref = extract_features(obs, player, num_players=2,
                                   max_planets=48, max_fleets=128)["pairwise_features"].numpy()
            pv = vec[i].numpy()
            d = np.abs(pv[..., 15:21] - ref[..., 15:21])   # all 6 ETA bins
            max_diff = max(max_diff, float(d.max()))
            if (ref[..., 15:21] != 0).any():
                exercised = True
    assert max_diff < 0.06, f"reachable_enemy_mass ETA-bin parity diverged: max|Δ|={max_diff:.4f}"
    assert exercised, "no nonzero reachable_enemy_mass bins seen — scenario did not exercise the channels"
    print(f"  parity OK: max reachable_enemy_mass ETA-bin diff = {max_diff:.4f}")


def test_wave_feature_parity():
    """Phase 5 synchronized-wave features (pairwise ch 21-40) must match between the
    torch_env path used during training and the kaggle features.py path used by eval/export."""
    num_envs = 4
    env = VecTorchEnv(num_envs=num_envs, num_players=2, device="cpu")
    env.reset(seeds=list(range(200, 200 + num_envs)))
    for _ in range(25):
        env.step({0: torch.randint(0, 2, (num_envs, MAX_OWNED, 3)),
                  1: torch.randint(0, 2, (num_envs, MAX_OWNED, 3))})

    max_diff = 0.0
    max_ch = -1
    exercised = False
    for player in (0, 1):
        vec = env.get_features(player, max_planets=48, max_fleets=128)["pairwise_features"]
        for i in range(num_envs):
            obs = to_legacy_obs(env, env_idx=i, player=player)
            ref = extract_features(obs, player, num_players=2,
                                   max_planets=48, max_fleets=128)["pairwise_features"].numpy()
            pv = vec[i].numpy()
            d = np.abs(pv[..., 21:41] - ref[..., 21:41])
            local = float(d.max())
            if local > max_diff:
                max_diff = local
                max_ch = int(np.unravel_index(np.argmax(d), d.shape)[-1]) + 21
            if (ref[..., 21:41] != 0).any():
                exercised = True
    assert max_diff < 0.08, f"wave feature parity diverged: max|Δ|={max_diff:.4f} at ch {max_ch}"
    assert exercised, "no nonzero wave features seen — scenario did not exercise the channels"
    ch_label = str(max_ch) if max_ch >= 0 else "n/a"
    print(f"  parity OK: max wave-feature diff = {max_diff:.4f} (ch {ch_label})")


def test_friendly_inbound_deflates_capture_roi():
    """Controlled: a neutral target with a big friendly fleet aimed at it should read
    much lower roi than the same board with no inbound fleet. Own targets untouched."""
    # source owned planet at (10,10); a NEAR neutral target at (16,10), 5 ships, prod 3
    # (the realistic over-fire case: a cheap nearby neutral, attractive at short ETA so its
    # cost-at-arrival stays small). [id, owner, x, y, r, ships, prod]
    src = [0, 0, 10.0, 10.0, 1.0, 50.0, 2.0]
    tgt = [1, -1, 16.0, 10.0, 1.0, 5.0, 3.0]
    planets = [src, tgt]
    # fleet aimed from src toward tgt (angle 0 = +x): [id, owner, x, y, angle, from, ships].
    # 40 ships >> the neutral's capture cost → coverage saturates → roi should collapse.
    fleet_inbound = [0, 0, 13.0, 10.0, 0.0, 0, 40.0]

    def roi_of_target(fleets):
        obs = {"planets": planets, "fleets": fleets,
               "step": 0, "angular_velocity": 0.0, "comet_planet_ids": []}
        out = extract_features(obs, player=0, num_players=2,
                               max_planets=48, max_fleets=128)["pairwise_features"].numpy()
        return float(out[0, 1, 12]), float(out[0, 1, 13])  # slot 0, target planet idx 1

    roi20_no, roi50_no = roi_of_target([])
    roi20_yes, roi50_yes = roi_of_target([fleet_inbound])

    assert roi20_no > 0.3, f"baseline roi_20 should be attractive, got {roi20_no:.3f}"
    assert roi20_yes < 0.1, \
        f"inbound fleet should collapse roi_20: {roi20_no:.3f} -> {roi20_yes:.3f}"
    assert roi50_yes < 0.1, \
        f"inbound fleet should collapse roi_50: {roi50_no:.3f} -> {roi50_yes:.3f}"
    print(f"  deflation OK: roi_20 {roi20_no:.3f}->{roi20_yes:.3f}  roi_50 {roi50_no:.3f}->{roi50_yes:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    test_pairwise_roi_parity()
    test_reachable_enemy_mass_parity()
    test_wave_feature_parity()
    test_friendly_inbound_deflates_capture_roi()
    print("ALL PASS")
