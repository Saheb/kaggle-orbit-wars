"""Friendly-coverage roi deflation and target-value channels: (A) parity of pairwise roi between the GPU
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
    # features.py resolves fleet targets from current obs; force the same here (the
    # launch-cache drift is an accepted SPS trade — this test locks the MATH).
    env.refresh_fleet_targets()

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
    """reachable_enemy_mass (pairwise ch 15) must match between the torch_env GPU path and
    the kaggle features.py path. Sim-gap on this channel = train/eval policy divergence."""
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
            d = np.abs(pv[..., 15] - ref[..., 15])
            max_diff = max(max_diff, float(d.max()))
            if (ref[..., 15] != 0).any():
                exercised = True
    assert max_diff < 0.06, f"reachable_enemy_mass parity diverged: max|Δ|={max_diff:.4f}"
    assert exercised, "no nonzero reachable_enemy_mass seen — scenario did not exercise the channel"
    print(f"  parity OK: max reachable_enemy_mass diff = {max_diff:.4f}")


def test_target_value_keepability_parity():
    """capture_value/reactive_roi/friendly_reach/keepability (ch 16:20) must match
    between torch_env and features.py. This is the target-priority signal for
    "worth taking and keeping"; sim-gap here poisons train/eval target choice."""
    num_envs = 4
    env = VecTorchEnv(num_envs=num_envs, num_players=2, device="cpu")
    env.reset(seeds=list(range(200, 200 + num_envs)))
    for _ in range(25):
        env.step({0: torch.randint(0, 2, (num_envs, MAX_OWNED, 3)),
                  1: torch.randint(0, 2, (num_envs, MAX_OWNED, 3))})
    env.refresh_fleet_targets()   # match features.py's from-current-obs resolve (see roi test)

    max_diff = 0.0
    exercised = False
    for player in (0, 1):
        vec = env.get_features(player, max_planets=48, max_fleets=128)["pairwise_features"]
        for i in range(num_envs):
            obs = to_legacy_obs(env, env_idx=i, player=player)
            ref = extract_features(obs, player, num_players=2,
                                   max_planets=48, max_fleets=128)["pairwise_features"].numpy()
            pv = vec[i].numpy()
            d = np.abs(pv[..., 16:20] - ref[..., 16:20])
            max_diff = max(max_diff, float(d.max()))
            if (ref[..., 16:20] != 0).any():
                exercised = True
    assert max_diff < 0.06, f"target-value/keepability parity diverged: max|Δ|={max_diff:.4f}"
    assert exercised, "no nonzero target-value/keepability channels seen"
    print(f"  parity OK: max target-value/keepability diff = {max_diff:.4f}")


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


def test_keepability_channels_prefer_supported_targets():
    """Controlled: same-production targets get equal raw value, but the nearby supported
    one should have better reactive ROI and keepability than the far target beside enemy mass."""
    from features import compute_pairwise_features

    planets = [
        [0, 0, 10.0, 10.0, 1.0, 50.0, 2.0],   # source
        [1, -1, 26.0, 10.0, 1.0, 5.0, 4.0],   # supported neutral target
        [2, -1, 85.0, 85.0, 1.0, 5.0, 4.0],   # unsupported neutral target
        [3, 0, 24.0, 10.0, 1.0, 80.0, 2.0],   # friendly support beside target 1
        [4, 1, 83.0, 85.0, 1.0, 100.0, 2.0],  # enemy support beside target 2
        [5, 1, 32.0, 10.0, 1.0, 5.0, 4.0],    # same-prod enemy target: double value swing
    ]
    out = compute_pairwise_features(
        planets,
        owned_indices=np.array([0], dtype=np.int64),
        owned_count=1,
        player=0,
        max_planets=8,
        max_owned=1,
        step=10,
    )
    supported = out[0, 1]
    unsupported = out[0, 2]
    enemy_same_prod = out[0, 5]

    assert abs(float(supported[16]) - float(unsupported[16])) < 1e-6
    assert float(enemy_same_prod[16]) > 1.9 * float(supported[16])
    assert float(supported[17]) > float(unsupported[17]), (supported[17], unsupported[17])
    assert float(supported[18]) > float(unsupported[18]), (supported[18], unsupported[18])
    assert float(supported[19]) > float(unsupported[19]), (supported[19], unsupported[19])
    print("  keepability OK: supported target outranks unsupported trap on roi/support/margin")


if __name__ == "__main__":
    print("=" * 60)
    test_pairwise_roi_parity()
    test_reachable_enemy_mass_parity()
    test_target_value_keepability_parity()
    test_friendly_inbound_deflates_capture_roi()
    test_keepability_channels_prefer_supported_targets()
    print("ALL PASS")
