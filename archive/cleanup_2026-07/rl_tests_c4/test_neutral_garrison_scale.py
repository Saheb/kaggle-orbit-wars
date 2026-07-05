"""Test the --neutral-garrison-scale board-curriculum flag (torch_env.reset).

Asserts that:
  1. neutral garrison is scaled by the factor (×3 → neutral ships 3× original)
  2. home planets are NOT scaled (always 10 ships, overwritten after scaling)
  3. symmetry preserved (the 4-fold symmetric planet groups have identical scaled ships)
  4. scale=1.0 (default) is a no-op

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_neutral_garrison_scale.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from torch_env import VecTorchEnv


def _neutral_ships(env, env_i=0):
    """Return (planet_ids, ships) for all neutral planets in env_i."""
    p = env.planets[env_i].cpu().numpy()
    alive = env.planet_alive[env_i].cpu().numpy()
    out = []
    for i in range(len(p)):
        if alive[i] and p[i, 1] == -1:
            out.append((int(p[i, 0]), float(p[i, 5])))
    return out


def _home_ships(env, env_i=0):
    """Return ships for home planets (owner 0 and 1)."""
    p = env.planets[env_i].cpu().numpy()
    alive = env.planet_alive[env_i].cpu().numpy()
    out = {}
    for i in range(len(p)):
        if alive[i] and p[i, 1] in (0, 1):
            out[int(p[i, 1])] = float(p[i, 5])
    return out


def test_scale_3x_neutrals_scaled_home_unchanged():
    """×3 scale: neutral ships tripled, home planets stay at 10."""
    env_default = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                              action_decode="target", neutral_garrison_scale=1.0)
    env_default.reset(seeds=[42])

    env_scaled = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                             action_decode="target", neutral_garrison_scale=3.0)
    env_scaled.reset(seeds=[42])

    default_neutrals = _neutral_ships(env_default)
    scaled_neutrals = _neutral_ships(env_scaled)

    assert len(default_neutrals) == len(scaled_neutrals), \
        f"neutral count mismatch: {len(default_neutrals)} vs {len(scaled_neutrals)}"

    for (pid_d, ships_d), (pid_s, ships_s) in zip(default_neutrals, scaled_neutrals):
        assert pid_d == pid_s, f"planet id mismatch: {pid_d} vs {pid_s}"
        expected = float(int(ships_d * 3.0))
        assert ships_s == expected, \
            f"planet {pid_d}: scaled {ships_s} != expected {expected} (default {ships_d} × 3)"

    home_default = _home_ships(env_default)
    home_scaled = _home_ships(env_scaled)
    for owner in (0, 1):
        assert home_default[owner] == 10.0, f"default home owner {owner} != 10"
        assert home_scaled[owner] == 10.0, f"scaled home owner {owner} != 10"

    print(f"PASS ×3: {len(scaled_neutrals)} neutrals scaled, "
          f"home planets unchanged (10/10)")


def test_scale_1x_noop():
    """scale=1.0 is a no-op (identical to default)."""
    env_a = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                        action_decode="target", neutral_garrison_scale=1.0)
    env_a.reset(seeds=[99])
    env_b = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                        action_decode="target", neutral_garrison_scale=1.0)
    env_b.reset(seeds=[99])

    a_neutrals = _neutral_ships(env_a)
    b_neutrals = _neutral_ships(env_b)
    assert a_neutrals == b_neutrals, "scale=1.0 should be identical"
    print(f"PASS ×1.0 no-op: {len(a_neutrals)} neutrals identical")


def test_symmetry_preserved():
    """The 4-fold symmetric planet groups have identical scaled ships.

    generate_planets creates planets in groups of 4 (Q1 + 3 symmetric copies).
    After scaling, all 4 in a group should have the same ships.
    """
    env = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                      action_decode="target", neutral_garrison_scale=3.0)
    env.reset(seeds=[7])

    p = env.planets[0].cpu().numpy()
    alive = env.planet_alive[0].cpu().numpy()
    n = int(alive.sum())
    num_groups = n // 4

    for g in range(num_groups):
        base = g * 4
        ships = [float(p[base + j, 5]) for j in range(4)]
        owners = [int(p[base + j, 1]) for j in range(4)]
        # If all 4 are neutral, ships must be identical (symmetric scaling)
        if all(o == -1 for o in owners):
            assert len(set(ships)) == 1, \
                f"group {g}: symmetric neutrals have different ships: {ships}"
        # Home planets (owner 0/1) are always 10
        for j, o in enumerate(owners):
            if o in (0, 1):
                assert ships[j] == 10.0, \
                    f"group {g} planet {base+j} owner {o}: ships {ships[j]} != 10"

    print(f"PASS symmetry: {num_groups} groups, 4-fold neutral ships identical")


def test_auto_reset_preserves_scaling():
    """Auto-reset must apply the same reset-time board curriculum as initial reset."""
    env = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                      action_decode="target", neutral_garrison_scale=3.0)
    env.reset(seeds=[5])
    env._auto_reset(torch.tensor([True]))

    p = env.planets[0].cpu().numpy()
    alive = env.planet_alive[0].cpu().numpy()
    neutral_ships = [float(p[i, 5]) for i in range(len(p)) if alive[i] and p[i, 1] == -1]
    assert neutral_ships, "auto-reset board should have neutrals"
    assert all(float(int(s / 3.0) * 3.0) == s for s in neutral_ships), \
        "auto-reset neutrals should still be integer-scaled by ×3"
    print(f"PASS auto-reset ×3: {len(neutral_ships)} neutrals remain scaled")


if __name__ == "__main__":
    test_scale_1x_noop()
    test_scale_3x_neutrals_scaled_home_unchanged()
    test_symmetry_preserved()
    test_auto_reset_preserves_scaling()
    print("\nAll tests passed.")
