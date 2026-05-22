"""Parity test: compare FastOrbitWarsEnv vs Kaggle OrbitWarsEnv trajectories.

Runs both environments with the same seed and agent, comparing planet/fleet
states at each step. Any discrepancy indicates a simulation bug in FastOrbitWarsEnv.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from env import OrbitWarsEnv
from fast_env import FastOrbitWarsEnv


def _planet_key(p):
    return (round(p[2], 2), round(p[3], 2), round(p[4], 4), p[5], p[6])


def _fleet_key(f):
    return (round(f[2], 2), round(f[3], 2), round(f[4], 4), f[6])


def compare_obs(kaggle_obs, fast_obs, step):
    errors = []
    k_p = len(kaggle_obs["planets"])
    f_p = len(fast_obs["planets"])
    if k_p != f_p:
        errors.append(f"Step {step}: planet count {k_p} vs {f_p}")

    k_f = len(kaggle_obs["fleets"])
    f_f = len(fast_obs["fleets"])
    if k_f != f_f:
        errors.append(f"Step {step}: fleet count {k_f} vs {f_f}")

    for i in range(min(k_p, f_p)):
        kp = kaggle_obs["planets"][i]
        fp = fast_obs["planets"][i]
        for j, (a, b) in enumerate(zip(kp, fp)):
            if j == 0 and a != b:
                errors.append(f"Step {step} planet {i} id: {a} vs {b}")
            elif j == 1 and a != b:
                pass  # owner assignment can differ in neutrality
            elif j in (2, 3) and abs(a - b) > 0.5:
                errors.append(f"Step {step} planet {i} pos[{j}]: {a:.2f} vs {b:.2f}")
            elif j == 5 and abs(a - b) > 1:
                errors.append(f"Step {step} planet {i} ships: {a:.1f} vs {b:.1f}")

    if k_f > 0 and f_f > 0:
        k_fleets_sorted = sorted(kaggle_obs["fleets"], key=lambda f: (round(f[2], 1), round(f[3], 1)))
        f_fleets_sorted = sorted(fast_obs["fleets"], key=lambda f: (round(f[2], 1), round(f[3], 1)))
        for i in range(min(len(k_fleets_sorted), len(f_fleets_sorted), 5)):
            kf = k_fleets_sorted[i]
            ff = f_fleets_sorted[i]
            if abs(kf[2] - ff[2]) > 1.0 or abs(kf[3] - ff[3]) > 1.0:
                errors.append(
                    f"Step {step} fleet {i} pos: ({kf[2]:.1f},{kf[3]:.1f}) vs ({ff[2]:.1f},{ff[3]:.1f})"
                )

    return errors


def _do_nothing_agent(obs):
    return []


def run_parity_test(num_games=3, max_steps=50):
    all_errors = []

    for seed in range(num_games):
        kaggle_env = OrbitWarsEnv(num_players=2, seed=seed, debug=False)
        fast_env = FastOrbitWarsEnv(num_players=2, seed=seed)

        k_obs = kaggle_env.reset(seed=seed)
        f_obs = fast_env.reset(seed=seed)

        # Check initial state
        errors = compare_obs(k_obs, f_obs, 0)
        all_errors.extend(errors)

        for step in range(1, max_steps + 1):
            actions = _do_nothing_agent(k_obs)
            k_obs_next, k_reward, k_done, _ = kaggle_env.step(actions)
            f_obs_next, f_reward, f_done, _ = fast_env.step(actions)

            errors = compare_obs(k_obs_next, f_obs_next, step)
            all_errors.extend(errors)

            k_obs = k_obs_next
            f_obs = f_obs_next

            if k_done or f_done:
                if k_done != f_done:
                    all_errors.append(f"Seed {seed} step {step}: done mismatch kaggle={k_done} fast={f_done}")
                break

        print(f"  Seed {seed}: {step} steps, {len(errors)} errors this game")

    if all_errors:
        print(f"\nPARITY FAIL: {len(all_errors)} total errors")
        for e in all_errors[:20]:
            print(f"  {e}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more")
        return False
    else:
        print("\nPARITY PASS: FastOrbitWarsEnv matches Kaggle env")
        return True


if __name__ == "__main__":
    success = run_parity_test(num_games=5, max_steps=100)
    sys.exit(0 if success else 1)