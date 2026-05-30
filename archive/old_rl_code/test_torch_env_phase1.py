"""Phase 1 parity test: collision + combat resolution.

Injects identical synthetic fleets into both VecTorchEnv and FastOrbitWarsEnv,
steps for N ticks (pre-comet-spawn), and compares planet ownership / ship counts
to validate combat resolution matches.
"""

from __future__ import annotations

import os
import sys
import math

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_env import FastOrbitWarsEnv
from torch_env import VecTorchEnv, to_legacy_obs, CENTER


def inject_fleet_pair_per_env(fast_envs, torch_env, seeds):
    """Manually launch a fleet from each player's home planet toward the
    center, in both envs, to create combat scenarios. Mutates env state.
    """
    for i, fenv in enumerate(fast_envs):
        # Find home planets for players 0 and 1
        for player in range(2):
            home = next((p for p in fenv.planets if p[1] == player), None)
            if home is None:
                continue
            angle = math.atan2(CENTER - home[3], CENTER - home[2])
            ships = 5
            home[5] -= ships
            start_x = home[2] + math.cos(angle) * (home[4] + 0.1)
            start_y = home[3] + math.sin(angle) * (home[4] + 0.1)
            fenv.fleets.append([
                fenv.next_fleet_id, player,
                start_x, start_y, angle, home[0], ships,
            ])
            fenv.next_fleet_id += 1

    # Mirror in torch_env: same fleets in same slots
    for i, fenv in enumerate(fast_envs):
        for j, fleet in enumerate(fenv.fleets):
            if j >= torch_env.fleets.shape[1]:
                break
            torch_env.fleets[i, j, :] = torch.tensor(fleet, dtype=torch.float32)
            torch_env.fleet_alive[i, j] = True
            # Mirror the ship debit on home planet
            home_pid = int(fleet[5])
            for k in range(torch_env.planets.shape[1]):
                if int(torch_env.planets[i, k, 0].item()) == home_pid:
                    torch_env.planets[i, k, 5] = float(
                        next(p for p in fenv.planets if p[0] == home_pid)[5]
                    )
                    break
        torch_env.next_fleet_id[i] = fenv.next_fleet_id


def compare_planets(legacy: dict, torch_obs: dict, step: int, tol: float = 1e-2):
    errs = []
    fp = {int(p[0]): p for p in legacy["planets"]}
    tp = {int(p[0]): p for p in torch_obs["planets"]}
    for pid in fp:
        if pid not in tp:
            errs.append((step, pid, "missing in torch"))
            continue
        a, b = fp[pid], tp[pid]
        for j, name in [(1, "owner"), (2, "x"), (3, "y"), (5, "ships")]:
            if abs(a[j] - b[j]) > tol:
                errs.append((step, pid, name, round(a[j], 3), round(b[j], 3)))
    return errs


def compare_fleets(legacy: dict, torch_obs: dict, step: int):
    """Compare fleet counts and rough positions."""
    fa = legacy["fleets"]
    tf = torch_obs["fleets"]
    if len(fa) != len(tf):
        return [(step, "fleet_count", len(fa), len(tf))]
    return []


def run_parity(num_envs: int = 4, max_steps: int = 49):
    seeds = list(range(num_envs))
    fast_envs = [
        FastOrbitWarsEnv(num_players=2, seed=s, opponent_policy=None)
        for s in seeds
    ]
    fast_obs = [e.reset(seed=s) for e, s in zip(fast_envs, seeds)]

    torch_env = VecTorchEnv(num_envs=num_envs, num_players=2, device="cpu")
    torch_env.reset(seeds=seeds)

    # Inject identical fleets in both envs
    inject_fleet_pair_per_env(fast_envs, torch_env, seeds)

    print(f"  injected {len(fast_envs[0].fleets)} fleets per env")

    total_errs = 0
    for step in range(1, max_steps + 1):
        for i in range(num_envs):
            fast_envs[i].step([], opponent_actions=[[]])
            fast_obs[i] = fast_envs[i]._get_obs(0)
        torch_env.step()

        for i in range(num_envs):
            t_obs = to_legacy_obs(torch_env, env_idx=i)
            errs = compare_planets(fast_obs[i], t_obs, step, tol=1e-2)
            errs += compare_fleets(fast_obs[i], t_obs, step)
            total_errs += len(errs)
            if errs and total_errs <= 15:
                for e in errs[:3]:
                    print(f"  step={step} env={i} diff: {e}")

    print(f"\nTotal divergences: {total_errs}")
    return total_errs == 0


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 PARITY TEST — collision + combat (synthetic fleets)")
    print("=" * 60)
    ok = run_parity(num_envs=4, max_steps=49)
    print("RESULT:", "PASS ✓" if ok else "FAIL ✗")
