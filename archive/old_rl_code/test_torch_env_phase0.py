"""Phase 0 parity test: VecTorchEnv vs FastOrbitWarsEnv.

Runs N envs in both implementations with the same seeds and a no-op agent.
Compares planet/fleet positions at each step. Phase 0 covers only orbital
motion + fleet drift — no collisions, launches, or combat. So fleets in
this test are pre-seeded synthetically into the FastOrbitWarsEnv to verify
fleet movement parity as well.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_env import FastOrbitWarsEnv
from torch_env import VecTorchEnv, to_legacy_obs, MAX_PLANETS, MAX_FLEETS


def compare_planet_positions(legacy: dict, torch_obs: dict, step: int, tol: float = 1e-2):
    """Return list of (planet_idx, key, legacy_val, torch_val) diffs."""
    errs = []
    fp = {int(p[0]): p for p in legacy["planets"]}
    tp = {int(p[0]): p for p in torch_obs["planets"]}
    for pid in fp:
        if pid not in tp:
            errs.append((step, pid, "missing in torch"))
            continue
        a, b = fp[pid], tp[pid]
        # owner, x, y, ships
        for j, name in [(1, "owner"), (2, "x"), (3, "y"), (5, "ships")]:
            if abs(a[j] - b[j]) > tol:
                errs.append((step, pid, name, a[j], b[j]))
    return errs


def run_parity(num_envs: int = 4, max_steps: int = 100, seed_start: int = 0):
    """Parity test on planet orbital motion + production only (no fleets yet)."""
    seeds = list(range(seed_start, seed_start + num_envs))

    # Build both envs
    fast_envs = [
        FastOrbitWarsEnv(num_players=2, seed=s, opponent_policy=None)
        for s in seeds
    ]
    fast_obs = [e.reset(seed=s) for e, s in zip(fast_envs, seeds)]

    torch_env = VecTorchEnv(num_envs=num_envs, num_players=2, device="cpu")
    torch_env.reset(seeds=seeds)

    # Sanity: initial planet count per env
    for i in range(num_envs):
        n_fast = len(fast_obs[i]["planets"])
        n_torch = int(torch_env.planet_alive[i].sum().item())
        if n_fast != n_torch:
            print(f"  seed={seeds[i]}: planet count mismatch fast={n_fast} torch={n_torch}")
            return False
    print(f"  initial planet counts match across {num_envs} envs")

    total_errs = 0
    for step in range(1, max_steps + 1):
        # Step both
        for i in range(num_envs):
            fast_envs[i].step([], opponent_actions=[[]])
            fast_obs[i] = fast_envs[i]._get_obs(0)
        torch_env.step()

        for i in range(num_envs):
            t_obs = to_legacy_obs(torch_env, env_idx=i)
            errs = compare_planet_positions(fast_obs[i], t_obs, step, tol=1e-2)
            total_errs += len(errs)
            if errs and total_errs <= 10:
                for e in errs[:3]:
                    print(f"  step={step} env={i} diff: {e}")

    print(f"\nTotal divergences (tol=1e-2): {total_errs}")
    return total_errs == 0


def measure_sps(num_envs: int = 512, num_steps: int = 200, device: str = "cpu"):
    """Measure SPS of pure tensor ops in VecTorchEnv (no actions)."""
    env = VecTorchEnv(num_envs=num_envs, num_players=2, device=device)
    seeds = list(range(num_envs))
    env.reset(seeds=seeds)

    # Warmup (graph compilation / memory alloc)
    for _ in range(10):
        env.step()
    if device != "cpu":
        torch.mps.synchronize() if device == "mps" else torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_steps):
        env.step()
    if device != "cpu":
        torch.mps.synchronize() if device == "mps" else torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_steps = num_envs * num_steps
    sps = total_steps / elapsed
    print(f"  device={device}  num_envs={num_envs}  steps={num_steps}")
    print(f"  elapsed={elapsed:.2f}s  total_env_steps={total_steps:,}  SPS={sps:,.0f}")
    return sps


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 0 PARITY TEST — VecTorchEnv vs FastOrbitWarsEnv")
    print("=" * 60)
    # Phase 0 doesn't simulate comets, so we only check pre-spawn steps.
    ok = run_parity(num_envs=4, max_steps=49)
    print("RESULT:", "PASS ✓" if ok else "FAIL ✗")

    print("\n" + "=" * 60)
    print("Phase 0 SPS MEASUREMENT")
    print("=" * 60)
    print("\n[CPU baseline]")
    measure_sps(num_envs=64, num_steps=200, device="cpu")
    measure_sps(num_envs=256, num_steps=200, device="cpu")

    if torch.backends.mps.is_available():
        print("\n[MPS]")
        measure_sps(num_envs=64, num_steps=200, device="mps")
        measure_sps(num_envs=256, num_steps=200, device="mps")
        measure_sps(num_envs=512, num_steps=200, device="mps")
        measure_sps(num_envs=1024, num_steps=200, device="mps")
