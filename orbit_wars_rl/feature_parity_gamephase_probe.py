"""Game-phase feature-parity probe: VecTorchEnv.get_features() vs features.extract_features()
with the Stage-B game-phase channels ON.

Companion to feature_parity_comet_probe.py. The game-phase features add 4 global channels
(11-13 = early<50 / mid50-100 / late>=100 one-hot; 14 = normalized steps-to-next-comet-spawn),
computed in TWO places that must stay byte-identical:
    - features.game_phase_channels   (scalar, eval/export path)
    - torch_env.get_features          (vectorized, training path)
This probe drives one torch_env forward across the phase boundaries (50, 100) AND the comet
spawn boundaries (50, 150), and at each sampled step diffs all 15 global channels between the two
extractors (aligned per env via to_legacy_obs). Expected verdict: ALL channels match within tol —
the existing 11 unchanged (regression guard) and the 4 new ones identical.

Run: orbit_wars_rl/.venv/bin/python orbit_wars_rl/feature_parity_gamephase_probe.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features
from features import extract_features, set_game_phase_features, game_phase_channels, _COMET_SPAWN_STEPS
from torch_env import VecTorchEnv, to_legacy_obs, COMET_SPAWN_STEPS

TOL = 1e-4
SAMPLE_STEPS = [10, 49, 50, 51, 60, 99, 100, 101, 149, 150, 151]
N_ENVS = 4
PLAYER = 0


def main():
    assert _COMET_SPAWN_STEPS == COMET_SPAWN_STEPS, \
        f"spawn-step constants drifted: features {_COMET_SPAWN_STEPS} vs torch {COMET_SPAWN_STEPS}"

    set_game_phase_features(True)  # eval-path flag ON for this probe

    # Sanity: expected one-hot + comet-cycle at boundary steps.
    assert game_phase_channels(49) == [1.0, 0.0, 0.0, (50 - 49) / 100.0]
    assert game_phase_channels(50) == [0.0, 1.0, 0.0, (150 - 50) / 100.0]
    assert game_phase_channels(100) == [0.0, 0.0, 1.0, (150 - 100) / 100.0]
    assert game_phase_channels(450) == [0.0, 0.0, 1.0, 1.0]   # no spawn remains
    print("scalar game_phase_channels boundary checks: OK")

    env = VecTorchEnv(num_envs=N_ENVS, num_players=2, device="cpu",
                      action_decode="target", allow_reinforce=True,
                      reinforce_gate_min_planets=3, reinforce_garrison_floor=0.0,
                      game_phase_features=True)
    env.reset(seeds=list(range(N_ENVS)))

    target = set(SAMPLE_STEPS)
    # no-op actions for both seats: {player: (N, MAX_OWNED, 4) zeros (fire=0)}
    from torch_env import MAX_OWNED
    noop = {p: torch.zeros(N_ENVS, MAX_OWNED, 4, dtype=torch.long) for p in range(2)}

    max_dim_seen = 0
    worst = 0.0
    checked = 0
    while target:
        step_now = int(env.step_count[0].item())
        if step_now in target:
            target.discard(step_now)
            gf_torch = env.get_features(PLAYER, max_planets=48, max_fleets=128)["global_features"]
            max_dim_seen = gf_torch.shape[1]
            for e in range(N_ENVS):
                obs = to_legacy_obs(env, env_idx=e, player=PLAYER)
                feat = extract_features(obs, PLAYER, num_players=2, max_planets=48, max_fleets=128)
                g_eval = feat["global_features"].numpy()
                g_torch = gf_torch[e].numpy()
                assert g_eval.shape == g_torch.shape == (15,), (g_eval.shape, g_torch.shape)
                d = float(np.max(np.abs(g_eval - g_torch)))
                worst = max(worst, d)
                checked += 1
                if d > TOL:
                    print(f"  MISMATCH step={step_now} env={e} maxdiff={d:.5f}")
                    print("    eval  globals:", np.round(g_eval, 4))
                    print("    torch globals:", np.round(g_torch, 4))
            # show the 4 new channels for the first env at this step
            print(f"step {step_now:3d}: globals dim {max_dim_seen}  "
                  f"phase+comet[11:15] = {np.round(gf_torch[0, 11:15].numpy(), 3)}  maxdiff {worst:.2e}")
        env.step(noop)
        if int(env.step_count[0].item()) > max(SAMPLE_STEPS) + 2:
            break

    # Regression guard: flag OFF → 11 globals.
    env_off = VecTorchEnv(num_envs=2, num_players=2, device="cpu", action_decode="target")
    env_off.reset(seeds=[0, 1])
    dim_off = env_off.get_features(0)["global_features"].shape[1]

    print("=" * 72)
    print(f"checked {checked} (step,env) pairs across {len(SAMPLE_STEPS)} steps · "
          f"on-dim={max_dim_seen} off-dim={dim_off} · worst maxdiff {worst:.2e}")
    ok = worst <= TOL and max_dim_seen == 15 and dim_off == 11
    print("VERDICT:", "CLEAN ✅ (parity holds, 11<->15 gating correct)" if ok else "FAILED ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
