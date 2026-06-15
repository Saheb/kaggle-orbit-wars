"""Correctness test for the batched external-opponent obs extraction (SPS patch B).

`to_legacy_obs_batch(env, ids, player)` must produce byte-identical obs dicts to
`[to_legacy_obs(env, e, player) for e in ids]` — it only changes TRANSPORT (one batched
GPU->CPU copy per field instead of ~8 tiny syncs per env), never the values fed to the
external opponent. Covers planets, fleets, initial_planets, AND the comet path (the env is
stepped past the step-50 comet spawn so comets are live), with a non-contiguous/reordered
id set (the per-episode pool-grouping case).

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_to_legacy_obs_batch.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

from torch_env import VecTorchEnv, to_legacy_obs, to_legacy_obs_batch, MAX_OWNED


def _run(num_envs=8, steps=70):
    torch.manual_seed(0)
    env = VecTorchEnv(num_envs=num_envs, num_players=2, device="cpu", action_decode="angle")
    env.reset(seeds=list(range(num_envs)))
    for _ in range(steps):  # past step 50 -> comets live (exercises the comet branch)
        env.step({0: torch.randint(0, 2, (num_envs, MAX_OWNED, 3)),
                  1: torch.randint(0, 2, (num_envs, MAX_OWNED, 3))})
    return env


def test_batch_matches_per_env():
    env = _run()
    assert getattr(env, "_has_comets", False), "comets should be live (stepped past 50)"
    ids = [5, 1, 6, 2, 0]           # non-contiguous + reordered
    for player in (0, 1):
        ref = [to_legacy_obs(env, env_idx=e, player=player) for e in ids]
        got = to_legacy_obs_batch(env, ids, player)
        assert len(got) == len(ref)
        for j, (r, g) in enumerate(zip(ref, got)):
            assert r == g, f"player {player} id {ids[j]} mismatch:\n ref={r}\n got={g}"
        # at least one env should carry a live comet, or the comet branch is untested
    any_comet = any(o["comet_planet_ids"] for o in to_legacy_obs_batch(env, list(range(env.num_envs)), 0))
    assert any_comet, "no live comets in any env — comet branch not exercised"
    print("ok: to_legacy_obs_batch == per-env to_legacy_obs (planets/fleets/comets, reordered ids)")


def test_tensor_ids_and_singleton():
    env = _run(num_envs=4, steps=10)
    for ids in (torch.tensor([3, 0], dtype=torch.long), [2]):
        ref = [to_legacy_obs(env, env_idx=int(e), player=0) for e in (ids.tolist() if torch.is_tensor(ids) else ids)]
        got = to_legacy_obs_batch(env, ids, 0)
        assert ref == got, f"mismatch for ids={ids}"
    print("ok: accepts LongTensor + list ids, singleton group")


if __name__ == "__main__":
    test_batch_matches_per_env()
    test_tensor_ids_and_singleton()
    print("ALL PASS")
