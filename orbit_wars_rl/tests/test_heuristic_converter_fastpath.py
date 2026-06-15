"""Correctness test for the heuristic-move converter fast path (SPS patch C).

`_heuristic_moves_to_action_tensor(..., owned_idx=, slot_valid=, src_pids=)` must produce
the IDENTICAL (fire, angle_bin, ship_bin) action tensor + continuous-angle override as the
original env-derived path. The fast path only swaps HOW owned_idx/slot_valid/source-planet-ids
are obtained (reuse cached feats + one gather, vs recompute owned_indices_for + full
env.planets.cpu()) — never which slot a move maps to. get_features calls owned_indices_for,
so cached feats are byte-identical to the recompute (verified here end-to-end).

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_heuristic_converter_fastpath.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

from torch_env import VecTorchEnv, MAX_OWNED
from train_torch import _heuristic_moves_to_action_tensor


def _run(num_envs=8, steps=40):
    torch.manual_seed(1)
    env = VecTorchEnv(num_envs=num_envs, num_players=2, device="cpu", action_decode="angle")
    env.reset(seeds=list(range(num_envs)))
    for _ in range(steps):
        env.step({0: torch.randint(0, 2, (num_envs, MAX_OWNED, 3)),
                  1: torch.randint(0, 2, (num_envs, MAX_OWNED, 3))})
    return env


def test_fastpath_equals_original():
    env = _run()
    player = 0
    oi, sv = env.owned_indices_for(player)
    # Synthetic moves: from each env's first valid owned planet, send 20 ships at angle ~1.3 rad.
    src = env.planets.gather(1, oi.unsqueeze(-1).expand(-1, -1, 7))
    moves_per_env = []
    for e in range(env.num_envs):
        mv = []
        for k in range(MAX_OWNED):
            if bool(sv[e, k]):
                mv.append([int(src[e, k, 0].item()), 1.3 + 0.1 * k, 20])
        moves_per_env.append(mv)

    # Original env-derived path (what frozen_vs_deb_torch uses)
    act_old, cont_old = _heuristic_moves_to_action_tensor(moves_per_env, env, player, "cpu")
    # Fast path: precomputed owned_idx/slot_valid + gathered src planet ids
    src_pids = env.planets[:, :, 0].gather(1, oi)
    act_new, cont_new = _heuristic_moves_to_action_tensor(
        moves_per_env, env, player, "cpu", owned_idx=oi, slot_valid=sv, src_pids=src_pids)

    assert torch.equal(act_old, act_new), "action tensor differs between paths"
    # cont_angle has NaNs where no launch — compare with equal_nan
    assert torch.allclose(cont_old, cont_new, equal_nan=True), "cont_angle differs between paths"
    assert (act_new[:, :, 0] == 1).sum() > 0, "test built no launches"
    print("ok: converter fast path == original env path (actions + continuous angle)")


def _build_moves(env, oi, sv, src):
    moves = []
    for e in range(env.num_envs):
        mv = []
        for k in range(MAX_OWNED):
            if bool(sv[e, k]):
                mv.append([int(src[e, k, 0].item()), 1.3 + 0.1 * k, 20])
        moves.append(mv)
    return moves


def test_fastpath_subset_reordered():
    """The ACTUAL per-episode pool-grouping path: owned_idx/slot_valid/src_pids sliced to a
    non-contiguous, reordered env_ids subset. Each subset row must equal the full-env row."""
    env = _run()
    player = 0
    oi, sv = env.owned_indices_for(player)
    src_pids_full = env.planets[:, :, 0].gather(1, oi)
    src = env.planets.gather(1, oi.unsqueeze(-1).expand(-1, -1, 7))
    moves_full = _build_moves(env, oi, sv, src)

    act_full, cont_full = _heuristic_moves_to_action_tensor(
        moves_full, env, player, "cpu", owned_idx=oi, slot_valid=sv, src_pids=src_pids_full)

    ids = torch.tensor([5, 1, 6, 2], dtype=torch.long)   # non-contiguous + reordered
    moves_sub = [moves_full[e] for e in ids.tolist()]
    act_sub, cont_sub = _heuristic_moves_to_action_tensor(
        moves_sub, env, player, "cpu",
        owned_idx=oi[ids], slot_valid=sv[ids], src_pids=src_pids_full[ids])

    assert torch.equal(act_sub, act_full[ids]), "subset action tensor != full[ids]"
    assert torch.allclose(cont_sub, cont_full[ids], equal_nan=True), "subset cont_angle != full[ids]"
    assert (act_sub[:, :, 0] == 1).sum() > 0, "subset built no launches"
    print("ok: converter subset/reordered fast path == full[ids] (per-episode grouping path)")


if __name__ == "__main__":
    test_fastpath_equals_original()
    test_fastpath_subset_reordered()
    print("ALL PASS")
