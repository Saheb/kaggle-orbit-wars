"""torch_env wiring test for the reverse-edge reinforce cooldown: the get_features logit-mask
blocks the reverse edge, and auto-reset clears the per-edge history at the episode boundary
(same bug class as the decmass prev_decisive_suff re-arm).

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_reverse_edge_cooldown_env.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch_env import VecTorchEnv, _REINF_CD_NEVER, MAX_OWNED

K = 3


def _env():
    env = VecTorchEnv(num_envs=1, num_players=2, device="cpu", action_decode="target",
                      allow_reinforce=True, reverse_edge_cooldown=K)
    env.reset(seeds=[0])
    # planet 0 = A (ours), planet 1 = B (ours), planet 2 = enemy; rest dead.
    env.planet_alive[0, :] = False
    for i, own in enumerate([0, 0, 1]):
        env.planets[0, i, 1] = own
        env.planet_alive[0, i] = True
    return env


def test_cd_allocated_never():
    env = _env()
    P = env.planets.shape[1]
    assert env.reinf_cd is not None
    assert tuple(env.reinf_cd.shape) == (1, 2, P, P)
    assert bool((env.reinf_cd == _REINF_CD_NEVER).all()), "fresh state must be all-NEVER"


def test_mask_blocks_reverse_edge():
    env = _env()
    env.step_count[0] = 5
    env.reinf_cd[0, 0, 0, 1] = 5          # we reinforced A(0)->B(1) at step 5
    feats = env.get_features(0)
    oi = feats["owned_indices"][0]         # slot -> planet index
    tm = feats["target_mask"][0]           # (MAX_OWNED, P) bool
    slotB = int((oi == 1).nonzero()[0].item())
    slotA = int((oi == 0).nonzero()[0].item())
    assert not bool(tm[slotB, 0]), "B->A (reverse of recorded A->B) must be masked"
    assert bool(tm[slotA, 1]), "A->B (same direction) must stay legal"
    # window expiry: 4 steps later the reverse edge is free again
    env.step_count[0] = 5 + K + 1
    tm2 = env.get_features(0)["target_mask"][0]
    assert bool(tm2[slotB, 0]), "B->A must be legal once the K-step window passes"


def test_real_fired_reinforce_records_then_blocks():
    # End-to-end through the REAL _apply_actions launch path: fire a reinforce A->B, assert the
    # edge is recorded at the launch step, then assert the reverse B->A is masked the next step.
    env = _env()
    env.planets[0, 0, 5] = 200.0     # A: plenty of ships to send
    env.planets[0, 1, 5] = 20.0      # B
    env.planets[0, 2, 5] = 10.0      # C (enemy, keeps the game alive: n_alive == 2)
    A, B = 0, 1
    oi, _ = env.owned_indices_for(0)
    slotA = int((oi[0] == A).nonzero()[0].item())
    # action cols: [fire, angle_bin, ship_bin, target_idx]; ship_bin 0 = a small, sendable count.
    act0 = torch.zeros(1, MAX_OWNED, 4, dtype=torch.long)
    act0[0, slotA, 0] = 1            # fire
    act0[0, slotA, 3] = B           # target = B  -> a reinforce A->B
    actions = {0: act0, 1: torch.zeros(1, MAX_OWNED, 4, dtype=torch.long)}

    t0 = int(env.step_count[0].item())          # _apply_actions records BEFORE step_count += 1
    env.step(actions)
    assert int(env.reinf_cd[0, 0, A, B].item()) == t0, \
        f"a fired reinforce A->B must record reinf_cd[A,B] == {t0}, got {int(env.reinf_cd[0,0,A,B])}"

    # Next step: the reverse edge B->A must be masked (we're within K of the A->B launch).
    feats = env.get_features(0)
    oi2 = feats["owned_indices"][0]
    tm = feats["target_mask"][0]
    slotB = int((oi2 == B).nonzero()[0].item())
    assert not bool(tm[slotB, A]), "B->A must be masked the step after a real A->B reinforce fired"


def test_boundary_clears_state():
    env = _env()
    env.reinf_cd[0, 0, 0, 1] = 3
    env.step_count[0] = int(env.episode_steps)   # force time_up termination
    env.step()                                   # actions=None -> no launches; runs auto-reset
    assert bool((env.reinf_cd[0] == _REINF_CD_NEVER).all()), \
        "auto-reset must clear reinf_cd for the done env"


if __name__ == "__main__":
    test_cd_allocated_never()
    test_mask_blocks_reverse_edge()
    test_real_fired_reinforce_records_then_blocks()
    test_boundary_clears_state()
    print("PASS: reverse-edge cooldown (torch_env) — alloc, mask, window, REAL fired-reinforce record+block, boundary")
