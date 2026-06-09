"""Stage-2 VDN per-planet GAE math.

Verifies `compute_gae_per_planet`:
  1. EQUIVALENCE — a single always-owned planet reduces to standard GAE.
  2. SUM CONSISTENCY — with all planets owned, sum_k A_k == A_total (the global
     GAE on V_total = sum_k V_k), so the decomposition is consistent with the
     centralised critic.
  3. OWNERSHIP GATING — unowned planets get exactly zero advantage.

Run:  python orbit_wars_rl/tests/test_vdn_gae.py
"""
from __future__ import annotations
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_torch import compute_gae, compute_gae_per_planet


def test_single_planet_equals_standard_gae():
    torch.manual_seed(0)
    T, B = 6, 4
    rewards = torch.randn(T, B)
    v = torch.randn(T, B)
    dones = torch.zeros(T, B, dtype=torch.bool); dones[-1] = True
    next_v = torch.randn(B)
    adv_std, _ = compute_gae(rewards, v, dones, next_v, gamma=0.99, lam=0.95)

    owned = torch.ones(T, B, 1)
    adv_pid = compute_gae_per_planet(
        rewards, v.unsqueeze(-1), owned, dones,
        next_v.unsqueeze(-1), torch.ones(B, 1), gamma=0.99, lam=0.95)
    assert torch.allclose(adv_pid.squeeze(-1), adv_std, atol=1e-5), \
        "single always-owned planet must reduce to standard GAE"
    print("  [1] single-planet == standard GAE  ✓")


def test_sum_over_planets_equals_total():
    torch.manual_seed(1)
    T, B, K = 8, 5, 3
    rewards = torch.randn(T, B)
    v_pid = torch.randn(T, B, K)
    owned = torch.ones(T, B, K)            # all owned all steps
    dones = torch.zeros(T, B, dtype=torch.bool); dones[-1] = True
    next_v_pid = torch.randn(B, K)
    adv_pid = compute_gae_per_planet(
        rewards, v_pid, owned, dones, next_v_pid, torch.ones(B, K),
        gamma=0.99, lam=0.95)

    # A_total via standard GAE on V_total = sum_k V_k
    v_tot = v_pid.sum(-1)
    adv_tot, _ = compute_gae(rewards, v_tot, dones, next_v_pid.sum(-1),
                             gamma=0.99, lam=0.95)
    assert torch.allclose(adv_pid.sum(-1), adv_tot, atol=1e-5), \
        "sum_k A_k must equal the global advantage A_total"
    print("  [2] sum_k A_k == A_total  ✓")


def test_unowned_planets_zero():
    torch.manual_seed(2)
    T, B, K = 5, 2, 3
    rewards = torch.randn(T, B)
    v_pid = torch.randn(T, B, K)
    owned = torch.ones(T, B, K)
    owned[:, :, 2] = 0.0                   # planet 2 never owned
    owned[0, 0, 1] = 0.0                   # planet 1 unowned at (t0, env0)
    v_pid = v_pid * owned                  # value 0 where unowned (as in rollout)
    dones = torch.zeros(T, B, dtype=torch.bool); dones[-1] = True
    next_owned = owned[-1].clone()
    adv_pid = compute_gae_per_planet(
        rewards, v_pid, owned, dones, torch.randn(B, K) * next_owned, next_owned,
        gamma=0.99, lam=0.95)
    assert float(adv_pid[:, :, 2].abs().max()) == 0.0, "never-owned planet must be 0"
    assert float(adv_pid[0, 0, 1].abs().max()) == 0.0, "unowned slot must be 0"
    print("  [3] unowned planets get 0 advantage  ✓")


if __name__ == "__main__":
    test_single_planet_equals_standard_gae()
    test_sum_over_planets_equals_total()
    test_unowned_planets_zero()
    print("OK")
