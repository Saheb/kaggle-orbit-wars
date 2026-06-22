"""Unit test for the COMA counterfactual Q-head (model.EntityTransformer.q_counterfactual).

The wall is a credit-assignment fixed point: ppo.py puts ONE scalar advantage on the summed
joint log-prob, so a source-slot's fire is never credited its marginal contribution (docs/q-head.md).
The Q-head prices each slot's marginal via the additive mean-pool counterfactual. This test pins the
two things that have to be exact for the gradient to mean anything:

  (1) the additive-pool delta equals a brute-force full-pool recompute, per slot, for FIRE and IDLE
      (so A_i = q_sa - (p·q_fire + (1-p)·q_idle) is the true counterfactual, not an approximation);
  (2) q_all_idle equals the brute-force all-valid-slots-idle pool (the Q(s,a)-Q(s,all-idle) probe);
  (3) q_out zero-init → every Q output is exactly 0 → A_i=0 at step 0 (no policy disruption pre-warmup);
  (4) the new q_* keys are registered so a pre-Q-head checkpoint still load_state_dict(strict=False)s.

Run:  <venv>/bin/python orbit_wars_rl/tests/test_q_head.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn as nn

from config import ModelConfig
from model import EntityTransformer, PHASE4_COMPAT_MISSING_KEYS

torch.manual_seed(0)

B, N_p, N_f = 4, 12, 6


def _build():
    cfg = ModelConfig()
    model = EntityTransformer(cfg).eval()
    D = cfg.entity_dim
    MO = cfg.max_owned_planets
    planet_features = torch.randn(B, N_p, cfg.planet_feature_dim)
    fleet_features = torch.randn(B, N_f, cfg.fleet_feature_dim)
    global_features = torch.randn(B, cfg.global_feature_dim)
    planet_mask = torch.ones(B, N_p, dtype=torch.bool)
    fleet_mask = torch.ones(B, N_f, dtype=torch.bool)

    # Per-row varying owned counts (3..6) → exercises the n_valid denominator + masking.
    slot_valid = torch.zeros(B, MO, dtype=torch.bool)
    owned_indices = torch.zeros(B, MO, dtype=torch.long)
    fire_bit = torch.zeros(B, MO, dtype=torch.long)
    ship_bin = torch.zeros(B, MO, dtype=torch.long)
    target_idx = torch.zeros(B, MO, dtype=torch.long)
    for r in range(B):
        n = 3 + r
        slot_valid[r, :n] = True
        owned_indices[r, :n] = torch.arange(n)
        fire_bit[r, :n] = torch.randint(0, 2, (n,))
        ship_bin[r, :n] = torch.randint(0, cfg.num_ship_bins, (n,))
        target_idx[r, :n] = torch.randint(0, N_p, (n,))

    encoded = model.encode_state(
        planet_features, fleet_features, global_features,
        planet_mask, fleet_mask, slot_valid=slot_valid, owned_indices=owned_indices,
    )
    return model, encoded, fire_bit, ship_bin, target_idx, slot_valid, D, MO


def test_shapes_and_zero_init():
    """Fresh model: q_out is zero-init → all Q outputs are exactly 0 (A_i=0 at step 0)."""
    model, encoded, fire_bit, ship_bin, target_idx, slot_valid, D, MO = _build()
    q = model.q_counterfactual(encoded, fire_bit, ship_bin, target_idx, slot_valid)
    assert q["q_sa"].shape == (B,)
    assert q["q_all_idle"].shape == (B,)
    assert q["q_fire"].shape == (B, MO)
    assert q["q_idle"].shape == (B, MO)
    for k, v in q.items():
        assert torch.allclose(v, torch.zeros_like(v)), f"{k} must be 0 under q_out zero-init, got {v}"
    print("OK shapes + q_out zero-init → Q≡0 (A_i=0 at step 0)")


def test_counterfactual_pool_delta_parity():
    """The additive mean-pool delta must equal a brute-force full-pool recompute, per slot,
    for BOTH force-fire and force-idle. Randomize q_out/q_fc so Q is non-trivial."""
    model, encoded, fire_bit, ship_bin, target_idx, slot_valid, D, MO = _build()
    nn.init.normal_(model.q_out.weight, std=0.5)
    nn.init.normal_(model.q_out.bias, std=0.5)
    nn.init.normal_(model.q_fc.weight, std=0.3)

    q = model.q_counterfactual(encoded, fire_bit, ship_bin, target_idx, slot_valid)

    oe = encoded["owned_enriched"]
    pep = encoded["planet_emb"]
    gt = encoded["global_token"]
    sa_fire, sa_idle = model._q_slot_tokens(oe, pep, ship_bin, target_idx)
    sv = slot_valid.float()
    n_valid = sv.sum(dim=1).clamp(min=1.0)
    sa_taken = torch.where((fire_bit > 0).unsqueeze(-1), sa_fire, sa_idle)

    def brute_pool_q(sa_swapped):
        pool = (sa_swapped * sv.unsqueeze(-1)).sum(dim=1) / n_valid.unsqueeze(-1)
        return model._q_from_pool(gt, pool)

    for i in range(MO):
        sa_bf_fire = sa_taken.clone(); sa_bf_fire[:, i, :] = sa_fire[:, i, :]
        sa_bf_idle = sa_taken.clone(); sa_bf_idle[:, i, :] = sa_idle[:, i, :]
        assert torch.allclose(brute_pool_q(sa_bf_fire), q["q_fire"][:, i], atol=1e-4), f"q_fire slot {i}"
        assert torch.allclose(brute_pool_q(sa_bf_idle), q["q_idle"][:, i], atol=1e-4), f"q_idle slot {i}"

    # invalid slots (sv_i=0) must be no-ops → q_fire_i == q_idle_i == q_sa.
    for r in range(B):
        n = 3 + r
        for i in range(n, MO):
            assert torch.allclose(q["q_fire"][r, i], q["q_sa"][r], atol=1e-4)
            assert torch.allclose(q["q_idle"][r, i], q["q_sa"][r], atol=1e-4)
    print("OK additive-pool delta == brute-force recompute (fire & idle, valid & invalid slots)")


def test_all_idle_parity():
    """q_all_idle == brute-force pool of every valid slot's idle token."""
    model, encoded, fire_bit, ship_bin, target_idx, slot_valid, D, MO = _build()
    nn.init.normal_(model.q_out.weight, std=0.5)
    nn.init.normal_(model.q_out.bias, std=0.5)
    q = model.q_counterfactual(encoded, fire_bit, ship_bin, target_idx, slot_valid)

    _, sa_idle = model._q_slot_tokens(
        encoded["owned_enriched"], encoded["planet_emb"], ship_bin, target_idx)
    sv = slot_valid.float()
    pool = (sa_idle * sv.unsqueeze(-1)).sum(dim=1) / sv.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
    q_all_idle_bf = model._q_from_pool(encoded["global_token"], pool)
    assert torch.allclose(q_all_idle_bf, q["q_all_idle"], atol=1e-4)
    print("OK q_all_idle == brute-force all-idle pool")


def test_checkpoint_compat():
    """A pre-Q-head checkpoint (state_dict without q_* keys) must still load strict=False with
    every missing key registered in PHASE4_COMPAT_MISSING_KEYS (else resume/eval/export raise)."""
    cfg = ModelConfig()
    model = EntityTransformer(cfg)
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith("q_")}
    fresh = EntityTransformer(cfg)
    missing, unexpected = fresh.load_state_dict(sd, strict=False)
    q_missing = [k for k in missing if k.startswith("q_")]
    assert set(missing) == set(q_missing), f"only q_* keys may be missing, got {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    bad = [k for k in missing if k not in PHASE4_COMPAT_MISSING_KEYS]
    assert not bad, f"unregistered missing keys would abort resume/eval/export: {bad}"
    print(f"OK checkpoint-compat: {len(q_missing)} q_* keys all registered, no abort")


if __name__ == "__main__":
    test_shapes_and_zero_init()
    test_counterfactual_pool_delta_parity()
    test_all_idle_parity()
    test_checkpoint_compat()
    print("PASS: COMA Q-head — additive-pool delta parity, all-idle probe, zero-init, checkpoint-compat")
