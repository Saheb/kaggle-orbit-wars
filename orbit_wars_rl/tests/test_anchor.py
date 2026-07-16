"""Best-checkpoint anchor (Isaiah #1 / Yijie #13) — unit checks.

Verifies: (a) no anchor installed ⇒ inert; (b) anchor == live weights ⇒ KL exactly 0 (the
identity case that makes a fresh promotion a no-op); (c) a different anchor ⇒ KL > 0 and the
loss grows by coef·KL; (d) the value term is the MSE to the frozen best's value; (e) the KL's
gradient pulls the live policy back toward the anchor — the actual anti-drift mechanism;
(f) non-binary mode raises rather than silently anchoring the wrong distribution.
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from model import EntityTransformer
from ppo import PPOLearner
from torch_env import VecTorchEnv


def _build_batch(seed=50, n=8):
    """Binary-mode batch: the anchor is implemented over the exact NOOP/COMMIT distribution."""
    torch.manual_seed(42)
    cfg = Config(); cfg.device = "cpu"
    cfg.model.ship_bin_mode = "binary"
    model = EntityTransformer(cfg.model)
    learner = PPOLearner(model, cfg, device="cpu")
    env = VecTorchEnv(num_envs=n, num_players=2, device="cpu",
                      episode_steps=500, action_decode="target",
                      ship_bin_mode="binary")
    env.reset(seeds=list(range(seed, seed + n)))
    feats = env.get_features(0)

    slot_valid = feats["slot_valid"]
    target_mask = feats["target_mask"]
    fire_action = feats["fire_mask"].long()
    target_action = target_mask.long().argmax(dim=-1)

    batch = {
        "planet_features": feats["planet_features"], "fleet_features": feats["fleet_features"],
        "global_features": feats["global_features"], "planet_mask": feats["planet_mask"],
        "fleet_mask": feats["fleet_mask"], "fire_mask": feats["fire_mask"],
        "target_mask": target_mask, "slot_valid": slot_valid,
        "owned_indices": feats["owned_indices"], "owned_count": feats["owned_count"],
        **({"pairwise_features": feats["pairwise_features"]} if "pairwise_features" in feats else {}),
        "actions": {"fire": fire_action, "ship": torch.zeros_like(fire_action),
                    "target": target_action},
        "old_log_probs": {"fire": torch.zeros_like(fire_action, dtype=torch.float),
                          "ships": torch.zeros_like(fire_action, dtype=torch.float),
                          "target": torch.zeros_like(fire_action, dtype=torch.float)},
        "advantages": torch.zeros(n), "returns": torch.zeros(n), "old_values": torch.zeros(n),
    }
    return cfg, model, learner, batch


def _perturbed(model, scale=0.5):
    """A state_dict that is a genuinely different policy from `model`."""
    torch.manual_seed(7)
    return {k: v + scale * torch.randn_like(v) if v.is_floating_point() else v
            for k, v in model.state_dict().items()}


def test_no_anchor_is_inert():
    cfg, model, learner, batch = _build_batch()
    cfg.ppo.anchor_kl_coef = 0.1
    cfg.ppo.anchor_value_coef = 0.5
    _, m = learner.compute_loss(batch, return_metrics=True)   # set_anchor never called
    assert m["anchor_kl"] == 0.0 and m["anchor_value"] == 0.0
    assert math.isfinite(m["loss"])


def test_identical_anchor_has_zero_kl():
    """Anchor == live ⇒ KL 0 and the value MSE 0: a just-promoted anchor adds no force."""
    cfg, model, learner, batch = _build_batch()
    learner.set_anchor(model.state_dict())
    cfg.ppo.anchor_kl_coef = 0.1
    cfg.ppo.anchor_value_coef = 0.5
    _, m = learner.compute_loss(batch, return_metrics=True)
    assert m["anchor_kl"] < 1e-6, m["anchor_kl"]
    assert m["anchor_value"] < 1e-6, m["anchor_value"]


def test_different_anchor_adds_coef_times_kl():
    cfg, model, learner, batch = _build_batch()
    cfg.ppo.anchor_kl_coef = 0.0
    cfg.ppo.anchor_value_coef = 0.0
    loss_off, _ = learner.compute_loss(batch, return_metrics=True)

    learner.set_anchor(_perturbed(model))
    cfg.ppo.anchor_kl_coef = 0.25
    loss_on, m = learner.compute_loss(batch, return_metrics=True)

    assert m["anchor_kl"] > 0.0, "a different policy must have positive KL"
    assert abs((loss_on.item() - loss_off.item()) - 0.25 * m["anchor_kl"]) < 1e-4
    print(f"PASS anchor KL={m['anchor_kl']:.4f}")


def test_value_term_is_mse_to_anchor():
    cfg, model, learner, batch = _build_batch()
    cfg.ppo.anchor_kl_coef = 0.0
    cfg.ppo.anchor_value_coef = 0.0
    loss_off, _ = learner.compute_loss(batch, return_metrics=True)

    learner.set_anchor(_perturbed(model))
    cfg.ppo.anchor_value_coef = 0.5
    loss_on, m = learner.compute_loss(batch, return_metrics=True)
    assert m["anchor_value"] > 0.0
    assert abs((loss_on.item() - loss_off.item()) - 0.5 * m["anchor_value"]) < 1e-4


def test_kl_gradient_pulls_toward_anchor():
    """THE mechanism: descending the anchor KL moves the live policy toward the anchor.

    Isolated — advantages are zero and every other coefficient is off, so the anchor KL is the
    only force on the weights.
    """
    cfg, model, learner, batch = _build_batch()
    cfg.ppo.anchor_kl_coef = 1.0
    cfg.ppo.anchor_value_coef = 0.0
    cfg.ppo.value_coef = 0.0
    cfg.ppo.entropy_coef_fire = 0.0
    cfg.ppo.entropy_coef_target = 0.0
    cfg.ppo.entropy_coef_ships = 0.0
    cfg.ppo.noop_kl_coef = 0.0
    cfg.ppo.ship_kl_coef = 0.0
    learner.set_anchor(_perturbed(model, scale=0.2))

    _, m_before = learner.compute_loss(batch, return_metrics=True)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    for _ in range(5):
        loss, _ = learner.compute_loss(batch, return_metrics=True)
        opt.zero_grad(); loss.backward(); opt.step()
    _, m_after = learner.compute_loss(batch, return_metrics=True)

    assert m_after["anchor_kl"] < m_before["anchor_kl"], (
        f"KL did not fall: {m_before['anchor_kl']:.5f} → {m_after['anchor_kl']:.5f}")
    print(f"PASS KL {m_before['anchor_kl']:.5f} → {m_after['anchor_kl']:.5f}")


def test_non_binary_anchor_raises():
    """Factorized fire/ship/target anchoring is not wired — fail loudly, never silently anchor
    the wrong distribution."""
    torch.manual_seed(42)
    cfg = Config(); cfg.device = "cpu"
    model = EntityTransformer(cfg.model)          # default ship_bin_mode = absolute
    learner = PPOLearner(model, cfg, device="cpu")
    learner.set_anchor(model.state_dict())
    cfg.ppo.anchor_kl_coef = 0.1

    _, _, _, binary_batch = _build_batch()
    with pytest.raises(NotImplementedError, match="binary"):
        learner.compute_loss(binary_batch, return_metrics=True)


if __name__ == "__main__":
    test_no_anchor_is_inert()
    test_identical_anchor_has_zero_kl()
    test_different_anchor_adds_coef_times_kl()
    test_value_term_is_mse_to_anchor()
    test_kl_gradient_pulls_toward_anchor()
    test_non_binary_anchor_raises()
    print("OK")
