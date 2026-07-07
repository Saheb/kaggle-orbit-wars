"""No-op KL bias (Jake Will Rank-2 anti-spray lever) — unit checks.

Verifies: (a) off ⇒ no effect on the loss + zeroed metrics; (b) on ⇒ the added loss equals
coef·KL(Bern(p_bar)‖Bern(q)) with p_bar = the reported mean launch rate; (c) the term's gradient
pulls an above-target launch rate DOWN toward the prior.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from model import EntityTransformer
from ppo import PPOLearner
from torch_env import VecTorchEnv


def _build_batch(seed=50):
    torch.manual_seed(42)
    cfg = Config(); cfg.device = "cpu"
    model = EntityTransformer(cfg.model)
    learner = PPOLearner(model, cfg, device="cpu")
    env = VecTorchEnv(num_envs=8, num_players=2, device="cpu",
                      episode_steps=500, action_decode="target")
    env.reset(seeds=list(range(seed, seed + 8)))
    feats = env.get_features(0)

    slot_valid = feats["slot_valid"]
    target_mask = feats["target_mask"]
    fire_action = slot_valid.long()
    target_action = target_mask.long().argmax(dim=-1)
    ship_action = torch.zeros_like(fire_action)

    def _fwd():
        return model(
            feats["planet_features"], feats["fleet_features"], feats["global_features"],
            feats["planet_mask"], feats["fleet_mask"], fire_mask=feats["fire_mask"],
            slot_valid=slot_valid, owned_indices=feats["owned_indices"],
            owned_count=feats["owned_count"], pairwise_features=feats.get("pairwise_features"),
        )

    outs = _fwd()
    tl = outs["target_logits"].masked_fill(~target_mask, -1e9)
    target_dist = torch.distributions.Categorical(logits=tl)
    gi = target_action.unsqueeze(-1)
    fl = torch.gather(outs["fire_logits"], -1, gi).squeeze(-1).masked_fill(~feats["fire_mask"], -1e9)
    sl = torch.gather(outs["ship_logits"], 2,
                      gi.unsqueeze(-1).expand(-1, -1, 1, outs["ship_logits"].shape[-1])).squeeze(2)
    sv = slot_valid.float()
    fired = fire_action.float() * sv
    lp_fire = torch.distributions.Bernoulli(logits=fl).log_prob(fire_action.float()) * sv
    lp_ship = torch.distributions.Categorical(logits=sl).log_prob(ship_action) * fired
    lp_target = target_dist.log_prob(target_action) * sv

    batch = {
        "planet_features": feats["planet_features"], "fleet_features": feats["fleet_features"],
        "global_features": feats["global_features"], "planet_mask": feats["planet_mask"],
        "fleet_mask": feats["fleet_mask"], "fire_mask": feats["fire_mask"],
        "target_mask": target_mask, "slot_valid": slot_valid,
        "owned_indices": feats["owned_indices"], "owned_count": feats["owned_count"],
        **({"pairwise_features": feats["pairwise_features"]} if "pairwise_features" in feats else {}),
        "actions": {"fire": fire_action, "ship": ship_action, "target": target_action},
        "old_log_probs": {"fire": lp_fire.detach(), "ships": lp_ship.detach(),
                          "target": lp_target.detach()},
        "advantages": torch.zeros(8), "returns": torch.zeros(8), "old_values": torch.zeros(8),
    }
    return cfg, model, learner, batch


def _bern_kl(p, q):
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def test_off_is_inert():
    """coef=0 ⇒ metrics zeroed and loss finite (the term never enters)."""
    cfg, model, learner, batch = _build_batch()
    cfg.ppo.noop_kl_coef = 0.0
    _, m = learner.compute_loss(batch, return_metrics=True)
    assert m["noop_kl"] == 0.0
    assert m["mean_launch_rate"] == 0.0
    assert math.isfinite(m["loss"])
    print("PASS off_is_inert")


def test_on_adds_exact_kl():
    """coef>0 ⇒ loss = loss_off + coef·KL, and the reported KL matches the closed form."""
    cfg, model, learner, batch = _build_batch()
    cfg.ppo.noop_kl_coef = 0.0
    loss_off, _ = learner.compute_loss(batch, return_metrics=True)
    cfg.ppo.noop_kl_coef = 0.5
    cfg.ppo.noop_target_launch_rate = 0.10
    loss_on, m = learner.compute_loss(batch, return_metrics=True)

    p_bar = m["mean_launch_rate"]
    assert 0.0 < p_bar < 1.0
    expected_kl = _bern_kl(p_bar, 0.10)
    assert abs(m["noop_kl"] - expected_kl) < 1e-4, (m["noop_kl"], expected_kl)
    assert abs((loss_on.item() - loss_off.item()) - 0.5 * m["noop_kl"]) < 1e-4
    print(f"PASS on_adds_exact_kl (p_bar={p_bar:.3f}, kl={m['noop_kl']:.4f})")


def test_gradient_lowers_launch_rate():
    """The term's gradient pulls an above-target launch rate down toward the prior.

    Isolated: advantages=0 (no policy grad) and fire entropy off, so only the no-op KL drives
    the fire logits.
    """
    cfg, model, learner, batch = _build_batch()
    cfg.ppo.entropy_coef_fire = 0.0
    cfg.ppo.noop_kl_coef = 1.0
    cfg.ppo.noop_target_launch_rate = 0.10
    _, m0 = learner.compute_loss(batch, return_metrics=True)
    lr_before = m0["mean_launch_rate"]
    assert lr_before > 0.10, f"fixture launch rate {lr_before} not above target; test invalid"
    for _ in range(5):
        learner.update([batch], kl_target=float("inf"))
    _, m1 = learner.compute_loss(batch, return_metrics=True)
    lr_after = m1["mean_launch_rate"]
    print(f"PASS gradient_lowers_launch_rate ({lr_before:.3f} → {lr_after:.3f})")
    assert lr_after < lr_before


if __name__ == "__main__":
    test_off_is_inert()
    test_on_adds_exact_kl()
    test_gradient_lowers_launch_rate()
