"""Stage-1 planet-centric PPO: per-slot (per-planet) clipped surrogate.

Verifies the delta in ppo.compute_loss that replaces the joint log-prob ratio
(product over all owned planets) with an independent per-slot ratio sharing the
global advantage (MAPPO factorisation).

Key property under test: clip_frac now reflects the PER-SLOT importance ratio,
independent of how many planets fire — so it is no longer mechanically inflated
by empire size (the pathology the old clip_frac_fire metric was a workaround for).

Run:  python -m pytest orbit_wars_rl/tests/test_per_slot_ppo.py -x -q
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from model import EntityTransformer
from ppo import PPOLearner
from torch_env import VecTorchEnv


def _multi_planet_state(num_envs=8, n_owned=4):
    """Build a state where player 0 owns `n_owned` planets (→ n_owned valid slots)."""
    env = VecTorchEnv(num_envs=num_envs, num_players=2, device="cpu",
                      episode_steps=500, action_decode="target")
    env.reset(seeds=list(range(100, 100 + num_envs)))
    # Hand player 0 the first n_owned (alive) planets, player 1 the next one.
    for i in range(n_owned):
        env.planets[:, i, 1] = 0
        env.planets[:, i, 5] = 80.0
    env.planets[:, n_owned, 1] = 1
    env.planets[:, n_owned, 5] = 80.0
    return env


def _build_controlled_batch(env, model, delta: float, head: str = "fire"):
    """Fire from every owned planet; set old_log_probs = new − delta on `head` per
    slot so each valid slot's per-slot importance ratio on that head is exp(delta).
    head ∈ {"fire", "ship", "target"}."""
    feats = env.get_features(0, max_planets=48, max_fleets=128)
    with torch.no_grad():
        out = model(
            feats["planet_features"], feats["fleet_features"], feats["global_features"],
            feats["planet_mask"], feats["fleet_mask"],
            fire_mask=feats["fire_mask"], slot_valid=feats["slot_valid"],
            owned_indices=feats["owned_indices"], owned_count=feats["owned_count"],
            pairwise_features=feats.get("pairwise_features"),
        )
    sv = feats["slot_valid"].float()
    fire_action = feats["fire_mask"].long()                      # fire every valid slot
    fired = fire_action.float() * sv
    # Match compute_loss exactly: it masks target_logits with target_mask before
    # building the target distribution (ppo.py). Replicate so the ship/target
    # ratios are exactly 1 and only the fire head carries the controlled delta.
    tgt_logits = out["target_logits"]
    if feats.get("target_mask") is not None:
        tgt_logits = tgt_logits.masked_fill(~feats["target_mask"], -1e9)
    ship_action = out["ship_logits"].argmax(-1)
    target_action = tgt_logits.argmax(-1)

    fire_lp = torch.distributions.Bernoulli(logits=out["fire_logits"]).log_prob(fire_action.float())
    ship_lp = torch.distributions.Categorical(logits=out["ship_logits"]).log_prob(ship_action)
    target_lp = torch.distributions.Categorical(logits=tgt_logits).log_prob(target_action)

    # old = new − delta on the chosen head → that head's per-slot ratio = exp(delta).
    d_fire = delta if head == "fire" else 0.0
    d_ship = delta if head == "ship" else 0.0
    d_tgt  = delta if head == "target" else 0.0
    batch = {
        "planet_features": feats["planet_features"], "fleet_features": feats["fleet_features"],
        "global_features": feats["global_features"], "planet_mask": feats["planet_mask"],
        "fleet_mask": feats["fleet_mask"], "fire_mask": feats["fire_mask"],
        "target_mask": feats.get("target_mask"), "slot_valid": feats["slot_valid"],
        "owned_indices": feats["owned_indices"], "owned_count": feats["owned_count"],
        "pairwise_features": feats.get("pairwise_features"),
        "actions": {"fire": fire_action, "ship": ship_action, "target": target_action},
        "old_log_probs": {
            "fire": (fire_lp - d_fire).detach(),
            "ships": (ship_lp - d_ship).detach(),
            "target": (target_lp - d_tgt).detach(),
        },
        "advantages": torch.ones(env.num_envs),     # positive advantage
        "returns": torch.zeros(env.num_envs),
        "old_values": torch.zeros(env.num_envs),
    }
    n_valid = sv.sum(-1)
    return batch, n_valid


def test_clip_frac_is_per_slot_not_joint():
    """With every env owning ≥2 firing planets and per-slot ratio inside the clip
    band, clip_frac must be ~0 — proving it is NOT the joint product (which would
    exceed the band and clip)."""
    torch.manual_seed(0)
    cfg = Config(); cfg.device = "cpu"
    model = EntityTransformer(cfg.model); model.eval()
    learner = PPOLearner(model, cfg, device="cpu")
    env = _multi_planet_state(num_envs=8, n_owned=4)

    clip_eps = cfg.ppo.clip_eps  # 0.2 → band [0.8, 1.2]

    # delta inside the per-slot band: exp(0.15)=1.162 < 1.2. Joint over 4 slots
    # would be exp(0.60)=1.82 ≫ 1.2 → would clip if the ratio were joint.
    batch_in, n_valid = _build_controlled_batch(env, model, delta=0.15)
    assert (n_valid >= 2).all(), "test needs multi-planet envs"
    _, m_in = learner.compute_loss(batch_in, return_metrics=True)
    assert math.isfinite(m_in["loss"])
    assert m_in["clip_frac"] < 0.05, (
        f"per-slot ratio 1.162 is inside band → clip_frac should be ~0, "
        f"got {m_in['clip_frac']:.3f} (joint product would have clipped)")

    # delta outside the per-slot band: exp(0.30)=1.35 > 1.2 → every valid slot clips.
    batch_out, _ = _build_controlled_batch(env, model, delta=0.30)
    _, m_out = learner.compute_loss(batch_out, return_metrics=True)
    assert m_out["clip_frac"] > 0.95, (
        f"per-slot ratio 1.35 exceeds band → clip_frac should be ~1, "
        f"got {m_out['clip_frac']:.3f}")

    # approx_kl (joint) must still be the slot-count-scaled sum (early-stop relies
    # on it); approx_kl_slot is the calibrated per-slot value (magnitude ~delta).
    # Sign is negative here (old = new − delta), so compare magnitudes.
    assert abs(m_in["approx_kl"]) > abs(m_in["approx_kl_slot"]) * 1.5
    assert abs(abs(m_in["approx_kl_slot"]) - 0.15) < 0.05
    print(f"  clip_frac: in-band={m_in['clip_frac']:.3f}  out-band={m_out['clip_frac']:.3f}  "
          f"approx_kl joint={m_in['approx_kl']:.3f} slot={m_in['approx_kl_slot']:.3f}")


def test_ship_credit_is_joint_not_per_slot():
    """Ship-size is credited with a JOINT ratio (summed over fired slots), NOT
    per-slot. With ≥2 firing planets and a per-slot ship delta INSIDE the band,
    the joint ratio exp(k·delta) exceeds the band → clip_frac_ship ≈ 1, proving it
    is not per-slot (which would stay ≈ 0). Meanwhile clip_frac (fire/target) stays
    ≈ 0, since the ship delta does not enter the fire/target surrogate."""
    torch.manual_seed(0)
    cfg = Config(); cfg.device = "cpu"
    model = EntityTransformer(cfg.model); model.eval()
    learner = PPOLearner(model, cfg, device="cpu")
    env = _multi_planet_state(num_envs=8, n_owned=4)

    # delta inside the per-slot band: exp(0.15)=1.162 < 1.2. Joint over ≥2 fired
    # slots: exp(2·0.15)=1.35 > 1.2 → joint ship ratio clips; per-slot would not.
    batch, n_valid = _build_controlled_batch(env, model, delta=0.15, head="ship")
    assert (n_valid >= 2).all(), "test needs ≥2 firing planets so k·delta exits band"
    _, m = learner.compute_loss(batch, return_metrics=True)
    assert math.isfinite(m["loss"])
    assert m["clip_frac_ship"] > 0.95, (
        f"joint ship ratio exp(k·0.15) exceeds band → clip_frac_ship≈1, "
        f"got {m['clip_frac_ship']:.3f} (per-slot ship credit would be ≈0)")
    assert m["clip_frac"] < 0.05, (
        f"ship delta must not enter the fire/target surrogate → clip_frac≈0, "
        f"got {m['clip_frac']:.3f}")
    print(f"  clip_frac_ship(joint)={m['clip_frac_ship']:.3f}  "
          f"clip_frac_ft(per-slot)={m['clip_frac']:.3f}")


def test_gradient_direction_per_slot():
    """Positive advantage on fired planets must increase their fire probability."""
    torch.manual_seed(1)
    cfg = Config(); cfg.device = "cpu"; cfg.ppo.bc_coef = 0.0
    # All envs share one advantage value, so normalization (zero std) would zero
    # the policy gradient — disable it to isolate the sign of the update.
    cfg.ppo.normalize_advantages = False
    model = EntityTransformer(cfg.model)
    learner = PPOLearner(model, cfg, device="cpu")
    env = _multi_planet_state(num_envs=8, n_owned=4)

    batch, _ = _build_controlled_batch(env, model, delta=0.0)  # ratio 1 at start
    batch["advantages"] = torch.full((env.num_envs,), 5.0)

    feats_in = {k: batch[k] for k in ("planet_features", "fleet_features", "global_features",
                                       "planet_mask", "fleet_mask", "fire_mask", "slot_valid",
                                       "owned_indices", "owned_count")}
    feats_in["pairwise_features"] = batch["pairwise_features"]
    with torch.no_grad():
        p_before = torch.sigmoid(model(**feats_in)["fire_logits"])
    for _ in range(5):
        learner.update([batch], kl_target=float("inf"))
    with torch.no_grad():
        p_after = torch.sigmoid(model(**feats_in)["fire_logits"])

    sv = batch["slot_valid"].float()
    delta = (((p_after - p_before) * sv).sum() / sv.sum()).item()
    print(f"  mean fire-prob delta on valid slots: {delta:+.4f}")
    assert delta > 0.02, f"+adv must raise fire prob, got {delta:+.4f}"


if __name__ == "__main__":
    test_clip_frac_is_per_slot_not_joint()
    test_ship_credit_is_joint_not_per_slot()
    test_gradient_direction_per_slot()
    print("OK")
