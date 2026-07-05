"""Pre-training sanity checks for the VecTorchEnv + PPO self-play loop (target-decode).

Two validations before committing GPU time:
  A. Env symmetry — random-init-model vs random-init-model should be near 50/50
     under seat swap (catches gross seat asymmetry in the env/reward path).
  B. Gradient direction — positive advantage on an action must increase
     that action's probability after a PPO update.

Run:
    python tests/test_train_validation.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from model import EntityTransformer
from ppo import PPOLearner
from torch_env import VecTorchEnv, MAX_OWNED
from train_torch import sample_action_batched


def _forward(model, feats):
    with torch.no_grad():
        return model(
            feats["planet_features"], feats["fleet_features"], feats["global_features"],
            feats["planet_mask"], feats["fleet_mask"],
            fire_mask=feats["fire_mask"],
            slot_valid=feats["slot_valid"], owned_indices=feats["owned_indices"],
            owned_count=feats["owned_count"],
            pairwise_features=feats.get("pairwise_features"),
        )


# ----------------------------------------------------------------------------
# Test A — Env symmetry under seat swap
# ----------------------------------------------------------------------------

def _act(model, env, player):
    feats = env.get_features(player)
    outs = _forward(model, feats)
    fire_a, angle_a, ship_a, target_a, *_ = sample_action_batched(
        outs, feats["fire_mask"], feats.get("target_mask"))
    return torch.stack([fire_a, angle_a, ship_a, target_a], dim=-1)


def _play_set(m_p0, m_p1, n_games=64, seed_base=1000, episode_steps=200):
    env = VecTorchEnv(num_envs=n_games, num_players=2, device="cpu",
                      episode_steps=episode_steps, action_decode="target")
    env.reset(seeds=list(range(seed_base, seed_base + n_games)))
    w0 = w1 = d = 0
    done_count = 0
    for _ in range(episode_steps + 50):
        a0 = _act(m_p0, env, 0)
        a1 = _act(m_p1, env, 1)
        _, rewards, done = env.step({0: a0, 1: a1})
        for i in torch.where(done)[0].tolist():
            r0, r1 = rewards[i, 0].item(), rewards[i, 1].item()
            if r0 > r1: w0 += 1
            elif r1 > r0: w1 += 1
            else: d += 1
            done_count += 1
        if done_count >= n_games:
            break
    return w0, w1, d, done_count


def test_env_symmetry():
    print("=" * 60)
    print("Test A: Env symmetry under seat swap (random-init models, n=64)")
    print("=" * 60)
    cfg = Config()
    torch.manual_seed(1); m1 = EntityTransformer(cfg.model); m1.eval()
    torch.manual_seed(2); m2 = EntityTransformer(cfg.model); m2.eval()

    a0, a1, _, na = _play_set(m1, m2, seed_base=1000)
    b0, b1, _, nb = _play_set(m2, m1, seed_base=2000)
    print(f"  m1@seat0 vs m2@seat1: P0 wins {a0}/{na} = {a0/na:.1%}")
    print(f"  m2@seat0 vs m1@seat1: P0 wins {b0}/{nb} = {b0/nb:.1%}")
    # Symmetric env → both rates near 50%. n=64 → std err ~6%; allow ±15% (this is a
    # gross-asymmetry smoke test, not a statistical one).
    ok = abs(a0/na - 0.5) < 0.15 and abs(b0/nb - 0.5) < 0.15
    print("PASS" if ok else "FAIL", "— P0 win rates near 50% (allowance 15%)")
    assert ok
    return ok


# ----------------------------------------------------------------------------
# Test B — Gradient direction sanity
# ----------------------------------------------------------------------------

def test_gradient_direction():
    print("\n" + "=" * 60)
    print("Test B: Positive advantage on fire-action ⇒ fire prob increases")
    print("=" * 60)
    torch.manual_seed(42)
    cfg = Config(); cfg.device = "cpu"

    model = EntityTransformer(cfg.model)
    learner = PPOLearner(model, cfg, device="cpu")
    env = VecTorchEnv(num_envs=8, num_players=2, device="cpu",
                      episode_steps=500, action_decode="target")
    env.reset(seeds=list(range(50, 58)))

    feats = env.get_features(0)
    out0 = _forward(model, feats)

    slot_valid = feats["slot_valid"]                                    # (B, MO) bool
    target_mask = feats["target_mask"]                                  # (B, MO, P) bool
    # Deterministic joint action: fire on every valid slot, first legal target, ship bin 0.
    fire_action = slot_valid.long()
    target_action = target_mask.long().argmax(dim=-1)                   # first legal target
    ship_action = torch.zeros_like(fire_action)

    def _log_probs(outs):
        tl = outs["target_logits"].masked_fill(~target_mask, -1e9)
        target_dist = torch.distributions.Categorical(logits=tl)
        gi = target_action.unsqueeze(-1)
        fl = torch.gather(outs["fire_logits"], -1, gi).squeeze(-1)
        fl = fl.masked_fill(~feats["fire_mask"], -1e9)
        sl = torch.gather(outs["ship_logits"], 2,
                          gi.unsqueeze(-1).expand(-1, -1, 1, outs["ship_logits"].shape[-1])
                          ).squeeze(2)
        sv = slot_valid.float()
        fired = fire_action.float() * sv
        lp_fire = torch.distributions.Bernoulli(logits=fl).log_prob(fire_action.float()) * sv
        lp_ship = torch.distributions.Categorical(logits=sl).log_prob(ship_action) * fired
        lp_target = target_dist.log_prob(target_action) * sv
        return fl, lp_fire, lp_ship, lp_target

    fl0, lp_fire, lp_ship, lp_target = _log_probs(out0)

    batch = {
        "planet_features": feats["planet_features"],
        "fleet_features":  feats["fleet_features"],
        "global_features": feats["global_features"],
        "planet_mask":     feats["planet_mask"],
        "fleet_mask":      feats["fleet_mask"],
        "fire_mask":       feats["fire_mask"],
        "target_mask":     target_mask,
        "slot_valid":      slot_valid,
        "owned_indices":   feats["owned_indices"],
        "owned_count":     feats["owned_count"],
        **({"pairwise_features": feats["pairwise_features"]}
           if "pairwise_features" in feats else {}),
        "actions": {"fire": fire_action, "ship": ship_action, "target": target_action},
        "old_log_probs": {
            "fire":   lp_fire.detach(),
            "ships":  lp_ship.detach(),
            "target": lp_target.detach(),
        },
        "advantages": torch.full((8,), 5.0),
        "returns":    torch.full((8,), 5.0),
        "old_values": torch.zeros(8),
    }

    prob_before = torch.sigmoid(fl0[:, 0]).detach().clone()
    for _ in range(4):
        learner.update([batch], kl_target=float("inf"))
    out1 = _forward(model, feats)
    fl1, *_ = _log_probs(out1)
    prob_after = torch.sigmoid(fl1[:, 0])
    delta = (prob_after - prob_before).mean().item()
    print(f"  Fire prob[slot 0] mean delta (after-before): {delta:+.4f}")
    # Direction is the property under test. The magnitude saturates: old_log_probs are
    # fixed across the 4 update calls, so the PPO ratio clip caps total movement
    # (~+0.04 at the current clip/LR config).
    ok = delta > 0.02
    print("PASS" if ok else "FAIL", "— +adv must push fire prob up by >0.02")
    assert ok
    return ok


if __name__ == "__main__":
    a = test_env_symmetry()
    b = test_gradient_direction()
    print("\n" + "=" * 60)
    print(f"OVERALL: {'PASS ✓' if (a and b) else 'FAIL ✗'}")
