"""Pre-training sanity checks for the VecTorchEnv + PPO self-play loop.

Two validations before committing GPU time:
  A. Env symmetry — random-vs-random play should be near 50/50 with seat-swap.
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


# ----------------------------------------------------------------------------
# Test A — Env symmetry under seat swap
# ----------------------------------------------------------------------------

def _act(model, env, player):
    feats = env.get_features(player)
    with torch.no_grad():
        out = model(
            feats["planet_features"], feats["fleet_features"], feats["global_features"],
            feats["planet_mask"], feats["fleet_mask"],
            fire_mask=feats["fire_mask"], angle_mask=feats["angle_mask"],
            slot_valid=feats["slot_valid"], owned_indices=feats["owned_indices"],
            owned_count=feats["owned_count"],
        )
    fl = out["fire_logits"].masked_fill(~feats["fire_mask"], -1e9)
    al = out["angle_logits"].masked_fill(~feats["angle_mask"], -1e9)
    f = torch.distributions.Bernoulli(logits=fl).sample().long()
    a = torch.distributions.Categorical(logits=al).sample()
    s = torch.distributions.Categorical(logits=out["ship_logits"]).sample()
    return torch.stack([f, a, s], dim=-1)


def _play_set(m_p0, m_p1, n_games=128, seed_base=1000):
    env = VecTorchEnv(num_envs=n_games, num_players=2, device="cpu", episode_steps=500)
    env.reset(seeds=list(range(seed_base, seed_base + n_games)))
    w0 = w1 = d = 0
    done_count = 0
    for _ in range(600):
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
    print("Test A: Env symmetry under seat swap (random vs random, n=128)")
    print("=" * 60)
    cfg = Config()
    torch.manual_seed(1); m1 = EntityTransformer(cfg.model); m1.eval()
    torch.manual_seed(2); m2 = EntityTransformer(cfg.model); m2.eval()

    a0, a1, _, na = _play_set(m1, m2, seed_base=1000)
    b0, b1, _, nb = _play_set(m2, m1, seed_base=2000)
    print(f"  m1@slot0 vs m2@slot1: P0 wins {a0}/{na} = {a0/na:.1%}")
    print(f"  m2@slot0 vs m1@slot1: P0 wins {b0}/{nb} = {b0/nb:.1%}")
    # Symmetric env → both rates near 50% ± 4% (std err for n=128).
    # Up to ±10% acceptable (game-state variance can be high).
    ok = abs(a0/na - 0.5) < 0.10 and abs(b0/nb - 0.5) < 0.10
    print("PASS" if ok else "FAIL", "— P0 win rates near 50% (allowance 10%)")
    return ok


# ----------------------------------------------------------------------------
# Test B — Gradient direction sanity
# ----------------------------------------------------------------------------

def test_gradient_direction():
    print("\n" + "=" * 60)
    print("Test B: Positive advantage on fire-action ⇒ fire prob increases")
    print("=" * 60)
    torch.manual_seed(42)
    cfg = Config(); cfg.device = "cpu"; cfg.ppo.bc_coef = 0.0

    model = EntityTransformer(cfg.model)
    learner = PPOLearner(model, cfg, device="cpu")
    env = VecTorchEnv(num_envs=8, num_players=2, device="cpu", episode_steps=500)
    env.reset(seeds=list(range(50, 58)))

    feats = env.get_features(0)
    with torch.no_grad():
        out0 = model(
            feats["planet_features"], feats["fleet_features"], feats["global_features"],
            feats["planet_mask"], feats["fleet_mask"],
            fire_mask=feats["fire_mask"], angle_mask=feats["angle_mask"],
            slot_valid=feats["slot_valid"], owned_indices=feats["owned_indices"],
            owned_count=feats["owned_count"],
        )

    fire_action = feats["fire_mask"].long()
    angle_action = torch.zeros(8, MAX_OWNED, dtype=torch.long)
    ship_action = torch.zeros(8, MAX_OWNED, dtype=torch.long)

    fl, al, sl = out0["fire_logits"], out0["angle_logits"], out0["ship_logits"]
    old_fire_lp = (
        torch.distributions.Bernoulli(logits=fl).log_prob(fire_action.float())
        * feats["slot_valid"].float()
    )
    fired = fire_action.float() * feats["slot_valid"].float()
    old_angle_lp = torch.distributions.Categorical(logits=al).log_prob(angle_action) * fired
    old_ship_lp = torch.distributions.Categorical(logits=sl).log_prob(ship_action) * fired

    batch = {
        "planet_features": feats["planet_features"],
        "fleet_features":  feats["fleet_features"],
        "global_features": feats["global_features"],
        "planet_mask":     feats["planet_mask"],
        "fleet_mask":      feats["fleet_mask"],
        "fire_mask":       feats["fire_mask"],
        "angle_mask":      feats["angle_mask"],
        "slot_valid":      feats["slot_valid"],
        "owned_indices":   feats["owned_indices"],
        "owned_count":     feats["owned_count"],
        "actions": {"fire": fire_action, "angle": angle_action, "ship": ship_action},
        "old_log_probs": {
            "fire": old_fire_lp.detach(),
            "angle": old_angle_lp.detach(),
            "ships": old_ship_lp.detach(),
        },
        "advantages": torch.full((8,), 5.0),
        "returns":    torch.full((8,), 5.0),
        "old_values": torch.zeros(8),
    }

    prob_before = torch.sigmoid(fl[:, 0]).detach().clone()
    for _ in range(4):
        learner.update([batch], kl_target=float("inf"))
    with torch.no_grad():
        out1 = model(
            feats["planet_features"], feats["fleet_features"], feats["global_features"],
            feats["planet_mask"], feats["fleet_mask"],
            fire_mask=feats["fire_mask"], angle_mask=feats["angle_mask"],
            slot_valid=feats["slot_valid"], owned_indices=feats["owned_indices"],
            owned_count=feats["owned_count"],
        )
    prob_after = torch.sigmoid(out1["fire_logits"][:, 0])
    delta = (prob_after - prob_before).mean().item()
    print(f"  Fire prob[slot 0] mean delta (after-before): {delta:+.4f}")
    ok = delta > 0.05
    print("PASS" if ok else "FAIL", "— +adv must push fire prob up by >0.05")
    return ok


if __name__ == "__main__":
    a = test_env_symmetry()
    b = test_gradient_direction()
    print("\n" + "=" * 60)
    print(f"OVERALL: {'PASS ✓' if (a and b) else 'FAIL ✗'}")
