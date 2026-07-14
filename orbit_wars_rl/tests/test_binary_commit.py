"""Binary NOOP/COMMIT action semantics and train/eval resolver parity."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from action_mask import (actions_from_target_policy, compute_action_masks,
                         resolve_binary_commit_np)
from binary_policy import (binary_action_entropy, binary_action_log_probs,
                           binary_taken_log_prob)
from config import Config
from model import EntityTransformer
from ppo import PPOLearner
from torch_env import VecTorchEnv, _resolve_binary_commit
from train_torch import binary_rollout_metrics, sample_action_batched


def _pairwise(src_ships, target_ships, *, enemy=False, own=False, production=0):
    pw = np.zeros((1, 2, 26), dtype=np.float32)
    pw[0, 1, 5] = float(own)
    pw[0, 1, 6] = float(enemy)
    pw[0, 1, 7] = float(not own and not enemy)
    pw[0, 1, 8] = production / 5.0
    pw[0, 1, 10] = target_ships / 200.0
    pw[0, 1, 24] = 1.0 / 200.0
    pw[0, :, 25] = src_ships / 200.0
    return pw


def test_binary_resolver_semantics_and_torch_parity():
    cases = [
        (_pairwise(1, 0), 1, 0),        # low-source launch is always NOOP
        (_pairwise(10, 6), 10, 10),     # affordable neutral attack is all-in
        (_pairwise(5, 6), 5, 0),        # unaffordable attack is NOOP
        (_pairwise(20, 4, enemy=True, production=2), 20, 20),
        (_pairwise(10, 0, own=True), 10, 0),  # maintain=1 is below the commit floor
    ]
    for pw, source, expected_target in cases:
        np_sizes, np_ok = resolve_binary_commit_np(pw, np.array([source], dtype=np.float32))
        t_sizes, t_ok = _resolve_binary_commit(
            torch.from_numpy(pw), torch.tensor([source], dtype=torch.float32))
        assert np.array_equal(np_sizes, t_sizes.numpy())
        assert np.array_equal(np_ok, t_ok.numpy())
        assert np_sizes[0, 1] == expected_target


def _decode(source_ships):
    obs = {
        "player": 0,
        "step": 0,
        "planets": [
            [0, 0, 20.0, 50.0, 2.0, source_ships, 1],
            [1, -1, 30.0, 50.0, 2.0, 6, 1],
        ],
        "fleets": [],
        "angular_velocity": 0.0,
    }
    masks = compute_action_masks(obs, 0)
    fire = torch.full((1, 16, 2), 20.0)
    target = torch.full((1, 16, 2), -20.0)
    target[:, :, 1] = 20.0
    ships = torch.zeros(1, 16, 2, 4)
    pairwise = np.zeros((16, 2, 26), dtype=np.float32)
    pairwise[0] = _pairwise(source_ships, 6)[0]
    return actions_from_target_policy(
        fire, target, ships, masks, obs, 0,
        ship_bin_mode="binary", pairwise_features=pairwise,
    )


def test_binary_eval_decode_all_in_or_noop():
    assert _decode(5) == []
    moves = _decode(10)
    assert len(moves) == 1
    assert moves[0][2] == 10


def test_binary_sampler_has_no_ship_action_and_no_idle_target_credit():
    outputs = {
        "target_logits": torch.zeros(1, 2, 2),
        "fire_logits": torch.tensor([[[100.0, 100.0], [-100.0, -100.0]]]),
        "ship_logits": torch.randn(1, 2, 2, 4),
    }
    fire, _, ship, _, lp_action, lp_ship, lp_target = sample_action_batched(
        outputs, torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, 2, dtype=torch.bool), "binary",
    )
    assert torch.equal(fire, torch.tensor([[1, 0]]))
    assert torch.equal(ship, torch.zeros_like(ship))
    assert torch.equal(lp_ship, torch.zeros_like(lp_ship))
    assert lp_target[0, 1] == 0
    assert torch.isclose(lp_action[0, 0].exp(), torch.tensor(0.5), atol=1e-5)
    assert torch.isclose(lp_action[0, 1], torch.tensor(0.0), atol=1e-5)


def test_binary_executed_action_distribution_is_exactly_marginalized():
    target_logits = torch.log(torch.tensor([[[0.75, 0.25]]]))
    fire_logits = torch.logit(torch.tensor([[[0.10, 0.90]]]))
    log_noop, log_commit = binary_action_log_probs(
        target_logits, fire_logits, fire_mask=torch.tensor([[True]]))

    assert torch.allclose(log_commit.exp(), torch.tensor([[[0.075, 0.225]]]), atol=1e-6)
    assert torch.allclose(log_noop.exp(), torch.tensor([[0.700]]), atol=1e-6)
    assert torch.allclose(
        log_noop.exp() + log_commit.exp().sum(dim=-1), torch.ones_like(log_noop), atol=1e-6)
    assert torch.allclose(
        binary_taken_log_prob(log_noop, log_commit, torch.tensor([[0]]), torch.tensor([[1]])).exp(),
        torch.tensor([[0.700]]), atol=1e-6)
    expected_h = -(torch.tensor([0.700, 0.075, 0.225])
                   * torch.tensor([0.700, 0.075, 0.225]).log()).sum()
    assert torch.allclose(binary_action_entropy(log_noop, log_commit), expected_h.view(1, 1))


def test_binary_rollout_and_ppo_log_prob_parity():
    torch.manual_seed(7)
    cfg = Config()
    cfg.device = "cpu"
    cfg.model.ship_bin_mode = "binary"
    model = EntityTransformer(cfg.model)
    learner = PPOLearner(model, cfg, device="cpu")
    env = VecTorchEnv(num_envs=4, num_players=2, device="cpu", episode_steps=100,
                      action_decode="target", ship_bin_mode="binary")
    env.reset(seeds=[10, 11, 12, 13])
    feats = env.get_features(0)
    outputs = model(
        feats["planet_features"], feats["fleet_features"], feats["global_features"],
        feats["planet_mask"], feats["fleet_mask"], fire_mask=feats["fire_mask"],
        slot_valid=feats["slot_valid"], owned_indices=feats["owned_indices"],
        owned_count=feats["owned_count"], pairwise_features=feats["pairwise_features"],
    )
    fire, _, ship, target, lp_action, lp_ship, lp_target = sample_action_batched(
        outputs, feats["fire_mask"], feats["target_mask"], "binary")
    batch = {
        "planet_features": feats["planet_features"],
        "fleet_features": feats["fleet_features"],
        "global_features": feats["global_features"],
        "planet_mask": feats["planet_mask"],
        "fleet_mask": feats["fleet_mask"],
        "fire_mask": feats["fire_mask"],
        "target_mask": feats["target_mask"],
        "slot_valid": feats["slot_valid"],
        "owned_indices": feats["owned_indices"],
        "owned_count": feats["owned_count"],
        "pairwise_features": feats["pairwise_features"],
        "actions": {"fire": fire, "ship": ship, "target": target},
        "old_log_probs": {"fire": lp_action.detach(), "ships": lp_ship.detach(),
                          "target": lp_target.detach()},
        "advantages": torch.zeros(4),
        "returns": torch.zeros(4),
        "old_values": outputs["value"].detach(),
    }
    _, metrics = learner.compute_loss(batch, return_metrics=True)
    assert abs(metrics["approx_kl"]) < 1e-6


def test_binary_metrics_report_actionable_noop_and_commit_mass():
    pairwise = torch.zeros(1, 2, 2, 26)
    pairwise[0, 0, 1, 7] = 1.0
    pairwise[0, 1, 1, 5] = 1.0
    metrics = binary_rollout_metrics({
        "slot_valid": torch.ones(1, 2, dtype=torch.bool),
        "fire_mask": torch.ones(1, 2, dtype=torch.bool),
        "fire_a": torch.tensor([[1, 0]]),
        "ship_count_a": torch.tensor([[20.0, 0.0]]),
        "target_a": torch.tensor([[1, 1]]),
        "pairwise_features": pairwise,
    })
    assert metrics["binary_actionable_source_rate"] == 1.0
    assert metrics["binary_noop_rate"] == 0.5
    assert metrics["binary_commit_ships_mean"] == 20.0
    assert metrics["binary_attack_share"] == 1.0
