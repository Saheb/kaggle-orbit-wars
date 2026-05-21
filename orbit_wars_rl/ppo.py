"""PPO trainer for Orbit Wars Entity Transformer.

Clipped PPO with per-head entropy coefficients, linear warmup + cosine decay,
value clipping, and gradient clipping.
"""

from __future__ import annotations

import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import EntityTransformer, NUM_ANGLE_BINS, NUM_SHIP_BINS
from config import Config


class PPOLearner:
    def __init__(self, model, cfg, device="cpu"):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.ppo.learning_rate,
            eps=1e-5,
        )
        self.total_steps = 0
        self.update_count = 0

    def compute_loss(self, batch, return_metrics=False):
        """Compute PPO clipped loss on a batch.

        batch keys:
            - planet_features, fleet_features, global_features
            - planet_mask, fleet_mask
            - fire_mask, angle_mask, slot_valid, owned_indices, owned_count
            - actions: {fire, angle, ship} — taken action indices
            - old_log_probs: {fire, angle, ships}
            - advantages, returns, old_values
        """
        cfg = self.cfg.ppo

        # Move to device
        def to_dev(x):
            return x.to(self.device) if isinstance(x, torch.Tensor) else x

        planet_features = to_dev(batch["planet_features"])
        fleet_features = to_dev(batch["fleet_features"])
        global_features = to_dev(batch["global_features"])
        planet_mask = to_dev(batch["planet_mask"])
        fleet_mask = to_dev(batch["fleet_mask"])
        fire_mask = to_dev(batch["fire_mask"])
        angle_mask = to_dev(batch["angle_mask"])
        owned_indices = batch["owned_indices"]

        slot_valid_2d = to_dev(batch["slot_valid"])  # (B, max_owned) bool

        outputs = self.model(
            planet_features, fleet_features, global_features,
            planet_mask, fleet_mask,
            fire_mask=fire_mask, angle_mask=angle_mask,
            slot_valid=slot_valid_2d, owned_indices=owned_indices,
            owned_count=batch.get("owned_count"),
        )

        fire_logits = outputs["fire_logits"]
        angle_logits = outputs["angle_logits"]
        ship_logits = outputs["ship_logits"]
        values = outputs["value"]

        # Action distributions
        fire_dist = torch.distributions.Bernoulli(logits=fire_logits)
        angle_dist = torch.distributions.Categorical(logits=angle_logits)
        ship_dist = torch.distributions.Categorical(logits=ship_logits)

        # Actions taken
        fire_action = to_dev(batch["actions"]["fire"])
        angle_action = to_dev(batch["actions"]["angle"])
        ship_action = to_dev(batch["actions"]["ship"])

        # New log probs. Angle/ship choices are only part of the executed action
        # for slots that actually fire; when fire=0 the env ignores them.
        slot_valid = slot_valid_2d.unsqueeze(-1)  # (B, max_owned, 1)
        fired_slots = fire_action.float() * slot_valid.squeeze(-1)

        new_log_prob_fire = fire_dist.log_prob(fire_action.float()) * slot_valid.squeeze(-1)
        new_log_prob_angle = angle_dist.log_prob(angle_action) * fired_slots
        new_log_prob_ships = ship_dist.log_prob(ship_action) * fired_slots

        # Sum across planet slots: (B, max_owned) -> (B,)
        new_log_prob = (new_log_prob_fire + new_log_prob_angle + new_log_prob_ships).sum(dim=-1)

        # Old log probs — same treatment
        old_fire = to_dev(batch["old_log_probs"]["fire"]) * slot_valid.squeeze(-1)
        old_angle = to_dev(batch["old_log_probs"]["angle"]) * fired_slots
        old_ships = to_dev(batch["old_log_probs"]["ships"]) * fired_slots
        old_log_prob = (old_fire + old_angle + old_ships).sum(dim=-1)  # (B,)

        # Advantages
        advantages = to_dev(batch["advantages"])
        if cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss (clipped)
        ratio = torch.exp(new_log_prob - torch.clamp(old_log_prob, min=-50))
        ratio = torch.clamp(ratio, 0.0, 10.0)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        returns = to_dev(batch["returns"])
        old_values = to_dev(batch["old_values"])
        if cfg.clip_value:
            values_clipped = old_values + torch.clamp(values - old_values, -cfg.clip_eps, cfg.clip_eps)
            value_loss = torch.max((values - returns) ** 2, (values_clipped - returns) ** 2).mean()
        else:
            value_loss = ((values - returns) ** 2).mean()

        # Entropy bonuses
        fire_entropy = fire_dist.entropy().mean()
        angle_entropy = angle_dist.entropy().mean()
        ship_entropy = ship_dist.entropy().mean()

        bc_loss = torch.tensor(0.0, device=self.device)
        if cfg.bc_coef > 0 and "bc_targets" in batch:
            bc_fire_target = to_dev(batch["bc_targets"]["fire"]).float()
            bc_angle_target = to_dev(batch["bc_targets"]["angle"])
            bc_ship_target = to_dev(batch["bc_targets"]["ship"])
            bc_fired = (bc_fire_target > 0.5).float() * slot_valid.squeeze(-1)

            bc_fire_loss = F.binary_cross_entropy_with_logits(
                fire_logits.clamp(-30, 30),
                bc_fire_target,
                reduction="none",
            )
            bc_fire_loss = (bc_fire_loss * slot_valid.squeeze(-1)).sum() / slot_valid.squeeze(-1).sum().clamp(min=1)

            B, max_owned, _ = angle_logits.shape
            bc_angle_loss = F.cross_entropy(
                angle_logits.reshape(B * max_owned, -1),
                bc_angle_target.reshape(B * max_owned),
                reduction="none",
            ).reshape(B, max_owned)
            bc_angle_loss = (bc_angle_loss * bc_fired).sum() / bc_fired.sum().clamp(min=1)

            bc_ship_loss = F.cross_entropy(
                ship_logits.reshape(B * max_owned, -1),
                bc_ship_target.reshape(B * max_owned),
                reduction="none",
            ).reshape(B, max_owned)
            bc_ship_loss = (bc_ship_loss * bc_fired).sum() / bc_fired.sum().clamp(min=1)
            bc_loss = bc_fire_loss + bc_angle_loss + bc_ship_loss

        loss = (policy_loss
                + cfg.value_coef * value_loss
                + cfg.bc_coef * bc_loss
                - cfg.entropy_coef_fire * fire_entropy
                - cfg.entropy_coef_angle * angle_entropy
                - cfg.entropy_coef_ships * ship_entropy)

        if return_metrics:
            clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()
            metrics = {
                "loss": loss.item(),
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "fire_entropy": fire_entropy.item(),
                "angle_entropy": angle_entropy.item(),
                "ship_entropy": ship_entropy.item(),
                "bc_loss": bc_loss.item(),
                "clip_frac": clip_frac.item(),
                "approx_kl": (old_log_prob - new_log_prob).mean().item(),
                "mean_advantage": advantages.mean().item(),
                "mean_value": values.mean().item(),
                "mean_return": returns.mean().item(),
            }
            return loss, metrics

        return loss

    def update(self, batches, scheduler=None, ppo_epochs=None):
        """Run PPO update on a list of minibatches."""
        cfg = self.cfg.ppo
        epochs = ppo_epochs or cfg.ppo_epochs

        total_metrics = {}
        for epoch in range(epochs):
            for batch in batches:
                loss, metrics = self.compute_loss(batch, return_metrics=True)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v

        n_updates = epochs * len(batches)
        avg_metrics = {k: v / n_updates for k, v in total_metrics.items()}
        avg_metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        avg_metrics["grad_norm"] = total_metrics.get("grad_norm", 0) / max(n_updates, 1)
        self.update_count += n_updates
        return avg_metrics

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.total_steps = state_dict.get("total_steps", 0)
        self.update_count = state_dict.get("update_count", 0)
