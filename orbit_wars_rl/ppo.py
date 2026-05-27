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
    def __init__(self, model, cfg, device="cpu", frozen_il_model=None):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        # Frozen reference policy for IL regularization (KL-to-BC penalty).
        # Held in eval mode, no gradients. Set via training entrypoint when
        # cfg.ppo.il_lambda > 0.
        self.frozen_il_model = frozen_il_model
        if self.frozen_il_model is not None:
            self.frozen_il_model.to(device).eval()
            for p in self.frozen_il_model.parameters():
                p.requires_grad_(False)
        # Current IL coefficient — schedule updated externally each iter
        self.il_coef = float(getattr(cfg.ppo, "il_lambda", 0.0)) if frozen_il_model is not None else 0.0

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.ppo.learning_rate,
            eps=1e-5,
        )
        self.total_steps = 0
        self.update_count = 0

    def set_il_coef(self, coef: float) -> None:
        """Update the IL coefficient (typically called per iter for schedule)."""
        self.il_coef = float(coef)

    def _il_kl_penalty(self, batch, current_outputs) -> torch.Tensor:
        """KL(π_current || π_frozen_BC) on fire/ship/target heads, masked to valid
        slots. Returns scalar tensor; 0.0 if no frozen ref.

        Angle is not part of the executed policy (target-decode only) so its KL
        is not included.
        """
        if self.frozen_il_model is None or self.il_coef <= 0:
            return torch.zeros((), device=self.device)

        def to_dev(x):
            return x.to(self.device) if isinstance(x, torch.Tensor) else x

        with torch.no_grad():
            pairwise = batch.get("pairwise_features")
            if pairwise is not None:
                pairwise = to_dev(pairwise)
            frozen_out = self.frozen_il_model(
                to_dev(batch["planet_features"]),
                to_dev(batch["fleet_features"]),
                to_dev(batch["global_features"]),
                to_dev(batch["planet_mask"]),
                to_dev(batch["fleet_mask"]),
                fire_mask=to_dev(batch["fire_mask"]),
                angle_mask=to_dev(batch["angle_mask"]),
                slot_valid=to_dev(batch["slot_valid"]),
                owned_indices=batch["owned_indices"],
                pairwise_features=pairwise,
            )

        slot_valid = to_dev(batch["slot_valid"]).float()    # (B, MO)
        sv_sum = slot_valid.sum().clamp(min=1)

        # Fire: Bernoulli KL per slot, averaged over valid slots.
        # KL(Bern(p) || Bern(q)) = p log(p/q) + (1-p) log((1-p)/(1-q))
        p_curr = torch.sigmoid(current_outputs["fire_logits"]).clamp(1e-6, 1 - 1e-6)
        p_froz = torch.sigmoid(frozen_out["fire_logits"]).clamp(1e-6, 1 - 1e-6)
        fire_kl = (p_curr * (p_curr / p_froz).log()
                   + (1 - p_curr) * ((1 - p_curr) / (1 - p_froz)).log())
        fire_kl = (fire_kl * slot_valid).sum() / sv_sum

        # Ship / target: Categorical KL on logits, only for valid slots.
        def cat_kl(curr_logits, froz_logits, slot_mask):
            log_curr = torch.log_softmax(curr_logits, dim=-1)
            log_froz = torch.log_softmax(froz_logits, dim=-1)
            p_curr_ = log_curr.exp()
            kl_per = (p_curr_ * (log_curr - log_froz)).sum(dim=-1)  # (B, MO)
            return (kl_per * slot_mask).sum() / slot_mask.sum().clamp(min=1)

        ship_kl = cat_kl(current_outputs["ship_logits"], frozen_out["ship_logits"], slot_valid)
        target_kl = cat_kl(current_outputs["target_logits"], frozen_out["target_logits"], slot_valid)

        return fire_kl + ship_kl + target_kl

    def compute_loss(self, batch, return_metrics=False):
        """Compute PPO clipped loss on a batch (target-decode only).

        batch keys:
            - planet_features, fleet_features, global_features
            - planet_mask, fleet_mask, angle_mask (passed to model; not used for action)
            - fire_mask, slot_valid, owned_indices, owned_count
            - target_mask
            - actions: {fire, ship, target}
            - old_log_probs: {fire, ships, target}
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
        angle_mask = to_dev(batch["angle_mask"])  # still passed to model forward
        target_mask = batch.get("target_mask")
        if target_mask is not None:
            target_mask = to_dev(target_mask)
        owned_indices = batch["owned_indices"]

        slot_valid_2d = to_dev(batch["slot_valid"])  # (B, max_owned) bool

        pairwise = batch.get("pairwise_features")
        if pairwise is not None:
            pairwise = to_dev(pairwise)
        outputs = self.model(
            planet_features, fleet_features, global_features,
            planet_mask, fleet_mask,
            fire_mask=fire_mask, angle_mask=angle_mask,
            slot_valid=slot_valid_2d, owned_indices=owned_indices,
            owned_count=batch.get("owned_count"),
            pairwise_features=pairwise,
        )

        fire_logits = outputs["fire_logits"]
        ship_logits = outputs["ship_logits"]
        target_logits = outputs["target_logits"]
        if target_mask is not None:
            target_logits = target_logits.masked_fill(~target_mask, -1e9)
        values = outputs["value"]

        # Action distributions (target-decode: fire, ship, target only — no angle).
        fire_dist   = torch.distributions.Bernoulli(logits=fire_logits)
        ship_dist   = torch.distributions.Categorical(logits=ship_logits)
        target_dist = torch.distributions.Categorical(logits=target_logits)

        # Actions taken
        fire_action   = to_dev(batch["actions"]["fire"])
        ship_action   = to_dev(batch["actions"]["ship"])
        target_action = to_dev(batch["actions"]["target"])

        # Log probs — ship/target only matter for slots that actually fired.
        slot_valid  = slot_valid_2d.unsqueeze(-1)   # (B, max_owned, 1)
        fired_slots = fire_action.float() * slot_valid.squeeze(-1)

        new_log_prob_fire   = fire_dist.log_prob(fire_action.float()) * slot_valid.squeeze(-1)
        new_log_prob_ships  = ship_dist.log_prob(ship_action)  * fired_slots
        new_log_prob_target = target_dist.log_prob(target_action) * fired_slots

        # Sum across planet slots: (B, max_owned) -> (B,)
        new_log_prob = (new_log_prob_fire + new_log_prob_ships + new_log_prob_target).sum(dim=-1)

        # Old log probs (stored at rollout time)
        old_fire   = to_dev(batch["old_log_probs"]["fire"])  * slot_valid.squeeze(-1)
        old_ships  = to_dev(batch["old_log_probs"]["ships"]) * fired_slots
        old_target = to_dev(batch["old_log_probs"]["target"]) * fired_slots
        old_log_prob = (old_fire + old_ships + old_target).sum(dim=-1)  # (B,)

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

        # Entropy bonuses (direction = target)
        fire_entropy   = fire_dist.entropy().mean()
        ship_entropy   = ship_dist.entropy().mean()
        target_entropy = target_dist.entropy().mean()

        # IL regularization: KL(π_current || π_frozen_BC) on rollout states.
        il_kl = self._il_kl_penalty(batch, outputs)

        loss = (policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef_fire  * fire_entropy
                - cfg.entropy_coef_angle * target_entropy   # cfg key reused for direction head
                - cfg.entropy_coef_ships * ship_entropy
                + self.il_coef * il_kl)

        if return_metrics:
            clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()
            with torch.no_grad():
                sv = slot_valid_2d.float()                              # (B, MO)
                sv_sum = sv.sum().clamp(min=1)
                fire_probs = torch.sigmoid(fire_logits)                 # (B, MO)
                slot_valid_count = sv.sum(dim=0).clamp(min=1)           # (MO,)
                per_slot_fire = (fire_probs * sv).sum(dim=0) / slot_valid_count
                fired_mask = (fire_probs > 0.5).float() * sv
                fire_rate_overall = fired_mask.sum() / sv_sum
                fires_per_state = fired_mask.sum(dim=-1)
                multi_owned = (sv.sum(dim=-1) >= 2).float()
                multi_owned_sum = multi_owned.sum().clamp(min=1)
                avg_sources_when_multi = (fires_per_state * multi_owned).sum() / multi_owned_sum

                ship_argmax = ship_logits.argmax(dim=-1)
                weighted = (ship_argmax == 0).float() * fired_mask
                ship_bin0_rate = weighted.sum() / fired_mask.sum().clamp(min=1)
                mean_ship_bin = (ship_argmax.float() * fired_mask).sum() / fired_mask.sum().clamp(min=1)

            metrics = {
                "loss": loss.item(),
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "fire_entropy": fire_entropy.item(),
                "target_entropy": target_entropy.item(),
                "ship_entropy": ship_entropy.item(),
                "clip_frac": clip_frac.item(),
                "approx_kl": (old_log_prob - new_log_prob).mean().item(),
                "mean_advantage": advantages.mean().item(),
                "mean_value": values.mean().item(),
                "mean_return": returns.mean().item(),
                "fire_rate_overall": fire_rate_overall.item(),
                "avg_sources_multi": avg_sources_when_multi.item(),
                "ship_bin0_rate": ship_bin0_rate.item(),
                "mean_ship_bin": mean_ship_bin.item(),
                "il_kl": il_kl.item() if isinstance(il_kl, torch.Tensor) else float(il_kl),
                "il_coef": self.il_coef,
                "per_slot_fire_probs": per_slot_fire.detach().cpu().tolist(),
            }
            return loss, metrics

        return loss

    def update(self, batches, scheduler=None, ppo_epochs=None,
               kl_target: float = 0.05, bc_batch: dict | None = None):
        """Run PPO update on a list of minibatches.

        kl_target:  stop epoch loop early if mean approx-KL exceeds this value,
                    preventing destructive policy updates.
                    Good range: 0.01–0.05. Pass float('inf') to disable.
        bc_batch:   optional dict of BC training samples (from bc._collate).
                    If provided, a BC forward pass is combined into the same
                    backward as each PPO minibatch (single optimizer step),
                    preventing competing gradient updates.
        """
        cfg = self.cfg.ppo
        epochs = ppo_epochs or cfg.ppo_epochs

        sum_metrics: dict[str, float] = {}
        n_updates = 0
        early_stopped = False

        for epoch in range(epochs):
            epoch_kl = 0.0
            for batch in batches:
                ppo_loss, metrics = self.compute_loss(batch, return_metrics=True)

                # BC regularization: separate forward pass, combined backward.
                # This avoids the batch-size mismatch of embedding bc_targets
                # inside the PPO forward, while still using a single optimizer step.
                bc_loss_val = 0.0
                if bc_batch is not None and cfg.bc_coef > 0:
                    from bc import bc_loss as _bc_loss
                    bc_pairwise = bc_batch.get("pairwise_features")
                    if bc_pairwise is not None:
                        bc_pairwise = bc_pairwise.to(self.device)
                    bc_out = self.model(
                        bc_batch["planet_features"].to(self.device),
                        bc_batch["fleet_features"].to(self.device),
                        bc_batch["global_features"].to(self.device),
                        bc_batch["planet_mask"].to(self.device),
                        bc_batch["fleet_mask"].to(self.device),
                        fire_mask=bc_batch["fire_mask"].to(self.device),
                        angle_mask=bc_batch["angle_mask"].to(self.device),
                        slot_valid=bc_batch["slot_valid"].to(self.device),
                        owned_indices=bc_batch["owned_indices"],
                        pairwise_features=bc_pairwise,
                    )
                    bc_loss_tensor, bc_m = _bc_loss(bc_out, {
                        k: v.to(self.device) for k, v in bc_batch.items()
                    })
                    total_loss = ppo_loss + cfg.bc_coef * bc_loss_tensor
                    bc_loss_val = bc_m["loss"]
                    for k, v in bc_m.items():
                        metrics[f"bc_{k}"] = v
                else:
                    total_loss = ppo_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                for k, v in metrics.items():
                    if isinstance(v, list):
                        # Element-wise sum for list-valued metrics (per-slot etc.)
                        existing = sum_metrics.get(k)
                        if existing is None:
                            sum_metrics[k] = list(v)
                        else:
                            sum_metrics[k] = [a + b for a, b in zip(existing, v)]
                    else:
                        sum_metrics[k] = sum_metrics.get(k, 0.0) + v
                epoch_kl += metrics.get("approx_kl", 0.0)
                n_updates += 1

            # KL early stopping: if this epoch's mean KL exceeded the target,
            # bail out before the next epoch to prevent policy collapse.
            if (epoch_kl / max(len(batches), 1)) > kl_target:
                early_stopped = True
                break

        n = max(n_updates, 1)
        avg_metrics = {
            k: ([x / n for x in v] if isinstance(v, list) else v / n)
            for k, v in sum_metrics.items()
        }
        avg_metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        avg_metrics["kl_early_stop"] = float(early_stopped)
        self.update_count += n_updates
        return avg_metrics

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self):
        # Save model-arch config alongside weights so loaders can rebuild the
        # right shape (num_ship_bins for fraction-head, min_ship_bin for mask).
        model_cfg = getattr(self.cfg, "model", None)
        cfg_blob = {}
        if model_cfg is not None:
            cfg_blob = {
                "num_ship_bins": int(getattr(model_cfg, "num_ship_bins", 32)),
                "min_ship_bin": int(getattr(model_cfg, "min_ship_bin", 0)),
                "ship_bin_mode": str(getattr(model_cfg, "ship_bin_mode", "absolute")),
                "action_decode": str(getattr(model_cfg, "action_decode", "angle")),
            }
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "update_count": self.update_count,
            "config": cfg_blob,
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.total_steps = state_dict.get("total_steps", 0)
        self.update_count = state_dict.get("update_count", 0)
