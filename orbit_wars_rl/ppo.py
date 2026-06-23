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


def _gather_target_logits(per_target_logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(per_target_logits, -1, target_idx.unsqueeze(-1)).squeeze(-1)


def _gather_target_ship_logits(per_target_logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        per_target_logits,
        2,
        target_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, per_target_logits.shape[-1]),
    ).squeeze(2)


class PPOLearner:
    def __init__(self, model, cfg, device="cpu", frozen_il_model=None,
                 roi_heads=None, aux_roi_coef=0.0):
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

        # ROI auxiliary regression heads — keep the original ROI/contest pairwise columns
        # (pair_kv.weight[:, 108:111] and target_scorer[0].weight[:, 204:207])
        # anchored to encoding roi_20/roi_50/enemy_contest throughout PPO.
        # Transient: not saved to checkpoint, discarded after training.
        self.roi_heads = roi_heads   # dict with keys "kv" and "ts", or None
        self.aux_roi_coef = float(aux_roi_coef)
        if self.roi_heads is not None:
            for h in self.roi_heads.values():
                h.to(device)

        self.phase4_residual_lr_mult = float(getattr(cfg.ppo, "phase4_residual_lr_mult", 1.0))
        residual_prefixes = (
            "fire_q.", "fire_k.", "fire_scorer.",
            "ship_q.", "ship_k.", "ship_scorer.",
        )
        residual_param_ids = {
            id(p) for name, p in model.named_parameters()
            if any(name.startswith(prefix) for prefix in residual_prefixes)
        }
        base_params = []
        residual_params = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            (residual_params if id(p) in residual_param_ids else base_params).append(p)
        if self.roi_heads is not None and self.aux_roi_coef > 0.0:
            for h in self.roi_heads.values():
                for p in h.parameters():
                    if p.requires_grad:
                        base_params.append(p)

        param_groups = [{"params": base_params, "lr": cfg.ppo.learning_rate}]
        if residual_params:
            param_groups.append({
                "params": residual_params,
                "lr": cfg.ppo.learning_rate * self.phase4_residual_lr_mult,
            })

        self.optimizer = torch.optim.Adam(param_groups, eps=1e-5)
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
                slot_valid=to_dev(batch["slot_valid"]),
                owned_indices=batch["owned_indices"],
                pairwise_features=pairwise,
            )

        slot_valid = to_dev(batch["slot_valid"]).float()    # (B, MO)
        sv_sum = slot_valid.sum().clamp(min=1)

        target_action = to_dev(batch["actions"]["target"])
        curr_fire_logits = _gather_target_logits(current_outputs["fire_logits"], target_action)
        froz_fire_logits = _gather_target_logits(frozen_out["fire_logits"], target_action)
        curr_ship_logits = _gather_target_ship_logits(current_outputs["ship_logits"], target_action)
        froz_ship_logits = _gather_target_ship_logits(frozen_out["ship_logits"], target_action)

        # Fire: Bernoulli KL at the sampled target, averaged over valid slots.
        # KL(Bern(p) || Bern(q)) = p log(p/q) + (1-p) log((1-p)/(1-q))
        p_curr = torch.sigmoid(curr_fire_logits).clamp(1e-6, 1 - 1e-6)
        p_froz = torch.sigmoid(froz_fire_logits).clamp(1e-6, 1 - 1e-6)
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

        ship_kl = cat_kl(curr_ship_logits, froz_ship_logits, slot_valid)
        target_kl = cat_kl(current_outputs["target_logits"], frozen_out["target_logits"], slot_valid)

        return fire_kl + ship_kl + target_kl

    def compute_loss(self, batch, return_metrics=False, value_only=False):
        """Compute PPO clipped loss on a batch (target-decode only).

        batch keys:
            - planet_features, fleet_features, global_features
            - planet_mask, fleet_mask
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
            fire_mask=fire_mask,
            slot_valid=slot_valid_2d, owned_indices=owned_indices,
            owned_count=batch.get("owned_count"),
            pairwise_features=pairwise,
        )

        fire_logits_target = outputs["fire_logits"]
        ship_logits_target = outputs["ship_logits"]
        target_logits = outputs["target_logits"]
        if target_mask is not None:
            target_logits = target_logits.masked_fill(~target_mask, -1e9)
        values = outputs["value"]

        # Actions taken
        fire_action   = to_dev(batch["actions"]["fire"])
        ship_action   = to_dev(batch["actions"]["ship"])
        target_action = to_dev(batch["actions"]["target"])
        fire_logits = _gather_target_logits(fire_logits_target, target_action)
        ship_logits = _gather_target_ship_logits(ship_logits_target, target_action)

        # Action distributions (target-decode: fire, ship, target only — no angle).
        fire_dist   = torch.distributions.Bernoulli(logits=fire_logits)
        ship_dist   = torch.distributions.Categorical(logits=ship_logits)
        target_dist = torch.distributions.Categorical(logits=target_logits)

        # Target is part of the sampled joint action even when fire=0.
        slot_valid  = slot_valid_2d.unsqueeze(-1)   # (B, max_owned, 1)
        fired_slots = fire_action.float() * slot_valid.squeeze(-1)

        new_log_prob_fire   = fire_dist.log_prob(fire_action.float()) * slot_valid.squeeze(-1)
        new_log_prob_ships  = ship_dist.log_prob(ship_action)  * fired_slots
        new_log_prob_target = target_dist.log_prob(target_action) * slot_valid.squeeze(-1)

        # Sum across planet slots: (B, max_owned) -> (B,)
        new_log_prob = (new_log_prob_fire + new_log_prob_ships + new_log_prob_target).sum(dim=-1)

        # Old log probs (stored at rollout time)
        old_fire   = to_dev(batch["old_log_probs"]["fire"])  * slot_valid.squeeze(-1)
        old_ships  = to_dev(batch["old_log_probs"]["ships"]) * fired_slots
        old_target = to_dev(batch["old_log_probs"]["target"]) * slot_valid.squeeze(-1)
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
        # Skipped in value_only (critic warmup) — the policy is frozen, so the IL
        # anchor and entropy/policy terms are irrelevant (and the frozen-IL forward
        # would be wasted compute).
        il_kl = self._il_kl_penalty(batch, outputs) if not value_only else outputs["fire_logits"].new_zeros(())

        if value_only:
            loss = cfg.value_coef * value_loss
        else:
            loss = (policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef_fire  * fire_entropy
                    - cfg.entropy_coef_target * target_entropy
                    - cfg.entropy_coef_ships * ship_entropy
                    + self.il_coef * il_kl)

        if return_metrics:
            clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()
            # Per-slot fire clip_frac: measures whether individual fire decisions
            # would be clipped, decoupled from multi-slot joint-log-prob amplification.
            # The joint clip_frac is mechanically inflated when many owned slots are
            # present (16 slots × per-slot Δlog_p summing up), so this per-slot metric
            # gives a calibrated read of actual policy change per decision.
            with torch.no_grad():
                new_lp_fire_s = fire_dist.log_prob(fire_action.float()) * slot_valid_2d
                old_lp_fire_s = to_dev(batch["old_log_probs"]["fire"]) * slot_valid_2d
                ratio_fire_s = torch.exp(
                    new_lp_fire_s - torch.clamp(old_lp_fire_s, min=-50)
                )
                sv_f = slot_valid_2d.float()
                clip_frac_fire = (
                    ((ratio_fire_s - 1.0).abs() > cfg.clip_eps).float() * sv_f
                ).sum() / sv_f.sum().clamp(min=1)
            with torch.no_grad():
                sv = slot_valid_2d.float()                              # (B, MO)
                sv_sum = sv.sum().clamp(min=1)
                fire_probs = torch.sigmoid(fire_logits)                 # (B, MO)
                slot_valid_count = sv.sum(dim=0).clamp(min=1)           # (MO,)
                per_slot_fire = (fire_probs * sv).sum(dim=0) / slot_valid_count
                fired_mask = (fire_probs > 0.5).float() * sv
                fire_rate_overall = fired_mask.sum() / sv_sum
                fires_per_state = fired_mask.sum(dim=-1)
                # owned_planets: mean planets the agent owns (expansion — the win driver).
                # fire_fraction: on firing steps, fraction of owned planets that fired.
                # This is the TRUE carpet-bomb signal (->1.0 = fire from everything).
                # (srcs_multi removed — empire-size-confounded, never moved real wins.)
                owned_per_state = sv.sum(dim=-1)
                owned_planets = owned_per_state.mean()
                firing = (fires_per_state > 0).float()
                fire_fraction = ((fires_per_state / owned_per_state.clamp(min=1)) * firing).sum() / firing.sum().clamp(min=1)

                ship_argmax = ship_logits.argmax(dim=-1)
                weighted = (ship_argmax == 0).float() * fired_mask
                ship_bin0_rate = weighted.sum() / fired_mask.sum().clamp(min=1)
                mean_ship_bin = (ship_argmax.float() * fired_mask).sum() / fired_mask.sum().clamp(min=1)
                target_valid = target_logits > -1e8
                fire_target_mean = (fire_logits_target.masked_fill(~target_valid, 0.0).sum(dim=-1)
                                    / target_valid.float().sum(dim=-1).clamp(min=1.0))
                fire_target_var = (((fire_logits_target - fire_target_mean.unsqueeze(-1)).masked_fill(~target_valid, 0.0) ** 2).sum(dim=-1)
                                   / target_valid.float().sum(dim=-1).clamp(min=1.0))
                fire_target_std = ((fire_target_var * sv).sum() / sv_sum).sqrt()
                ship_target_scores = ship_logits_target.amax(dim=-1)  # (B, MO, max_planets)
                ship_target_mean = (ship_target_scores.masked_fill(~target_valid, 0.0).sum(dim=-1)
                                    / target_valid.float().sum(dim=-1).clamp(min=1.0))
                ship_target_var = (((ship_target_scores - ship_target_mean.unsqueeze(-1)).masked_fill(~target_valid, 0.0) ** 2).sum(dim=-1)
                                   / target_valid.float().sum(dim=-1).clamp(min=1.0))
                ship_target_std = ((ship_target_var * sv).sum() / sv_sum).sqrt()
                fire_prior = outputs.get("_phase4_fire_prior")
                fire_residual = outputs.get("_phase4_fire_residual")
                ship_prior = outputs.get("_phase4_ship_prior")
                ship_residual = outputs.get("_phase4_ship_residual")
                fire_prior_rms = fire_resid_rms = fire_resid_ratio = fire_resid_abs_mean = 0.0
                ship_prior_rms = ship_resid_rms = ship_resid_ratio = ship_resid_abs_mean = 0.0
                fire_decision_flip = ship_decision_flip = 0.0
                if fire_prior is not None and fire_residual is not None:
                    valid_targets = target_valid & slot_valid_2d.unsqueeze(-1)
                    valid_targets_f = valid_targets.float()
                    vt_sum = valid_targets_f.sum().clamp(min=1.0)
                    fire_prior_rms = (((fire_prior * valid_targets_f) ** 2).sum() / vt_sum).sqrt()
                    fire_resid_rms = (((fire_residual * valid_targets_f) ** 2).sum() / vt_sum).sqrt()
                    fire_resid_ratio = fire_resid_rms / fire_prior_rms.clamp(min=1e-6)
                    fire_resid_abs_mean = (fire_residual.abs() * valid_targets_f).sum() / vt_sum
                    fire_prior_logits = _gather_target_logits(fire_prior, target_action)
                    fire_decision_flip = (
                        (((fire_prior_logits > 0) != (fire_logits > 0)).float() * sv).sum()
                        / sv_sum
                    )
                if ship_prior is not None and ship_residual is not None:
                    valid_targets_bins = target_valid.unsqueeze(-1) & slot_valid_2d.unsqueeze(-1).unsqueeze(-1)
                    min_ship_bin = int(getattr(self.model.cfg, "min_ship_bin", 0))
                    if min_ship_bin > 0:
                        valid_targets_bins = valid_targets_bins.clone()
                        valid_targets_bins[..., :min_ship_bin] = False
                    valid_targets_bins_f = valid_targets_bins.float()
                    vtb_sum = valid_targets_bins_f.sum().clamp(min=1.0)
                    ship_prior_rms = (((ship_prior * valid_targets_bins_f) ** 2).sum() / vtb_sum).sqrt()
                    ship_resid_rms = (((ship_residual * valid_targets_bins_f) ** 2).sum() / vtb_sum).sqrt()
                    ship_resid_ratio = ship_resid_rms / ship_prior_rms.clamp(min=1e-6)
                    ship_resid_abs_mean = (ship_residual.abs() * valid_targets_bins_f).sum() / vtb_sum
                    ship_prior_logits = _gather_target_ship_logits(ship_prior, target_action)
                    ship_slots = (((fire_logits > 0) | (fire_prior_logits > 0)).float() * sv
                                  if fire_prior is not None else sv)
                    ship_decision_flip = (
                        ((ship_prior_logits.argmax(dim=-1) != ship_logits.argmax(dim=-1)).float() * ship_slots).sum()
                        / ship_slots.sum().clamp(min=1.0)
                    )

            metrics = {
                "loss": loss.item(),
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "fire_entropy": fire_entropy.item(),
                "target_entropy": target_entropy.item(),
                "ship_entropy": ship_entropy.item(),
                "clip_frac": clip_frac.item(),
                "clip_frac_fire": clip_frac_fire.item(),
                "approx_kl": (old_log_prob - new_log_prob).mean().item(),
                "mean_advantage": advantages.mean().item(),
                "mean_value": values.mean().item(),
                "mean_return": returns.mean().item(),
                "fire_rate_overall": fire_rate_overall.item(),
                "owned_planets": owned_planets.item(),
                "fire_fraction": fire_fraction.item(),
                "ship_bin0_rate": ship_bin0_rate.item(),
                "mean_ship_bin": mean_ship_bin.item(),
                "fire_target_std": fire_target_std.item(),
                "ship_target_std": ship_target_std.item(),
                "phase4_fire_prior_rms": float(fire_prior_rms),
                "phase4_ship_prior_rms": float(ship_prior_rms),
                "phase4_fire_resid_rms": float(fire_resid_rms),
                "phase4_ship_resid_rms": float(ship_resid_rms),
                "phase4_fire_resid_ratio": float(fire_resid_ratio),
                "phase4_ship_resid_ratio": float(ship_resid_ratio),
                "phase4_fire_resid_abs_mean": float(fire_resid_abs_mean),
                "phase4_ship_resid_abs_mean": float(ship_resid_abs_mean),
                "phase4_fire_decision_flip": float(fire_decision_flip),
                "phase4_ship_decision_flip": float(ship_decision_flip),
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

                # ROI auxiliary loss: keep pair_kv.weight[:, 108:111] and
                # target_scorer[0].weight[:, 204:207] encoding roi/contest info.
                # Gradient flows only through those columns (other weights not in graph).
                if self.roi_heads is not None and self.aux_roi_coef > 0.0:
                    pairwise = batch.get("pairwise_features")
                    if pairwise is not None:
                        pairwise = pairwise.to(self.device)          # (B, MO, N_p, F_pair)
                        x_new = pairwise[..., 12:15].reshape(-1, 3)  # (B*MO*N_p, 3)
                        # pair_kv branch — only new columns in computational graph
                        contrib_kv = F.linear(x_new, self.model.pair_kv.weight[:, 108:111])
                        pred_kv = self.roi_heads["kv"](contrib_kv)
                        # target_scorer branch
                        contrib_ts = F.linear(x_new, self.model.target_scorer[0].weight[:, 204:207])
                        pred_ts = self.roi_heads["ts"](contrib_ts)
                        aux_loss = F.mse_loss(pred_kv, x_new) + F.mse_loss(pred_ts, x_new)
                        total_loss = total_loss + self.aux_roi_coef * aux_loss

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
        avg_metrics["phase4_residual_learning_rate"] = self.get_phase4_residual_lr()
        avg_metrics["kl_early_stop"] = float(early_stopped)
        self.update_count += n_updates
        return avg_metrics

    # Value-head parameter prefixes — the ONLY params trained during critic warmup.
    # The shared trunk + policy heads stay frozen so the BC policy is byte-for-byte
    # untouched while the value head fits the frozen features.
    _VALUE_HEAD_PREFIXES = ("value_fc1", "value_fc2", "value_out")

    def _set_value_only(self, on: bool) -> None:
        for name, p in self.model.named_parameters():
            p.requires_grad = (name.split(".")[0] in self._VALUE_HEAD_PREFIXES) if on else True

    def value_warmup_update(self, batches):
        """Critic-only warmup (BC warmstart: trained policy + UNtrained critic).
        Freeze the trunk + policy heads; fit ONLY the value head on the frozen BC
        features for one pass over the minibatches. No scheduler step (keep LR
        steady) and no IL/entropy/policy terms. Returns minimal logging metrics."""
        self._set_value_only(True)
        sum_vl, n = 0.0, 0
        for batch in batches:
            loss, metrics = self.compute_loss(batch, return_metrics=True, value_only=True)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.cfg.ppo.max_grad_norm)
            self.optimizer.step()
            sum_vl += metrics["value_loss"]
            n += 1
        self._set_value_only(False)
        self.update_count += n
        return {"value_loss": sum_vl / max(n, 1),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "phase4_residual_learning_rate": self.get_phase4_residual_lr(),
                "kl_early_stop": 0.0, "critic_warmup": 1.0}

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def get_phase4_residual_lr(self):
        if len(self.optimizer.param_groups) < 2:
            return self.optimizer.param_groups[0]["lr"]
        return self.optimizer.param_groups[1]["lr"]

    def state_dict(self):
        # Save model-arch config alongside weights so loaders can rebuild the
        # right shape (num_ship_bins for fraction-head, min_ship_bin for mask).
        model_cfg = getattr(self.cfg, "model", None)
        cfg_blob = {}
        if model_cfg is not None:
            cfg_blob = {
                "num_ship_bins": int(getattr(model_cfg, "num_ship_bins", 32)),
                "min_ship_bin": int(getattr(model_cfg, "min_ship_bin", 0)),
                "pairwise_feature_dim": int(getattr(model_cfg, "pairwise_feature_dim", 0)),
                "ship_bin_mode": str(getattr(model_cfg, "ship_bin_mode", "absolute")),
                "action_decode": str(getattr(model_cfg, "action_decode", "angle")),
                "allow_reinforce": bool(getattr(model_cfg, "allow_reinforce", False)),
                "game_phase_features": bool(getattr(model_cfg, "game_phase_features", False)),
                # Reinforce / sufficient-commit DISCIPLINE — eval & export must mask the SAME way
                # the ckpt was trained or the policy self-sabotages. Persist so they auto-load
                # instead of relying on CLI flags being remembered (a panel/submission footgun).
                "reinforce_gate_min_planets": int(getattr(model_cfg, "reinforce_gate_min_planets", 0)),
                "reinforce_forward_only": bool(getattr(model_cfg, "reinforce_forward_only", False)),
                "reverse_edge_cooldown": int(getattr(model_cfg, "reverse_edge_cooldown", 0)),
                "reinforce_garrison_floor": float(getattr(model_cfg, "reinforce_garrison_floor", 0.0)),
                "sufficient_commit_factor": float(getattr(model_cfg, "sufficient_commit_factor", 0.0)),
                "redundant_target_factor": float(getattr(model_cfg, "redundant_target_factor", 0.0)),
                # provenance: how the ckpt was trained (eval always clamps, so not an eval-contract field)
                "ship_overflow_mode": str(getattr(model_cfg, "ship_overflow_mode", "drop")),
                # reward-shaping provenance only; eval/export do not consume these.
                "capture_utility_coef": float(getattr(model_cfg, "capture_utility_coef", 0.0)),
                "capture_utility_window": int(getattr(model_cfg, "capture_utility_window", 30)),
                "capture_idle_penalty": float(getattr(model_cfg, "capture_idle_penalty", 0.0)),
                "phase4_residual_init_std": float(getattr(model_cfg, "phase4_residual_init_std", 0.0)),
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
