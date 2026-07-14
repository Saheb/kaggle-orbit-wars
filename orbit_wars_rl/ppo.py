"""PPO trainer for Orbit Wars Entity Transformer.

Clipped PPO with per-head entropy coefficients, linear warmup + cosine decay,
value clipping, and gradient clipping.
"""

from __future__ import annotations

import functools
import math
import time
from collections import deque

import torch
import torch.nn as nn
import numpy as np

from binary_policy import (binary_action_entropy, binary_action_log_probs,
                           binary_taken_log_prob)
from model import EntityTransformer, NUM_ANGLE_BINS, NUM_SHIP_BINS, SHIP_COUNTS
from config import Config


@functools.lru_cache(maxsize=8)
def _ship_log_prior_cpu(exp: float, num_bins: int) -> torch.Tensor:
    """Log of the full-send-biased ship-size prior: w_i ∝ SHIP_COUNTS[i]**exp, normalized.
    Cached (fixed vector); moved to the loss device/dtype by _ship_log_prior."""
    counts = torch.tensor(SHIP_COUNTS[:num_bins], dtype=torch.float64)
    w = counts ** exp
    return (w / w.sum()).log().float()


def _ship_log_prior(exp: float, num_bins: int, device, dtype) -> torch.Tensor:
    return _ship_log_prior_cpu(float(exp), int(num_bins)).to(device=device, dtype=dtype)


def _gather_target_logits(per_target_logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(per_target_logits, -1, target_idx.unsqueeze(-1)).squeeze(-1)


def _gather_target_ship_logits(per_target_logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        per_target_logits,
        2,
        target_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, per_target_logits.shape[-1]),
    ).squeeze(2)


def _fire_target_conditioning_metrics(target_logits: torch.Tensor,
                                      fire_prior: torch.Tensor,
                                      fire_residual: torch.Tensor,
                                      fire_mask: torch.Tensor,
                                      slot_valid: torch.Tensor) -> dict[str, torch.Tensor]:
    """Target-policy-weighted fire flips on sources that can actually commit.

    Uses the unmasked conditioned logit (prior + residual), so legality masks cannot
    masquerade as target-conditioning decisions. A flip means crossing the 0-logit
    / 0.5-probability boundary relative to the slot-only prior.
    """
    valid_targets = target_logits > -1e8
    actionable = fire_mask.bool() & slot_valid.bool() & valid_targets.any(dim=-1)
    actionable_f = actionable.float()
    denom = actionable_f.sum().clamp(min=1.0)

    target_probs = torch.softmax(target_logits.masked_fill(~valid_targets, -1e9), dim=-1)
    target_probs = target_probs * valid_targets.float()
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp(min=1.0)

    conditioned = fire_prior + fire_residual
    prior_commit = fire_prior > 0
    conditioned_commit = conditioned > 0
    noop_to_commit = ~prior_commit & conditioned_commit & valid_targets
    commit_to_noop = prior_commit & ~conditioned_commit & valid_targets

    def expected_rate(events: torch.Tensor) -> torch.Tensor:
        per_source = (target_probs * events.float()).sum(dim=-1)
        return (per_source * actionable_f).sum() / denom

    has_commit = (conditioned_commit & valid_targets).any(dim=-1)
    has_noop = ((~conditioned_commit) & valid_targets).any(dim=-1)
    straddles = has_commit & has_noop
    n2c = expected_rate(noop_to_commit)
    c2n = expected_rate(commit_to_noop)
    return {
        "flip_prob": n2c + c2n,
        "noop_to_commit_prob": n2c,
        "commit_to_noop_prob": c2n,
        "straddle_rate": (straddles.float() * actionable_f).sum() / denom,
    }


class PPOLearner:
    def __init__(self, model, cfg, device="cpu"):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        # bf16 autocast for the update forward (set by the trainer; default off). No GradScaler:
        # bf16 shares fp32's exponent range so gradients don't underflow the way fp16's do.
        self.amp_enabled = False
        self.amp_dtype = torch.bfloat16

        self.phase4_residual_lr_mult = float(getattr(cfg.ppo, "phase4_residual_lr_mult", 1.0))
        residual_prefixes = (
            "fire_q.", "fire_k.", "fire_scorer.",
            "ship_q.", "ship_k.", "ship_scorer.",
        )
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if self.phase4_residual_lr_mult == 1.0:
            # Preserve the legacy single-group parameter order so mature checkpoints
            # can warm-resume their Adam moments. A second group has no effect at x1.
            param_groups = [{"params": trainable_params, "lr": cfg.ppo.learning_rate}]
        else:
            residual_param_ids = {
                id(p) for name, p in model.named_parameters()
                if any(name.startswith(prefix) for prefix in residual_prefixes)
            }
            base_params = [p for p in trainable_params if id(p) not in residual_param_ids]
            residual_params = [p for p in trainable_params if id(p) in residual_param_ids]
            param_groups = [{"params": base_params, "lr": cfg.ppo.learning_rate}]
            if residual_params:
                param_groups.append({
                    "params": residual_params,
                    "lr": cfg.ppo.learning_rate * self.phase4_residual_lr_mult,
                })

        self.optimizer = torch.optim.Adam(param_groups, eps=1e-5)
        self.total_steps = 0
        self.update_count = 0

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
        policy_model = getattr(self.model, "_orig_mod", self.model)
        binary_mode = getattr(policy_model.cfg, "ship_bin_mode", "absolute") == "binary"
        # Autocast the model forward to bf16 (the dominant matmul cost). Then upcast the
        # float outputs back to fp32 so the log-prob/ratio/exp arithmetic below runs in fp32
        # (bf16's 8-bit mantissa would inject noise into PPO ratios). The bf16 matmuls and
        # their backward graph are unaffected — the upcast is just a cast node the gradient
        # flows back through. No-op when amp is off (outputs already fp32).
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
            outputs = self.model(
                planet_features, fleet_features, global_features,
                planet_mask, fleet_mask,
                fire_mask=fire_mask,
                slot_valid=slot_valid_2d, owned_indices=owned_indices,
                owned_count=batch.get("owned_count"),
                pairwise_features=pairwise,
            )
        if self.amp_enabled:
            outputs = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                       for k, v in outputs.items()}

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
        slot_valid = slot_valid_2d.unsqueeze(-1)   # (B, max_owned, 1)
        decision_valid = fire_mask.float() if binary_mode else slot_valid.squeeze(-1)
        fired_slots = fire_action.float() * decision_valid

        binary_log_noop = binary_log_commit = binary_launch_probs = None
        if binary_mode:
            binary_log_noop, binary_log_commit = binary_action_log_probs(
                target_logits, fire_logits_target, target_mask, fire_mask)
            new_log_prob_fire = binary_taken_log_prob(
                binary_log_noop, binary_log_commit, fire_action, target_action) * decision_valid
            new_log_prob_ships = torch.zeros_like(new_log_prob_fire)
            new_log_prob_target = torch.zeros_like(new_log_prob_fire)
            binary_launch_probs = binary_log_commit.exp().sum(dim=-1)
        else:
            new_log_prob_fire = fire_dist.log_prob(fire_action.float()) * decision_valid
            new_log_prob_ships = ship_dist.log_prob(ship_action) * fired_slots
            new_log_prob_target = target_dist.log_prob(target_action) * slot_valid.squeeze(-1)

        # Sum across planet slots: (B, max_owned) -> (B,)
        new_log_prob = (new_log_prob_fire + new_log_prob_ships + new_log_prob_target).sum(dim=-1)

        # Old log probs (stored at rollout time)
        old_fire = to_dev(batch["old_log_probs"]["fire"]) * decision_valid
        old_ships = (torch.zeros_like(old_fire) if binary_mode else
                     to_dev(batch["old_log_probs"]["ships"]) * fired_slots)
        old_target = (torch.zeros_like(old_fire) if binary_mode else
                      to_dev(batch["old_log_probs"]["target"]) * slot_valid.squeeze(-1))
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
        if binary_mode:
            # One exact entropy over the executed {NOOP, COMMIT(target)} distribution.
            # It is a categorical target choice with NOOP as an extra category, so the
            # existing target entropy coefficient controls it; fire/ship terms are zero.
            fire_entropy = torch.zeros((), device=ship_logits.device)
            ship_entropy = torch.zeros((), device=ship_logits.device)
            per_source_action_entropy = binary_action_entropy(binary_log_noop, binary_log_commit)
            target_entropy = ((per_source_action_entropy * decision_valid).sum()
                              / decision_valid.sum().clamp(min=1))
        else:
            fire_entropy = fire_dist.entropy().mean()
            ship_entropy = ship_dist.entropy().mean()
            target_entropy = target_dist.entropy().mean()

        # No-op KL bias (Jake Will Rank-2 lever): anchor the BATCH-MEAN launch rate to a low
        # prior. p_bar = mean fire prob over valid owned slots (WITH grad); KL(Bern(p_bar) ‖
        # Bern(q)) pulls it toward q. Anchoring the MEAN (not per-sample) lets individual turns
        # fire at 100% as long as others average out — kills spray without killing decisiveness.
        mean_launch_rate = 0.0
        noop_kl = 0.0
        if cfg.noop_kl_coef > 0.0:
            q = cfg.noop_target_launch_rate
            sv_lr = decision_valid if binary_mode else slot_valid_2d.float()  # (B, MO)
            launch_probs = binary_launch_probs if binary_mode else torch.sigmoid(fire_logits)
            p_bar = (launch_probs * sv_lr).sum() / sv_lr.sum().clamp(min=1)
            p_bar = p_bar.clamp(1e-6, 1.0 - 1e-6)
            noop_kl = (p_bar * (p_bar / q).log()
                       + (1.0 - p_bar) * ((1.0 - p_bar) / (1.0 - q)).log())
            mean_launch_rate = p_bar.detach().item()

        # Ship-size KL-to-prior (Ender lever): pull the per-draw ship-count distribution toward a
        # full-send-biased prior (w_i ∝ SHIP_COUNTS[i]**ship_kl_prior_exp), on fired slots only.
        # KL(π ‖ prior) = Σ π_i (log π_i − log prior_i). Replaces (set entropy_coef_ships=0) the
        # uniform-seeking ship entropy bonus — see the ship_kl_coef config note.
        ship_kl = 0.0
        if cfg.ship_kl_coef > 0.0 and not binary_mode:
            log_prior = _ship_log_prior(cfg.ship_kl_prior_exp, ship_logits.shape[-1],
                                        ship_logits.device, ship_logits.dtype)   # (num_bins,)
            log_q = torch.log_softmax(ship_logits, dim=-1)                        # (B, MO, num_bins)
            kl_per_slot = (log_q.exp() * (log_q - log_prior)).sum(dim=-1)         # (B, MO)
            ship_kl = (kl_per_slot * fired_slots).sum() / fired_slots.sum().clamp(min=1)

        if value_only:
            loss = cfg.value_coef * value_loss
        else:
            loss = (policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef_fire  * fire_entropy
                    - cfg.entropy_coef_target * target_entropy
                    - cfg.entropy_coef_ships * ship_entropy)
            if cfg.noop_kl_coef > 0.0:
                loss = loss + cfg.noop_kl_coef * noop_kl
            if cfg.ship_kl_coef > 0.0:
                loss = loss + cfg.ship_kl_coef * ship_kl

        if return_metrics:
            clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()
            # Per-slot fire clip_frac: measures whether individual fire decisions
            # would be clipped, decoupled from multi-slot joint-log-prob amplification.
            # The joint clip_frac is mechanically inflated when many owned slots are
            # present (16 slots × per-slot Δlog_p summing up), so this per-slot metric
            # gives a calibrated read of actual policy change per decision.
            with torch.no_grad():
                fire_valid = decision_valid if binary_mode else slot_valid_2d.float()
                new_lp_fire_s = (new_log_prob_fire if binary_mode else
                                 fire_dist.log_prob(fire_action.float()) * fire_valid)
                old_lp_fire_s = to_dev(batch["old_log_probs"]["fire"]) * fire_valid
                ratio_fire_s = torch.exp(
                    new_lp_fire_s - torch.clamp(old_lp_fire_s, min=-50)
                )
                sv_f = fire_valid
                clip_frac_fire = (
                    ((ratio_fire_s - 1.0).abs() > cfg.clip_eps).float() * sv_f
                ).sum() / sv_f.sum().clamp(min=1)
            with torch.no_grad():
                sv = slot_valid_2d.float()                              # (B, MO)
                sv_sum = sv.sum().clamp(min=1)
                fire_probs = (binary_launch_probs if binary_mode else
                              torch.sigmoid(fire_logits))                # (B, MO)
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
                # Entropy ceiling for the target head: its uniform max is ln(#legal
                # targets), which moves with game state — batch mean over valid slots.
                # Fire (ln 2) and ship (ln num_bins) ceilings are constants; all three
                # are logged as entropy/*_frac so the wandb curves read as
                # fraction-of-uniform-max on one 0-1 scale (raw entropies are NOT
                # cross-head comparable: ln2 vs ~ln40 vs ln32).
                tgt_legal = target_valid.float().sum(dim=-1).clamp(min=1.0)   # (B, MO)
                if binary_mode:
                    tgt_legal = tgt_legal + 1.0  # NOOP is an additional executed action
                    target_entropy_max = ((tgt_legal.log() * decision_valid).sum()
                                          / decision_valid.sum().clamp(min=1.0))
                else:
                    target_entropy_max = (tgt_legal.log() * sv).sum() / sv_sum
                fire_prior = outputs.get("_phase4_fire_prior")
                fire_residual = outputs.get("_phase4_fire_residual")
                ship_prior = outputs.get("_phase4_ship_prior")
                ship_residual = outputs.get("_phase4_ship_residual")
                fire_prior_rms = fire_resid_rms = fire_resid_ratio = 0.0
                ship_prior_rms = ship_resid_rms = ship_resid_ratio = 0.0
                fire_flip_prob = fire_noop_to_commit = fire_commit_to_noop = fire_straddle_rate = 0.0
                ship_decision_flip = 0.0
                if fire_prior is not None and fire_residual is not None:
                    valid_targets = target_valid & slot_valid_2d.unsqueeze(-1)
                    valid_targets_f = valid_targets.float()
                    vt_sum = valid_targets_f.sum().clamp(min=1.0)
                    fire_prior_rms = (((fire_prior * valid_targets_f) ** 2).sum() / vt_sum).sqrt()
                    fire_resid_rms = (((fire_residual * valid_targets_f) ** 2).sum() / vt_sum).sqrt()
                    fire_resid_ratio = fire_resid_rms / fire_prior_rms.clamp(min=1e-6)
                    fire_conditioning = _fire_target_conditioning_metrics(
                        target_logits, fire_prior, fire_residual, fire_mask, slot_valid_2d)
                    fire_flip_prob = fire_conditioning["flip_prob"]
                    fire_noop_to_commit = fire_conditioning["noop_to_commit_prob"]
                    fire_commit_to_noop = fire_conditioning["commit_to_noop_prob"]
                    fire_straddle_rate = fire_conditioning["straddle_rate"]
                if ship_prior is not None and ship_residual is not None:
                    valid_targets_bins = target_valid.unsqueeze(-1) & slot_valid_2d.unsqueeze(-1).unsqueeze(-1)
                    valid_targets_bins_f = valid_targets_bins.float()
                    vtb_sum = valid_targets_bins_f.sum().clamp(min=1.0)
                    ship_prior_rms = (((ship_prior * valid_targets_bins_f) ** 2).sum() / vtb_sum).sqrt()
                    ship_resid_rms = (((ship_residual * valid_targets_bins_f) ** 2).sum() / vtb_sum).sqrt()
                    ship_resid_ratio = ship_resid_rms / ship_prior_rms.clamp(min=1e-6)
                    ship_prior_logits = _gather_target_ship_logits(ship_prior, target_action)
                    fire_prior_logits = (_gather_target_logits(fire_prior, target_action)
                                         if fire_prior is not None else None)
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
                # Normalized to each head's uniform ceiling (same numerator conventions
                # as the raw entropies above) — cross-head comparable, 1.0 = uniform.
                "fire_entropy_frac": fire_entropy.item() / math.log(2.0),
                "target_entropy_frac": target_entropy.item() / max(target_entropy_max.item(), 1e-6),
                "ship_entropy_frac": ship_entropy.item() / math.log(ship_logits.shape[-1]),
                "target_entropy_max": target_entropy_max.item(),
                "noop_kl": float(noop_kl.item() if torch.is_tensor(noop_kl) else noop_kl),
                "mean_launch_rate": mean_launch_rate,
                "ship_kl": float(ship_kl.item() if torch.is_tensor(ship_kl) else ship_kl),
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
                "phase4_fire_resid_ratio": float(fire_resid_ratio),
                "phase4_ship_resid_ratio": float(ship_resid_ratio),
                "fire_target_flip_prob": float(fire_flip_prob),
                "fire_target_noop_to_commit_prob": float(fire_noop_to_commit),
                "fire_target_commit_to_noop_prob": float(fire_commit_to_noop),
                "fire_target_straddle_rate": float(fire_straddle_rate),
                "phase4_ship_decision_flip": float(ship_decision_flip),
                "per_slot_fire_probs": per_slot_fire.detach().cpu().tolist(),
            }
            return loss, metrics

        return loss

    def update(self, batches, scheduler=None, ppo_epochs=None,
               kl_target: float = 0.05, timers=None, sync=None,
               lean_metrics: bool = False):
        """Run PPO update on a list of minibatches.

        kl_target:  stop epoch loop early if mean approx-KL exceeds this value,
                    preventing destructive policy updates.
                    Good range: 0.01–0.05. Pass float('inf') to disable.
        timers:     optional dict accumulating wall-time into "upd_fwd" (compute_loss,
                    incl. metrics .item() syncs), "upd_bwd" (backward), "upd_opt"
                    (grad-clip + optimizer/scheduler step). `sync` is called at each
                    boundary (pass torch.cuda.synchronize for true attribution).
        lean_metrics: THROUGHPUT PROBE ONLY. Skip the per-minibatch metrics block
                    (~40 .item() GPU→CPU syncs each) on all but the final update, so
                    only ONE full metrics dict is computed per rollout. Gradients/loss
                    are byte-identical (metrics are no_grad + detached) — this isolates
                    the logging-sync tax. Disables KL early-stopping (needs per-minibatch
                    KL). Do NOT use for real training runs (loses the KL guard).
        """
        cfg = self.cfg.ppo
        epochs = ppo_epochs or cfg.ppo_epochs
        if sync is None:
            sync = lambda: None

        sum_metrics: dict[str, float] = {}
        n_updates = 0
        n_metrics = 0
        early_stopped = False

        for epoch in range(epochs):
            epoch_kl = 0.0
            last_epoch = epoch == epochs - 1
            for i, batch in enumerate(batches):
                # Lean mode: only the very last update of the rollout carries metrics.
                want_metrics = (not lean_metrics) or (last_epoch and i == len(batches) - 1)
                _t0 = time.perf_counter()
                if want_metrics:
                    ppo_loss, metrics = self.compute_loss(batch, return_metrics=True)
                else:
                    ppo_loss = self.compute_loss(batch, return_metrics=False)
                    metrics = None
                total_loss = ppo_loss
                if timers is not None:
                    sync(); timers["upd_fwd"] += time.perf_counter() - _t0
                    _t0 = time.perf_counter()

                self.optimizer.zero_grad()
                total_loss.backward()
                if timers is not None:
                    sync(); timers["upd_bwd"] += time.perf_counter() - _t0
                    _t0 = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if timers is not None:
                    sync(); timers["upd_opt"] += time.perf_counter() - _t0

                n_updates += 1
                if metrics is None:
                    continue
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
                n_metrics += 1

            # KL early stopping: if this epoch's mean KL exceeded the target,
            # bail out before the next epoch to prevent policy collapse.
            # (Skipped under lean_metrics — no per-minibatch KL available.)
            if not lean_metrics and (epoch_kl / max(len(batches), 1)) > kl_target:
                early_stopped = True
                break

        n = max(n_metrics, 1)
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
        # right shape (num_ship_bins for fraction-head).
        model_cfg = getattr(self.cfg, "model", None)
        cfg_blob = {}
        if model_cfg is not None:
            cfg_blob = {
                "num_ship_bins": int(getattr(model_cfg, "num_ship_bins", 32)),
                "pairwise_feature_dim": int(getattr(model_cfg, "pairwise_feature_dim", 0)),
                "ship_bin_mode": str(getattr(model_cfg, "ship_bin_mode", "absolute")),
                "action_decode": str(getattr(model_cfg, "action_decode", "angle")),
                "allow_reinforce": bool(getattr(model_cfg, "allow_reinforce", False)),
                # Blessed feature semantics (2026-07 cleanup): recorded so loaders can verify.
                "game_phase_features": True,
                "pressure_precise_resolver": True,
                "feature_config": "blessed-2026-07",
                # Projected-future timeline channels (planet dim 20→116, 2026-07-10).
                "timeline_features": True,
                # Reinforce / sufficient-commit DISCIPLINE — eval & export must mask the SAME way
                # the ckpt was trained or the policy self-sabotages. Persist so they auto-load
                # instead of relying on CLI flags being remembered (a panel/submission footgun).
                "reinforce_gate_min_planets": int(getattr(model_cfg, "reinforce_gate_min_planets", 0)),
                "reinforce_forward_only": bool(getattr(model_cfg, "reinforce_forward_only", False)),
                "reverse_edge_cooldown": int(getattr(model_cfg, "reverse_edge_cooldown", 0)),
                "reinforce_garrison_floor": float(getattr(model_cfg, "reinforce_garrison_floor", 0.0)),
                "sufficient_commit_factor": float(getattr(model_cfg, "sufficient_commit_factor", 0.0)),
                # provenance: how the ckpt was trained (eval always clamps, so not an eval-contract field)
                "ship_overflow_mode": str(getattr(model_cfg, "ship_overflow_mode", "drop")),
                "phase4_residual_init_std": float(getattr(model_cfg, "phase4_residual_init_std", 0.0)),
            }
        # Save the UNCOMPILED model's state_dict. torch.compile wraps the model and prefixes
        # every key with "_orig_mod." — persisting that breaks eval/export/resume (which load
        # plain uncompiled models). Unwrap so checkpoints are always canonical.
        save_model = getattr(self.model, "_orig_mod", self.model)
        return {
            "model": save_model.state_dict(),
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
