"""Exact executed-action distribution for binary NOOP/COMMIT policies."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_action_log_probs(target_logits: torch.Tensor,
                            fire_logits_target: torch.Tensor,
                            target_mask: torch.Tensor | None = None,
                            fire_mask: torch.Tensor | None = None,
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log P(NOOP) and log P(COMMIT(target)) for each source.

    The model factorizes target first and target-conditioned fire second:
      P(COMMIT(t)) = P(t) P(fire | t)
      P(NOOP)      = sum_t P(t) P(noop | t)
    The returned tensors are the normalized distribution over executed actions.
    """
    valid_target = ((target_logits > -1e8) if target_mask is None
                    else target_mask.bool() & (target_logits > -1e8))
    has_target = valid_target.any(dim=-1)
    actionable = has_target if fire_mask is None else has_target & fire_mask.bool()

    log_target = torch.log_softmax(target_logits.masked_fill(~valid_target, -1e9), dim=-1)
    log_target = log_target.masked_fill(~valid_target, -1e9)
    log_commit = log_target + F.logsigmoid(fire_logits_target)
    log_noop = torch.logsumexp(log_target + F.logsigmoid(-fire_logits_target), dim=-1)

    # A source without a feasible commit has the singleton distribution P(NOOP)=1.
    log_commit = log_commit.masked_fill(~actionable.unsqueeze(-1), -1e9)
    log_noop = torch.where(actionable, log_noop, torch.zeros_like(log_noop))
    return log_noop, log_commit


def binary_taken_log_prob(log_noop: torch.Tensor, log_commit: torch.Tensor,
                          fire_action: torch.Tensor,
                          target_action: torch.Tensor) -> torch.Tensor:
    """Log probability of the executed NOOP or COMMIT(target) action."""
    target = target_action.long().clamp(0, log_commit.shape[-1] - 1)
    chosen_commit = torch.gather(log_commit, -1, target.unsqueeze(-1)).squeeze(-1)
    return torch.where(fire_action.bool(), chosen_commit, log_noop)


def binary_action_entropy(log_noop: torch.Tensor, log_commit: torch.Tensor) -> torch.Tensor:
    """Entropy of the normalized {NOOP, COMMIT(target)} distribution per source."""
    log_actions = torch.cat([log_noop.unsqueeze(-1), log_commit], dim=-1)
    probs = log_actions.exp()
    return -(probs * log_actions).sum(dim=-1)
