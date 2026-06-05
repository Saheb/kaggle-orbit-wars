from __future__ import annotations

import torch
import torch.nn as nn

from orbit_wars_rl.model import EntityTransformer, NUM_SHIP_BINS


class JointActionRanker(nn.Module):
    """Shadow joint head over candidate `(source, target, ship_bin)` actions."""

    def __init__(self, backbone: EntityTransformer, ship_emb_dim: int = 16, hidden_dim: int | None = None):
        super().__init__()
        self.backbone = backbone
        D = backbone.cfg.entity_dim
        F_pair = getattr(backbone.cfg, "pairwise_feature_dim", 0)
        self.ship_emb = nn.Embedding(NUM_SHIP_BINS, ship_emb_dim)
        self.hidden_dim = hidden_dim or D
        self.extra_feat_dim = 10
        in_dim = D + D + D + F_pair + ship_emb_dim + 4 + self.extra_feat_dim
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def score_actions(
        self,
        batch: dict,
        slot_idx: torch.Tensor,
        ship_bin: torch.Tensor,
        target_idx: torch.Tensor,
        action_extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        enc = self.backbone.encode_state(
            batch["planet_features"],
            batch["fleet_features"],
            batch["global_features"],
            batch["planet_mask"],
            batch["fleet_mask"],
            slot_valid=batch["slot_valid"],
            owned_indices=batch["owned_indices"],
            pairwise_features=batch["pairwise_features"],
        )
        owned = enc["owned_enriched"]        # (B, MO, D)
        planets = enc["planet_emb"]          # (B, Np, D)
        global_token = enc["global_token"]   # (B, D)
        pairwise = enc["pairwise_features"]  # (B, MO, Np, F)

        b = torch.arange(slot_idx.shape[0], device=slot_idx.device)
        src = owned[b, slot_idx]
        tgt = planets[b, target_idx]
        g = global_token[b]
        pw = pairwise[b, slot_idx, target_idx]
        se = self.ship_emb(ship_bin)

        # A few explicit numeric cues to help the scorer not relearn simple scales.
        num_feats = torch.stack([
            ship_bin.float() / max(1, self.ship_emb.num_embeddings - 1),
            target_idx.float() / max(1, planets.shape[1] - 1),
            slot_idx.float() / max(1, owned.shape[1] - 1),
            batch["slot_valid"][b, slot_idx].float(),
        ], dim=-1)
        if action_extra is None:
            action_extra = torch.zeros((slot_idx.shape[0], self.extra_feat_dim), device=slot_idx.device, dtype=src.dtype)
        inp = torch.cat([src, tgt, g, pw, se, num_feats, action_extra], dim=-1)
        return self.scorer(inp).squeeze(-1)
