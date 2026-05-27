"""Entity Transformer in PyTorch for Orbit Wars.

Architecture (ADR-001/002/003 documented):
- 72-bin discretized angles (ADR-001)
- Shared backbone + mode token for 2p/4p (ADR-002)
- Baked-in geometric features, discovered strategy (ADR-003)

Phase 1 feature dims: planet=20, fleet=13, global=11, pairwise=12, max_owned=16.
Value head: concat(global_token, owned_pool) → 2D → D → D/2 → 1.
~350K params: 3 layers, 96 dim, 4 heads, 3x MLP expansion.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ANGLE_BINS = 144
NUM_SHIP_BINS = 32
ANGLE_BIN_WIDTH = 2 * math.pi / NUM_ANGLE_BINS
SHIP_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 19, 22, 26, 30, 35, 42, 50, 60, 72, 86, 102, 122, 145, 173, 206, 245, 290, 350, 420]


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_expansion=3, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # x: (B, N, D), mask: (B, N) bool — True = valid
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.mlp(x))
        return x


class EntityTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D = cfg.entity_dim

        # Entity projections
        self.planet_proj = nn.Linear(cfg.planet_feature_dim, D)
        self.fleet_proj = nn.Linear(cfg.fleet_feature_dim, D)
        self.global_proj = nn.Linear(cfg.global_feature_dim, D)
        self.mode_proj = nn.Linear(cfg.global_feature_dim, D)

        # Transformer layers
        self.blocks = nn.ModuleList([
            TransformerBlock(D, cfg.num_heads, cfg.mlp_expansion, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])

        # Pairwise cross-attention: for each owned slot, attend to all planets using
        # explicit (src, tgt) geometric features. Closes the trig gap exposed by the
        # prior BC angle-head failure (0.08 reduction vs 0.40 gate).
        F_pair = getattr(cfg, "pairwise_feature_dim", 0)
        self.use_pairwise = F_pair > 0
        if self.use_pairwise:
            self.pair_kv = nn.Linear(D + F_pair, 2 * D)
            self.pair_q = nn.Linear(D, D)
            self.pair_out = nn.Linear(D, D)
            self.pair_ln = nn.LayerNorm(D)

        # Action heads (per owned planet)
        self.fire_head = nn.Linear(D, 1)
        self.num_angle_bins = getattr(cfg, "num_angle_bins", NUM_ANGLE_BINS)
        self.angle_head = nn.Linear(D, self.num_angle_bins)
        # Ship head: bin count is configurable so the fraction-head experiment
        # can swap to 10 fraction bins. Default 32 = legacy absolute counts.
        self.num_ship_bins = getattr(cfg, "num_ship_bins", NUM_SHIP_BINS)
        self.ship_head = nn.Linear(D, self.num_ship_bins)
        # Target-index head. When pairwise features are available we score each
        # (slot, target) pair from per-target inputs — see docs/bugs.md (target-head
        # collapse). When pairwise is disabled we fall back to a slot-only Linear.
        self.max_planets = cfg.max_planets if hasattr(cfg, "max_planets") else 48
        if self.use_pairwise:
            self.tgt_q = nn.Linear(D, D)
            self.tgt_k = nn.Linear(D, D)
            tgt_in = D + D + F_pair
            tgt_hidden = D
            self.target_scorer = nn.Sequential(
                nn.Linear(tgt_in, tgt_hidden),
                nn.GELU(),
                nn.Linear(tgt_hidden, 1),
            )
        else:
            self.target_head = nn.Linear(D, self.max_planets)

        # Value head: concat global token + owned pool → Linear(2D→D) by default.
        # value_head_in=0 means auto (2*D); load_checkpoint sets it to D for
        # pre-Phase-1 checkpoints that used mean-pool-all-entities (D→D).
        _vh_in = getattr(cfg, "value_head_in", 0) or (2 * D)
        self.value_fc1 = nn.Linear(_vh_in, D)
        self.value_fc2 = nn.Linear(D, D // 2)
        self.value_out = nn.Linear(D // 2, 1)

    def forward(self, planet_features, fleet_features, global_features,
                planet_mask, fleet_mask, fire_mask=None, angle_mask=None,
                slot_valid=None, owned_indices=None, owned_count=None,
                pairwise_features=None):
        """
        Args:
            planet_features: (B, N_p, D_p)
            fleet_features: (B, N_f, D_f)
            global_features: (B, D_g)
            planet_mask: (B, N_p) bool, True = real entity
            fleet_mask: (B, N_f) bool, True = real entity
            fire_mask: (B, max_owned) bool, True = can fire
            angle_mask: (B, max_owned, 72) bool, True = legal angle
            slot_valid: (B, max_owned) bool, True = real owned planet slot
            owned_indices: (B, max_owned) int, indices into planet array
            owned_count: (B,) int

        Returns dict with fire_logits, angle_logits, ship_logits, value.
        """
        B = planet_features.shape[0]
        D = self.cfg.entity_dim
        max_owned = self.cfg.max_owned_planets

        # Project entities
        planet_emb = self.planet_proj(planet_features)
        fleet_emb = self.fleet_proj(fleet_features)
        global_emb = self.global_proj(global_features) + self.mode_proj(global_features)

        # Concatenate: [global, planets, fleets]
        all_entities = torch.cat([
            global_emb.unsqueeze(1),
            planet_emb,
            fleet_emb,
        ], dim=1)  # (B, 1+N_p+N_f, D)

        # Attention mask: True = VALID (for key_padding_mask, True = PAD/IGNORE)
        total_entities = 1 + planet_features.shape[1] + fleet_features.shape[1]
        attn_mask = torch.ones(B, total_entities, dtype=torch.bool, device=planet_features.device)
        attn_mask[:, 0] = False  # global token is always valid
        # planet_mask is True for real, we need True for padding
        attn_mask[:, 1:1+planet_features.shape[1]] = ~planet_mask
        attn_mask[:, 1+planet_features.shape[1]:] = ~fleet_mask

        # Transformer
        x = all_entities
        for block in self.blocks:
            x = block(x, mask=attn_mask)

        # Extract owned planet entity representations
        if owned_indices is None:
            owned_indices = torch.zeros(B, max_owned, dtype=torch.long, device=planet_features.device)

        # owned_indices point into planet_features array, offset by 1 for global token
        full_indices = (owned_indices + 1).clamp(0, x.shape[1] - 1)  # (B, max_owned)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, max_owned)
        owned_entities = x[batch_idx, full_indices]  # (B, max_owned, D)

        # Pairwise cross-attention: enrich each slot with explicit (src, tgt) geometry
        # for fire/angle/ship heads. Target head scores per-(slot, target) directly
        # from the same per-target inputs — see docs/bugs.md.
        target_logits = None
        if self.use_pairwise and pairwise_features is not None:
            N_p = planet_features.shape[1]
            planet_emb_post = x[:, 1:1 + N_p, :]                                # (B, N_p, D)
            planet_per_slot = planet_emb_post.unsqueeze(1).expand(-1, max_owned, -1, -1)
            kv_input = torch.cat([planet_per_slot, pairwise_features], dim=-1)  # (B, MO, N_p, D+F)
            kv = self.pair_kv(kv_input)                                         # (B, MO, N_p, 2D)
            k, v = kv.chunk(2, dim=-1)
            q = self.pair_q(owned_entities).unsqueeze(2)                        # (B, MO, 1, D)
            scale = D ** -0.5
            scores = (q @ k.transpose(-2, -1)).squeeze(2) * scale               # (B, MO, N_p)
            tgt_valid = planet_mask.unsqueeze(1).expand(-1, max_owned, -1)
            scores = scores.masked_fill(~tgt_valid, -1e4)
            attn = F.softmax(scores, dim=-1).unsqueeze(2)
            enriched = (attn @ v).squeeze(2)
            owned_entities = self.pair_ln(owned_entities + self.pair_out(enriched))

            # Per-target scoring head: each (slot, target) gets its own logit from
            # [q_slot, k_target, pair_features]. This is the fix for the collapse
            # documented in docs/bugs.md — the prior Linear(D, max_planets) head
            # had no per-target conditioning, capping target_top1 near random.
            q_tgt = self.tgt_q(owned_entities).unsqueeze(2).expand(-1, -1, N_p, -1)
            k_tgt = self.tgt_k(planet_emb_post).unsqueeze(1).expand(-1, max_owned, -1, -1)
            scorer_in = torch.cat([q_tgt, k_tgt, pairwise_features], dim=-1)
            tgt_scores = self.target_scorer(scorer_in).squeeze(-1)              # (B, MO, N_p)
            # Pad to max_planets width if needed
            if N_p < self.max_planets:
                pad = torch.full(
                    (B, max_owned, self.max_planets - N_p),
                    -100.0, device=tgt_scores.device, dtype=tgt_scores.dtype,
                )
                tgt_scores = torch.cat([tgt_scores, pad], dim=-1)
            elif N_p > self.max_planets:
                tgt_scores = tgt_scores[..., :self.max_planets]
            target_logits = tgt_scores
        elif self.use_pairwise:
            # Some legacy tests/callers do not pass pairwise_features and do
            # not consume target_logits. Keep forward usable for those paths
            # without adding fallback parameters that would break checkpoints.
            target_logits = torch.zeros(
                B, max_owned, self.max_planets,
                device=owned_entities.device, dtype=owned_entities.dtype,
            )

        # Action heads
        fire_logits = self.fire_head(owned_entities).squeeze(-1)  # (B, max_owned)
        angle_logits = self.angle_head(owned_entities)  # (B, max_owned, 144)
        ship_logits = self.ship_head(owned_entities)  # (B, max_owned, num_ship_bins)
        # min_ship_bin: bins below this are masked to -inf so they're never
        # sampled / argmaxed. For the fraction head (10 bins on [0.1, 1.0]),
        # setting min_ship_bin=1 removes the "10%-of-source" trap that PPO
        # collapses to in cold-start self-play (every prior run hit ship0>0.5
        # by iter 6-7 regardless of LR or IL/BC anchor strength).
        min_ship_bin = int(getattr(self.cfg, "min_ship_bin", 0))
        if min_ship_bin > 0:
            ship_logits[..., :min_ship_bin] = -100.0
        if target_logits is None:
            target_logits = self.target_head(owned_entities)  # (B, max_owned, max_planets)
        # Mask out invalid planet slots (padded) so target softmax only sees real planets
        if planet_mask is not None:
            tgt_mask = planet_mask.unsqueeze(1).expand(-1, max_owned, -1)  # (B, MO, N_p)
            # If max_planets in model > N_p in obs, pad the mask to model width
            mp = target_logits.shape[-1]
            np_obs = tgt_mask.shape[-1]
            if mp > np_obs:
                pad = torch.zeros(B, max_owned, mp - np_obs, dtype=torch.bool, device=tgt_mask.device)
                tgt_mask = torch.cat([tgt_mask, pad], dim=-1)
            elif mp < np_obs:
                tgt_mask = tgt_mask[..., :mp]
            target_logits = target_logits.masked_fill(~tgt_mask, -100.0)

        # Apply masks (-100 is safe in float16 on MPS; -1e9 overflows)
        if fire_mask is not None:
            fire_logits = fire_logits.masked_fill(~fire_mask, -100.0)
        if angle_mask is not None:
            if angle_mask.shape[-1] != angle_logits.shape[-1]:
                angle_mask = angle_mask[..., :angle_logits.shape[-1]]
            angle_logits = angle_logits.masked_fill(~angle_mask, -100.0)
        if slot_valid is not None:
            fire_logits = fire_logits.masked_fill(~slot_valid, -100.0)
            angle_logits = angle_logits.masked_fill(~slot_valid.unsqueeze(-1), -100.0)
            ship_logits = ship_logits.masked_fill(~slot_valid.unsqueeze(-1), -100.0)
            target_logits = target_logits.masked_fill(~slot_valid.unsqueeze(-1), -100.0)

        # Value head: new=concat(global_token, owned_pool) [2D], old=mean-pool all [D].
        if self.value_fc1.in_features == D:
            # Pre-Phase-1 checkpoint: mean-pool all valid entities
            valid_float = (~attn_mask).float()
            value_input = (x * valid_float.unsqueeze(-1)).sum(1) / valid_float.sum(1, keepdim=True).clamp(min=1)
        else:
            global_token = x[:, 0, :]                                    # (B, D)
            if slot_valid is not None:
                owned_float = slot_valid.float().unsqueeze(-1)
                n_owned = owned_float.sum(dim=1).clamp(min=1)
                owned_pool = (owned_entities * owned_float).sum(dim=1) / n_owned
            else:
                owned_pool = owned_entities.mean(dim=1)
            value_input = torch.cat([global_token, owned_pool], dim=-1)  # (B, 2D)
        value = self.value_out(F.gelu(self.value_fc2(F.gelu(self.value_fc1(value_input))))).squeeze(-1)

        return {
            "fire_logits": fire_logits,
            "angle_logits": angle_logits,
            "ship_logits": ship_logits,
            "target_logits": target_logits,
            "value": value,
        }


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def angle_bin_to_radians(bin_idx):
    return (bin_idx.float() + 0.5) * ANGLE_BIN_WIDTH


def ship_bin_to_count(bin_idx, max_ships):
    counts = torch.tensor(SHIP_COUNTS, dtype=torch.long, device=bin_idx.device)
    return counts[bin_idx].clamp(max=max_ships.long())


if __name__ == "__main__":
    from config import ModelConfig
    cfg = ModelConfig()
    model = EntityTransformer(cfg)
    print(f"Model params: {count_params(model):,}")
    B = 2
    N_p, N_f = 20, 10
    planet_features = torch.randn(B, N_p, cfg.planet_feature_dim)
    fleet_features = torch.randn(B, N_f, cfg.fleet_feature_dim)
    global_features = torch.randn(B, cfg.global_feature_dim)
    planet_mask = torch.ones(B, N_p, dtype=torch.bool)
    fleet_mask = torch.ones(B, N_f, dtype=torch.bool)
    out = model(planet_features, fleet_features, global_features, planet_mask, fleet_mask)
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
