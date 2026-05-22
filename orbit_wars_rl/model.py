"""Entity Transformer in PyTorch for Orbit Wars.

Architecture (ADR-001/002/003 documented):
- 72-bin discretized angles (ADR-001)
- Shared backbone + mode token for 2p/4p (ADR-002)
- Baked-in geometric features, discovered strategy (ADR-003)

~307K params: 3 layers, 96 dim, 4 heads, 3x MLP expansion.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ANGLE_BINS = 72
NUM_SHIP_BINS = 16
ANGLE_BIN_WIDTH = 2 * math.pi / NUM_ANGLE_BINS
SHIP_COUNTS = [1, 2, 3, 5, 8, 13, 20, 30, 45, 65, 90, 120, 160, 200, 250, 300]


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

        # Action heads (per owned planet)
        self.fire_head = nn.Linear(D, 1)
        self.angle_head = nn.Linear(D, NUM_ANGLE_BINS)
        self.ship_head = nn.Linear(D, NUM_SHIP_BINS)

        # Value head
        self.value_fc1 = nn.Linear(D, D)
        self.value_fc2 = nn.Linear(D, D // 2)
        self.value_out = nn.Linear(D // 2, 1)

    def forward(self, planet_features, fleet_features, global_features,
                planet_mask, fleet_mask, fire_mask=None, angle_mask=None,
                slot_valid=None, owned_indices=None, owned_count=None):
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

        # Action heads
        fire_logits = self.fire_head(owned_entities).squeeze(-1)  # (B, max_owned)
        angle_logits = self.angle_head(owned_entities)  # (B, max_owned, 72)
        ship_logits = self.ship_head(owned_entities)  # (B, max_owned, 16)

        # Apply masks (-100 is safe in float16 on MPS; -1e9 overflows)
        if fire_mask is not None:
            fire_logits = fire_logits.masked_fill(~fire_mask, -100.0)
        if angle_mask is not None:
            angle_logits = angle_logits.masked_fill(~angle_mask, -100.0)
        if slot_valid is not None:
            fire_logits = fire_logits.masked_fill(~slot_valid, -100.0)
            angle_logits = angle_logits.masked_fill(~slot_valid.unsqueeze(-1), -100.0)
            ship_logits = ship_logits.masked_fill(~slot_valid.unsqueeze(-1), -100.0)

        # Value head: mean-pool valid entities
        valid_float = (~attn_mask).float()  # (B, N) 1=valid, 0=pad
        pooled = (x * valid_float.unsqueeze(-1)).sum(dim=1) / valid_float.sum(dim=1, keepdim=True).clamp(min=1)
        value = self.value_out(F.gelu(self.value_fc2(F.gelu(self.value_fc1(pooled))))).squeeze(-1)

        return {
            "fire_logits": fire_logits,
            "angle_logits": angle_logits,
            "ship_logits": ship_logits,
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