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
PHASE4_COMPAT_MISSING_KEYS = {
    "fire_q.weight", "fire_q.bias",
    "fire_k.weight", "fire_k.bias",
    "fire_scorer.0.weight", "fire_scorer.0.bias",
    "fire_scorer.2.weight", "fire_scorer.2.bias",
    "ship_q.weight", "ship_q.bias",
    "ship_k.weight", "ship_k.bias",
    "ship_scorer.0.weight", "ship_scorer.0.bias",
    "ship_scorer.2.weight", "ship_scorer.2.bias",
}


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

        # Reinforcement curriculum: an annealed additive bias on OWN-target logits
        # (own planets, pairwise is_mine==1). Set externally per-iter by the training
        # loop (negative → 0 over training). Plain float, not a Parameter, so it never
        # enters state_dict — checkpoint-compatible. 0.0 = no effect (default/eval).
        self.reinforce_logit_bias = 0.0

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

        # Action heads (per owned planet).
        # NB: the angle head was removed — Phase 1 decodes fire direction from the
        # target via orbital-intercept geometry (target-decode), so the head was
        # dead weight (never sampled, no gradient). NUM_ANGLE_BINS is still used by
        # the env/action-mask geometry. Legacy checkpoints' angle_head.* keys are
        # dropped in load_state_dict below.
        # Ship head: bin count is configurable so the fraction-head experiment
        # can swap to 10 fraction bins. Default 32 = legacy absolute counts.
        self.num_ship_bins = getattr(cfg, "num_ship_bins", NUM_SHIP_BINS)
        self.max_planets = cfg.max_planets if hasattr(cfg, "max_planets") else 48
        # Phase 5: synthetic NO_OP target column ("do nothing this step"), always legal
        # for a valid source slot, at index == max_planets. Target width = max_planets+1.
        self.no_op_idx = self.max_planets
        if self.use_pairwise:
            # Direct per-(source, target) heads (Phase 5): fire/ship are scored per
            # target from the same [q_slot, k_target, pairwise] inputs as target
            # selection — NO slot-only prior + residual (that formulation left the
            # fire residual inert; see docs/phase5.md §1). The scorers ARE the heads.
            self.tgt_q = nn.Linear(D, D)
            self.tgt_k = nn.Linear(D, D)
            tgt_in = D + D + F_pair
            tgt_hidden = D
            self.target_scorer = nn.Sequential(
                nn.Linear(tgt_in, tgt_hidden),
                nn.GELU(),
                nn.Linear(tgt_hidden, 1),
            )
            self.fire_q = nn.Linear(D, D)
            self.fire_k = nn.Linear(D, D)
            self.fire_scorer = nn.Sequential(
                nn.Linear(tgt_in, tgt_hidden),
                nn.GELU(),
                nn.Linear(tgt_hidden, 1),
            )
            self.ship_q = nn.Linear(D, D)
            self.ship_k = nn.Linear(D, D)
            self.ship_scorer = nn.Sequential(
                nn.Linear(tgt_in, tgt_hidden),
                nn.GELU(),
                nn.Linear(tgt_hidden, self.num_ship_bins),
            )
            # Learned NO_OP target logit per source slot (the "do nothing" column key).
            self.no_op_head = nn.Linear(D, 1)
            # Fresh model: small-init the head output layers so they start near-neutral
            # but off dead-zero. phase4_residual_init_std reused as the init scale.
            resid_init_std = float(getattr(cfg, "phase4_residual_init_std", 0.0))
            if resid_init_std > 0.0:
                nn.init.normal_(self.fire_scorer[-1].weight, mean=0.0, std=resid_init_std)
                nn.init.normal_(self.ship_scorer[-1].weight, mean=0.0, std=resid_init_std)
            else:
                nn.init.zeros_(self.fire_scorer[-1].weight)
                nn.init.zeros_(self.ship_scorer[-1].weight)
            nn.init.zeros_(self.fire_scorer[-1].bias)
            nn.init.zeros_(self.ship_scorer[-1].bias)
        else:
            # Legacy non-pairwise fallback (tests / value-only): slot-only heads,
            # broadcast across targets. No NO_OP column on this path.
            self.target_head = nn.Linear(D, self.max_planets)
            self.fire_head = nn.Linear(D, 1)
            self.ship_head = nn.Linear(D, self.num_ship_bins)

        # Value head: concat global token + owned pool → Linear(2D→D) by default.
        # value_head_in=0 means auto (2*D); load_checkpoint sets it to D for
        # pre-Phase-1 checkpoints that used mean-pool-all-entities (D→D).
        _vh_in = getattr(cfg, "value_head_in", 0) or (2 * D)
        self.value_fc1 = nn.Linear(_vh_in, D)
        self.value_fc2 = nn.Linear(D, D // 2)
        self.value_out = nn.Linear(D // 2, 1)

    def encode_state(self, planet_features, fleet_features, global_features,
                     planet_mask, fleet_mask, slot_valid=None, owned_indices=None,
                     pairwise_features=None):
        """Encode a state and expose backbone embeddings for auxiliary heads.

        Returns a dict with:
        - ``global_token``: (B, D)
        - ``planet_emb``:   (B, N_p, D) post-transformer planet embeddings
        - ``owned_entities``: (B, max_owned, D) slot embeddings
        - ``owned_enriched``: (B, max_owned, D) after pairwise enrichment
        - ``pairwise_features`` passthrough
        - ``attn_mask`` and ``x`` for value head compatibility
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
        planet_emb_post = x[:, 1:1 + planet_features.shape[1], :]
        owned_enriched = owned_entities

        # Pairwise cross-attention: enrich each slot with explicit (src, tgt) geometry
        # for fire/angle/ship heads. Target head scores per-(slot, target) directly
        # from the same per-target inputs — see docs/bugs.md.
        if self.use_pairwise and pairwise_features is not None:
            N_p = planet_features.shape[1]
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
            owned_enriched = self.pair_ln(owned_entities + self.pair_out(enriched))

        return {
            "x": x,
            "attn_mask": attn_mask,
            "global_token": x[:, 0, :],
            "planet_emb": planet_emb_post,
            "owned_entities": owned_entities,
            "owned_enriched": owned_enriched,
            "pairwise_features": pairwise_features,
        }

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
            angle_mask: accepted for caller compatibility but unused (angle head removed)
            slot_valid: (B, max_owned) bool, True = real owned planet slot
            owned_indices: (B, max_owned) int, indices into planet array
            owned_count: (B,) int

        Returns dict with per-target fire/ship logits, target_logits, value.
        """
        encoded = self.encode_state(
            planet_features, fleet_features, global_features,
            planet_mask, fleet_mask,
            slot_valid=slot_valid, owned_indices=owned_indices,
            pairwise_features=pairwise_features,
        )
        x = encoded["x"]
        attn_mask = encoded["attn_mask"]
        planet_emb_post = encoded["planet_emb"]
        owned_entities = encoded["owned_entities"]
        owned_enriched = encoded["owned_enriched"]
        B = planet_features.shape[0]
        max_owned = owned_enriched.shape[1]
        D = x.shape[-1]
        mp = self.max_planets

        def _pad_last(t, fill):
            """Right-pad a (..., N_p) tensor to width mp (or truncate)."""
            if t.shape[-1] < mp:
                pad = torch.full((*t.shape[:-1], mp - t.shape[-1]), fill,
                                 device=t.device, dtype=t.dtype)
                return torch.cat([t, pad], dim=-1)
            return t[..., :mp]

        if self.use_pairwise and pairwise_features is not None:
            # ---- Direct per-(slot, target) heads from [q_slot, k_target, pairwise] ----
            # Each of target/fire/ship is scored per real planet, then a synthetic
            # NO_OP column (idx == max_planets) is appended (Phase 5; no slot prior).
            N_p = planet_features.shape[1]
            q_tgt = self.tgt_q(owned_enriched).unsqueeze(2).expand(-1, -1, N_p, -1)
            k_tgt = self.tgt_k(planet_emb_post).unsqueeze(1).expand(-1, max_owned, -1, -1)
            tgt_scores = self.target_scorer(torch.cat([q_tgt, k_tgt, pairwise_features], dim=-1)).squeeze(-1)
            # Reinforcement curriculum: bias own-target logits only (is_mine, idx 5).
            # Negative bias suppresses reinforcement early; annealed → 0 so RL learns
            # the reinforce value from reward. Enemy/neutral (is_mine==0) untouched.
            if self.reinforce_logit_bias != 0.0:
                tgt_scores = tgt_scores + self.reinforce_logit_bias * pairwise_features[..., 5]

            q_fire = self.fire_q(owned_enriched).unsqueeze(2).expand(-1, -1, N_p, -1)
            k_fire = self.fire_k(planet_emb_post).unsqueeze(1).expand(-1, max_owned, -1, -1)
            fire_scores = self.fire_scorer(torch.cat([q_fire, k_fire, pairwise_features], dim=-1)).squeeze(-1)

            q_ship = self.ship_q(owned_enriched).unsqueeze(2).expand(-1, -1, N_p, -1)
            k_ship = self.ship_k(planet_emb_post).unsqueeze(1).expand(-1, max_owned, -1, -1)
            ship_scores = self.ship_scorer(torch.cat([q_ship, k_ship, pairwise_features], dim=-1))  # (B,MO,N_p,bins)

            target_real = _pad_last(tgt_scores, -100.0)        # (B, MO, mp)
            fire_logits = _pad_last(fire_scores, 0.0)          # (B, MO, mp)  (pad cols never gathered)
            if ship_scores.shape[-2] < mp:
                pad = torch.zeros(*ship_scores.shape[:-2], mp - ship_scores.shape[-2],
                                  ship_scores.shape[-1], device=ship_scores.device, dtype=ship_scores.dtype)
                ship_logits = torch.cat([ship_scores, pad], dim=-2)
            else:
                ship_logits = ship_scores[..., :mp, :]

            # Mask non-real planet columns in the TARGET head only (fire/ship are
            # gathered at an already-legal chosen target / NO_OP, never at a pad col).
            if planet_mask is not None:
                tgt_mask = planet_mask.unsqueeze(1).expand(-1, max_owned, -1)
                np_obs = tgt_mask.shape[-1]
                if mp > np_obs:
                    padm = torch.zeros(B, max_owned, mp - np_obs, dtype=torch.bool, device=tgt_mask.device)
                    tgt_mask = torch.cat([tgt_mask, padm], dim=-1)
                elif mp < np_obs:
                    tgt_mask = tgt_mask[..., :mp]
                target_real = target_real.masked_fill(~tgt_mask, -100.0)

            # ---- Append NO_OP column (always legal for valid slots) ----
            noop_t = self.no_op_head(owned_enriched)                                    # (B, MO, 1)
            target_logits = torch.cat([target_real, noop_t], dim=-1)                    # (B, MO, mp+1)
            noop_fire = torch.full((B, max_owned, 1), -100.0,
                                   device=fire_logits.device, dtype=fire_logits.dtype)  # fire forced 0 at NO_OP
            fire_logits = torch.cat([fire_logits, noop_fire], dim=-1)                   # (B, MO, mp+1)
            noop_ship = torch.zeros(B, max_owned, 1, ship_logits.shape[-1],
                                    device=ship_logits.device, dtype=ship_logits.dtype)
            ship_logits = torch.cat([ship_logits, noop_ship], dim=-2)                   # (B, MO, mp+1, bins)
        elif self.use_pairwise:
            # Pairwise model called without pairwise_features (value-only / legacy
            # test). Produce neutral logits at NO_OP width so callers don't crash.
            target_logits = torch.zeros(B, max_owned, mp + 1,
                                        device=owned_enriched.device, dtype=owned_enriched.dtype)
            fire_logits = torch.full((B, max_owned, mp + 1), -100.0,
                                     device=owned_enriched.device, dtype=owned_enriched.dtype)
            ship_logits = torch.zeros(B, max_owned, mp + 1, self.num_ship_bins,
                                      device=owned_enriched.device, dtype=owned_enriched.dtype)
        else:
            # Legacy non-pairwise fallback: slot-only heads broadcast across targets.
            fire_slot = self.fire_head(owned_enriched).squeeze(-1)              # (B, MO)
            ship_slot = self.ship_head(owned_enriched)                         # (B, MO, bins)
            fire_logits = fire_slot.unsqueeze(-1).expand(-1, -1, mp).clone()
            ship_logits = ship_slot.unsqueeze(2).expand(-1, -1, mp, -1).clone()
            target_logits = self.target_head(owned_entities)                   # (B, MO, mp)
            if planet_mask is not None:
                tgt_mask = planet_mask.unsqueeze(1).expand(-1, max_owned, -1)
                np_obs = tgt_mask.shape[-1]
                if mp > np_obs:
                    padm = torch.zeros(B, max_owned, mp - np_obs, dtype=torch.bool, device=tgt_mask.device)
                    tgt_mask = torch.cat([tgt_mask, padm], dim=-1)
                elif mp < np_obs:
                    tgt_mask = tgt_mask[..., :mp]
                target_logits = target_logits.masked_fill(~tgt_mask, -100.0)

        # min_ship_bin: bins below this are masked to -inf so they're never
        # sampled / argmaxed (removes the degenerate "10%-of-source" 1-ship trap).
        min_ship_bin = int(getattr(self.cfg, "min_ship_bin", 0))
        if min_ship_bin > 0:
            ship_logits = ship_logits.clone()
            ship_logits[..., :min_ship_bin] = -100.0

        # Apply slot masks across the full target width incl. NO_OP (-100 is float16-safe).
        if fire_mask is not None:
            fire_logits = fire_logits.masked_fill(~fire_mask.unsqueeze(-1), -100.0)
        if slot_valid is not None:
            fire_logits = fire_logits.masked_fill(~slot_valid.unsqueeze(-1), -100.0)
            ship_logits = ship_logits.masked_fill(~slot_valid.unsqueeze(-1).unsqueeze(-1), -100.0)
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
                owned_pool = (owned_enriched * owned_float).sum(dim=1) / n_owned
            else:
                owned_pool = owned_enriched.mean(dim=1)
            value_input = torch.cat([global_token, owned_pool], dim=-1)  # (B, 2D)
        value = self.value_out(F.gelu(self.value_fc2(F.gelu(self.value_fc1(value_input))))).squeeze(-1)

        return {
            "fire_logits": fire_logits,
            "ship_logits": ship_logits,
            "target_logits": target_logits,
            "value": value,
        }

    def load_state_dict(self, state_dict, strict=True):
        # Legacy checkpoints carry a now-removed angle head; drop those keys so
        # resume/eval/export from pre-removal checkpoints (rev38, rev32b, ...) work.
        if any(k.startswith("angle_head.") for k in state_dict):
            state_dict = {k: v for k, v in state_dict.items()
                          if not k.startswith("angle_head.")}
        # Pairwise-channel growth (e.g. adding reachable_enemy_mass = ch15): the per-target
        # scorers' first Linear gains trailing input columns. Zero-pad the new columns of an
        # older checkpoint so the new feature contributes nothing at step 0 (decode + log-prob
        # match the source model). pairwise is the last block in [q, k, pairwise], so new
        # channels are the LAST columns → right-pad. Same parity-init idea as the Phase 4 heads.
        own = self.state_dict()
        for name in ("pair_kv.weight", "target_scorer.0.weight",
                     "fire_scorer.0.weight", "ship_scorer.0.weight"):
            if name in state_dict and name in own:
                ck, md = state_dict[name], own[name]
                if ck.dim() == 2 and ck.shape[0] == md.shape[0] and ck.shape[1] < md.shape[1]:
                    pad = torch.zeros(ck.shape[0], md.shape[1] - ck.shape[1],
                                      dtype=ck.dtype, device=ck.device)
                    state_dict = dict(state_dict)
                    state_dict[name] = torch.cat([ck, pad], dim=1)
        return super().load_state_dict(state_dict, strict=strict)


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
