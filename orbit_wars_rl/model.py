"""Entity Transformer in PyTorch for Orbit Wars.

Architecture (ADR-001/002/003 documented):
- 72-bin discretized angles (ADR-001)
- Shared backbone + mode token for 2p/4p (ADR-002)
- Baked-in geometric features, discovered strategy (ADR-003)

Phase 1 feature dims: planet=20, fleet=13, global=11, pairwise=20, max_owned=16.
Value head: concat(global_token, owned_pool) → 2D → D → D/2 → 1.
~350K params: 3 layers, 96 dim, 4 heads, 3x MLP expansion.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from action_mask import SHIP_COUNTS  # single source of truth; re-exported for `from model import SHIP_COUNTS`

NUM_ANGLE_BINS = 144
NUM_SHIP_BINS = len(SHIP_COUNTS)
ANGLE_BIN_WIDTH = 2 * math.pi / NUM_ANGLE_BINS
PHASE4_COMPAT_MISSING_KEYS = {
    "fire_q.weight", "fire_q.bias",
    "fire_k.weight", "fire_k.bias",
    "fire_scorer.0.weight", "fire_scorer.0.bias",
    "fire_scorer.2.weight", "fire_scorer.2.bias",
    "ship_q.weight", "ship_q.bias",
    "ship_k.weight", "ship_k.bias",
    "ship_scorer.0.weight", "ship_scorer.0.bias",
    "ship_scorer.2.weight", "ship_scorer.2.bias",
    # COMA counterfactual Q-head (docs/q-head.md) — absent from every pre-Q-head
    # checkpoint; registered so resume/eval/export don't abort on the missing keys.
    "q_fire_embed.weight", "q_ship_embed.weight",
    "q_tgt_proj.weight", "q_tgt_proj.bias",
    "q_sa_mlp.weight", "q_sa_mlp.bias",
    "q_fc.weight", "q_fc.bias",
    "q_out.weight", "q_out.bias",
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
        assert F_pair > 0, (
            "pairwise_feature_dim must be > 0 — the pairwise cross-attention target head "
            "is mandatory since the always-pairwise cleanup. Pre-pairwise checkpoints "
            "(pairwise_feature_dim=0) are unsupported; load them from git tag pre-cleanup-2026-07."
        )
        self.pair_kv = nn.Linear(D + F_pair, 2 * D)
        self.pair_q = nn.Linear(D, D)
        self.pair_out = nn.Linear(D, D)
        self.pair_ln = nn.LayerNorm(D)

        # Action heads (per owned planet)
        self.fire_head = nn.Linear(D, 1)
        # NB: the angle head was removed — Phase 1 decodes fire direction from the
        # target via orbital-intercept geometry (target-decode), so the head was
        # dead weight (never sampled, no gradient). NUM_ANGLE_BINS is still used by
        # the env/action-mask geometry.
        # Ship head: bin count is configurable so the fraction-head experiment
        # can swap to 10 fraction bins. Default 32 = legacy absolute counts.
        self.num_ship_bins = getattr(cfg, "num_ship_bins", NUM_SHIP_BINS)
        self.ship_head = nn.Linear(D, self.num_ship_bins)
        # Target-index head: score each (slot, target) pair from per-target inputs —
        # see docs/bugs.md (target-head collapse).
        self.max_planets = cfg.max_planets if hasattr(cfg, "max_planets") else 48
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
        resid_init_std = float(getattr(cfg, "phase4_residual_init_std", 0.0))
        if resid_init_std > 0.0:
            nn.init.normal_(self.fire_scorer[-1].weight, mean=0.0, std=resid_init_std)
            nn.init.normal_(self.ship_scorer[-1].weight, mean=0.0, std=resid_init_std)
        else:
            nn.init.zeros_(self.fire_scorer[-1].weight)
            nn.init.zeros_(self.ship_scorer[-1].weight)
        nn.init.zeros_(self.fire_scorer[-1].bias)
        nn.init.zeros_(self.ship_scorer[-1].bias)

        # Value head: concat global token + owned pool → Linear(2D→D) by default.
        # value_head_in=0 means auto (2*D); load_checkpoint sets it to D for
        # pre-Phase-1 checkpoints that used mean-pool-all-entities (D→D).
        _vh_in = getattr(cfg, "value_head_in", 0) or (2 * D)
        self.value_fc1 = nn.Linear(_vh_in, D)
        self.value_fc2 = nn.Linear(D, D // 2)
        self.value_out = nn.Linear(D // 2, 1)

        # --- COMA counterfactual Q-head (docs/q-head.md) ----------------------
        # Shape-twin of the value head with a per-slot ACTION stream. Each owned
        # slot gets a state-action token  sa_i = q_sa_mlp([owned_enriched_i, a_emb_i])
        # that BINDS the slot's action to its own state; the mean over valid slots
        # is action_pool, and Q = q_out(gelu(q_fc([global_token, action_pool]))).
        # The additive mean-pool makes a per-slot counterfactual a one-term delta
        # (q_counterfactual). fire/idle are MARGINALIZED, not silenced, so idle slots
        # get a gradient to START firing (idle gates target/ship off → idle = the
        # idle fire-embed only). q_out zero-init → Q≈0 and A_i≈0 at step 0.
        self.q_fire_embed = nn.Embedding(2, D)                 # idle(0) / fire(1)
        self.q_ship_embed = nn.Embedding(self.num_ship_bins, D)
        self.q_tgt_proj = nn.Linear(D, D)                      # projects the gathered target embedding
        self.q_sa_mlp = nn.Linear(2 * D, D)                    # per-slot state-action token
        self.q_fc = nn.Linear(2 * D, D)                        # twin of value_fc1
        self.q_out = nn.Linear(D, 1)                           # twin of value_out
        nn.init.zeros_(self.q_out.weight)
        nn.init.zeros_(self.q_out.bias)

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
        if pairwise_features is not None:
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
                planet_mask, fleet_mask, fire_mask=None,
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
        target_logits = None

        # Per-target scoring head: each (slot, target) gets its own logit from
        # [q_slot, k_target, pair_features]. This is the fix for the collapse
        # documented in docs/bugs.md — the prior Linear(D, max_planets) head
        # had no per-target conditioning, capping target_top1 near random.
        if pairwise_features is not None:
            N_p = planet_features.shape[1]
            q_tgt = self.tgt_q(owned_enriched).unsqueeze(2).expand(-1, -1, N_p, -1)
            k_tgt = self.tgt_k(planet_emb_post).unsqueeze(1).expand(-1, max_owned, -1, -1)
            scorer_in = torch.cat([q_tgt, k_tgt, pairwise_features], dim=-1)
            tgt_scores = self.target_scorer(scorer_in).squeeze(-1)              # (B, MO, N_p)
            # Reinforcement curriculum: bias own-target logits only (is_mine, idx 5).
            # Negative bias suppresses reinforcement early; annealed → 0 so RL learns
            # the reinforce value from reward. Enemy/neutral (is_mine==0) untouched.
            if self.reinforce_logit_bias != 0.0:
                tgt_scores = tgt_scores + self.reinforce_logit_bias * pairwise_features[..., 5]
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
        else:
            # Some tests/callers do not pass pairwise_features and do not consume
            # target_logits. Keep forward usable for those paths with a zeros target.
            target_logits = torch.zeros(
                B, max_owned, self.max_planets,
                device=owned_enriched.device, dtype=owned_enriched.dtype,
            )

        # Action heads. Fire/ship now condition on the chosen target via the same
        # per-(slot, target) pairwise path as target selection. The legacy slot-only
        # heads remain as residual priors so old checkpoints keep step-0 behavior.
        fire_logits_slot = self.fire_head(owned_enriched).squeeze(-1)  # (B, max_owned)
        ship_logits_slot = self.ship_head(owned_enriched)  # (B, max_owned, num_ship_bins)
        fire_prior = fire_logits_slot.unsqueeze(-1).expand(-1, -1, self.max_planets)
        ship_prior = ship_logits_slot.unsqueeze(2).expand(-1, -1, self.max_planets, -1)
        fire_residual = torch.zeros_like(fire_prior)
        ship_residual = torch.zeros_like(ship_prior)
        fire_logits = fire_prior
        ship_logits = ship_prior
        if pairwise_features is not None:
            N_p = planet_features.shape[1]
            q_fire = self.fire_q(owned_enriched).unsqueeze(2).expand(-1, -1, N_p, -1)
            k_fire = self.fire_k(planet_emb_post).unsqueeze(1).expand(-1, max_owned, -1, -1)
            fire_in = torch.cat([q_fire, k_fire, pairwise_features], dim=-1)
            fire_resid_live = self.fire_scorer(fire_in).squeeze(-1)

            q_ship = self.ship_q(owned_enriched).unsqueeze(2).expand(-1, -1, N_p, -1)
            k_ship = self.ship_k(planet_emb_post).unsqueeze(1).expand(-1, max_owned, -1, -1)
            ship_in = torch.cat([q_ship, k_ship, pairwise_features], dim=-1)
            ship_resid_live = self.ship_scorer(ship_in)

            fire_logits = fire_logits.clone()
            ship_logits = ship_logits.clone()
            fire_residual[..., :N_p] = fire_resid_live
            ship_residual[..., :N_p, :] = ship_resid_live
            fire_logits[..., :N_p] = fire_logits[..., :N_p] + fire_resid_live
            ship_logits[..., :N_p, :] = ship_logits[..., :N_p, :] + ship_resid_live
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
            "_phase4_fire_prior": fire_prior,
            "_phase4_ship_prior": ship_prior,
            "_phase4_fire_residual": fire_residual,
            "_phase4_ship_residual": ship_residual,
        }

    # --- COMA counterfactual Q-head ------------------------------------------
    def _q_slot_tokens(self, owned_enriched, planet_emb_post, ship_bin, target_idx):
        """Per-slot state-action tokens under FORCE-FIRE and FORCE-IDLE for every
        slot, binding each slot's action to its own state. Returns (sa_fire, sa_idle),
        each (B, MO, D). The target embedding is the GATHERED post-transformer planet
        embedding (not an index lookup). Idle = the idle fire-embed only (target/ship
        gated off), so 'idle' is a well-defined no-op rather than a removed slot."""
        B, MO, D = owned_enriched.shape
        Np = planet_emb_post.shape[1]
        tgt_clamped = target_idx.long().clamp(0, Np - 1)                            # (B, MO)
        tgt_emb = torch.gather(planet_emb_post, 1,
                               tgt_clamped.unsqueeze(-1).expand(-1, -1, D))         # (B, MO, D)
        ship_emb = self.q_ship_embed(ship_bin.long().clamp(0, self.num_ship_bins - 1))
        a_emb_fire = self.q_fire_embed.weight[1] + self.q_tgt_proj(tgt_emb) + ship_emb
        a_emb_idle = self.q_fire_embed.weight[0].view(1, 1, D).expand(B, MO, D)
        sa_fire = self.q_sa_mlp(torch.cat([owned_enriched, a_emb_fire], dim=-1))    # (B, MO, D)
        sa_idle = self.q_sa_mlp(torch.cat([owned_enriched, a_emb_idle], dim=-1))    # (B, MO, D)
        return sa_fire, sa_idle

    def _q_from_pool(self, global_token, action_pool):
        return self.q_out(F.gelu(self.q_fc(torch.cat([global_token, action_pool], dim=-1)))).squeeze(-1)

    def q_counterfactual(self, encoded, fire_bit, ship_bin, target_idx, slot_valid):
        """COMA Q-head: Q(s,a) and the per-slot fire/idle counterfactuals via the
        additive mean-pool delta. Returns a dict:
           q_sa       (B,)      Q of the taken joint action
           q_fire     (B, MO)   Q with slot i forced to fire (its sampled tgt/ship)
           q_idle     (B, MO)   Q with slot i forced idle
           q_all_idle (B,)      Q with every valid slot idle (global sensitivity probe)
        The caller forms the COMA advantage
           A_i = q_sa − (p_i·q_fire_i + (1−p_i)·q_idle_i),   p_i = sigmoid(fire_logit_i)
        which on an idle slot is −p_i·(q_fire_i − q_idle_i) → pushes P(idle) down when
        firing would help (the gradient the scalar-advantage surrogate is missing)."""
        owned_enriched = encoded["owned_enriched"]            # (B, MO, D)
        planet_emb_post = encoded["planet_emb"]               # (B, Np, D)
        global_token = encoded["global_token"]                # (B, D)
        B, MO, D = owned_enriched.shape
        sv = slot_valid.float()                               # (B, MO)
        n_valid = sv.sum(dim=1).clamp(min=1.0)                # (B,)

        sa_fire, sa_idle = self._q_slot_tokens(owned_enriched, planet_emb_post, ship_bin, target_idx)
        fired = (fire_bit > 0).unsqueeze(-1)                  # (B, MO, 1)
        sa_taken = torch.where(fired, sa_fire, sa_idle)       # (B, MO, D)

        action_pool = (sa_taken * sv.unsqueeze(-1)).sum(dim=1) / n_valid.unsqueeze(-1)   # (B, D)
        q_sa = self._q_from_pool(global_token, action_pool)   # (B,)

        # Additive mean-pool delta: swapping ONLY slot i changes the mean by
        # sv_i·(sa_i^cf − sa_taken_i)/n_valid  (exact; invalid slots → 0 change).
        coef = (sv / n_valid.unsqueeze(-1)).unsqueeze(-1)     # (B, MO, 1)
        pool_fire = action_pool.unsqueeze(1) + coef * (sa_fire - sa_taken)   # (B, MO, D)
        pool_idle = action_pool.unsqueeze(1) + coef * (sa_idle - sa_taken)   # (B, MO, D)
        gt = global_token.unsqueeze(1).expand(-1, MO, -1).reshape(B * MO, D)
        q_fire = self._q_from_pool(gt, pool_fire.reshape(B * MO, D)).view(B, MO)
        q_idle = self._q_from_pool(gt, pool_idle.reshape(B * MO, D)).view(B, MO)

        # Every valid slot idle — the Q(s,a) − Q(s, all-idle) global sensitivity probe.
        pool_all_idle = (sa_idle * sv.unsqueeze(-1)).sum(dim=1) / n_valid.unsqueeze(-1)
        q_all_idle = self._q_from_pool(global_token, pool_all_idle)

        return {"q_sa": q_sa, "q_fire": q_fire, "q_idle": q_idle, "q_all_idle": q_all_idle}

    def load_state_dict(self, state_dict, strict=True):
        # Input-channel growth (e.g. adding target-value / keepability pairwise channels, or
        # game-phase globals 11-14 on top of an 11-global checkpoint): the affected input Linear
        # gains trailing input columns. Zero-pad the new columns of an older checkpoint so the
        # new feature contributes nothing at step 0 (decode + log-prob match the source model).
        # New channels are appended LAST (pairwise is the last block in [q, k, pairwise]; phase
        # globals are extend()-ed after the base globals) → right-pad. Same parity-init idea as
        # the Phase 4 heads. global_proj/mode_proj both consume the global vector (see forward).
        own = self.state_dict()
        for name in ("pair_kv.weight", "target_scorer.0.weight",
                     "fire_scorer.0.weight", "ship_scorer.0.weight",
                     "global_proj.weight", "mode_proj.weight"):
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
