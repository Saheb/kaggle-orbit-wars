"""Export trained PyTorch model to standalone main.py for Kaggle submission.

Embeds model parameters as a base64-encoded torch state_dict and inlines the
features / action_mask code so the output file is fully self-contained.

Usage:
    python export_agent.py --checkpoint checkpoints/final.pt --output main_submitted.py
"""

from __future__ import annotations

import argparse
import base64
import io
import os

import torch
import numpy as np

from config import Config, ModelConfig
from model import EntityTransformer, NUM_ANGLE_BINS, NUM_SHIP_BINS, ANGLE_BIN_WIDTH


# ---------------------------------------------------------------------------
# Agent template
# The exported file is fully self-contained: no imports from this repo.
# ModelConfig is inlined as _Cfg; params are embedded as base64 bytes.
# ---------------------------------------------------------------------------

AGENT_TEMPLATE = '''\
"""Orbit Wars — Entity Transformer Agent (auto-exported, do not edit)"""

import base64
import io
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kaggle_environments.envs.orbit_wars.orbit_wars import CENTER, ROTATION_RADIUS_LIMIT

# --- Hyperparameters (inlined from ModelConfig) ---
NUM_ANGLE_BINS = {num_angle_bins}
NUM_SHIP_BINS = {num_ship_bins}
ANGLE_BIN_WIDTH = 2 * math.pi / NUM_ANGLE_BINS
BOARD_SIZE = 100.0
SUN_RADIUS = 10.0
CENTER_XY = 50.0
MAX_SPEED = 6.0
MAX_OWNED = 10
_ENTITY_DIM = {entity_dim}
_NUM_HEADS = {num_heads}
_NUM_LAYERS = {num_layers}
_MLP_EXP = {mlp_expansion}
_PLANET_DIM = {planet_feature_dim}
_FLEET_DIM = {fleet_feature_dim}
_GLOBAL_DIM = {global_feature_dim}
_MAX_PLANETS = 48
_MAX_FLEETS = 128


# --- Embedded parameters (base64-encoded torch state_dict) ---
_PARAMS_B64 = """{params_b64}"""


# --- Model architecture (mirrors model.py) ---

class _Block(nn.Module):
    def __init__(self, dim, heads, mlp_exp, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_exp), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * mlp_exp, dim), nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.ln1(x + a)
        return self.ln2(x + self.mlp(x))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        D = _ENTITY_DIM
        self.planet_proj = nn.Linear(_PLANET_DIM, D)
        self.fleet_proj = nn.Linear(_FLEET_DIM, D)
        self.global_proj = nn.Linear(_GLOBAL_DIM, D)
        self.mode_proj = nn.Linear(_GLOBAL_DIM, D)
        self.blocks = nn.ModuleList([_Block(D, _NUM_HEADS, _MLP_EXP) for _ in range(_NUM_LAYERS)])
        self.fire_head = nn.Linear(D, 1)
        self.angle_head = nn.Linear(D, NUM_ANGLE_BINS)
        self.ship_head = nn.Linear(D, NUM_SHIP_BINS)
        self.value_fc1 = nn.Linear(D, D)
        self.value_fc2 = nn.Linear(D, D // 2)
        self.value_out = nn.Linear(D // 2, 1)

    def forward(self, pf, ff, gf, pm, fm, fire_mask=None, angle_mask=None,
                slot_valid=None, owned_indices=None):
        B, D = pf.shape[0], _ENTITY_DIM
        planet_emb = self.planet_proj(pf)
        fleet_emb = self.fleet_proj(ff)
        global_emb = self.global_proj(gf) + self.mode_proj(gf)
        entities = torch.cat([global_emb.unsqueeze(1), planet_emb, fleet_emb], dim=1)

        N = entities.shape[1]
        attn_mask = torch.ones(B, N, dtype=torch.bool)
        attn_mask[:, 0] = False
        attn_mask[:, 1:1+pf.shape[1]] = ~pm
        attn_mask[:, 1+pf.shape[1]:] = ~fm

        x = entities
        for block in self.blocks:
            x = block(x, mask=attn_mask)

        if owned_indices is None:
            owned_indices = torch.zeros(B, MAX_OWNED, dtype=torch.long)
        fi = (owned_indices + 1).clamp(0, x.shape[1] - 1)
        bi = torch.arange(B).unsqueeze(1).expand(-1, MAX_OWNED)
        oe = x[bi, fi]

        fl = self.fire_head(oe).squeeze(-1)
        al = self.angle_head(oe)
        sl = self.ship_head(oe)

        if fire_mask is not None:
            fl = fl.masked_fill(~fire_mask, -1e9)
        if angle_mask is not None:
            al = al.masked_fill(~angle_mask, -1e9)
        if slot_valid is not None:
            fl = fl.masked_fill(~slot_valid, -1e9)
            al = al.masked_fill(~slot_valid.unsqueeze(-1), -1e9)
            sl = sl.masked_fill(~slot_valid.unsqueeze(-1), -1e9)

        vf = (~attn_mask).float()
        pooled = (x * vf.unsqueeze(-1)).sum(1) / vf.sum(1, keepdim=True).clamp(min=1)
        v = self.value_out(F.gelu(self.value_fc2(F.gelu(self.value_fc1(pooled))))).squeeze(-1)
        return dict(fire_logits=fl, angle_logits=al, ship_logits=sl, value=v)


# --- Lazy model loader ---
_model_cache = [None]

def _get_model():
    if _model_cache[0] is not None:
        return _model_cache[0]
    m = _Model()
    buf = io.BytesIO(base64.b64decode(_PARAMS_B64))
    sd = torch.load(buf, map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    _model_cache[0] = m
    return m


# --- Feature extraction (inlined from features.py) ---

{features_code}


# --- Action masks (inlined from action_mask.py) ---

{action_mask_code}


# --- Kaggle agent entry point ---

def agent(obs):
    model = _get_model()

    if not isinstance(obs, dict):
        obs = {{
            "step": int(getattr(obs, "step", 0)),
            "player": int(getattr(obs, "player", 0)),
            "planets": [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
                        for p in obs.planets],
            "fleets": [[f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships]
                       for f in obs.fleets],
            "angular_velocity": float(getattr(obs, "angular_velocity", 0.0)),
            "initial_planets": [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
                                 for p in getattr(obs, "initial_planets", obs.planets)],
            "comet_planet_ids": list(getattr(obs, "comet_planet_ids", [])),
        }}

    player = obs["player"]
    features = extract_features(obs, player, num_players=2)
    masks = compute_action_masks(obs, player)

    with torch.no_grad():
        out = model(
            features["planet_features"].unsqueeze(0),
            features["fleet_features"].unsqueeze(0),
            features["global_features"].unsqueeze(0),
            features["planet_mask"].unsqueeze(0),
            features["fleet_mask"].unsqueeze(0),
            fire_mask=masks["fire_mask"],
            angle_mask=masks["angle_mask"],
            slot_valid=masks["slot_valid"],
            owned_indices=masks["owned_indices"].unsqueeze(0),
        )

    return actions_from_policy(
        out["fire_logits"],
        out["angle_logits"],
        out["ship_logits"],
        {{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()}},
        obs, player,
    )
'''


# ---------------------------------------------------------------------------
# Export logic
# ---------------------------------------------------------------------------

def _read_module_body(filepath: str, strip_imports: bool = True) -> str:
    """Read a Python file and optionally strip top-level import lines."""
    with open(filepath) as f:
        lines = f.readlines()

    if not strip_imports:
        return "".join(lines)

    out = []
    in_docstring = False
    for line in lines:
        stripped = line.strip()
        # Skip module-level imports and the module docstring
        if stripped.startswith('"""') or stripped.startswith("'''"):
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if stripped.startswith("from __future__"):
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            # Keep relative imports that refer to game constants
            if "kaggle_environments" in stripped:
                continue  # the template already imports CENTER / ROTATION_RADIUS_LIMIT
            continue
        out.append(line)

    return "".join(out)


def load_model(checkpoint_path: str, cfg: Config) -> EntityTransformer:
    model = EntityTransformer(cfg.model)
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    model.eval()
    return model


def export_agent(checkpoint_path: str, output_path: str, cfg: Config):
    model = load_model(checkpoint_path, cfg)

    # Encode state_dict as base64
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    params_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # Read feature / action-mask code (strip duplicate imports; agent template provides them)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    features_code = _read_module_body(os.path.join(src_dir, "features.py"))
    action_mask_code = _read_module_body(os.path.join(src_dir, "action_mask.py"))

    m = cfg.model
    agent_code = AGENT_TEMPLATE.format(
        num_angle_bins=NUM_ANGLE_BINS,
        num_ship_bins=NUM_SHIP_BINS,
        angle_bin_width=ANGLE_BIN_WIDTH,
        entity_dim=m.entity_dim,
        num_heads=m.num_heads,
        num_layers=m.num_layers,
        mlp_expansion=m.mlp_expansion,
        planet_feature_dim=m.planet_feature_dim,
        fleet_feature_dim=m.fleet_feature_dim,
        global_feature_dim=m.global_feature_dim,
        params_b64=params_b64,
        features_code=features_code,
        action_mask_code=action_mask_code,
    )

    with open(output_path, "w") as f:
        f.write(agent_code)

    print(f"Exported agent → {output_path}")
    print(f"  Param bytes (base64): {len(params_b64):,}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="main_submitted.py")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config()
    cfg.seed = args.seed
    export_agent(args.checkpoint, args.output, cfg)
