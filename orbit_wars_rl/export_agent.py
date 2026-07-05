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
from model import EntityTransformer, NUM_ANGLE_BINS, NUM_SHIP_BINS, ANGLE_BIN_WIDTH, PHASE4_COMPAT_MISSING_KEYS
from features import PAIRWISE_FEATURE_DIM


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
import os

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
MAX_OWNED = {max_owned_planets}
_ENTITY_DIM = {entity_dim}
_NUM_HEADS = {num_heads}
_NUM_LAYERS = {num_layers}
_MLP_EXP = {mlp_expansion}
_PLANET_DIM = {planet_feature_dim}
_FLEET_DIM = {fleet_feature_dim}
_GLOBAL_DIM = {global_feature_dim}
_MAX_PLANETS = 48
_MAX_FLEETS = 128
_PAIRWISE_DIM = {pairwise_feature_dim}
_MIN_SHIP_BIN = {min_ship_bin}
_FIRE_THRESHOLD = {fire_threshold}
_SHIP_BIN_MODE = {ship_bin_mode}
_TARGET_DECODE = {target_decode}
_ALLOW_REINFORCE = {allow_reinforce}
# Reinforce-DISCIPLINE masks — MUST match training values (not stored in the checkpoint).
# Inert unless _ALLOW_REINFORCE; constrain only own (reinforce) targets.
_REINFORCE_GATE_MIN = {reinforce_gate_min_planets}
_REINFORCE_FORWARD_ONLY = {reinforce_forward_only}
_REINFORCE_GARRISON_FLOOR = {reinforce_garrison_floor}
# Sufficient-commit mask (ATTACKS) — also MUST match training. Independent of reinforce.
_SUFFICIENT_COMMIT_FACTOR = {sufficient_commit_factor}
# Path-obstruction mask (INFERENCE bugfix) — mask targets screened by an uncapturable planet so
# the head retargets. Independent of training; safe on any checkpoint.
_PATH_OBSTRUCTION_MASK = {path_obstruction_mask}
# Reverse-edge reinforce cooldown (MUST match training; auto-loaded from ckpt). 0 = off.
_REVERSE_EDGE_COOLDOWN = {reverse_edge_cooldown}


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
        self.use_pairwise = _PAIRWISE_DIM > 0
        if self.use_pairwise:
            self.pair_kv = nn.Linear(D + _PAIRWISE_DIM, 2 * D)
            self.pair_q = nn.Linear(D, D)
            self.pair_out = nn.Linear(D, D)
            self.pair_ln = nn.LayerNorm(D)
        self.fire_head = nn.Linear(D, 1)
        self.ship_head = nn.Linear(D, NUM_SHIP_BINS)
        if self.use_pairwise:
            self.tgt_q = nn.Linear(D, D)
            self.tgt_k = nn.Linear(D, D)
            self.target_scorer = nn.Sequential(
                nn.Linear(D + D + _PAIRWISE_DIM, D),
                nn.GELU(),
                nn.Linear(D, 1),
            )
            self.fire_q = nn.Linear(D, D)
            self.fire_k = nn.Linear(D, D)
            self.fire_scorer = nn.Sequential(
                nn.Linear(D + D + _PAIRWISE_DIM, D),
                nn.GELU(),
                nn.Linear(D, 1),
            )
            self.ship_q = nn.Linear(D, D)
            self.ship_k = nn.Linear(D, D)
            self.ship_scorer = nn.Sequential(
                nn.Linear(D + D + _PAIRWISE_DIM, D),
                nn.GELU(),
                nn.Linear(D, NUM_SHIP_BINS),
            )
            nn.init.zeros_(self.fire_scorer[-1].weight)
            nn.init.zeros_(self.fire_scorer[-1].bias)
            nn.init.zeros_(self.ship_scorer[-1].weight)
            nn.init.zeros_(self.ship_scorer[-1].bias)
        else:
            self.target_head = nn.Linear(D, _MAX_PLANETS)
        self.value_fc1 = nn.Linear(2 * D, D)
        self.value_fc2 = nn.Linear(D, D // 2)
        self.value_out = nn.Linear(D // 2, 1)

    def forward(self, pf, ff, gf, pm, fm, fire_mask=None, angle_mask=None,
                slot_valid=None, owned_indices=None, pairwise_features=None):
        B, D = pf.shape[0], _ENTITY_DIM
        max_owned = MAX_OWNED
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

        target_logits = None
        if self.use_pairwise and pairwise_features is not None:
            n_p = pf.shape[1]
            planet_emb_post = x[:, 1:1 + n_p, :]
            planet_per_slot = planet_emb_post.unsqueeze(1).expand(-1, max_owned, -1, -1)
            kv_input = torch.cat([planet_per_slot, pairwise_features], dim=-1)
            kv = self.pair_kv(kv_input)
            k, v = kv.chunk(2, dim=-1)
            q = self.pair_q(oe).unsqueeze(2)
            scores = (q @ k.transpose(-2, -1)).squeeze(2) * (D ** -0.5)
            tgt_valid = pm.unsqueeze(1).expand(-1, max_owned, -1)
            scores = scores.masked_fill(~tgt_valid, -1e4)
            attn = F.softmax(scores, dim=-1).unsqueeze(2)
            enriched = (attn @ v).squeeze(2)
            oe = self.pair_ln(oe + self.pair_out(enriched))

            q_tgt = self.tgt_q(oe).unsqueeze(2).expand(-1, -1, n_p, -1)
            k_tgt = self.tgt_k(planet_emb_post).unsqueeze(1).expand(-1, max_owned, -1, -1)
            scorer_in = torch.cat([q_tgt, k_tgt, pairwise_features], dim=-1)
            tgt_scores = self.target_scorer(scorer_in).squeeze(-1)
            if n_p < _MAX_PLANETS:
                pad = torch.full(
                    (B, max_owned, _MAX_PLANETS - n_p),
                    -100.0, device=tgt_scores.device, dtype=tgt_scores.dtype,
                )
                tgt_scores = torch.cat([tgt_scores, pad], dim=-1)
            else:
                tgt_scores = tgt_scores[..., :_MAX_PLANETS]
            target_logits = tgt_scores
        elif self.use_pairwise:
            target_logits = torch.zeros(
                B, max_owned, _MAX_PLANETS,
                device=oe.device, dtype=oe.dtype,
            )

        fl_slot = self.fire_head(oe).squeeze(-1)
        sl_slot = self.ship_head(oe)
        fl = fl_slot.unsqueeze(-1).expand(-1, -1, _MAX_PLANETS)
        sl = sl_slot.unsqueeze(2).expand(-1, -1, _MAX_PLANETS, -1)
        if self.use_pairwise and pairwise_features is not None:
            q_fire = self.fire_q(oe).unsqueeze(2).expand(-1, -1, n_p, -1)
            k_fire = self.fire_k(planet_emb_post).unsqueeze(1).expand(-1, max_owned, -1, -1)
            fire_in = torch.cat([q_fire, k_fire, pairwise_features], dim=-1)
            fire_resid = self.fire_scorer(fire_in).squeeze(-1)
            q_ship = self.ship_q(oe).unsqueeze(2).expand(-1, -1, n_p, -1)
            k_ship = self.ship_k(planet_emb_post).unsqueeze(1).expand(-1, max_owned, -1, -1)
            ship_in = torch.cat([q_ship, k_ship, pairwise_features], dim=-1)
            ship_resid = self.ship_scorer(ship_in)
            fl = fl.clone()
            sl = sl.clone()
            fl[..., :n_p] = fl[..., :n_p] + fire_resid
            sl[..., :n_p, :] = sl[..., :n_p, :] + ship_resid
        if _MIN_SHIP_BIN > 0:
            sl = sl.clone()
            sl[..., :_MIN_SHIP_BIN] = -100.0
        if target_logits is None:
            target_logits = self.target_head(oe)
        if pm is not None:
            tgt_mask = pm.unsqueeze(1).expand(-1, max_owned, -1)
            mp = target_logits.shape[-1]
            np_obs = tgt_mask.shape[-1]
            if mp > np_obs:
                pad = torch.zeros(B, max_owned, mp - np_obs, dtype=torch.bool, device=tgt_mask.device)
                tgt_mask = torch.cat([tgt_mask, pad], dim=-1)
            elif mp < np_obs:
                tgt_mask = tgt_mask[..., :mp]
            target_logits = target_logits.masked_fill(~tgt_mask, -100.0)

        if fire_mask is not None:
            fl = fl.masked_fill(~fire_mask.unsqueeze(-1), -100.0)
        if slot_valid is not None:
            fl = fl.masked_fill(~slot_valid.unsqueeze(-1), -100.0)
            sl = sl.masked_fill(~slot_valid.unsqueeze(-1).unsqueeze(-1), -100.0)
            target_logits = target_logits.masked_fill(~slot_valid.unsqueeze(-1), -100.0)

        global_token = x[:, 0, :]
        if slot_valid is not None:
            sv_f = slot_valid.float().unsqueeze(-1)
            n_owned = sv_f.sum(dim=1).clamp(min=1)
            owned_pool = (oe * sv_f).sum(dim=1) / n_owned
        else:
            owned_pool = oe.mean(dim=1)
        v = self.value_out(F.gelu(self.value_fc2(F.gelu(self.value_fc1(
            torch.cat([global_token, owned_pool], dim=-1)
        ))))).squeeze(-1)
        return dict(fire_logits=fl, ship_logits=sl,
                    target_logits=target_logits, value=v)


# --- Lazy model loader ---
_model_cache = [None]

def _get_model():
    if _model_cache[0] is not None:
        return _model_cache[0]
    m = _Model()
    buf = io.BytesIO(base64.b64decode(_PARAMS_B64))
    try:
        sd = torch.load(buf, map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(buf, map_location="cpu", weights_only=False)
    # strict=False: the embedded state_dict may carry inference-unused params
    # (COMA q_* head, VDN value_pp_*) the slim _Model doesn't define.
    m.load_state_dict(sd, strict=False)
    m.eval()
    _model_cache[0] = m
    return m


# --- Feature extraction (inlined from features.py; blessed-2026-07 semantics hard-coded) ---

{features_code}


# --- Reverse-edge cooldown rule (inlined from reinforce_cooldown.py) ---

{reinforce_cooldown_code}
_cd_is_blocked = is_blocked
_cd_record = record
_cd_on_loss = on_ownership_loss


# --- Action masks (inlined from action_mask.py) ---

{action_mask_code}


# --- Kaggle agent entry point ---

# Reverse-edge cooldown per-game state (canonical rule in reinforce_cooldown.py; mirrors
# eval.build_agent_fn). Cleared when the step counter resets (a new game in the same process).
_CD = {{"last": {{}}, "prev_step": -1}}


def agent(obs, cfg=None):
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
    if _REVERSE_EDGE_COOLDOWN > 0:
        _step_now = int(obs.get("step", 0))
        if _step_now <= _CD["prev_step"]:
            _CD["last"].clear()
        _CD["prev_step"] = _step_now
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
            pairwise_features=features["pairwise_features"].unsqueeze(0)
                if "pairwise_features" in features else None,
        )

    if _TARGET_DECODE:
        return actions_from_target_policy(
            out["fire_logits"],
            out["target_logits"],
            out["ship_logits"],
            {{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()}},
            obs, player,
            fire_threshold=_FIRE_THRESHOLD,
            ship_bin_mode=_SHIP_BIN_MODE,
            allow_reinforce=_ALLOW_REINFORCE,
            reinforce_gate_min_planets=_REINFORCE_GATE_MIN,
            reinforce_forward_only=_REINFORCE_FORWARD_ONLY,
            reinforce_garrison_floor=_REINFORCE_GARRISON_FLOOR,
            sufficient_commit_factor=_SUFFICIENT_COMMIT_FACTOR,
            path_obstruction_mask=_PATH_OBSTRUCTION_MASK,
            reverse_edge_cooldown=_REVERSE_EDGE_COOLDOWN,
            cooldown_last=_CD["last"] if _REVERSE_EDGE_COOLDOWN > 0 else None,
            cooldown_step=int(obs.get("step", 0)),
        )
    else:
        raise NotImplementedError(
            "angle-decode removed; this agent was exported for target-decode only"
        )
'''


# ---------------------------------------------------------------------------
# Export logic
# ---------------------------------------------------------------------------

def _strip_module_docstring(source: str) -> str:
    """Remove only the module-level docstring from Python source."""
    lines = source.split('\n')
    # Find the module docstring: skip blanks/comments, then find """ or '''
    start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s == '' or s.startswith('#'):
            continue
        if s.startswith('"""') or s.startswith("'''"):
            start = i
            break
        break  # first real line is not a docstring
    if start is None:
        return source

    delim = '"""' if lines[start].strip().startswith('"""') else "'''"
    stripped_start = lines[start].strip()
    # Check single-line docstring: """...all on one line..."""
    if stripped_start.count(delim) >= 2 and stripped_start.endswith(delim) and len(stripped_start) > 3:
        lines[start] = ''
        return '\n'.join(lines)

    # Multi-line: find closing delimiter
    end = None
    for i in range(start + 1, len(lines)):
        if lines[i].strip() == delim or (lines[i].strip().endswith(delim) and lines[i].strip().startswith(delim) == False):
            # Simple closing on its own line, or closing at end of line
            if lines[i].strip() == delim:
                end = i
                break
            # Closing delimiter at end of content line
            end = i
            break
        if lines[i].strip().endswith(delim):
            end = i
            break

    if end is not None:
        for i in range(start, end + 1):
            lines[i] = ''
    return '\n'.join(lines)


def _strip_imports(source: str) -> str:
    """Strip top-level import lines from Python source."""
    lines = source.split('\n')
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("from __future__"):
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            if "kaggle_environments" in stripped:
                continue
            continue
        out.append(line)
    return '\n'.join(out)


def _read_module_body(filepath: str, strip_imports: bool = True) -> str:
    """Read a Python file and strip module docstring + top-level imports."""
    with open(filepath) as f:
        source = f.read()
    if strip_imports:
        source = _strip_module_docstring(source)
        source = _strip_imports(source)
    return source


def _load_checkpoint(checkpoint_path: str):
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _apply_checkpoint_model_config(checkpoint, cfg: Config) -> dict:
    """Update cfg.model with architecture metadata saved in a checkpoint."""
    ckpt_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    if "num_ship_bins" in ckpt_cfg:
        cfg.model.num_ship_bins = int(ckpt_cfg["num_ship_bins"])
    elif isinstance(state_dict, dict) and "ship_head.weight" in state_dict:
        cfg.model.num_ship_bins = int(state_dict["ship_head.weight"].shape[0])

    if "min_ship_bin" in ckpt_cfg:
        cfg.model.min_ship_bin = int(ckpt_cfg["min_ship_bin"])
    if "ship_bin_mode" in ckpt_cfg:
        cfg.model.ship_bin_mode = str(ckpt_cfg["ship_bin_mode"])
    # Blessed feature config guard (2026-07 cleanup): the inlined extractor hard-codes
    # game-phase 15-global + precise resolver; refuse checkpoints trained otherwise. An ABSENT
    # pressure_precise_resolver counts as OFF (same as eval/train_torch — a 15-global corridor
    # checkpoint like stageb4/bc_snowball_15global must not silently export resolver features).
    # Blessed-era saves that lack the flag due to the old cfg_blob persistence bug need it
    # backfilled (see final_submissions/presres1_0.5M_backfilled_resolver.pt).
    if isinstance(state_dict, dict) and "global_proj.weight" in state_dict:
        _gdim = int(state_dict["global_proj.weight"].shape[1])
        if _gdim != 15:
            raise RuntimeError(
                f"Checkpoint has {_gdim}-global weights; the blessed config is 15-global. "
                f"Export it from the pre-cleanup git tag (pre-cleanup-2026-07) instead.")
    _stale = {k: bool(ckpt_cfg.get(k, False))
              for k in ("roi_enemy_deflate", "zero_roi_channels", "threat_eta_surface")
              if bool(ckpt_cfg.get(k, False))}
    if _stale or not bool(ckpt_cfg.get("pressure_precise_resolver", False)):
        raise RuntimeError(
            f"Checkpoint feature semantics (stale toggles: {_stale or 'resolver OFF/absent'}) "
            f"predate the 2026-07 cleanup — export it from the pre-cleanup git tag "
            f"(pre-cleanup-2026-07) instead.")
    # Pairwise input width must match what features.py emits (PAIRWISE_FEATURE_DIM), NOT the
    # checkpoint's stored width — narrower/older checkpoints are zero-padded by
    # EntityTransformer.load_state_dict. Stated explicitly (mirrors eval.load_checkpoint) so
    # export doesn't silently rely on the Config default tracking the feature width.
    if isinstance(state_dict, dict):
        cfg.model.pairwise_feature_dim = PAIRWISE_FEATURE_DIM if "pair_kv.weight" in state_dict else 0
    # Reinforce / sufficient-commit DISCIPLINE (persisted by ppo.state_dict) so the exported
    # mask matches training without relying on remembered CLI flags. Absent in old ckpts → 0/False.
    cfg.model._discipline_persisted = ("reinforce_gate_min_planets" in ckpt_cfg)
    cfg.model.allow_reinforce = bool(ckpt_cfg.get("allow_reinforce", False))
    cfg.model.reinforce_gate_min_planets = int(ckpt_cfg.get("reinforce_gate_min_planets", 0))
    cfg.model.reinforce_forward_only = bool(ckpt_cfg.get("reinforce_forward_only", False))
    cfg.model.reinforce_garrison_floor = float(ckpt_cfg.get("reinforce_garrison_floor", 0.0))
    cfg.model.sufficient_commit_factor = float(ckpt_cfg.get("sufficient_commit_factor", 0.0))

    return state_dict


def load_model(checkpoint_path: str, cfg: Config) -> EntityTransformer:
    checkpoint = _load_checkpoint(checkpoint_path)
    sd = _apply_checkpoint_model_config(checkpoint, cfg)
    model = EntityTransformer(cfg.model)
    if "model" in sd:
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if k not in PHASE4_COMPAT_MISSING_KEYS]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"checkpoint/model mismatch during export: missing={bad_missing}, unexpected={list(unexpected)}"
        )
    model.eval()
    return model


def export_agent(checkpoint_path: str, output_path: str, cfg: Config, fire_threshold: float = 0.5,
                 target_decode: bool = False,
                 reinforce_gate_min_planets: int = None,
                 reinforce_forward_only: bool = None,
                 reinforce_garrison_floor: float = None,
                 sufficient_commit_factor: float = None,
                 path_obstruction_mask: bool = False):
    model = load_model(checkpoint_path, cfg)
    # Discipline masks: explicit arg overrides; else use what the checkpoint was trained with
    # (load_model populated cfg.model via _apply_checkpoint_model_config). For OLD reinforce ckpts
    # that never persisted the discipline, the gate CAN'T be inferred → require it explicitly
    # (CONSISTENT with eval; previously export silently baked legacy 2 while eval used 0 — itself a
    # footgun where a ckpt evaluated one way and submitted another).
    _persisted = bool(getattr(cfg.model, "_discipline_persisted", False))
    if (bool(getattr(cfg.model, "allow_reinforce", False)) and not _persisted
            and reinforce_gate_min_planets is None):
        raise SystemExit(
            "Checkpoint has allow_reinforce=True but NO persisted reinforce discipline (pre-2026-06-15 "
            "ckpt). Pass --reinforce-gate-min-planets (and floor/forward) explicitly — it can't be "
            "inferred and guessing self-sabotages the exported agent.")
    if reinforce_gate_min_planets is None:
        reinforce_gate_min_planets = int(cfg.model.reinforce_gate_min_planets)
    if reinforce_forward_only is None:
        reinforce_forward_only = bool(cfg.model.reinforce_forward_only)
    if reinforce_garrison_floor is None:
        reinforce_garrison_floor = float(cfg.model.reinforce_garrison_floor)
    if sufficient_commit_factor is None:
        sufficient_commit_factor = float(cfg.model.sufficient_commit_factor)

    # Encode state_dict as base64. Drop inference-unused params the slim exported
    # _Model doesn't define (COMA q_* counterfactual head, VDN value_pp_*) — else
    # the exported loader sees them as unexpected keys.
    sd_export = {k: v for k, v in model.state_dict().items()
                 if not (k.startswith("q_") or k.startswith("value_pp_"))}
    buf = io.BytesIO()
    torch.save(sd_export, buf)
    params_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # Read feature / action-mask code (strip duplicate imports; agent template provides them)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    features_code = _read_module_body(os.path.join(src_dir, "features.py"))
    action_mask_code = _read_module_body(os.path.join(src_dir, "action_mask.py"))
    reinforce_cooldown_code = _read_module_body(os.path.join(src_dir, "reinforce_cooldown.py"))

    m = cfg.model
    # Auto-detect reinforcement from the checkpoint so the exported mask matches how
    # the policy was trained (baked into the checkpoint config by ppo.state_dict).
    _ckpt = _load_checkpoint(checkpoint_path)
    allow_reinforce = bool(_ckpt.get("config", {}).get("allow_reinforce", False)) if isinstance(_ckpt, dict) else False
    # Path-obstruction mask: bake if the run trained with it (persisted) OR the CLI flag is set
    # (inference-only enablement on a checkpoint that didn't train with it — safe either way).
    path_obstruction_mask = bool(path_obstruction_mask) or (
        bool(_ckpt.get("config", {}).get("path_obstruction_mask", False)) if isinstance(_ckpt, dict) else False)
    # Reverse-edge cooldown is persisted in the ckpt config (not a CLI flag here) → bake it so
    # the submission applies the SAME reinforce-ping-pong veto the policy trained+evaluated with.
    reverse_edge_cooldown = int(_ckpt.get("config", {}).get("reverse_edge_cooldown", 0)) if isinstance(_ckpt, dict) else 0
    if reverse_edge_cooldown > 0:
        print(f"  Reverse-edge cooldown: {reverse_edge_cooldown} (baked, matches training)")
    if allow_reinforce:
        print("  Reinforcement: ON (own planets are legal targets)")
        print(f"  Reinforce DISCIPLINE (must match training): gate_min_planets="
              f"{reinforce_gate_min_planets} forward_only={reinforce_forward_only} "
              f"garrison_floor={reinforce_garrison_floor}")
    agent_code = AGENT_TEMPLATE.format(
        num_angle_bins=NUM_ANGLE_BINS,
        num_ship_bins=m.num_ship_bins,
        angle_bin_width=ANGLE_BIN_WIDTH,
        entity_dim=m.entity_dim,
        num_heads=m.num_heads,
        num_layers=m.num_layers,
        mlp_expansion=m.mlp_expansion,
        max_owned_planets=m.max_owned_planets,
        planet_feature_dim=m.planet_feature_dim,
        fleet_feature_dim=m.fleet_feature_dim,
        global_feature_dim=m.global_feature_dim,
        pairwise_feature_dim=m.pairwise_feature_dim,
        min_ship_bin=m.min_ship_bin,
        fire_threshold=fire_threshold,
        ship_bin_mode=repr(m.ship_bin_mode),
        target_decode=target_decode,
        allow_reinforce=allow_reinforce,
        reinforce_gate_min_planets=reinforce_gate_min_planets,
        reinforce_forward_only=reinforce_forward_only,
        reinforce_garrison_floor=reinforce_garrison_floor,
        sufficient_commit_factor=sufficient_commit_factor,
        path_obstruction_mask=path_obstruction_mask,
        reverse_edge_cooldown=reverse_edge_cooldown,
        params_b64=params_b64,
        features_code=features_code,
        action_mask_code=action_mask_code,
        reinforce_cooldown_code=reinforce_cooldown_code,
    )

    with open(output_path, "w") as f:
        f.write(agent_code)

    print(f"Exported agent → {output_path}")
    print(f"  Param bytes (base64): {len(params_b64):,}")
    print(f"  Fire threshold: {fire_threshold}")
    print(f"  Target decode: {target_decode}")
    print(f"  Path-obstruction mask: {'ON' if path_obstruction_mask else 'off'}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")

    # --- Sanity eval: play the EXPORTED file vs zach to catch export bugs ---
    print("\nRunning sanity eval on exported agent (vs zach)...")
    _sanity_eval(output_path)


def _sanity_eval(output_path: str, games: int = 4,
                 opponent: str = "opponents/candidate_zach_public.py"):
    """Sanity-check the EXPORTED submission file: play a few games vs `opponent`
    (default: zach) and report wins. This runs the real submission code path, so
    it catches export bugs a checkpoint-only eval would miss. Falls back to a
    random opponent if the opponent file is unavailable. Game count is bounded
    (no subprocess timeout), so it can't hang the export."""
    import kaggle_environments as ke

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    opp = opponent
    if opp != "random":
        opp_path = opp if os.path.isabs(opp) else os.path.join(repo_root, opp)
        if os.path.exists(opp_path):
            opp = opp_path
        else:
            print(f"  [sanity] opponent {opponent} not found — using random")
            opp = "random"
    label = "random" if opp == "random" else os.path.basename(opp)[:-3]

    g = {}
    exec(open(output_path).read(), g)
    agent_fn = g["agent"]

    wins = 0
    for i in range(games):
        try:
            result = ke.make("orbit_wars").run([agent_fn, opp])
            r = result[-1][0]["reward"]
            if r is not None and r > 0:
                wins += 1
        except Exception as e:
            print(f"  [sanity] game {i} ERROR: {type(e).__name__}: {e} "
                  "— do NOT submit without investigating")
            return
    print(f"  [sanity] exported agent vs {label}: {wins}/{games} wins")
    if wins == 0:
        print("  [sanity] WARNING: 0 wins — do NOT submit without investigating")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="main_submitted.py")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fire-threshold", type=float, default=0.5)
    parser.add_argument("--target-decode", action="store_true",
                        help="Use target-planet aiming (actions_from_target_policy). "
                             "Required for checkpoints trained with --action-decode target.")
    # Reinforce-discipline masks. Defaults = the current locked Tier-1/Phase-2 training values
    # (p2rev4/p2rev5: forward-only DROPPED so the policy can pull ships back to defend, floor=0).
    # Inert for non-reinforce checkpoints; for reinforce checkpoints they MUST match
    # training or the policy self-sabotages (reinforces <gate planets, backward, drains source).
    # For OLD forward-only/floor=10 checkpoints (p2rev1-3) pass --reinforce-forward-only
    # --reinforce-garrison-floor 10 explicitly.
    # Defaults None → auto-load from the checkpoint config (persisted by ppo.state_dict); pass a
    # flag to override. Old ckpts without persisted discipline fall back to legacy values.
    parser.add_argument("--reinforce-gate-min-planets", type=int, default=None)
    parser.add_argument("--reinforce-forward-only", action=argparse.BooleanOptionalAction,
                        default=None, help="Own reinforce target must be closer to enemy than source.")
    parser.add_argument("--reinforce-garrison-floor", type=float, default=None)
    # Sufficient-commit mask (ATTACKS). MUST match training (p2rev6 = 1.0). 0 = off.
    parser.add_argument("--sufficient-commit-factor", type=float, default=None)
    parser.add_argument("--path-obstruction-mask", action="store_true",
                        help="INFERENCE bugfix: bake the path-obstruction target mask into the "
                             "submission (mask launches screened by an uncapturable planet so the "
                             "head retargets). Independent of training; safe on any checkpoint.")
    args = parser.parse_args()

    cfg = Config()
    cfg.seed = args.seed
    export_agent(args.checkpoint, args.output, cfg,
                 fire_threshold=args.fire_threshold,
                 target_decode=args.target_decode,
                 reinforce_gate_min_planets=args.reinforce_gate_min_planets,
                 reinforce_forward_only=args.reinforce_forward_only,
                 reinforce_garrison_floor=args.reinforce_garrison_floor,
                 sufficient_commit_factor=args.sufficient_commit_factor,
                 path_obstruction_mask=args.path_obstruction_mask)
