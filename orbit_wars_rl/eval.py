"""Evaluation: pit trained PyTorch policy against baselines."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
from statistics import mean, median

import torch
import numpy as np

from config import Config
from model import EntityTransformer, PHASE4_COMPAT_MISSING_KEYS
from features import extract_features, PAIRWISE_FEATURE_DIM
from action_mask import compute_action_masks, actions_from_target_policy
from torch_env import MAX_SHIP_SPEED as _DM_MAX_SPEED
from kaggle_environments.envs.orbit_wars.orbit_wars import CENTER, ROTATION_RADIUS_LIMIT


_BUNDLED_OPPONENT_ASSETS = {
    "candidate_ender.py": {
        2: ("ender_bundle/checkpoint_2p.pt",),
        4: ("ender_bundle/checkpoint_4p.pt",),
    },
    "candidate_yijie.py": {
        2: (
            "yijie_bundle/inference_2p/weights/weights_2p_u53000.npz",
            "yijie_bundle/inference_2p/weights/weights_2p_u55000.npz",
        ),
    },
    "candidate_sub_presres05.py": {
        2: ("../final_submissions/submission_presres05.tar.gz",),
    },
    "candidate_sub_stgpr1.py": {
        2: ("../final_submissions/submission_stgpr1.tar.gz",),
    },
}


def validate_opponent_assets(opponent: str, num_players: int) -> dict[str, str]:
    """Fail before eval if a path opponent or one of its declared assets is missing."""
    if opponent == "random":
        return {}
    path = Path(opponent).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"opponent agent does not exist: {path}")
    required = _BUNDLED_OPPONENT_ASSETS.get(path.name, {}).get(num_players, ())
    if path.name in _BUNDLED_OPPONENT_ASSETS and not required:
        raise RuntimeError(f"{path.name} has no declared {num_players}-player asset set")
    assets = [path.parent / rel for rel in required]
    missing = [p for p in assets if not p.is_file()]
    if missing:
        joined = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"opponent {path.name} is incomplete; required assets are missing:\n  {joined}\n"
            "Do not trust this eval. Re-sync bundled opponent assets before retrying."
        )
    manifest = {}
    for name, asset in zip(required, assets):
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()
        manifest[name] = digest
    if manifest:
        summary = ", ".join(f"{name}={digest[:12]}" for name, digest in manifest.items())
        print(f"Opponent asset manifest: {summary}", flush=True)
    return manifest


def _assert_game_completed(env, context: str) -> None:
    statuses = [str(state.status) for state in env.steps[-1]]
    if any(status != "DONE" for status in statuses):
        raise RuntimeError(f"{context} ended with agent statuses {statuses}; eval aborted")


def load_checkpoint(path: str, cfg: Config) -> tuple[dict, str]:
    """Load a checkpoint and patch cfg.model dims to match the saved weights.

    Returns (state_dict, action_decode).  Modifies cfg.model in-place so that
    EntityTransformer(cfg.model) builds the correct architecture for this
    checkpoint, regardless of what config.py currently says.
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    # Strip torch.compile's "_orig_mod." key prefix — checkpoints from --compile training runs
    # (pre-ppo.py fix) carry it and would otherwise fail the strict load below.
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}

    # --- head / bin dims from saved config or weight shapes ---
    if "num_ship_bins" in ckpt_cfg:
        cfg.model.num_ship_bins = int(ckpt_cfg["num_ship_bins"])
    elif "ship_head.weight" in sd:
        cfg.model.num_ship_bins = int(sd["ship_head.weight"].shape[0])

    if "angle_head.weight" in sd:
        n = int(sd["angle_head.weight"].shape[0])
        if n != cfg.model.num_angle_bins:
            cfg.model.num_angle_bins = n

    if "ship_bin_mode" in ckpt_cfg:
        cfg.model.ship_bin_mode = str(ckpt_cfg["ship_bin_mode"])
    # Binary commit gates are a MASK contract: evaluating a "minimal"-trained checkpoint under
    # "full" would delete 80% of the actions it learned to use. Absent => legacy "full".
    cfg.model.binary_commit_gates = str(ckpt_cfg.get("binary_commit_gates", "full"))

    # --- feature projection dims: always infer from weight shapes ---
    if "planet_proj.weight" in sd:
        cfg.model.planet_feature_dim = int(sd["planet_proj.weight"].shape[1])
    if "fleet_proj.weight" in sd:
        cfg.model.fleet_feature_dim = int(sd["fleet_proj.weight"].shape[1])
    if "global_proj.weight" in sd:
        cfg.model.global_feature_dim = int(sd["global_proj.weight"].shape[1])
    # --- transformer capacity (width + depth): infer from weights, like the dims above, so
    # capacity-experiment checkpoints (e.g. 128d/6L) load without CLI flags. planet_proj maps
    # feature_dim -> entity_dim, so shape[0] is the model width; blocks.<i>.* gives the depth. ---
    if "planet_proj.weight" in sd:
        cfg.model.entity_dim = int(sd["planet_proj.weight"].shape[0])
    _block_idx = [int(k.split(".")[1]) for k in sd
                  if k.startswith("blocks.") and k.split(".")[1].isdigit()]
    if _block_idx:
        cfg.model.num_layers = max(_block_idx) + 1
    # features.py always emits PAIRWISE_FEATURE_DIM channels, so the model's pairwise input
    # must be that wide regardless of the checkpoint. Older/narrower checkpoints are zero-padded
    # by EntityTransformer.load_state_dict (new channels contribute nothing → identical
    # behaviour). Pairwise is mandatory since the always-pairwise cleanup; a checkpoint without
    # pair_kv is pre-pairwise and unsupported (it fails at load_state_dict with missing keys).
    cfg.model.pairwise_feature_dim = PAIRWISE_FEATURE_DIM

    # Detect value head version from fc1 input width (old=D, new=2D).
    if "value_fc1.weight" in sd:
        cfg.model.value_head_in = int(sd["value_fc1.weight"].shape[1])

    action_decode = str(ckpt_cfg.get("action_decode", "angle"))
    # Reinforcement: eval must mask targets the SAME way the checkpoint was trained.
    cfg.model.allow_reinforce = bool(ckpt_cfg.get("allow_reinforce", False))
    # Feature semantics are hard-coded in
    # features.py (game-phase 15-global ON, precise pressure resolver ON, friendly deflation ON,
    # enemy-deflate/zero-roi/surface-threat REMOVED). Evaluating a checkpoint trained under
    # different semantics would silently feed it wrong features — refuse instead.
    _blessed = {"game_phase_features": True, "pressure_precise_resolver": True,
                "roi_enemy_deflate": False, "zero_roi_channels": False,
                "threat_eta_surface": False}
    _mismatch = {k: bool(ckpt_cfg.get(k, False)) for k, want in _blessed.items()
                 if bool(ckpt_cfg.get(k, False)) != want}
    if _mismatch:
        raise RuntimeError(
            f"Checkpoint feature semantics {_mismatch} do not match the blessed config "
            f"{_blessed}. This checkpoint predates the 2026-07 cleanup — eval it from the "
            f"pre-cleanup git tag (pre-cleanup-2026-07) instead.")
    # Reinforce / sufficient-commit DISCIPLINE: persisted at train time so eval/export mask the
    # SAME way (else the policy self-sabotages). Absent in old ckpts → defaults (0/False) → those
    # still require CLI flags, as before. evaluate_checkpoint uses these unless CLI overrides.
    cfg.model.reinforce_gate_min_planets = int(ckpt_cfg.get("reinforce_gate_min_planets", 0))
    cfg.model.reinforce_forward_only = bool(ckpt_cfg.get("reinforce_forward_only", False))
    cfg.model.reverse_edge_cooldown = int(ckpt_cfg.get("reverse_edge_cooldown", 0))
    cfg.model.reinforce_garrison_floor = float(ckpt_cfg.get("reinforce_garrison_floor", 0.0))
    cfg.model.sufficient_commit_factor = float(ckpt_cfg.get("sufficient_commit_factor", 0.0))
    cfg.model._discipline_persisted = ("reinforce_gate_min_planets" in ckpt_cfg)
    return sd, action_decode


def load_eval_model(path: str, cfg: Config) -> tuple[EntityTransformer, str]:
    """THE one checkpoint → eval-ready model path. evaluate_checkpoint and every probe use it.

    build_agent_fn reads the mask contract (allow_reinforce, binary_commit_gates, reinforce
    discipline, reverse-edge cooldown) OFF THE MODEL OBJECT. Probes that hand-copied this setup
    forgot an attribute twice: allow_reinforce (Key Lesson 14), then binary_commit_gates (2026-10:
    the minimal-gates champion was probed under the legacy "full" gates). Everything here comes
    from the checkpoint; a caller that wants an override sets the attribute on the returned model.
    Returns (model, action_decode); modifies cfg.model in place (see load_checkpoint).
    """
    state_dict, action_decode = load_checkpoint(path, cfg)
    m = cfg.model
    model = EntityTransformer(m).to(torch.device(cfg.device))
    model.allow_reinforce = bool(m.allow_reinforce)
    model.binary_commit_gates = str(m.binary_commit_gates)
    model.reinforce_gate_min_planets = int(m.reinforce_gate_min_planets)
    model.reinforce_forward_only = bool(m.reinforce_forward_only)
    model.reverse_edge_cooldown = int(m.reverse_edge_cooldown)
    model.reinforce_garrison_floor = float(m.reinforce_garrison_floor)
    model.sufficient_commit_factor = float(m.sufficient_commit_factor)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in missing if k not in PHASE4_COMPAT_MISSING_KEYS]
    # VDN per-planet value head (Stage 2) is never used at eval — ignore it if the
    # checkpoint carries it but this (eval-time) model doesn't.
    bad_unexpected = [k for k in unexpected if not k.startswith("value_pp_")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch: missing={bad_missing}, unexpected={bad_unexpected}")
    model.eval()
    return model, action_decode


def build_agent_fn(model: EntityTransformer, device: torch.device,
                   fire_threshold: float = 0.5, sample: bool = False,
                   ship_bin_mode: str = "absolute",
                   target_decode: bool = False,
                   num_players: int = 2,
                   allow_reinforce: bool = False):
    """Return a kaggle_environments-compatible agent function wrapping the model.

    sample=True uses Bernoulli/Categorical sampling instead of threshold/argmax —
    helps when the training-time distribution is multi-modal but the mode is
    degenerate (e.g. 1-ship-fleet trap).
    """
    model.eval()
    # Reverse-edge cooldown state: a per-GAME edge-history dict (canonical rule in
    # reinforce_cooldown.py), kept in this closure across steps. Reset when the step counter
    # resets (new game / new seat run), so a prior game's edges never mis-block the next.
    _cd_K = int(getattr(model, "reverse_edge_cooldown", 0))
    _cd = {"last": {}, "prev_step": -1}
    # Projected-timeline channels: feed them iff the checkpoint was trained with them
    # (planet_proj input width 116 vs the pre-timeline 20).
    _timeline = int(model.planet_proj.in_features) > 20
    # Same for the projected economy series (global_proj width 63 vs the earlier 15).
    _global_econ = int(model.global_proj.in_features) > 15
    # Binary commit gates — a MASK contract that must match training. Read off the model (set by
    # evaluate_checkpoint from the checkpoint), like allow_reinforce above.
    _binary_gates = str(getattr(model, "binary_commit_gates", "full"))

    def agent_fn(obs):
        # obs may be a dict or an Observation namedtuple depending on caller
        if not isinstance(obs, dict):
            obs = {
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
            }

        player = obs["player"]
        # Reverse-edge cooldown: detect a new game (step counter reset) and clear the edge history.
        if _cd_K > 0:
            step_now = int(obs.get("step", 0))
            if step_now <= _cd["prev_step"]:
                _cd["last"].clear()
            _cd["prev_step"] = step_now
        features = extract_features(
            obs, player, num_players=num_players, timeline=_timeline,
            global_econ=_global_econ,
        )
        masks = compute_action_masks(obs, player)

        with torch.no_grad():
            outputs = model(
                features["planet_features"].unsqueeze(0).to(device),
                features["fleet_features"].unsqueeze(0).to(device),
                features["global_features"].unsqueeze(0).to(device),
                features["planet_mask"].unsqueeze(0).to(device),
                features["fleet_mask"].unsqueeze(0).to(device),
                fire_mask=masks["fire_mask"].to(device),
                slot_valid=masks["slot_valid"].to(device),
                owned_indices=masks["owned_indices"].to(device),
                owned_count=masks["owned_count"],
                pairwise_features=features["pairwise_features"].unsqueeze(0).to(device)
                    if "pairwise_features" in features else None,
            )

        if target_decode:
            return actions_from_target_policy(
                outputs["fire_logits"].cpu(),
                outputs["target_logits"].cpu(),
                (outputs["ship_logits"].cpu()
                 if outputs["ship_logits"] is not None else None),
                {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()},
                obs, player,
                fire_threshold=fire_threshold,
                sample=sample,
                ship_bin_mode=ship_bin_mode,
                binary_commit_gates=_binary_gates,
                pairwise_features=(features["pairwise_features"].cpu().numpy()
                                   if "pairwise_features" in features else None),
                allow_reinforce=getattr(model, "allow_reinforce", allow_reinforce),
                reinforce_gate_min_planets=getattr(model, "reinforce_gate_min_planets", 0),
                reinforce_forward_only=getattr(model, "reinforce_forward_only", False),
                reinforce_garrison_floor=getattr(model, "reinforce_garrison_floor", 0.0),
                sufficient_commit_factor=getattr(model, "sufficient_commit_factor", 0.0),
                reverse_edge_cooldown=_cd_K,
                cooldown_last=_cd["last"] if _cd_K > 0 else None,
                cooldown_step=int(obs.get("step", 0)),
            )

        raise NotImplementedError(
            "angle-decode path removed (angle head deleted); target-based checkpoints "
            "use target-decode. Pass target_decode=True (--target-decode)."
        )

    def _agent_fn_loud(obs):
        # kaggle_environments runs the agent inside its own try/except: a decode exception is
        # swallowed (the agent simply "makes no move"), which surfaces as a silent 0%/blank panel
        # — indistinguishable from a real loss. That cost hours once (an undefined `fleets` in the
        # sufficient-commit veto). Fail LOUD instead: dump the full traceback and hard-exit so a
        # CODE BUG can never be mistaken for a 0% score. os._exit bypasses kaggle's except.
        try:
            return agent_fn(obs)
        except Exception:
            import sys
            import traceback
            sys.stderr.write(
                "\n" + "!" * 78
                + "\n!!! AGENT DECODE CRASHED — this is a CODE BUG, not a 0% result.\n"
                + "!!! Aborting eval LOUDLY (kaggle would otherwise swallow it as 'no move').\n"
                + "!" * 78 + "\n" + traceback.format_exc() + "!" * 78 + "\n"
            )
            sys.stderr.flush()
            os._exit(1)

    return _agent_fn_loud


_CONV_MILESTONES = (16, 32, 50, 100)
_ECONOMY_MILESTONES = (32, 50, 100)
# The opening window isolates the phase that decides expansion: opening cap/atk-launch and
# caps_early/atk_early are windowed to <50 (a whole-game fraction is inflated by benign late
# surplus re-fire in long won games). mid = [50, 100). (phase2 / metrics.md)
_LAUNCH_WINDOW = 50
_MID_WINDOW = 100      # mid-game cap/atk window = [_LAUNCH_WINDOW, _MID_WINDOW) = steps 50-100


# Orbit rate of the game currently being analysed, set once per game by game_conversion (it is
# constant for a game). Read by the lead-aware target resolvers below so they don't need it threaded
# through every helper signature. Default 0.0 = treat planets as static (safe: degrades to a
# distance-aware ray test, never worse than the old angle-only heuristic).
_CONV_ANGVEL = 0.0


def _planet_pos_at(p, t):
    """Planet `p`'s (x, y) `t` steps in the future along its orbit. Static at/beyond the rotation
    radius limit (the engine leaves those fixed). Orbit radius is rotation-invariant, so it can be
    read from the current position; phase advances by _CONV_ANGVEL*t (engine: angle = init + w*step)."""
    dx, dy = p[2] - CENTER, p[3] - CENTER
    orb = math.hypot(dx, dy)
    if orb + p[4] >= ROTATION_RADIUS_LIMIT:
        return p[2], p[3]
    ph = math.atan2(dy, dx) + _CONV_ANGVEL * t
    return CENTER + orb * math.cos(ph), CENTER + orb * math.sin(ph)


def _lead_collision_target(planets, x, y, angle, ships, skip_pid=None):
    """Planet a fleet at (x, y) heading `angle` (speed from `ships`) will physically collide with,
    accounting for the target's ORBITAL motion over the flight — a lead/intercept projection that
    mirrors the aimer the agent fires with (`_target_intercept_angle`). Picks the min-ETA hit among
    planets the straight-line heading reaches within radius; None if it hits nothing (flies to the
    void). Validated at 98.4% vs the true swept-collision on replay (the old angle-only heuristic was
    65.6%); the residual is grazing geometry. `skip_pid` excludes the source planet for a launch."""
    c, sn = math.cos(angle), math.sin(angle)
    speed = max(_ship_speed_py(ships), 1e-6)
    best, best_eta = None, None
    for p in planets:
        if skip_pid is not None and p[0] == skip_pid:
            continue
        pr = p[4]
        eta = max(0.0, (math.hypot(p[2] - x, p[3] - y) - pr) / speed)
        for _ in range(4):                          # converge ETA against the moving target
            lx, ly = _planet_pos_at(p, eta)
            eta = max(0.0, (math.hypot(lx - x, ly - y) - pr) / speed)
        lx, ly = _planet_pos_at(p, eta)
        vx, vy = lx - x, ly - y
        along = vx * c + vy * sn
        if along <= 0:                              # planet is behind the heading
            continue
        perp = abs(vx * sn - vy * c)
        if perp < pr + 0.5 and (best_eta is None or eta < best_eta):
            best_eta, best = eta, p
    return best


def _resolve_launch_target(planets, src, angle, ships):
    """Planet a launch from `src` at `angle` actually hits. `ships` (the launched ship count,
    needed for fleet speed) makes this the lead-aware collision resolver — the fleet flies
    straight and captures whatever it physically collides with, so distance / planet radius / the
    target's orbital motion all matter (none of which the old angle-only match saw)."""
    # Fleet spawns at the source surface + a small launch offset along the heading (engine:
    # start = planet + cos/sin(angle)*(radius + 0.1)), then flies straight.
    sx = src[2] + math.cos(angle) * (src[4] + 0.1)
    sy = src[3] + math.sin(angle) * (src[4] + 0.1)
    return _lead_collision_target(planets, sx, sy, angle, ships, skip_pid=src[0])


def _relative_economy_snapshot(obs, seat):
    """Production/material advantage for one side in a two-player observation.

    Both values are paired differences (ours - opponent), so board-scale differences do not
    create separate numerator/denominator aggregation artifacts. Material includes ships on
    planets and in flight.
    """
    opponent = 1 - seat
    planets = obs.get("planets") or []
    fleets = obs.get("fleets") or []
    our_prod = sum(float(p[6]) for p in planets if int(p[1]) == seat)
    opp_prod = sum(float(p[6]) for p in planets if int(p[1]) == opponent)
    our_material = sum(float(p[5]) for p in planets if int(p[1]) == seat)
    opp_material = sum(float(p[5]) for p in planets if int(p[1]) == opponent)
    our_material += sum(float(f[6]) for f in fleets if int(f[1]) == seat)
    opp_material += sum(float(f[6]) for f in fleets if int(f[1]) == opponent)
    return our_prod - opp_prod, our_material - opp_material


def _relative_economy_milestones(steps, seat):
    """Paired economy snapshots, carrying terminal state forward after an early finish."""
    out = {}
    if not steps or len(steps[0]) != 2:
        return {ms: None for ms in _ECONOMY_MILESTONES}
    for ms in _ECONOMY_MILESTONES:
        state = steps[min(ms, len(steps) - 1)][seat]
        out[ms] = _relative_economy_snapshot(state.observation, seat)
    return out


def _resolved_inbound_by_side(planets, fleets, tgt, seat):
    """Friendly/enemy fleet mass whose next resolved planet is `tgt`."""
    friendly = enemy = 0.0
    for f in fleets or []:
        owner = int(f[1])
        if owner < 0:
            continue
        hit = _lead_collision_target(planets, f[2], f[3], f[4], f[6])
        if hit is None or int(hit[0]) != int(tgt[0]):
            continue
        if owner == seat:
            friendly += float(f[6])
        else:
            enemy += float(f[6])
    return friendly, enemy


def _already_covered_neutral(planets, fleets, tgt, seat):
    """Whether known friendly inbound already flips an uncontested neutral.

    This is deliberately a coordination probe, not a waste label: an opponent can launch a
    simultaneous counter-wave that is absent from the decision-time observation.
    """
    if int(tgt[1]) != -1:
        return False
    friendly, enemy = _resolved_inbound_by_side(planets, fleets, tgt, seat)
    return enemy == 0.0 and friendly >= float(tgt[5]) + 1.0


def _ship_speed_py(ships):
    """Scalar mirror of torch_env._ship_speed (kaggle speed formula)."""
    s = max(float(ships), 1.0)
    base = (math.log(s) / math.log(1000.0)) ** 1.5
    return min(1.0 + (_DM_MAX_SPEED - 1.0) * base, _DM_MAX_SPEED)


def game_conversion(steps, seat):
    """Whole-game CONVERSION for `seat` from kaggle env.steps.

    capture        = a planet whose owner transitions TO `seat`.
    attack-launch  = a legal fire whose aimed target is NOT owned by `seat`.
                     Reinforce launches (target owned by `seat`) CANNOT capture,
                     so they are excluded from the cap/launch denominator and
                     counted separately (reinforce_launches). Launches whose
                     target can't be resolved by angle are skipped (matches the
                     replay analyzer), so eval numbers compare to Isaiah/Jake.
    Also records owned-planet count at step milestones (expansion/retention).
    Returns per-game counts; `add_conversion` aggregates across games.
    """
    # Orbit rate for the lead-aware target resolvers (constant per game) — set once here so the
    # per-launch / per-fleet resolution below sees the planets' motion over each flight.
    global _CONV_ANGVEL
    _CONV_ANGVEL = 0.0
    for _s in steps:
        if seat < len(_s):
            _av = _s[seat].observation.get("angular_velocity")
            if _av is not None:
                _CONV_ANGVEL = float(_av)
                break
    caps = atk = reinf = atk_ships = 0
    atk_early = caps_early = 0                                       # opening window (t < _LAUNCH_WINDOW)
    atk_mid = caps_mid = 0                                           # mid-game window [50, 100)
    reinf_early = 0                                                  # reinforce launches in the opening window
    # Retention: of the planets we CAPTURE, how many do we then lose, and how long did we hold
    # them? cap_step[pid] = step we (most recently) took pid; on a later loss we close the episode.
    # lost_caps/captures is the recapture/turnover rate — immune to the end->0 churn degeneracy.
    # Home/initial planets are excluded by construction (never entered cap_step).
    cap_step: dict = {}
    lost_caps = 0
    hold_durations: list = []   # steps held before losing (lost episodes only; held-to-end censored)
    # launch_rate / fire_frac (vs Isaiah 0.036 / 0.17): ALL legal launches (attack+reinforce),
    # counted BEFORE target resolution (a fire is a fire). launch_rate = launches /
    # owned-planet-steps; fire_frac = on firing steps, mean fraction of owned planets that fired.
    launch_states = launch_count = fire_steps = 0
    fire_frac_sum = 0.0
    # ship0 by phase × outcome (the panic hypothesis): is the 1-ship probe an END-GAME /
    # LOSING artifact rather than a genuine habit? Split legal launches into early<50 /
    # mid50-100 / late>=100, count sent==1 (the eval analog of training ship_bin0); the
    # panel routes won/lost. mean ships/launch per phase complements it (undercommit read).
    launches_ph = [0, 0, 0]
    ship1_ph = [0, 0, 0]
    ship_ph_sum = [0, 0, 0]
    # Coordination mechanism probe for the contested window. An "already-covered" neutral has no
    # visible enemy inbound and already has enough friendly fleet mass in flight to flip its
    # current garrison. This does NOT call the extra launch waste; simultaneous counter-launches
    # are unobservable at decision time.
    neutral_launches_u100 = neutral_ships_u100 = 0
    already_covered_neutral_launches_u100 = already_covered_neutral_ships_u100 = 0
    economy_at = _relative_economy_milestones(steps, seat)
    planets_at = {ms: None for ms in _CONV_MILESTONES}
    prev = {}
    last = None
    for t in range(1, len(steps)):
        if seat >= len(steps[t]) or seat >= len(steps[t - 1]):
            continue
        obs0 = steps[t - 1][seat].observation
        p0 = obs0.get("planets")
        fleets0 = obs0.get("fleets") or []
        p1 = steps[t][seat].observation.get("planets")
        acts = steps[t][seat].action or []
        if p1:
            owned_now = 0
            for p in p1:
                pid, own = p[0], int(p[1])
                if own == seat:
                    owned_now += 1
                was = prev.get(pid)
                if was is not None and was != seat and own == seat:
                    caps += 1
                    if t < _LAUNCH_WINDOW:
                        caps_early += 1                # opening captures (for opening cap/atk-launch)
                    elif t < _MID_WINDOW:
                        caps_mid += 1                  # mid-game (50-100) captures
                    cap_step[pid] = t                  # open a hold episode
                elif was == seat and own != seat and pid in cap_step:
                    hold_durations.append(t - cap_step[pid])   # lost what we took
                    lost_caps += 1
                    del cap_step[pid]
                prev[pid] = own
            last = p1
            if t in planets_at:
                planets_at[t] = owned_now
        if not p0:
            continue
        byid = {p[0]: p for p in p0}
        owned_dec = sum(1 for p in p0 if int(p[1]) == seat)  # empire size at decision
        launch_states += owned_dec
        fired_this_step = 0
        for mv in acts:
            if not mv or len(mv) < 3:
                continue
            src = byid.get(int(mv[0]))
            if src is None:
                continue
            sent, ssh = int(mv[2]), float(src[5])
            if not (ssh > 0 and sent <= ssh):       # legal launches only
                continue
            fired_this_step += 1                    # counted before target resolution
            _ph = 0 if t < _LAUNCH_WINDOW else (1 if t < _MID_WINDOW else 2)
            launches_ph[_ph] += 1
            ship_ph_sum[_ph] += sent
            if sent == 1:
                ship1_ph[_ph] += 1
            tgt = _resolve_launch_target(p0, src, float(mv[1]), sent)
            if tgt is None:
                continue                            # flew to the void / unresolvable → skip
            if int(tgt[1]) == seat:
                reinf += 1                          # reinforce: cannot capture
                if t < _LAUNCH_WINDOW:
                    reinf_early += 1                # reinforce in the opening window
            else:
                atk += 1
                atk_ships += sent
                if t < _LAUNCH_WINDOW:
                    atk_early += 1
                elif t < _MID_WINDOW:
                    atk_mid += 1                       # mid-game (50-100) attack launches
                if t < _MID_WINDOW and int(tgt[1]) == -1:
                    neutral_launches_u100 += 1
                    neutral_ships_u100 += sent
                    if _already_covered_neutral(p0, fleets0, tgt, seat):
                        already_covered_neutral_launches_u100 += 1
                        already_covered_neutral_ships_u100 += sent
        if fired_this_step > 0 and owned_dec > 0:
            fire_steps += 1
            fire_frac_sum += fired_this_step / owned_dec
        launch_count += fired_this_step
    end_planets = sum(1 for p in (last or []) if int(p[1]) == seat)
    out = {"captures": caps, "attack_launches": atk, "reinforce_launches": reinf,
           "attack_ships": atk_ships, "end_planets": end_planets,
           "atk_early": atk_early, "caps_early": caps_early, "atk_mid": atk_mid, "caps_mid": caps_mid,
           "reinf_early": reinf_early,
           "lost_caps": lost_caps, "hold_durations": hold_durations,
           "glen": len(steps),
           "launch_states": launch_states, "launch_count": launch_count,
           "fire_steps": fire_steps, "fire_frac_sum": fire_frac_sum,
           "launches_ph": launches_ph, "ship1_ph": ship1_ph, "ship_ph_sum": ship_ph_sum,
           "neutral_launches_u100": neutral_launches_u100,
           "neutral_ships_u100": neutral_ships_u100,
           "already_covered_neutral_launches_u100": already_covered_neutral_launches_u100,
           "already_covered_neutral_ships_u100": already_covered_neutral_ships_u100}
    for ms in _CONV_MILESTONES:
        out[f"p{ms}"] = planets_at[ms]
    for ms in _ECONOMY_MILESTONES:
        snap = economy_at[ms]
        out[f"prod_delta_{ms}"] = snap[0] if snap is not None else None
        out[f"material_delta_{ms}"] = snap[1] if snap is not None else None
    return out


def new_conversion_acc():
    acc = {"captures": 0, "attack_launches": 0, "reinforce_launches": 0,
           "attack_ships": 0, "end_planets": 0, "games": 0,
           "atk_early": 0, "caps_early": 0, "atk_mid": 0, "caps_mid": 0, "reinf_early": 0,
           "lost_caps": 0, "hold_durations": [],
           # elimination-depth: our final own-material in LOST games (0 = total wipeout). A GRADED
           # loss signal (out-massed% saturates vs strong play; this doesn't). See docs/metrics.md.
           "lost_material": [],
           "launch_states": 0, "launch_count": 0, "fire_steps": 0, "fire_frac_sum": 0.0,
           # fire-rate split by game outcome — fire_frac inflates on losses (cornered to few
           # planets → firing from "many of few"), so the won-game value is the honest spray read.
           "launch_states_won": 0, "launch_count_won": 0, "fire_steps_won": 0, "fire_frac_sum_won": 0.0,
           "launch_states_lost": 0, "launch_count_lost": 0, "fire_steps_lost": 0, "fire_frac_sum_lost": 0.0,
           # ship0 (1-ship probe) by phase × outcome — the panic hypothesis
           "launches_ph": [0, 0, 0], "ship1_ph": [0, 0, 0], "ship_ph_sum": [0, 0, 0],
           "launches_ph_won": [0, 0, 0], "ship1_ph_won": [0, 0, 0], "ship_ph_sum_won": [0, 0, 0],
           "launches_ph_lost": [0, 0, 0], "ship1_ph_lost": [0, 0, 0], "ship_ph_sum_lost": [0, 0, 0],
           # Mechanism-only coordination probe. Outcome splits prevent a changing win/loss mix
           # from masquerading as a policy change.
           "neutral_launches_u100": 0, "neutral_ships_u100": 0,
           "already_covered_neutral_launches_u100": 0, "already_covered_neutral_ships_u100": 0,
           "neutral_launches_u100_won": 0, "neutral_ships_u100_won": 0,
           "already_covered_neutral_launches_u100_won": 0, "already_covered_neutral_ships_u100_won": 0,
           "neutral_launches_u100_lost": 0, "neutral_ships_u100_lost": 0,
           "already_covered_neutral_launches_u100_lost": 0, "already_covered_neutral_ships_u100_lost": 0,
           # retention split by outcome — peel-rate → 1 on elimination (lose every planet because you
           # LOST the game), so the won-game value is the honest "can we hold mid-game?" read.
           "captures_won": 0, "captures_lost": 0, "lost_caps_won": 0, "lost_caps_lost": 0,
           "hold_durations_won": [], "hold_durations_lost": [],
           # per-game LENGTHS split by outcome → median game length for wins (stall-and-win vs
           # quick wrap-up) and losses.
           "game_len_won": [], "game_len_lost": [],
           # conversion + expansion split by outcome (aggregates are dominated by the majority class
           # — mostly losses vs a strong opp — so the won-game ramp is the real read)
           "attack_launches_won": 0, "attack_launches_lost": 0,
           "atk_early_won": 0, "atk_early_lost": 0, "caps_early_won": 0, "caps_early_lost": 0,
           "atk_mid_won": 0, "atk_mid_lost": 0, "caps_mid_won": 0, "caps_mid_lost": 0,
           "games_won": 0, "games_lost": 0}
    for ms in _CONV_MILESTONES:
        acc[f"p{ms}_sum"] = 0
        acc[f"p{ms}_n"] = 0
        acc[f"p{ms}_sum_won"] = 0; acc[f"p{ms}_n_won"] = 0
        acc[f"p{ms}_sum_lost"] = 0; acc[f"p{ms}_n_lost"] = 0
    for ms in _ECONOMY_MILESTONES:
        for metric in ("prod_delta", "material_delta"):
            acc[f"{metric}_{ms}"] = []
            acc[f"{metric}_{ms}_won"] = []
            acc[f"{metric}_{ms}_lost"] = []
    return acc


def add_conversion(acc, conv, won=None, material=None):
    if won is False and material is not None:
        acc["lost_material"].append(material)  # elimination-depth (graded loss signal)
    for k in ("captures", "attack_launches", "reinforce_launches", "attack_ships",
              "end_planets", "atk_early", "caps_early", "atk_mid", "caps_mid", "reinf_early",
              "lost_caps", "launch_states", "launch_count", "fire_steps", "fire_frac_sum",
              "neutral_launches_u100", "neutral_ships_u100",
              "already_covered_neutral_launches_u100", "already_covered_neutral_ships_u100"):
        acc[k] += conv.get(k, 0)
    # route the fire-rate + conversion fields into won/lost buckets so spray + the opening ramp can
    # be read free of the losing-position confound (won=None from non-eval callers → overall only)
    if won is not None:
        suf = "won" if won else "lost"
        for k in ("launch_states", "launch_count", "fire_steps", "fire_frac_sum",
                  "captures", "lost_caps", "attack_launches", "atk_early", "caps_early",
                  "atk_mid", "caps_mid", "neutral_launches_u100", "neutral_ships_u100",
                  "already_covered_neutral_launches_u100", "already_covered_neutral_ships_u100"):
            acc[f"{k}_{suf}"] += conv.get(k, 0)
        acc[f"hold_durations_{suf}"].extend(conv["hold_durations"])
        acc[f"game_len_{suf}"].append(conv["glen"])
        acc["games_won" if won else "games_lost"] += 1
    acc["hold_durations"].extend(conv["hold_durations"])
    for i in range(3):
        acc["launches_ph"][i] += conv["launches_ph"][i]
        acc["ship1_ph"][i] += conv["ship1_ph"][i]
        acc["ship_ph_sum"][i] += conv["ship_ph_sum"][i]
        if won is not None:
            suf = "won" if won else "lost"
            acc[f"launches_ph_{suf}"][i] += conv["launches_ph"][i]
            acc[f"ship1_ph_{suf}"][i] += conv["ship1_ph"][i]
            acc[f"ship_ph_sum_{suf}"][i] += conv["ship_ph_sum"][i]
    acc["games"] += 1
    for ms in _CONV_MILESTONES:
        v = conv[f"p{ms}"]
        if v is not None:
            acc[f"p{ms}_sum"] += v
            acc[f"p{ms}_n"] += 1
            if won is not None:
                suf = "won" if won else "lost"
                acc[f"p{ms}_sum_{suf}"] += v
                acc[f"p{ms}_n_{suf}"] += 1
    for ms in _ECONOMY_MILESTONES:
        for metric in ("prod_delta", "material_delta"):
            v = conv.get(f"{metric}_{ms}")
            if v is not None:
                acc[f"{metric}_{ms}"].append(v)
                if won is not None:
                    acc[f"{metric}_{ms}_{suf}"].append(v)


def _fmt_conversion(acc):
    """Two-line conversion summary. cap/launch counts ATTACK launches only
    (reinforce can't capture). Reference = Isaiah (#1 player)."""
    n = max(acc["games"], 1)
    c, al, rl = acc["captures"], acc["attack_launches"], acc["reinforce_launches"]
    pl = lambda ms: (f"{acc[f'p{ms}_sum']/acc[f'p{ms}_n']:.0f}" if acc[f"p{ms}_n"] else "—")
    plw = lambda ms: (f"{acc[f'p{ms}_sum_won']/acc[f'p{ms}_n_won']:.0f}" if acc[f"p{ms}_n_won"] else "—")
    pll = lambda ms: (f"{acc[f'p{ms}_sum_lost']/acc[f'p{ms}_n_lost']:.0f}" if acc[f"p{ms}_n_lost"] else "—")
    # Retention (denominator-free): of planets we CAPTURE, the fraction we then lose,
    # and the median steps we held a lost planet (short = peeled fast). lost-cap rate→1 = pure
    # capture-and-lose turnover (the "can't hold the midgame lead" disease); hold→game length = sticky.
    hd = acc["hold_durations"]
    lost_rate = acc["lost_caps"] / max(c, 1)
    med_hold = (sorted(hd)[len(hd) // 2] if hd else 0)
    # retention split by outcome (lost-cap → 1 on elimination = you lost the GAME, not a hold-skill
    # signal). Won-game lost-cap = do we drop planets even when winning? = the real retention read.
    _med = lambda h: (sorted(h)[len(h) // 2] if h else 0)
    lostr_w = acc["lost_caps_won"] / max(acc["captures_won"], 1)
    lostr_l = acc["lost_caps_lost"] / max(acc["captures_lost"], 1)
    medh_w, medh_l = _med(acc["hold_durations_won"]), _med(acc["hold_durations_lost"])
    # median game LENGTH split by outcome: long wins = stall-and-win (attrition), short = decisive snowball.
    medlen_w, medlen_l = _med(acc["game_len_won"]), _med(acc["game_len_lost"])
    # opening (t<50) cap/atk-launch: the whole-game value is PHASE-confounded (easy late-game
    # cleanup captures mask a catastrophic opening). The opening decides expansion → read this.
    # caps_early/atk_early; mild window-edge bias (a t~48 launch capturing at t~55 deflates it).
    cap_open = acc["caps_early"] / max(acc["atk_early"], 1)
    cap_open_w = acc["caps_early_won"] / max(acc["atk_early_won"], 1)
    cap_open_l = acc["caps_early_lost"] / max(acc["atk_early_lost"], 1)
    capw = acc["captures_won"] / max(acc["attack_launches_won"], 1)
    capl = acc["captures_lost"] / max(acc["attack_launches_lost"], 1)
    # mid-game (50-100) cap/atk — the collapse window; the missing read for "why planets go 6→4"
    cap_mid = acc["caps_mid"] / max(acc["atk_mid"], 1)
    cap_mid_w = acc["caps_mid_won"] / max(acc["atk_mid_won"], 1)
    cap_mid_l = acc["caps_mid_lost"] / max(acc["atk_mid_lost"], 1)
    # spray read: launch_rate = launches / owned-planet-steps; fire_frac = on firing steps,
    # mean fraction of owned planets that fired. Length-confound-free (rate, not total) BUT
    # WIN/LOSS-confounded: fire_frac inflates on losses (cornered to few planets). Read the
    # WON-game value as the honest "are we sprayers?" signal (snowball losers 0.31 vs winners 0.19).
    lr = acc["launch_count"] / max(acc["launch_states"], 1)
    ff = acc["fire_frac_sum"] / max(acc["fire_steps"], 1)
    lr_w = acc["launch_count_won"] / max(acc["launch_states_won"], 1)
    ff_w = acc["fire_frac_sum_won"] / max(acc["fire_steps_won"], 1)
    lr_l = acc["launch_count_lost"] / max(acc["launch_states_lost"], 1)
    ff_l = acc["fire_frac_sum_lost"] / max(acc["fire_steps_lost"], 1)
    gw, gl = acc["games_won"], acc["games_lost"]
    wl = (f"     WON({gw}g) lr {lr_w:.3f} ff {ff_w:.2f}  |  LOST({gl}g) lr {lr_l:.3f} ff {ff_l:.2f}"
          f"   (read WON; ff inflates on losses)\n") if (gw + gl) > 0 else ""
    rwl = (f"     WON({gw}g) peel-rate {lostr_w:.2f} hold {medh_w}st  |  LOST({gl}g) peel-rate {lostr_l:.2f} hold {medh_l}st"
           f"   (read WON; peel-rate→1 on elimination)\n") if (gw + gl) > 0 else ""
    pwl = (f"     WON({gw}g) {plw(16)}/{plw(32)}/{plw(50)}/{plw(100)}  cap/atk open<50 {cap_open_w:.2f} mid50-100 {cap_mid_w:.2f} (whole {capw:.2f})"
           f"\n     LOST({gl}g) {pll(16)}/{pll(32)}/{pll(50)}/{pll(100)}  cap/atk open<50 {cap_open_l:.2f} mid50-100 {cap_mid_l:.2f} (whole {capl:.2f})\n"
           if (gw + gl) > 0 else "")
    # ship0 (1-ship probe) by phase × outcome — tests the panic hypothesis: a 1-ship launch
    # concentrated in late/lost games is an end-game/losing artifact, not a policy habit.
    # mean = mean ships per launch in that phase (the undercommit complement).
    def _s0(suf):
        lp, s1, ss = acc[f"launches_ph{suf}"], acc[f"ship1_ph{suf}"], acc[f"ship_ph_sum{suf}"]
        return "  ".join(
            (f"{nm} {100*s1[i]/lp[i]:.0f}%(mean{ss[i]/lp[i]:.0f},n{lp[i]})" if lp[i] else f"{nm} —(n0)")
            for i, nm in enumerate(("early<50", "mid50-100", "late>=100")))
    s0wl = (f"\n     WON  {_s0('_won')}\n     LOST {_s0('_lost')}" if (gw + gl) > 0 else "")
    # Per-game paired advantages avoid the two failure modes that motivated this metric: separate
    # aggregate numerators can hide who was ahead in the same game, and late snapshots can select
    # only long survivors. Terminal states are carried forward in game_conversion.
    def _econ(metric, suffix):
        cells = []
        for ms in _ECONOMY_MILESTONES:
            values = acc[f"{metric}_{ms}{suffix}"]
            if not values:
                cells.append("—")
                continue
            lead = sum(v > 0 for v in values) / len(values)
            cells.append(f"{median(values):+.0f}({lead:.0%})")
        return "/".join(cells)

    econ = ""
    if gw + gl > 0:
        econ = (
            "  relative-economy median Δ ours−opp (% games ahead; terminal carried)\n"
            f"     WON  prod@32/50/100 {_econ('prod_delta', '_won')}  ·  "
            f"material {_econ('material_delta', '_won')}\n"
            f"     LOST prod@32/50/100 {_econ('prod_delta', '_lost')}  ·  "
            f"material {_econ('material_delta', '_lost')}\n"
        )

    def _already_covered(suffix):
        launches = acc[f"neutral_launches_u100{suffix}"]
        ships = acc[f"neutral_ships_u100{suffix}"]
        covered_launches = acc[f"already_covered_neutral_launches_u100{suffix}"]
        covered_ships = acc[f"already_covered_neutral_ships_u100{suffix}"]
        return (f"{covered_launches / max(launches, 1):.1%} launches "
                f"({covered_launches}/{launches}), {covered_ships / max(ships, 1):.1%} ships")

    already_covered = (
        "  ★ already-covered neutral follow-up <100 "
        f"{_already_covered('')}\n"
        f"     WON {_already_covered('_won')}  |  LOST {_already_covered('_lost')} "
        "(direction ambiguous; coordination tracker, not a quality score)\n"
    )
    # Trusted core only; detailed proxy diagnostics live in git history.
    # Cut: the force-concentration-wall microscopy (decisive-mass, hold-floor, triage, om32,
    # failed-attack), the reinf-* deep-dives, hoard-vs-Isaiah, near-vs-far, launch-waste
    # (self-flagged non-discriminating) — all elaborations of proxies that saturate vs strong
    # play / proved gameable (decmass). out-massed DEMOTED to one annotated floor number.
    # Elimination-depth added (graded loss signal). See docs/metrics.md + Ender calibration.
    lostmat = acc["lost_material"]
    ldepth = (f"  loss-depth  median own-material in LOST games {_med(lostmat):.0f} "
              f"(0 = total wipeout)  ·  wiped-to-0 {100*sum(1 for m in lostmat if m<=0)/len(lostmat):.0f}%\n"
              if lostmat else "")
    return (f"Conversion: caps/game {c/n:.1f}  atk-launch/game {al/n:.1f}  "
            f"cap/atk-launch {c/max(al,1):.3f} (open<50 {cap_open:.3f}  mid50-100 {cap_mid:.3f})  "
            f"ships/cap {acc['attack_ships']/max(c,1):.0f}  reinf_share {rl/max(al+rl,1):.2f}\n"
            f"  planets@16/32/50/100 {pl(16)}/{pl(32)}/{pl(50)}/{pl(100)}  end {acc['end_planets']/n:.1f}\n"
            f"  game-len  median WON {medlen_w}st ({acc['games_won']}g)  ·  LOST {medlen_l}st ({acc['games_lost']}g)\n"
            f"{pwl}"
            f"{econ}"
            f"  retention  peel-rate {lost_rate:.2f} ({acc['lost_caps']}/{c} caps lost)  median-hold {med_hold}st\n"
            f"{rwl}"
            f"{ldepth}"
            f"  fire-rate  launch_rate {lr:.3f}  fire_frac {ff:.2f}   [ref:Isaiah 0.036 / 0.17]\n"
            f"{wl}"
            f"  ship0 1-ship-probe by phase  {_s0('')}{s0wl}\n"
            f"{already_covered}")


def _fmt_tier_summary(acc):
    """⭐ TIERED SUMMARY — re-prints the highest-signal metrics in priority order so the
    decision-relevant numbers aren't buried in the ~30-line dump above. Values are DUPLICATED
    (not moved). Priority + confound notes per docs/metrics.md. Read top-down, stop when answered."""
    gw, gl = acc["games_won"], acc["games_lost"]
    wr = gw / max(gw + gl, 1)
    _med = lambda h: (sorted(h)[len(h) // 2] if h else 0)
    # Outcome-grounded metrics only; model-based force proxies were not
    # discriminating in matched play.
    lostmat = acc["lost_material"]
    lmed = _med(lostmat) if lostmat else 0
    wiped = (100 * sum(1 for m in lostmat if m <= 0) / len(lostmat)) if lostmat else 0.0
    peel = acc["lost_caps"] / max(acc["captures"], 1)                 # of captures, fraction we lose
    peel_w = acc["lost_caps_won"] / max(acc["captures_won"], 1)       # won-game (elimination-free) read
    hold_w = _med(acc["hold_durations_won"])
    cap_open_w = acc["caps_early_won"] / max(acc["atk_early_won"], 1)
    p50w = (acc["p50_sum_won"] / acc["p50_n_won"]) if acc["p50_n_won"] else 0.0
    endp = acc["end_planets"] / max(acc["games"], 1)
    # T3 — degeneracy tripwires (binary; normal = ignore)
    lr = acc["launch_count"] / max(acc["launch_states"], 1)
    ff_w = acc["fire_frac_sum_won"] / max(acc["fire_steps_won"], 1)
    s0 = sum(acc["ship1_ph"]) / max(sum(acc["launches_ph"]), 1)
    medlen_w = _med(acc["game_len_won"])
    def _lost_econ(metric):
        return "/".join(
            (f"{median(acc[f'{metric}_{ms}_lost']):+.0f}"
             if acc[f"{metric}_{ms}_lost"] else "—")
            for ms in _ECONOMY_MILESTONES
        )
    bar = "─" * 78
    return (
        f"\n{bar}\n"
        f"⭐ TIERED METRIC SUMMARY  (priority order; values duplicated from above — docs/metrics.md)\n"
        f"{bar}\n"
        f"  T1 ARBITER   win-rate {wr:.1%} ({gw}/{gw + gl})   ← the only absolute-regression signal\n"
        f"  T2 THE WALL  loss-depth med-material-in-loss {lmed:.0f} · wiped-to-0 {wiped:.0f}%  (graded; want ↑ material)\n"
        f"               LOST paired Δ@32/50/100 prod {_lost_econ('prod_delta')} · material {_lost_econ('material_delta')}  (want ↑)\n"
        f"               retention  peel-rate WON {peel_w:.2f} (all {peel:.2f}) · median-hold WON {hold_w}st  (want peel↓)\n"
        f"               expansion  planets@50 WON {p50w:.0f} · end {endp:.1f}   ·   open<50 cap/atk WON {cap_open_w:.2f}\n"
        f"  T3 TRIPWIRE  launch_rate {lr:.3f} (→0 passive)   fire_frac WON {ff_w:.2f} (→1 carpet-bomb)   "
        f"ship0 {s0:.0%} (high = 1-ship collapse)\n"
        f"  colour only  game-len WON {medlen_w}st  (symptom of the root, NOT a gate — don't bribe with speed_coef)\n"
        f"{bar}"
    )


def evaluate_against_baseline(
    model: EntityTransformer,
    device: torch.device,
    num_games: int = 32,
    seed_start: int = 0,
    opponent: str = "random",
    num_players: int = 2,
    fire_threshold: float = 0.5,
    sample: bool = False,
    ship_bin_mode: str = "absolute",
    target_decode: bool = False,
) -> dict:
    """Evaluate trained policy against a baseline using kaggle_environments.

    Args:
        opponent: "random" or path to a Python agent file (e.g. "main.py")
        num_players: 2 or 4
    """
    from kaggle_environments import make

    validate_opponent_assets(opponent, num_players)
    agent_fn = build_agent_fn(model, device, fire_threshold=fire_threshold, sample=sample,
                              ship_bin_mode=ship_bin_mode,
                              target_decode=target_decode,
                              num_players=num_players)
    opponents = [opponent] * (num_players - 1)
    agents = [agent_fn] + opponents

    wins = 0
    total_material = 0
    conv_tot = new_conversion_acc()
    results = []

    for seed in range(seed_start, seed_start + num_games):
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run(agents)
        _assert_game_completed(env, f"seed={seed}")
        final = env.steps[-1]
        rewards = [s.reward for s in final]

        obs = final[0].observation
        material = sum(p[5] for p in obs.planets if p[1] == 0)
        material += sum(f[6] for f in obs.fleets if f[1] == 0)

        # Rank by reward; player 0 wins if their reward is strictly highest
        my_reward = rewards[0] if rewards[0] is not None else 0.0
        best_opp = max((r for r in rewards[1:] if r is not None), default=0.0)
        is_win = my_reward > best_opp

        add_conversion(conv_tot, game_conversion(env.steps, 0), won=is_win, material=material)

        wins += int(is_win)
        total_material += material
        results.append({
            "seed": seed,
            "win": is_win,
            "material": material,
            "rewards": rewards,
        })

    return {
        "wins": wins,
        "total_games": num_games,
        "win_rate": wins / num_games,
        "avg_material": total_material / num_games,
        "conversion": conv_tot,
        "results": results,
    }


def _accumulate_panel_records(records: list) -> dict:
    """Build the panel result dict from a list of per-game records.

    Each record is {archetype, my_seat, is_win, material, conv}. Factored out of
    evaluate_panel so recompute_panel.py can replay saved --panel-out records with the
    CURRENT metric code (add_conversion is a pure additive accumulator).
    """
    from eval_panel import BY_ARCHETYPE
    per_arch = {arch: {"wins": 0, "total": 0,
                       "wins_seat0": 0, "wins_seat1": 0,
                       "total_seat0": 0, "total_seat1": 0,
                       "material_sum": 0}
                for arch in BY_ARCHETYPE}
    overall = {"wins": 0, "total": 0, "wins_seat0": 0, "wins_seat1": 0,
               "total_seat0": 0, "total_seat1": 0}
    conv_tot = new_conversion_acc()
    for r in records:
        arch = r["archetype"]; my_seat = r["my_seat"]
        is_win = r["is_win"]; material = r["material"]
        add_conversion(conv_tot, r["conv"], won=is_win, material=material)
        c = per_arch[arch]
        c["wins"] += int(is_win); c["total"] += 1
        c[f"wins_seat{my_seat}"] += int(is_win)
        c[f"total_seat{my_seat}"] += 1
        c["material_sum"] += material
        overall["wins"] += int(is_win); overall["total"] += 1
        overall[f"wins_seat{my_seat}"] += int(is_win)
        overall[f"total_seat{my_seat}"] += 1
    return {"overall": overall, "per_archetype": per_arch, "conversion": conv_tot}


def evaluate_panel(
    model: EntityTransformer,
    device: torch.device,
    opponent: str,
    fire_threshold: float = 0.5,
    sample: bool = False,
    ship_bin_mode: str = "absolute",
    target_decode: bool = False,
    collect_records: bool = False,
) -> dict:
    """Stratified eval over the 128-seed community panel, playing both seats.

    256 games per opponent (128 seeds × 2 seats). Aggregates wins per
    archetype (8 games per cell = 4 seeds × 2 seats) and per seat, so a
    +5pp overall regression hidden by an asymmetric or board-shape-specific
    weakness is visible.
    """
    from kaggle_environments import make
    from eval_panel import BY_ARCHETYPE

    validate_opponent_assets(opponent, 2)
    agent_fn = build_agent_fn(model, device, fire_threshold=fire_threshold, sample=sample,
                              ship_bin_mode=ship_bin_mode,
                              target_decode=target_decode)

    records: list = []
    total_games = sum(len(seeds) for seeds in BY_ARCHETYPE.values()) * 2

    print(f"Panel eval START — opponent: {opponent} | {total_games} games "
          f"(128 seeds × 2 seats) | decode={'target' if target_decode else 'argmax'} "
          f"fire_thr={fire_threshold}",
          flush=True)

    wins_running = 0
    for archetype, seeds in BY_ARCHETYPE.items():
        for seed in seeds:
            for my_seat in (0, 1):
                agents = [agent_fn, opponent] if my_seat == 0 else [opponent, agent_fn]
                env = make("orbit_wars", configuration={"seed": seed}, debug=False)
                env.run(agents)
                _assert_game_completed(
                    env, f"archetype={archetype} seed={seed} seat={my_seat}")
                final = env.steps[-1]
                rewards = [s.reward if s.reward is not None else 0.0 for s in final]
                my_reward = rewards[my_seat]
                opp_reward = rewards[1 - my_seat]
                is_win = my_reward > opp_reward
                conv = game_conversion(env.steps, my_seat)
                # Material on the model's side
                obs = final[0].observation
                material = sum(p[5] for p in obs.planets if p[1] == my_seat)
                material += sum(f[6] for f in obs.fleets if f[1] == my_seat)
                records.append({"archetype": archetype, "my_seat": my_seat,
                                "is_win": is_win, "material": material, "conv": conv})
                wins_running += int(is_win)
                if len(records) % 16 == 0 or len(records) == total_games:
                    print(f"  panel progress: {len(records)}/{total_games}  "
                          f"overall {wins_running}/{len(records)} "
                          f"({100*wins_running/max(len(records),1):.1f}%)",
                          flush=True)

    result = _accumulate_panel_records(records)
    if collect_records:
        result["_records"] = records
    return result


def print_panel_report(result: dict, opponent: str) -> None:
    """Pretty-print panel results."""
    o = result["overall"]
    print()
    print("=" * 78)
    print(f"Panel eval vs {opponent}")
    print("=" * 78)
    print(f"Overall:   {o['wins']}/{o['total']}  ({100*o['wins']/max(o['total'],1):.1f}%)")
    s0 = 100 * o['wins_seat0'] / max(o['total_seat0'], 1)
    s1 = 100 * o['wins_seat1'] / max(o['total_seat1'], 1)
    print(f"  seat 0:  {o['wins_seat0']}/{o['total_seat0']}  ({s0:.1f}%)")
    print(f"  seat 1:  {o['wins_seat1']}/{o['total_seat1']}  ({s1:.1f}%)")
    asym = s0 - s1
    print(f"  asymmetry (seat0 − seat1): {asym:+.1f}pp")
    if "conversion" in result:
        print(_fmt_conversion(result["conversion"]))
    print()
    print("Per archetype  (8 games each = 4 seeds × 2 seats):")
    print(f"  {'archetype':<48s}  {'WR':>6s}  {'s0/s1':>10s}  {'mat':>8s}")
    rows = []
    for arch, c in result["per_archetype"].items():
        wr = 100 * c["wins"] / max(c["total"], 1)
        s0 = 100 * c["wins_seat0"] / max(c["total_seat0"], 1)
        s1 = 100 * c["wins_seat1"] / max(c["total_seat1"], 1)
        mat = c["material_sum"] / max(c["total"], 1)
        rows.append((wr, arch, c, s0, s1, mat))
    # sort by winrate descending so worst cells stand out at the bottom
    rows.sort(key=lambda r: -r[0])
    for wr, arch, c, s0, s1, mat in rows:
        print(f"  {arch:<48s}  {wr:>5.1f}%  {s0:>4.0f}/{s1:>3.0f}  {mat:>8.0f}")
    # quick diagnostic
    worst = min(rows, key=lambda r: r[0])
    best = max(rows, key=lambda r: r[0])
    print()
    print(f"Best:  {best[1]}  ({best[0]:.1f}%)")
    print(f"Worst: {worst[1]}  ({worst[0]:.1f}%)")
    print(f"Spread: {best[0] - worst[0]:.1f}pp")
    if "conversion" in result:
        print(_fmt_tier_summary(result["conversion"]))


def evaluate_checkpoint(params_path: str, cfg: Config, num_games: int = 32,
                        seed_start: int = 0,
                        opponent: str = "random", fire_threshold: float = 0.5,
                        panel: bool = False, sample: bool = False,
                        target_decode: bool = False,
                        reinforce_gate_min_planets: int = None,
                        reinforce_forward_only: bool = None,
                        reinforce_garrison_floor: float = None,
                        sufficient_commit_factor: float = None,
                        collect_records: bool = False):
    """Load a checkpoint and evaluate it."""
    device = torch.device(cfg.device)

    model, ckpt_action_decode = load_eval_model(params_path, cfg)
    # Discipline masks: an explicit CLI value overrides; otherwise auto-load what the checkpoint
    # was trained with (load_checkpoint set these on cfg.model). Eliminates the "forgot the flag
    # → wrong panel/submission" footgun for masked runs. For OLD reinforce ckpts that never
    # persisted the discipline, the gate CAN'T be inferred (guessing self-sabotages) → require it.
    if (bool(cfg.model.allow_reinforce) and not bool(getattr(cfg.model, "_discipline_persisted", False))
            and reinforce_gate_min_planets is None):
        raise SystemExit(
            "Checkpoint has allow_reinforce=True but NO persisted reinforce discipline (pre-2026-06-15 "
            "ckpt). The gate/floor/forward values can't be inferred and guessing self-sabotages — pass "
            "--reinforce-gate-min-planets (and --reinforce-garrison-floor / --[no-]reinforce-forward-only) "
            "explicitly to match how it was trained.")
    # Each auto-loaded entry shows its value AND whether the mask is actually ACTIVE — so
    # "forward_only=False [off]" reads as "no mask applied", not "a mask got enabled". off =
    # the mask is a no-op at this value (gate≤0 / forward False / floor≤0 / suff≤0).
    _on = lambda active: "on" if active else "off"
    _from_ckpt = []
    if reinforce_gate_min_planets is None:
        reinforce_gate_min_planets = int(cfg.model.reinforce_gate_min_planets)
        _from_ckpt.append(f"gate={reinforce_gate_min_planets} [{_on(reinforce_gate_min_planets > 0)}]")
    if reinforce_forward_only is None:
        reinforce_forward_only = bool(cfg.model.reinforce_forward_only)
        _from_ckpt.append(f"forward_only={reinforce_forward_only} [{_on(reinforce_forward_only)}]")
    if reinforce_garrison_floor is None:
        reinforce_garrison_floor = float(cfg.model.reinforce_garrison_floor)
        _from_ckpt.append(f"floor={reinforce_garrison_floor} [{_on(reinforce_garrison_floor > 0)}]")
    if sufficient_commit_factor is None:
        sufficient_commit_factor = float(cfg.model.sufficient_commit_factor)
        _from_ckpt.append(f"sufficient_commit={sufficient_commit_factor} [{_on(sufficient_commit_factor > 0)}]")
    if _from_ckpt:
        print(f"Discipline auto-loaded from checkpoint: {', '.join(_from_ckpt)}")
        # Full resolved set in effect (incl. any CLI-set values), so train/eval parity is visible.
        print(f"  → discipline in effect: gate={reinforce_gate_min_planets} "
              f"forward_only={reinforce_forward_only} floor={reinforce_garrison_floor} "
              f"suff={sufficient_commit_factor}")
    if cfg.model.ship_bin_mode != "absolute":
        print(f"Checkpoint ship_bin_mode={cfg.model.ship_bin_mode}")
    # Auto-detect action_decode from checkpoint config; CLI --target-decode overrides.
    if not target_decode and ckpt_action_decode == "target":
        target_decode = True
        print("Checkpoint action_decode=target  →  enabling target_decode automatically")

    # Explicit CLI overrides of the discipline load_eval_model set from the checkpoint (a None
    # argument resolved to the checkpoint's own value above, so this is a no-op unless overridden).
    model.reinforce_gate_min_planets = int(reinforce_gate_min_planets)
    model.reinforce_forward_only = bool(reinforce_forward_only)
    model.reinforce_garrison_floor = float(reinforce_garrison_floor)
    model.sufficient_commit_factor = float(sufficient_commit_factor)
    if model.allow_reinforce:
        print(f"Reinforcement: ON (own planets are legal targets) | "
              f"gate>={model.reinforce_gate_min_planets} planets, "
              f"forward_only={model.reinforce_forward_only}, "
              f"garrison_floor={model.reinforce_garrison_floor}, "
              f"reverse_edge_cooldown={model.reverse_edge_cooldown}")
    if model.sufficient_commit_factor > 0.0:
        print(f"Sufficient-commit mask: ON | veto attacks with ships <= "
              f"target_defense × {model.sufficient_commit_factor}")

    if panel:
        results = evaluate_panel(model, device, opponent=opponent,
                                 fire_threshold=fire_threshold, sample=sample,
                                 ship_bin_mode=cfg.model.ship_bin_mode,
                                 target_decode=target_decode,
                                 collect_records=collect_records)
        print_panel_report(results, opponent)
        return results

    results = evaluate_against_baseline(
        model, device,
        ship_bin_mode=cfg.model.ship_bin_mode,
        target_decode=target_decode,
        num_games=num_games,
        seed_start=seed_start,
        opponent=opponent,
        num_players=cfg.env.num_players,
        fire_threshold=fire_threshold,
        sample=sample,
    )

    print(f"Win rate vs {opponent}: {results['win_rate']:.2%}  "
          f"({results['wins']}/{results['total_games']})")
    print(f"Fire threshold: {fire_threshold}")
    print(f"Target decode: {target_decode}")
    print(f"Avg material: {results['avg_material']:.1f}")
    print(_fmt_conversion(results["conversion"]))
    for r in results["results"][:5]:
        print(f"  seed={r['seed']} win={r['win']} "
              f"material={r['material']} rewards={r['rewards']}")
    print(_fmt_tier_summary(results["conversion"]))  # tiered summary LAST = bottom of output

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=0,
                        help="First seed for non-panel eval. Ignored by --panel, which uses the fixed archetype panel.")
    parser.add_argument("--opponent", default="random",
                        help="'random' or path to agent .py file")
    parser.add_argument("--num-players", type=int, choices=[2, 4], default=2)
    parser.add_argument("--fire-threshold", type=float, default=0.5)
    parser.add_argument("--panel", action="store_true",
                        help="Use 128-seed community panel with both-seat eval "
                             "(256 games, per-archetype breakdown).")
    parser.add_argument("--sample", action="store_true",
                        help="Sample from policy distribution instead of argmax. "
                             "Use when the mode is degenerate but distribution mass "
                             "is on competent bins (1-ship-fleet trap).")
    parser.add_argument("--target-decode", action="store_true",
                        help="Aim with target_logits plus orbital intercept.")
    parser.add_argument("--reinforce-gate-min-planets", type=int, default=None,
                        help="Reinforce-discipline parity: own targets legal only at "
                             ">= this many owned planets. Default=auto-load from checkpoint; "
                             "pass to override. MUST match training.")
    parser.add_argument("--reinforce-forward-only", action=argparse.BooleanOptionalAction, default=None,
                        help="Reinforce-discipline parity: own target legal only if closer "
                             "to the nearest enemy than the source. Default=auto-load from ckpt; "
                             "pass --reinforce-forward-only / --no-reinforce-forward-only to override.")
    parser.add_argument("--reinforce-garrison-floor", type=float, default=None,
                        help="Reinforce-discipline parity: veto a reinforce that drains the "
                             "source below this. Default=auto-load from checkpoint.")
    parser.add_argument("--sufficient-commit-factor", type=float, default=None,
                        help="Sufficient-commit parity: veto an attack whose ships <= target "
                             "defense × this factor. Default=auto-load from ckpt (1.0 = strict).")
    parser.add_argument("--panel-out", type=str, default=None,
                        help="Pickle the full --panel per-game records here (each game's conv dict "
                             "incl. dm_ratios), AND print the report normally. recompute_panel.py "
                             "re-derives any metric offline — so a later metric addition never needs a "
                             "panel re-run. No effect without --panel.")
    args = parser.parse_args()

    cfg = Config()
    cfg.env.num_players = args.num_players
    _eval_result = evaluate_checkpoint(
        args.checkpoint,
        cfg,
        num_games=args.games,
        seed_start=args.seed_start,
        opponent=args.opponent,
        fire_threshold=args.fire_threshold,
        panel=args.panel,
        sample=args.sample,
        target_decode=args.target_decode,
        reinforce_gate_min_planets=args.reinforce_gate_min_planets,
        reinforce_forward_only=args.reinforce_forward_only,
        reinforce_garrison_floor=args.reinforce_garrison_floor,
        sufficient_commit_factor=args.sufficient_commit_factor,
        collect_records=bool(args.panel_out),
    )
    if args.panel_out and _eval_result is not None:
        import pickle
        with open(args.panel_out, "wb") as _f:
            pickle.dump({"records": _eval_result.get("_records", []),
                         "opponent": args.opponent}, _f)
        print(f"PANEL RECORDS → {args.panel_out}: "
              f"{len(_eval_result.get('_records', []))} games "
              f"(recompute any metric: python orbit_wars_rl/recompute_panel.py {args.panel_out})",
              flush=True)
