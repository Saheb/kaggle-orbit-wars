"""Evaluation: pit trained PyTorch policy against baselines."""

from __future__ import annotations

import argparse
import math
import os
from statistics import mean

import torch
import numpy as np

from config import Config
from model import EntityTransformer, NUM_ANGLE_BINS, NUM_SHIP_BINS, ANGLE_BIN_WIDTH, PHASE4_COMPAT_MISSING_KEYS
from features import (extract_features, _ETA_PROBE_SPEED, set_game_phase_features,
                      set_ablate_roi, set_ablate_channels, PAIRWISE_FEATURE_DIM)
from action_mask import (compute_action_masks, actions_from_policy, actions_from_target_policy, _fleet_speed,
                         _ship_bin_to_count, _target_intercept_angle, MAX_OWNED_PLANETS)
# Decisive-mass floor constants — IMPORTED from torch_env so the eval dm_* gap diagnostic uses the
# EXACT same floor as the training reward/diag (they can never drift). project_force_concentration_wall.
from torch_env import (_DM_BETA, _DM_ETA_FREE, _DM_ETA_SCALE, _DM_HORIZON, _DM_OVERHEAD,
                       MAX_SHIP_SPEED as _DM_MAX_SPEED)
from kaggle_environments.envs.orbit_wars.orbit_wars import CENTER, ROTATION_RADIUS_LIMIT
# Reactive-margin weight for the eval dm_* floor. Defaults to torch_env's 2.2 (= training default).
# A decmass run trained with a NON-default --decisive-mass-beta must pass --decisive-mass-beta to
# eval for "eval floor == reward floor" to hold (beta is an env param, not persisted in the ckpt).
# The value used is printed on the `decisive-mass` line so the floor is never ambiguous.
_DM_BETA_EVAL = _DM_BETA


def load_checkpoint(path: str, cfg: Config) -> tuple[dict, str]:
    """Load a checkpoint and patch cfg.model dims to match the saved weights.

    Returns (state_dict, action_decode).  Modifies cfg.model in-place so that
    EntityTransformer(cfg.model) builds the correct architecture for this
    checkpoint — regardless of what config.py currently says.  This lets old
    (pre-Phase-1) checkpoints be evaluated after the config has been bumped.
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # --- head / bin dims from saved config or weight shapes ---
    if "num_ship_bins" in ckpt_cfg:
        cfg.model.num_ship_bins = int(ckpt_cfg["num_ship_bins"])
    elif "ship_head.weight" in sd:
        cfg.model.num_ship_bins = int(sd["ship_head.weight"].shape[0])

    if "angle_head.weight" in sd:
        n = int(sd["angle_head.weight"].shape[0])
        if n != cfg.model.num_angle_bins:
            cfg.model.num_angle_bins = n

    if "min_ship_bin" in ckpt_cfg:
        cfg.model.min_ship_bin = int(ckpt_cfg["min_ship_bin"])
    if "ship_bin_mode" in ckpt_cfg:
        cfg.model.ship_bin_mode = str(ckpt_cfg["ship_bin_mode"])

    # --- feature projection dims: always infer from weight shapes ---
    if "planet_proj.weight" in sd:
        cfg.model.planet_feature_dim = int(sd["planet_proj.weight"].shape[1])
    if "fleet_proj.weight" in sd:
        cfg.model.fleet_feature_dim = int(sd["fleet_proj.weight"].shape[1])
    if "global_proj.weight" in sd:
        cfg.model.global_feature_dim = int(sd["global_proj.weight"].shape[1])
    if "pair_kv.weight" in sd:
        D = int(sd["planet_proj.weight"].shape[0])
        ckpt_pw = int(sd["pair_kv.weight"].shape[1]) - D
        # features.py always emits PAIRWISE_FEATURE_DIM channels, so the model's pairwise
        # input must be that wide regardless of the checkpoint. Older/narrower checkpoints
        # are zero-padded by EntityTransformer.load_state_dict (new channels contribute
        # nothing → identical behaviour). Building to the checkpoint width would feed
        # PAIRWISE_FEATURE_DIM-channel features to a narrower model → shape mismatch at forward.
        cfg.model.pairwise_feature_dim = PAIRWISE_FEATURE_DIM if ckpt_pw > 0 else 0
    else:
        cfg.model.pairwise_feature_dim = 0

    # Detect value head version from fc1 input width (old=D, new=2D).
    if "value_fc1.weight" in sd:
        cfg.model.value_head_in = int(sd["value_fc1.weight"].shape[1])

    action_decode = str(ckpt_cfg.get("action_decode", "angle"))
    # Reinforcement: eval must mask targets the SAME way the checkpoint was trained.
    cfg.model.allow_reinforce = bool(ckpt_cfg.get("allow_reinforce", False))
    # Game-phase features: eval's extract_features must emit the SAME globals the ckpt was
    # trained on (11 vs 15). Set the module flag to match (off for all pre-Stage-B ckpts).
    cfg.model.game_phase_features = bool(ckpt_cfg.get("game_phase_features", False))
    set_game_phase_features(cfg.model.game_phase_features)
    # Reinforce / sufficient-commit DISCIPLINE: persisted at train time so eval/export mask the
    # SAME way (else the policy self-sabotages). Absent in old ckpts → defaults (0/False) → those
    # still require CLI flags, as before. evaluate_checkpoint uses these unless CLI overrides.
    cfg.model.reinforce_gate_min_planets = int(ckpt_cfg.get("reinforce_gate_min_planets", 0))
    cfg.model.reinforce_forward_only = bool(ckpt_cfg.get("reinforce_forward_only", False))
    cfg.model.reverse_edge_cooldown = int(ckpt_cfg.get("reverse_edge_cooldown", 0))
    cfg.model.reinforce_garrison_floor = float(ckpt_cfg.get("reinforce_garrison_floor", 0.0))
    cfg.model.sufficient_commit_factor = float(ckpt_cfg.get("sufficient_commit_factor", 0.0))
    cfg.model._discipline_persisted = ("reinforce_gate_min_planets" in ckpt_cfg)
    # provenance (inspectable; eval always clamps regardless of how training handled overflow)
    cfg.model.ship_overflow_mode = str(ckpt_cfg.get("ship_overflow_mode", "drop"))
    return sd, action_decode


# Fire-head isolation override (eval diagnostic). On sources the fire head VETOES (fire_prob <
# threshold) that have a high-holdable-ROI attack available, FORCE fire toward the head's own
# argmax target (fall back to the top-ROI target if the head's pick is illegal/own), with the
# head's own ship sizing. Isolates "does the fire veto cost us winnable attacks" from selection.
# WR rises => fire veto suppresses valuable attacks (audit right); ties/falls => vetoes correct.
_FORCE_FIRE = {"on": False, "roi": 0.3, "forced": 0, "states": 0, "to_head_tgt": 0}


def set_force_fire_high_roi(on: bool, roi_threshold: float = 0.3) -> None:
    _FORCE_FIRE.update(on=bool(on), roi=float(roi_threshold), forced=0, states=0, to_head_tgt=0)


def _apply_force_fire(moves, outputs, masks, obs, player, fire_threshold, ship_bin_mode):
    tgt_arg = torch.argmax(outputs["target_logits"][0], dim=-1).cpu().numpy()
    tgt_idx_t = torch.as_tensor(tgt_arg, device=outputs["target_logits"].device).unsqueeze(0)
    fire_logits = torch.gather(outputs["fire_logits"], -1, tgt_idx_t.unsqueeze(-1)).squeeze(-1)
    ship_logits = torch.gather(
        outputs["ship_logits"],
        2,
        tgt_idx_t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, outputs["ship_logits"].shape[-1]),
    ).squeeze(2)
    fire_p = torch.sigmoid(fire_logits[0]).cpu().numpy()
    ship_arg = torch.argmax(ship_logits[0], dim=-1).cpu().numpy()
    owned_idx = masks["owned_indices"].cpu().numpy()
    max_ships = masks["max_ships"].cpu().numpy().reshape(-1)
    planets = obs["planets"]
    fleets = obs.get("fleets") or []
    owned_count = int(masks["owned_count"])
    fired_src = {int(m[0]) for m in moves}
    _FORCE_FIRE["states"] += 1
    for slot in range(min(owned_count, fire_p.shape[0])):
        if len(moves) >= MAX_OWNED_PLANETS:
            break
        if fire_p[slot] >= fire_threshold:
            continue                                          # head already fires → not a veto
        pidx = int(owned_idx[slot])
        if pidx >= len(planets):
            continue
        src = planets[pidx]
        if int(src[0]) in fired_src or src[5] <= 0:
            continue
        best_roi = best_tgt = None                            # best holdable-ROI attack from this source
        for tgt in planets:
            if int(tgt[1]) == player or int(tgt[0]) == int(src[0]):
                continue
            hr = _holdable_roi(src, tgt, planets, fleets, player)
            if hr is not None and (best_roi is None or hr > best_roi):
                best_roi, best_tgt = hr, tgt
        if best_roi is None or best_roi < _FORCE_FIRE["roi"]:
            continue                                          # no worthwhile attack → don't force (avoid spray)
        ti = int(tgt_arg[slot])                               # keep head's target if it's a legal attack
        head_tgt = planets[ti] if 0 <= ti < len(planets) else None
        use_head = head_tgt is not None and int(head_tgt[1]) != player and int(head_tgt[0]) != int(src[0])
        use_tgt = head_tgt if use_head else best_tgt
        ships = min(int(_ship_bin_to_count(int(ship_arg[slot]), int(max_ships[slot]), mode=ship_bin_mode)), int(src[5]))
        if ships <= 0:
            continue
        angle = _target_intercept_angle(src, use_tgt, ships, obs)
        moves.append([int(src[0]), float(angle), int(ships)])
        fired_src.add(int(src[0]))
        _FORCE_FIRE["forced"] += 1
        if use_head:
            _FORCE_FIRE["to_head_tgt"] += 1
    return moves


# Retarget override (eval diagnostic, selection isolation). Leaves fire/ship as-is; for each ATTACK
# the policy actually launches, redirect its target to the top-holdable-ROI candidate from that
# source (keep source + ship count). No new launches (no spray), no fire change → clean test of
# "of the attacks we make, does picking the best target raise WR?" Raises best% to ~100%.
_RETARGET = {"on": False, "resize": False, "retargeted": 0, "attacks": 0, "uniq_sum": 0.0, "turns": 0}


def set_retarget_top_roi(on: bool, resize: bool = False) -> None:
    _RETARGET.update(on=bool(on), resize=bool(resize), retargeted=0, attacks=0, uniq_sum=0.0, turns=0)


def _apply_retarget(moves, obs, player):
    planets = obs["planets"]
    fleets = obs.get("fleets") or []
    byid = {int(p[0]): p for p in planets}
    for m in moves:
        src = byid.get(int(m[0]))
        if src is None:
            continue
        cur = _resolve_launch_target(planets, src, m[1])
        if cur is None or int(cur[1]) == player:
            continue                                          # leave reinforces / unresolved alone
        _RETARGET["attacks"] += 1
        best_roi = best = None
        for tgt in planets:
            if int(tgt[1]) == player or int(tgt[0]) == int(src[0]):
                continue
            hr = _holdable_roi(src, tgt, planets, fleets, player)
            if hr is not None and (best_roi is None or hr > best_roi):
                best_roi, best = hr, tgt
        if best is not None and int(best[0]) != int(cur[0]):
            ships = int(m[2])
            if _RETARGET["resize"]:                            # size to actually capture the NEW target
                need = int(best[5]) + (1 if int(best[1]) < 0 else int(best[6]) * 3 + 1)
                ships = min(int(src[5]), max(ships, need))
            m[2] = int(ships)
            m[1] = float(_target_intercept_angle(src, best, ships, obs))
            _RETARGET["retargeted"] += 1
    # target-funnel diagnostic: distinct attack targets this turn / number of attacks (1.0 = all distinct,
    # low = many sources piling on the same ROI-greedy target = no expansion spread)
    atk_tgts = []
    for m in moves:
        s = byid.get(int(m[0]))
        if s is None:
            continue
        rt = _resolve_launch_target(planets, s, m[1])
        if rt is not None and int(rt[1]) != player:
            atk_tgts.append(int(rt[0]))
    if atk_tgts:
        _RETARGET["uniq_sum"] += len(set(atk_tgts)) / len(atk_tgts)
        _RETARGET["turns"] += 1
    return moves


def build_agent_fn(model: EntityTransformer, device: torch.device,
                   fire_threshold: float = 0.5, sample: bool = False,
                   ship_bin_mode: str = "absolute",
                   target_decode: bool = False,
                   target_sanity_penalty: float = 0.0,
                   reserve_frac: float = 0.0,
                   allow_reinforce: bool = False,
                   veto_stats: dict = None,
                   defensive_reinforce_k: int = 0,
                   defensive_reinforce_beta: float = 2.2,
                   defensive_reinforce_max_targets: int = 1,
                   defensive_reinforce_value_margin: float | None = None,
                   defensive_reinforce_overfill: float = 1.0,
                   defensive_reinforce_stats: dict = None,
                   natural_head_audit_stats: dict = None,
                   natural_head_audit_beta: float = 2.2):
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
        features = extract_features(obs, player, num_players=2)
        masks = compute_action_masks(obs, player)

        with torch.no_grad():
            outputs = model(
                features["planet_features"].unsqueeze(0).to(device),
                features["fleet_features"].unsqueeze(0).to(device),
                features["global_features"].unsqueeze(0).to(device),
                features["planet_mask"].unsqueeze(0).to(device),
                features["fleet_mask"].unsqueeze(0).to(device),
                fire_mask=masks["fire_mask"].to(device),
                angle_mask=masks["angle_mask"].to(device),
                slot_valid=masks["slot_valid"].to(device),
                owned_indices=masks["owned_indices"].to(device),
                owned_count=masks["owned_count"],
                pairwise_features=features["pairwise_features"].unsqueeze(0).to(device)
                    if "pairwise_features" in features else None,
            )

        action_fn = actions_from_target_policy if target_decode else actions_from_policy
        if target_decode:
            moves = action_fn(
                outputs["fire_logits"].cpu(),
                outputs["target_logits"].cpu(),
                outputs["ship_logits"].cpu(),
                {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()},
                obs, player,
                fire_threshold=fire_threshold,
                sample=sample,
                ship_bin_mode=ship_bin_mode,
                target_sanity_penalty=target_sanity_penalty,
                reserve_frac=reserve_frac,
                allow_reinforce=getattr(model, "allow_reinforce", allow_reinforce),
                reinforce_gate_min_planets=getattr(model, "reinforce_gate_min_planets", 0),
                reinforce_forward_only=getattr(model, "reinforce_forward_only", False),
                reinforce_garrison_floor=getattr(model, "reinforce_garrison_floor", 0.0),
                sufficient_commit_factor=getattr(model, "sufficient_commit_factor", 0.0),
                reverse_edge_cooldown=_cd_K,
                cooldown_last=_cd["last"] if _cd_K > 0 else None,
                cooldown_step=int(obs.get("step", 0)),
                defensive_reinforce_k=defensive_reinforce_k,
                defensive_reinforce_beta=defensive_reinforce_beta,
                defensive_reinforce_max_targets=defensive_reinforce_max_targets,
                defensive_reinforce_value_margin=defensive_reinforce_value_margin,
                defensive_reinforce_overfill=defensive_reinforce_overfill,
                defensive_reinforce_stats=defensive_reinforce_stats,
                natural_head_audit_stats=natural_head_audit_stats,
                natural_head_audit_beta=natural_head_audit_beta,
                veto_stats=veto_stats,
            )
            if _RETARGET["on"] and not sample:
                _apply_retarget(moves, obs, player)
            if _FORCE_FIRE["on"] and not sample:
                _apply_force_fire(moves, outputs, masks, obs, player, fire_threshold, ship_bin_mode)
            return moves

        raise NotImplementedError(
            "angle-decode path removed (angle head deleted); Phase 1 checkpoints "
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
# redundant/underkill are windowed to the OPENING: late-game surplus production re-fires at the
# last enemy planets (benign — we've already won), which inflates a whole-game fraction in long
# won games. The launch waste we care about is in the opening, where wasted ships feed the
# mid-game collapse. <50 isolates that phase. (phase2 / metrics.md)
_LAUNCH_WINDOW = 50
_MID_WINDOW = 100      # mid-game cap/atk window = [_LAUNCH_WINDOW, _MID_WINDOW) = steps 50-100
_TRIAGE_LOSS_WINDOWS = (10, 20, 30)
_SAFE_REINF_LOOKAHEAD = 30
# Isaiah (#1 player) hoard reference at the same milestones. Contested phase (16-50)
# is the clean read: ~half the army deployed, ~11-22 ships/planet. The @100 jump
# (garr 0.87, 60 ships/planet) is won-game accumulation, not hoarding.
_ISAIAH_HOARD_REF = "garr_frac 0.50/0.51/0.54/0.87  ships/planet 11/15/22/60"

# reinforce-by-empire-size bins (owned planets AT LAUNCH TIME). The aggregate reinf_share
# is opponent/success-confounded (it co-moves with empire size — phase2 §6); bucketing by
# empire size makes it directly comparable to the top-player ramp (phase2 §2 / metrics.md):
# @1 ≈0.00, @2 ≈0.10, @9-12 ≈0.30, @13+ 0.34-0.61.
# @2 and @3 split (2026-06-13): with reinforce_gate_min_planets=3, reinforce is MASKED at
# 1-2 planets, so a combined "2-3" bin is diluted by gated 2s and under-reads true @3. The
# winner ramp (89 snowball replays) is @1:0.007 @2:0.10 @3:0.19 — so @3 ≈ 2× a combined 2-3.
_REINF_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 6, "4-6"),
               (7, 9, "7-9"), (10, 12, "10-12"), (13, 10**9, "13+")]
_REINF_RAMP_REF = "@1:0.01 @2:0.10 @3:0.19 @9-12:0.30 @13+:0.34-0.61"
_REINF_STEP_REF = "<50:0.29 · 50-100:0.41 · >100:0.31"   # snowball winners; peaks mid-game
_ATK_FAIL_LABELS = ("single", "wrong-src", "aggregate", "unafford", "sufficient", "redundant")


def _reinf_bin_idx(owned):
    for i, (lo, hi, _) in enumerate(_REINF_BINS):
        if lo <= owned <= hi:
            return i
    return 0  # owned 0 can't launch; guard


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


def _resolve_launch_target(planets, src, angle, ships=None):
    """Planet a launch from `src` at `angle` actually hits. With `ships` given (the launched ship
    count, needed for fleet speed) this is the lead-aware collision resolver — the fleet flies
    straight and captures whatever it physically collides with, so distance / planet radius / the
    target's orbital motion all matter (none of which the old angle-only match saw). `ships=None`
    falls back to the legacy angle-only match for the live retarget intent-probe (`_apply_retarget`),
    which has no ship count and only wants the aimed direction."""
    if ships is None:
        sx, sy = src[2], src[3]
        best, bd = None, 0.6
        for p in planets:
            if p[0] == src[0]:
                continue
            pa = math.atan2(p[3] - sy, p[2] - sx)
            dd = abs((pa - angle + math.pi) % (2 * math.pi) - math.pi)
            if dd < bd:
                bd, best = dd, p
        return best
    # Fleet spawns at the source surface + a small launch offset along the heading (engine:
    # start = planet + cos/sin(angle)*(radius + 0.1)), then flies straight.
    sx = src[2] + math.cos(angle) * (src[4] + 0.1)
    sy = src[3] + math.sin(angle) * (src[4] + 0.1)
    return _lead_collision_target(planets, sx, sy, angle, ships, skip_pid=src[0])


def _cap_cost_at_arrival(src, tgt, seat):
    """Ships needed to CAPTURE planet `tgt` from `src` by the time a fleet arrives — the
    SAME quantity the roi-deflation feature uses (features.py compute_pairwise_features), so
    `redundant`/`underkill` measure exactly what the deflation acts on. eta from straight-line
    dist (the feature adds a small rotation correction for orbiting planets — second-order on
    eta). ships_at_arrival = current + production·eta; neutral cost +1, enemy +prod·3+1.
    Returns 0 for an own target (can't 'capture' it)."""
    owner = int(tgt[1])
    if owner == seat:
        return 0.0
    dist = math.hypot(tgt[2] - src[2], tgt[3] - src[3])
    eta = max(1.0, math.ceil(dist / _ETA_PROBE_SPEED))
    ships_at_arrival = min(tgt[5] + tgt[6] * eta, 500.0)
    return ships_at_arrival + (1.0 if owner == -1 else tgt[6] * 3 + 1.0)


def _holdable_roi(src, tgt, planets, fleets, seat, beta=_DM_BETA):
    """Reactive-aware ROI of attacking `tgt` from `src`: value (prod·20) minus producer_v2's capture
    FLOOR — projected defenders + enemy inbound + beta·rho(eta)·reachable enemy PLANET mass + overhead,
    the SAME floor as the decisive-mass reward (torch_env._decisive_mass_fields). Unlike the static
    ch12 roi_20, the cost prices the REACTIVE peel, so a closer/richer-but-unholdable target scores
    LOW. Returns None for an own target (can't attack it)."""
    owner = int(tgt[1])
    if owner == seat:
        return None
    dist = math.hypot(tgt[2] - src[2], tgt[3] - src[3])
    eta = min(max(1.0, math.ceil(dist / _ETA_PROBE_SPEED)), _DM_HORIZON)
    inbound = _friendly_inbound(fleets, tgt, 1 - seat)        # enemy FLEET ships already inbound
    enemy_mass = 0.0                                          # reachable enemy PLANET mass (cheap_enemy_pressure)
    for ep in planets:
        eo = int(ep[1])
        if eo < 0 or eo == seat or int(ep[0]) == int(tgt[0]):
            continue                                          # enemy planets only (excl neutral/self/target)
        reach = max(_fleet_speed(int(ep[5])) * _DM_HORIZON, 1e-6)
        d = math.hypot(ep[2] - tgt[2], ep[3] - tgt[3])
        enemy_mass += ep[5] * max(1.0 - d / reach, 0.0)
    rho = min(max((eta - _DM_ETA_FREE) / _DM_ETA_SCALE, 0.0), 1.0)
    floor = tgt[5] + tgt[6] * eta + inbound + beta * rho * enemy_mass + _DM_OVERHEAD
    return (tgt[6] * 20.0 - floor) / max(floor, 1.0)


def _attack_capture_floor(src, tgt, planets, fleets, seat, beta=None):
    """Decision-time floor for a launched attack.

    Neutral targets use the static garrison floor from the neutral decisive-mass
    diagnostic. Enemy targets use the reactive decisive-mass floor, including
    production during ETA, enemy inbound, and reachable enemy planet mass.
    """
    if beta is None:
        beta = _DM_BETA_EVAL
    owner = int(tgt[1])
    if owner == seat:
        return 0.0
    dist = math.hypot(tgt[2] - src[2], tgt[3] - src[3])
    eta = min(max(1.0, math.ceil(dist / _ETA_PROBE_SPEED)), _DM_HORIZON)
    if owner < 0:
        return max(tgt[5] + _DM_OVERHEAD, 1e-6)
    inbound = _friendly_inbound(fleets, tgt, 1 - seat)
    enemy_mass = 0.0
    for ep in planets:
        eo = int(ep[1])
        if eo < 0 or eo == seat or int(ep[0]) == int(tgt[0]):
            continue
        reach = max(_fleet_speed(int(ep[5])) * _DM_HORIZON, 1e-6)
        d = math.hypot(ep[2] - tgt[2], ep[3] - tgt[3])
        enemy_mass += ep[5] * max(1.0 - d / reach, 0.0)
    rho = min(max((eta - _DM_ETA_FREE) / _DM_ETA_SCALE, 0.0), 1.0)
    return max(tgt[5] + tgt[6] * eta + inbound + beta * rho * enemy_mass + _DM_OVERHEAD, 1e-6)


def _attack_failure_bucket(planets, fleets, src, tgt, seat, sent, captured_soon):
    """Classify a failed attack launch into the fix it implies.

    Returns None for successful non-redundant attacks. For failures, the key split is:
    did the chosen source have enough and undersend, or did the target require
    cross-source aggregation that the factored action cannot express in one move?
    """
    floor = _attack_capture_floor(src, tgt, planets, fleets, seat)
    if floor <= 0:
        return None
    inbound_before = _friendly_inbound(fleets, tgt, seat)
    if inbound_before >= floor:
        return "redundant"
    if captured_soon:
        return None
    need = max(0.0, floor - inbound_before)
    if inbound_before + float(sent) >= floor:
        return "sufficient"
    owned_garrisons = [
        float(p[5])
        for p in planets
        if int(p[1]) == seat and int(p[0]) != int(tgt[0]) and float(p[5]) > 0
    ]
    max_single = max(owned_garrisons, default=0.0)
    total_owned = sum(owned_garrisons)
    if float(src[5]) >= need:
        return "single"
    if max_single >= need:
        return "wrong-src"
    if total_owned >= need:
        return "aggregate"
    return "unafford"


def _reachable_drainable_attack_mass(planets, fleets, tgt, seat, window, beta=None):
    """Reachable, drainable owned garrison for a same-turn aggregate attack on `tgt`.

    This is a stricter follow-up to the raw `aggregate` bucket. Raw aggregate only says total
    owned garrison can cover the missing floor; this asks whether mass is plausibly usable:
    source spare must be drainable under its own inbound threat, and it must arrive by `window`
    steps. It intentionally stays diagnostic-only; it is an estimate, not an action generator.
    """
    if beta is None:
        beta = _DM_BETA_EVAL
    enemy_in_src: dict = {}
    for f in (fleets or []):
        o = int(f[1])
        if o < 0 or o == seat:
            continue
        r = _dm_fleet_target(planets, f)
        if r is not None:
            enemy_in_src[r[0]] = enemy_in_src.get(r[0], 0.0) + f[6]
    avail = 0.0
    source_count = 0
    for src in planets:
        if int(src[1]) != seat or int(src[0]) == int(tgt[0]) or src[5] <= 0:
            continue
        g = float(src[5])
        own_threat = enemy_in_src.get(src[0], 0.0)
        if own_threat >= g:
            spare = g                                            # doomed source: recycle fully
        elif own_threat > 0:
            hold_floor = own_threat + beta * _reachable_enemy_mass(planets, src, seat) + _DM_OVERHEAD
            spare = max(0.0, g - hold_floor)
        else:
            spare = g                                            # no active inbound threat → all garrison is drainable
        if spare <= 0:
            continue
        dist = math.hypot(tgt[2] - src[2], tgt[3] - src[3])
        if dist / max(_ship_speed_py(spare), 1e-6) <= window:
            avail += spare
            source_count += 1
    return avail, source_count


def _friendly_inbound(fleets, tgt, seat):
    """Own (seat) ships in flight already HEADED toward planet `tgt` — same geometry the
    friendly-contest feature reads (along>0, perp < radius+1.5). Used to flag a *redundant*
    attack-launch: firing at a target a friendly fleet is already capturing. Decision-time
    obs (fleets@t-1) naturally excludes the launch being made this step."""
    if not fleets:
        return 0.0
    tx, ty, tr = tgt[2], tgt[3], tgt[4]
    s = 0.0
    for f in fleets:
        if int(f[1]) != seat:
            continue
        c, sn = math.cos(f[4]), math.sin(f[4])
        vx, vy = tx - f[2], ty - f[3]
        along = vx * c + vy * sn
        perp = abs(vx * sn - vy * c)
        if along > 0 and perp < tr + 1.5:
            s += f[6]
    return s


def _ship_speed_py(ships):
    """Scalar mirror of torch_env._ship_speed (kaggle speed formula)."""
    s = max(float(ships), 1.0)
    base = (math.log(s) / math.log(1000.0)) ** 1.5
    return min(1.0 + (_DM_MAX_SPEED - 1.0) * base, _DM_MAX_SPEED)


def _dm_fleet_target(planets, f):
    """Planet a fleet `f` is converging on — the lead-aware swept-collision target (min-ETA hit,
    accounting for the planet's orbital motion over the flight). Mirrors torch_env._fleet_target_idx
    so the eval dm_* gap diagnostic and the training reward attribute the SAME target; NOT
    _friendly_inbound (which sums to every roughly-aligned target). Returns planet record or None."""
    return _lead_collision_target(planets, f[2], f[3], f[4], f[6])


def _decisive_gap_step(planets, fleets, seat, beta=None, targets="enemy"):
    """Per attacked-enemy-target decisive-mass RATIO (mass/floor) for `seat` from one observation,
    using the EXACT floor of torch_env._decisive_mass_fields:
        floor = garr + prod*eta + enemy_inbound + beta*rho(eta)*reachable_enemy_mass + overhead
    (eta = MAX arrival ETA of our converging mass). `beta` defaults to _DM_BETA_EVAL (= the training
    default 2.2; pass explicitly to match a non-default-beta run). Returns a list of ratios, one per
    enemy planet we have inflight mass converging on. cross/gap/near-miss/overkill all derive from
    the ratio (cross = ratio>=1, gap = max(0,1-ratio)), so the eval read can't drift from the reward."""
    if beta is None:
        beta = _DM_BETA_EVAL
    if not planets:
        return []
    our_mass: dict = {}
    our_eta: dict = {}
    enemy_inbound: dict = {}
    for f in (fleets or []):
        own = int(f[1])
        if own < 0:
            continue
        tgt = _dm_fleet_target(planets, f)
        if tgt is None:
            continue
        tid = tgt[0]
        if own == seat:
            our_mass[tid] = our_mass.get(tid, 0.0) + f[6]
            dist = math.hypot(tgt[2] - f[2], tgt[3] - f[3])
            eta = min(dist / max(_ship_speed_py(f[6]), 1e-6), _DM_HORIZON)
            our_eta[tid] = max(our_eta.get(tid, 0.0), eta)             # MAX ETA over our fleets
        else:
            enemy_inbound[tid] = enemy_inbound.get(tid, 0.0) + f[6]
    if targets == "neutral":
        # NEUTRAL capture sufficiency: neutrals don't grow or reinforce, so the floor is just the
        # static garrison (+overhead) — the "did we send enough to TAKE this neutral" gap the enemy
        # dm line (owner>=0) skips. This is the opening land-grab undercommit (e.g. 16 ships at g25).
        ratios = []
        for tgt in planets:
            if int(tgt[1]) != -1:
                continue                                              # neutrals only
            mass = our_mass.get(tgt[0], 0.0)
            if mass <= 0:
                continue                                              # attacked-with-mass only
            floor = max(tgt[5] + _DM_OVERHEAD, 1e-6)                  # static garrison; no prod growth / no reactive term
            ratios.append(mass / floor)
        return ratios
    enemy_planets = [p for p in planets if int(p[1]) != seat and int(p[1]) >= 0]
    ratios = []
    for tgt in enemy_planets:
        tid = tgt[0]
        mass = our_mass.get(tid, 0.0)
        if mass <= 0:
            continue                                                  # attacked-with-mass only
        eta = our_eta.get(tid, 0.0)
        inbound = enemy_inbound.get(tid, 0.0)
        em = 0.0                                                       # reachable enemy PLANET mass
        for src in enemy_planets:
            if src[0] == tid:
                continue
            pd = math.hypot(src[2] - tgt[2], src[3] - tgt[3])
            reach = max(_ship_speed_py(src[5]) * _DM_HORIZON, 1e-6)
            em += src[5] * max(1.0 - pd / reach, 0.0)
        rho = min(max((eta - _DM_ETA_FREE) / _DM_ETA_SCALE, 0.0), 1.0)
        floor = max(tgt[5] + tgt[6] * eta + inbound + beta * rho * em + _DM_OVERHEAD, 1e-6)
        ratios.append(mass / floor)
    return ratios


def _hold_floor_step(planets, fleets, seat, cap_step, t, beta=None):
    """Defensive mirror of _decisive_gap_step: for each OWN planet under an actual inbound threat
    (enemy fleet converging on it — the same condition as the hold-loss 'out-massed' autopsy), the
    HOLD ratio = can we keep it?
        hold = (our_garrison + friendly_inbound) / (enemy_inbound + beta*reachable_enemy_mass + 1)
    ratio < 1 = under-defended (will likely be peeled). `beta`/overhead reuse the decisive-mass
    constants; reachable_enemy_mass = cheap_enemy_pressure (the forward-projected reinforcement
    behind the inbound — the out-massing mechanism). Returns [(ratio, age_after_capture)] per
    threatened own planet; age is None for home/initial planets (never captured → not in cap_step).
    Duration-weighted: a planet threatened N steps contributes N observations (like dm target-steps)."""
    if beta is None:
        beta = _DM_BETA_EVAL
    if not planets:
        return []
    our_in: dict = {}
    enemy_in: dict = {}
    for f in (fleets or []):
        own = int(f[1])
        if own < 0:
            continue
        tgt = _dm_fleet_target(planets, f)
        if tgt is None:
            continue
        tid = tgt[0]
        if own == seat:
            our_in[tid] = our_in.get(tid, 0.0) + f[6]
        else:
            enemy_in[tid] = enemy_in.get(tid, 0.0) + f[6]
    enemy_planets = [p for p in planets if int(p[1]) != seat and int(p[1]) >= 0]
    out = []
    for p in planets:
        if int(p[1]) != seat:
            continue                                                  # OUR planets only
        ein = enemy_in.get(p[0], 0.0)
        if ein <= 0:
            continue                                                  # threatened (inbound) only
        em = 0.0                                                       # reachable enemy PLANET mass
        for src in enemy_planets:
            pd = math.hypot(src[2] - p[2], src[3] - p[3])
            reach = max(_ship_speed_py(src[5]) * _DM_HORIZON, 1e-6)
            em += src[5] * max(1.0 - pd / reach, 0.0)
        floor = max(ein + beta * em + _DM_OVERHEAD, 1e-6)
        ratio = (p[5] + our_in.get(p[0], 0.0)) / floor
        age = (t - cap_step[p[0]]) if p[0] in cap_step else None
        out.append((ratio, age))
    return out


def _enemy_threat(planets, fleets, tgt, seat):
    """(enemy_inbound, threat_eta) on `tgt`: total enemy fleet ships converging on it + the SOONEST
    enemy arrival ETA (the window we must get defenders in by). threat_eta = +inf if none inbound."""
    ein = 0.0
    eta = math.inf
    for f in (fleets or []):
        o = int(f[1])
        if o < 0 or o == seat:
            continue
        r = _dm_fleet_target(planets, f)
        if r is None or r[0] != tgt[0]:
            continue
        ein += f[6]
        d = math.hypot(tgt[2] - f[2], tgt[3] - f[3])
        eta = min(eta, d / max(_ship_speed_py(f[6]), 1e-6))
    return ein, eta


def _reachable_enemy_mass(planets, tgt, seat):
    """cheap_enemy_pressure on `tgt`: reachable ENEMY planet mass (forward-projected reinforcement
    behind the inbound — the out-massing mechanism), same decay as the decisive-mass floor."""
    em = 0.0
    for src in planets:
        o = int(src[1])
        if o != seat and o >= 0 and src[0] != tgt[0]:
            pd = math.hypot(src[2] - tgt[2], src[3] - tgt[3])
            reach = max(_ship_speed_py(src[5]) * _DM_HORIZON, 1e-6)
            em += src[5] * max(1.0 - pd / reach, 0.0)
    return em


def _reachable_friendly_mass(planets, fleets, tgt, seat, threat_eta):
    """FAITHFUL (safe_drain-like) friendly mass that can defend `tgt` BEFORE the threat arrives: our
    fleets already inbound that ARRIVE in time + the SPARE garrison of owned sources that can reach it
    in time. Spare = full garrison for a DOOMED source (its own inbound ≥ its garrison → nothing to
    hold, recycle it all — orbit_lite.safe_drain's doomed-source rule), else garrison − own inbound
    (shed only what it can spare and still hold). This is the chosen FORK-#2 (a) faithful version —
    it answers 'could we have spared/recycled nearby mass?', not just 'was help already in flight?'."""
    if not math.isfinite(threat_eta):
        threat_eta = float(_DM_HORIZON)
    avail = 0.0
    enemy_in_src: dict = {}
    for f in (fleets or []):
        o = int(f[1])
        if o < 0:
            continue
        r = _dm_fleet_target(planets, f)
        if r is None:
            continue
        if o == seat and r[0] == tgt[0]:
            d = math.hypot(tgt[2] - f[2], tgt[3] - f[3])
            if d / max(_ship_speed_py(f[6]), 1e-6) <= threat_eta:
                avail += f[6]                                          # our reinforcement arriving in time
        elif o != seat:
            enemy_in_src[r[0]] = enemy_in_src.get(r[0], 0.0) + f[6]
    for src in planets:
        if int(src[1]) != seat or src[0] == tgt[0]:
            continue
        g = src[5]
        own_threat = enemy_in_src.get(src[0], 0.0)
        spare = g if own_threat >= g else (g - own_threat)            # doomed → drain fully
        if spare <= 0:
            continue
        d = math.hypot(src[2] - tgt[2], src[3] - tgt[3])
        if d / max(_ship_speed_py(spare), 1e-6) <= threat_eta:        # reachable BEFORE the threat
            avail += spare
    return avail


def _threat_class(planets, fleets, tgt, seat, beta=None):
    """Triage class of a threatened OWN planet `tgt`, mirroring producer_v2's 'can I profitably hold
    this, and can a neighbour spare the ships?' (safe_drain + roi gate):
        defense_cost = enemy_inbound + beta*reachable_enemy_mass + 1 − garrison   (hold-floor shortfall)
        friendly_available = _reachable_friendly_mass (faithful, ETA-aware)
        value = production * horizon   (ship-equivalent of the production stream we keep — the unit
                competitive_score works in; current garrison is sunk, not counted)
        save_ratio = defense_cost / max(value, 1)
    Returns 0 already-safe (cost<=0) / 1 cheap-save (savable, ratio<1) / 2 expensive-save (savable,
    ratio>=1) / 3 hopeless (friendly_available < defense_cost → planner abandons + recycles mass)."""
    if beta is None:
        beta = _DM_BETA_EVAL
    enemy_in, threat_eta = _enemy_threat(planets, fleets, tgt, seat)
    defense_cost = enemy_in + beta * _reachable_enemy_mass(planets, tgt, seat) + _DM_OVERHEAD - tgt[5]
    if defense_cost <= 0:
        return 0                                                      # already safe
    if _reachable_friendly_mass(planets, fleets, tgt, seat, threat_eta) < defense_cost:
        return 3                                                      # hopeless → recycle the mass
    value = max(tgt[6] * _DM_HORIZON, 1.0)
    return 1 if (defense_cost / value) < 1.0 else 2                   # cheap vs expensive save


def _reinf_reciprocity(reinf_edges):
    """Reinforce PING-PONG detector. `reinf_edges` = [(step, src_pid, tgt_pid), ...] of own-target
    reinforce launches. Returns (recip, recip3_ph):
      recip      = [r1, r2, r3] count of reinforces whose NEAREST reverse edge (tgt->src) lands
                   within 1 / 2 / 3 steps after it (cumulative).
      recip3_ph  = the within-3 count split by the FORWARD edge's phase (<50 / 50-100 / >=100).
    An A->B reinforce then a B->A reinforce a few turns later = ships oscillating between two own
    planets (pure waste). TEMPORAL loop (arrival, then reverse) — a same-turn role mutex misses it.
    Rank1 winners: reciprocal-within-3 is <1%; an observed ~43% is a real pathology."""
    from bisect import bisect_right
    recip = [0, 0, 0]
    recip3_ph = [0, 0, 0]
    if not reinf_edges:
        return recip, recip3_ph
    by_edge: dict = {}
    for (tt, a, b) in reinf_edges:
        by_edge.setdefault((a, b), []).append(tt)
    for lst in by_edge.values():
        lst.sort()
    for (tt, a, b) in reinf_edges:
        rev = by_edge.get((b, a))
        if not rev:
            continue
        j = bisect_right(rev, tt)                   # nearest reverse strictly after this launch
        if j < len(rev):
            d = rev[j] - tt
            if d <= 3:
                recip[2] += 1
                ph = 0 if tt < _LAUNCH_WINDOW else (1 if tt < _MID_WINDOW else 2)
                recip3_ph[ph] += 1
            if d <= 2: recip[1] += 1
            if d <= 1: recip[0] += 1
    return recip, recip3_ph


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
    caps = atk = reinf = atk_ships = redundant = underkill = 0
    atk_early = redundant_early = underkill_early = caps_early = 0   # opening window (t < _LAUNCH_WINDOW)
    atk_mid = caps_mid = 0                                           # mid-game window [50, 100)
    reinf_early = reinf_mid = 0                                      # reinforce launches by step-window
    reinf_fwd = reinf_rear = reinf_dirn = 0                          # reinforce direction (vs enemy centroid)
    reinf_edges: list = []   # (step, src_pid, tgt_pid) per reinforce launch → reciprocal ping-pong below
    # Phase-structured reinforce diagnostics (denom = reinf_n_ph per phase). Top-player mining shows
    # early reinforces are forward/outward but mid/late logistics are freer → LOG, don't hard-mask:
    #   fwde  = target closer to the NEAREST ENEMY planet than source (forward-to-enemy)
    #   cout  = target FURTHER from our OWN-empire centroid than source (mass pushed outward)
    #   ftop3 = target is one of our 3 frontline (closest-to-enemy) owned planets
    reinf_n_ph = [0, 0, 0]; reinf_fwde_ph = [0, 0, 0]; reinf_cout_ph = [0, 0, 0]; reinf_ftop3_ph = [0, 0, 0]
    # forward-to-enemy by reinforce SIZE (rank1: <=20 ships fwd 42% vs 51-100 fwd 71% — big sends move outward)
    reinf_n_sz = [0, 0, 0]; reinf_fwde_sz = [0, 0, 0]   # ship-size buckets: <=20 / 21-50 / 51+
    # Retention: of the planets we CAPTURE, how many do we then lose, and how long did we hold
    # them? cap_step[pid] = step we (most recently) took pid; on a later loss we close the episode.
    # lost_caps/captures is the recapture/turnover rate — immune to the end->0 churn degeneracy.
    # Home/initial planets are excluded by construction (never entered cap_step).
    cap_step: dict = {}
    cap_class: dict = {}         # triage class at capture time (capture-born doomed read)
    lost_caps = 0
    hold_durations: list = []   # steps held before losing (lost episodes only; held-to-end censored)
    # lost-capture AUTOPSY (why do held planets fall?): measured at the step of loss from the t-1
    # state, reusing _friendly_inbound geometry. mode = [abandoned, out-massed, too-late, other]:
    #   abandoned  = we left <=2 ships (captured and moved the army on)
    #   out-massed = garrison>2 but enemy inbound fleet > our garrison (under-massed vs the threat)
    #   too-late   = we had reinforcement inbound but not enough/in time (reactive)
    cap_garr: dict = {}          # ships on a planet right after we captured it (holding-surplus read)
    loss_mode = [0, 0, 0, 0]
    loss_garr_cap: list = []     # garrison at capture, per lost planet
    loss_garr_at: list = []      # garrison just before loss
    loss_enemy_in: list = []     # enemy ships inbound to it just before loss
    # REINFORCE TRIAGE / save-efficiency (the "are we reinforcing the wrong planets?" read):
    #   reinf_into[pid] = reinforce ships we've poured into a currently-held planet; on LOSS the tally
    #   goes to to_lost (good ships after bad), held-to-end → to_saved. reinf_class_ships buckets each
    #   reinforce launch by the TARGET's hold-triage class AT DECISION TIME (safe/cheap/expensive/
    #   hopeless — see _reinforce_target_class). Confirms the hunch if to_lost% and hopeless% are high.
    reinf_into: dict = {}
    reinf_to_lost = 0.0
    reinf_class_ships = [0.0, 0.0, 0.0, 0.0]   # reinforce ships by TARGET class at launch [safe,cheap,exp,hopeless]
    lost_by_class = [0, 0, 0, 0]               # planets we LOST, by their threat class at loss-time
    lost_by_class_cap_nt = [0, 0, 0, 0]        # captured planet losses only, excluding terminal collapse
    hopeless_lost_reinforced = 0               # hopeless losses we'd poured ships into (good-after-bad)
    hopeless_lost_abandoned = 0                # hopeless losses we correctly let go (recycled the mass)
    hopeless_lost_reinf_ships = 0.0            # ship-weighted version of hopeless_lost_reinforced
    reinf_events_by_pid: dict = {}             # pid -> [(launch_step, ships, class)] for loss-window reads
    reinf_lost_within = [0.0, 0.0, 0.0]        # reinforce ships whose target is lost within 10/20/30 steps
    safe_reinf_events: list = []               # (launch_step, target_pid, ships) for future utility read
    safe_reinf_util = [0.0, 0.0, 0.0]          # safe-class reinforce ships -> [later attack, frontline, idle]
    cap_born_class = [0, 0, 0, 0]              # captures by triage class immediately after capture
    cap_born_lost_nt = [0, 0, 0, 0]            # nonterminal captured losses, bucketed by birth class
    reinf_bin = [0] * len(_REINF_BINS)   # own-target launches by empire size at launch
    atk_bin = [0] * len(_REINF_BINS)     # attack launches by empire size at launch
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
    # attack target NEAR-vs-FAR (the "target head reaches for far planets when a nearer one is
    # available" question): per attack launch, compare the chosen target's distance-from-source to
    # the NEAREST legal attack target (any non-seat planet). nearest = chose the closest; ratio =
    # chosen_dist / nearest_dist (1.0 = nearest). Current-position approx (ignores orbital motion),
    # but the won/lost split controls for that — the approx applies equally to both buckets, so a
    # won-vs-lost gap is real: bigger far-reach in LOST games => losing-state artifact, not a head defect.
    atkfar_n_ph = [0, 0, 0]
    atkfar_nearest_ph = [0, 0, 0]
    atkfar_ratio_sum_ph = [0.0, 0.0, 0.0]
    # value-aware DOMINANCE: distance alone over-flags value-driven far-targeting (a far planet with
    # higher production is a correct pick). dominated = a strictly BETTER target was passed up — one
    # both CLOSER and at least as PRODUCTIVE as the chosen (p[6] = production). Far-but-richer is NOT
    # flagged; near-and-richer-passed-up IS → the unambiguous target-head defect rate (no value excuse).
    atkdom_ph = [0, 0, 0]
    # value-aware HOLDABLE-ROI rank: dom% above ranks by RAW production, so it over-flags a
    # closer-richer-but-UNHOLDABLE target the head correctly skipped. Here we rank candidates by
    # reactive-aware holdable ROI (prices the peel via the decisive-mass floor). best = chose the
    # top holdable-ROI target; top3 = within the top 3. LOST≫WON => value-aware targeting IS the
    # lever; WON≈LOST => systematic but not outcome-relevant (selection isn't the wall).
    atkh_n_ph = [0, 0, 0]
    atkh_best_ph = [0, 0, 0]
    atkh_top3_ph = [0, 0, 0]
    # Failed-attack decomposition: if an attack launch does not lead to ownership soon after ETA,
    # classify the implied fix. This separates per-source sizing from wrong-source, aggregate
    # coordination, unaffordable target, and "sent enough but still failed" cases.
    atk_n_ph = [0, 0, 0]
    atkfail_n_ph = [0, 0, 0]
    atkfail_bucket_ph = [[0, 0, 0] for _ in _ATK_FAIL_LABELS]
    atkagg_reach_ph = [0, 0, 0]
    planets_at = {ms: None for ms in _CONV_MILESTONES}
    garrison_at = {ms: None for ms in _CONV_MILESTONES}   # ships parked on owned planets
    inflight_at = {ms: None for ms in _CONV_MILESTONES}   # ships in owned fleets (deployed)
    # decisive-mass GAP: ratios (our inflight mass / producer_v2 capture floor) per attacked enemy
    # target, split by phase (<50/50-100/>=100). Survives-argmax read of the decmass target — does
    # the policy assemble force to the capture floor vs only improving adjacent competence?
    dm_ratios_ph = [[], [], []]
    dm_neutral_ratios_ph = [[], [], []]   # NEUTRAL capture sufficiency (mass / static garrison) by phase
    # hold-floor (DEFENSIVE mirror): hold ratio per threatened OWN planet, split by phase and by
    # age-after-capture (0-5 / 6-15 / 16+ steps). Tells whether we lose captures IMMEDIATELY (can't
    # route defense in time) or LATER (logistics/churn). Uses cap_step for age.
    hold_ph = [[], [], []]
    hold_age = [[], [], []]
    # Pre-scan ownership timeline (keyed by GLOBAL planet id → no slot-reorder issue) so a launch
    # can look FORWARD: did its target actually become ours shortly after arrival? Used for the
    # forward-looking underkill (a per-launch threshold mis-flags legit multi-wave as underkill).
    T = len(steps)
    us_pids_at = [set() for _ in range(T)]
    for s in range(T):
        if seat < len(steps[s]):
            ps = steps[s][seat].observation.get("planets")
            if ps:
                us_pids_at[s] = {p[0] for p in ps if int(p[1]) == seat}
    prev = {}
    last = None
    for t in range(1, len(steps)):
        if seat >= len(steps[t]) or seat >= len(steps[t - 1]):
            continue
        p0 = steps[t - 1][seat].observation.get("planets")
        p1 = steps[t][seat].observation.get("planets")
        acts = steps[t][seat].action or []
        if p1:
            terminal_loss_step = not any(int(q[1]) == seat for q in p1)
            owned_now = garrison_now = 0
            for p in p1:
                pid, own = p[0], int(p[1])
                if own == seat:
                    owned_now += 1
                    garrison_now += p[5]
                was = prev.get(pid)
                _captured_loss = False
                _born_cls = None
                if was is not None and was != seat and own == seat:
                    caps += 1
                    if t < _LAUNCH_WINDOW:
                        caps_early += 1                # opening captures (for opening cap/atk-launch)
                    elif t < _MID_WINDOW:
                        caps_mid += 1                  # mid-game (50-100) captures
                    cap_step[pid] = t                  # open a hold episode
                    cap_garr[pid] = p[5]               # garrison right after capture
                    _f1cap = steps[t][seat].observation.get("fleets") or []
                    _born_cls = _threat_class(p1, _f1cap, p, seat)
                    cap_class[pid] = _born_cls
                    cap_born_class[_born_cls] += 1
                elif was == seat and own != seat and pid in cap_step:
                    _captured_loss = True
                    _born_cls = cap_class.get(pid)
                    hold_durations.append(t - cap_step[pid])   # lost what we took
                    lost_caps += 1
                    # AUTOPSY: why? use t-1 state (just before the flip).
                    tgt0 = next((q for q in p0 if q[0] == pid), None) if p0 else None
                    if tgt0 is not None:
                        f0a = steps[t - 1][seat].observation.get("fleets") or []
                        g_loss = tgt0[5]
                        e_in = _friendly_inbound(f0a, tgt0, 1 - seat)
                        loss_garr_cap.append(cap_garr.get(pid, 0))
                        loss_garr_at.append(g_loss)
                        loss_enemy_in.append(e_in)
                        if g_loss <= 2:
                            loss_mode[0] += 1                          # ABANDONED
                        elif e_in > g_loss:
                            loss_mode[1] += 1                          # OUT-MASSED
                        elif _friendly_inbound(f0a, tgt0, seat) > 0:
                            loss_mode[2] += 1                          # TOO-LATE
                        else:
                            loss_mode[3] += 1                          # OTHER
                    del cap_step[pid]
                    cap_class.pop(pid, None)
                # TRIAGE on LOSS (any owned planet, incl. home): classify it at the loss decision
                # (t-1) → was it cheaply savable (a MISS) or hopeless (correctly let go)? + route the
                # reinforce ships we'd poured into it to "to_lost", and for hopeless losses record
                # whether we wasted ships on it (reinforced) or abandoned it (recycled the mass).
                if was == seat and own != seat:
                    _t0 = next((q for q in p0 if q[0] == pid), None) if p0 else None
                    if _t0 is not None:
                        _f0loss = steps[t - 1][seat].observation.get("fleets") or []
                        _cls = _threat_class(p0, _f0loss, _t0, seat)
                        lost_by_class[_cls] += 1
                        if _captured_loss and not terminal_loss_step:
                            lost_by_class_cap_nt[_cls] += 1
                            if _born_cls is not None:
                                cap_born_lost_nt[_born_cls] += 1
                        if _cls == 3:
                            _lost_reinf_ships = reinf_into.get(pid, 0.0)
                            if _lost_reinf_ships > 0:
                                hopeless_lost_reinforced += 1
                                hopeless_lost_reinf_ships += _lost_reinf_ships
                            else:
                                hopeless_lost_abandoned += 1
                    if pid in reinf_into:
                        reinf_to_lost += reinf_into.pop(pid)
                    if pid in reinf_events_by_pid:
                        for _rt, _rs, _rcls in reinf_events_by_pid.pop(pid):
                            _age = t - _rt
                            for _wi, _ww in enumerate(_TRIAGE_LOSS_WINDOWS):
                                if _age <= _ww:
                                    reinf_lost_within[_wi] += _rs
                prev[pid] = own
            last = p1
            if t in planets_at:
                fleets = steps[t][seat].observation.get("fleets") or []
                planets_at[t] = owned_now
                garrison_at[t] = garrison_now
                inflight_at[t] = sum(f[6] for f in fleets if int(f[1]) == seat)
            # decisive-mass gap on the post-step state (mirrors training's per-step measurement)
            _f_dm = steps[t][seat].observation.get("fleets") or []
            _ph_dm = 0 if t < _LAUNCH_WINDOW else (1 if t < _MID_WINDOW else 2)
            dm_ratios_ph[_ph_dm].extend(_decisive_gap_step(p1, _f_dm, seat))
            dm_neutral_ratios_ph[_ph_dm].extend(_decisive_gap_step(p1, _f_dm, seat, targets="neutral"))
            # hold-floor (defensive): cap_step here reflects current holdings (capture/loss for step t
            # processed above), so age-after-capture is correct. Bucket by phase + by age.
            for _hr, _age in _hold_floor_step(p1, _f_dm, seat, cap_step, t):
                hold_ph[_ph_dm].append(_hr)
                if _age is not None:
                    _ab = 0 if _age <= 5 else (1 if _age <= 15 else 2)
                    hold_age[_ab].append(_hr)
        if not p0:
            continue
        byid = {p[0]: p for p in p0}
        f0 = steps[t - 1][seat].observation.get("fleets") or []   # in-flight at decision time
        owned_dec = sum(1 for p in p0 if int(p[1]) == seat)  # empire size at decision
        bidx = _reinf_bin_idx(owned_dec)
        launch_states += owned_dec
        fired_this_step = 0
        # enemy centroid for reinforce-direction (forward-staging) — enemy = other players' planets
        _enemy = [p for p in p0 if int(p[1]) != seat and int(p[1]) >= 0]
        _ecx = sum(p[2] for p in _enemy) / len(_enemy) if _enemy else None
        _ecy = sum(p[3] for p in _enemy) / len(_enemy) if _enemy else None
        # own-empire centroid (centroid-outward ref) + frontline top-3 (closest-to-enemy owned)
        _own = [p for p in p0 if int(p[1]) == seat]
        _ocx = sum(p[2] for p in _own) / len(_own) if _own else None
        _ocy = sum(p[3] for p in _own) / len(_own) if _own else None
        _front3: set = set()
        if _own and _enemy:
            _de = lambda q: min(math.hypot(q[2] - e[2], q[3] - e[3]) for e in _enemy)
            _front3 = {p[0] for p in sorted(_own, key=_de)[:3]}
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
                reinf_bin[bidx] += 1
                reinf_edges.append((t, int(src[0]), int(tgt[0])))   # for reciprocal ping-pong
                # TRIAGE: tally these ships against the target planet (resolved to_lost/to_saved on
                # outcome) + bucket by the target's hold-triage class at decision time (obs t-1).
                reinf_into[tgt[0]] = reinf_into.get(tgt[0], 0.0) + sent
                _rcls = _threat_class(p0, f0, tgt, seat)
                reinf_class_ships[_rcls] += sent
                reinf_events_by_pid.setdefault(tgt[0], []).append((t, float(sent), _rcls))
                if _rcls == 0:
                    safe_reinf_events.append((t, int(tgt[0]), float(sent)))
                if _ecx is not None:               # direction: target vs source distance to enemy
                    dS = math.hypot(src[2] - _ecx, src[3] - _ecy)
                    dT = math.hypot(tgt[2] - _ecx, tgt[3] - _ecy)
                    reinf_dirn += 1
                    if dT < dS - 3: reinf_fwd += 1      # toward enemy (forward-staging)
                    elif dT > dS + 3: reinf_rear += 1   # away from enemy (rear-defense)
                # phase-structured: forward-to-nearest-enemy, centroid-outward, frontline-top3
                reinf_n_ph[_ph] += 1
                if _enemy:
                    deS = min(math.hypot(src[2] - e[2], src[3] - e[3]) for e in _enemy)
                    deT = min(math.hypot(tgt[2] - e[2], tgt[3] - e[3]) for e in _enemy)
                    sz = 0 if sent <= 20 else (1 if sent <= 50 else 2)
                    reinf_n_sz[sz] += 1
                    if deT < deS:
                        reinf_fwde_ph[_ph] += 1
                        reinf_fwde_sz[sz] += 1
                if _ocx is not None and math.hypot(tgt[2] - _ocx, tgt[3] - _ocy) > math.hypot(src[2] - _ocx, src[3] - _ocy):
                    reinf_cout_ph[_ph] += 1            # mass pushed outward from own centroid
                if int(tgt[0]) in _front3:
                    reinf_ftop3_ph[_ph] += 1
                if t < _LAUNCH_WINDOW:
                    reinf_early += 1                # reinforce by step-window (when does it matter?)
                elif t < _MID_WINDOW:
                    reinf_mid += 1
            else:
                atk += 1
                atk_ships += sent
                atk_bin[bidx] += 1
                atk_n_ph[_ph] += 1
                # near-vs-far: nearest legal attack target (any non-seat planet) from this source
                _cand = [q for q in p0 if int(q[1]) != seat]
                if _cand:
                    _d_near = min(math.hypot(q[2] - src[2], q[3] - src[3]) for q in _cand)
                    if _d_near > 1e-6:
                        _d_chosen = math.hypot(tgt[2] - src[2], tgt[3] - src[3])
                        atkfar_n_ph[_ph] += 1
                        atkfar_ratio_sum_ph[_ph] += _d_chosen / _d_near
                        if _d_chosen <= _d_near + 1e-6:
                            atkfar_nearest_ph[_ph] += 1
                        _p_chosen = tgt[6]
                        if any(int(q[0]) != int(tgt[0])
                               and math.hypot(q[2] - src[2], q[3] - src[3]) < _d_chosen - 1e-6
                               and q[6] >= _p_chosen for q in _cand):
                            atkdom_ph[_ph] += 1            # passed up a closer-AND-richer target
                        # holdable-ROI rank (reactive-aware): how many candidates beat the chosen?
                        if len(_cand) >= 2:
                            _hr_chosen = _holdable_roi(src, tgt, p0, f0, seat)
                            if _hr_chosen is not None:
                                _better = 0
                                for q in _cand:
                                    if int(q[0]) == int(tgt[0]):
                                        continue
                                    _hr_q = _holdable_roi(src, q, p0, f0, seat)
                                    if _hr_q is not None and _hr_q > _hr_chosen + 1e-9:
                                        _better += 1
                                atkh_n_ph[_ph] += 1
                                if _better == 0:
                                    atkh_best_ph[_ph] += 1   # chose the top holdable-ROI target
                                if _better < 3:
                                    atkh_top3_ph[_ph] += 1   # chosen within top-3 holdable-ROI
                early = t < _LAUNCH_WINDOW
                if early:
                    atk_early += 1
                elif t < _MID_WINDOW:
                    atk_mid += 1                       # mid-game (50-100) attack launches
                # Launch-waste trichotomy:
                #   redundant (OVERKILL) = target was ALREADY covered to capture by own fleets
                #     inbound BEFORE this launch (friendly_inbound >= cap_cost_at_arrival, the
                #     SAME quantity the roi-deflation zeroes) → pure surplus.
                #   underkill (INEFFECTIVE) = FORWARD-looking: the target never becomes ours
                #     within ~eta+10 steps of the launch → the ships didn't lead to a capture
                #     (the seed1030 18-at-23 lone-undercommit case). A per-launch threshold
                #     mis-flags legit multi-wave (each wave < cost) — forward-looking doesn't,
                #     since a target taken by a later wave reads as captured for all waves.
                #   (neither = an effective launch.)
                fin = _friendly_inbound(f0, tgt, seat)
                capcost = _cap_cost_at_arrival(src, tgt, seat)
                eta = max(1, int(math.ceil(
                    math.hypot(tgt[2] - src[2], tgt[3] - src[3]) / _ETA_PROBE_SPEED)))
                pid = tgt[0]
                captured_soon = any(pid in us_pids_at[s] for s in range(t + 1, min(t + eta + 11, T)))
                fail_bucket = _attack_failure_bucket(p0, f0, src, tgt, seat, sent, captured_soon)
                if fail_bucket is not None:
                    atkfail_n_ph[_ph] += 1
                    atkfail_bucket_ph[_ATK_FAIL_LABELS.index(fail_bucket)][_ph] += 1
                    if fail_bucket == "aggregate":
                        floor = _attack_capture_floor(src, tgt, p0, f0, seat)
                        need = max(0.0, floor - _friendly_inbound(f0, tgt, seat))
                        avail, _ = _reachable_drainable_attack_mass(p0, f0, tgt, seat, eta + 10)
                        if avail >= need:
                            atkagg_reach_ph[_ph] += 1
                if fin >= capcost > 0:
                    redundant += 1
                    if early:
                        redundant_early += 1
                else:
                    if not captured_soon:
                        underkill += 1
                        if early:
                            underkill_early += 1
        if fired_this_step > 0 and owned_dec > 0:
            fire_steps += 1
            fire_frac_sum += fired_this_step / owned_dec
        launch_count += fired_this_step
    for _rt, _pid, _ships in safe_reinf_events:
        _attacked = False
        _frontline = False
        for _s in range(_rt + 1, min(T, _rt + _SAFE_REINF_LOOKAHEAD + 1)):
            if seat >= len(steps[_s]):
                continue
            _src_prev = None
            if _s > 0 and seat < len(steps[_s - 1]):
                _ps_prev = steps[_s - 1][seat].observation.get("planets") or []
                _src_prev = next((q for q in _ps_prev if q[0] == _pid and int(q[1]) == seat), None)
            if _src_prev is not None:
                for _mv in (steps[_s][seat].action or []):
                    if not _mv or len(_mv) < 3 or int(_mv[0]) != _pid:
                        continue
                    _tgt_prev = _resolve_launch_target(_ps_prev, _src_prev, float(_mv[1]), int(_mv[2]))
                    if _tgt_prev is not None and int(_tgt_prev[1]) != seat:
                        _attacked = True
                        break
            if _attacked:
                break
            _ps_now = steps[_s][seat].observation.get("planets") or []
            _now = next((q for q in _ps_now if q[0] == _pid), None)
            if _now is None or int(_now[1]) != seat:
                break
            _enemy_now = [q for q in _ps_now if int(q[1]) != seat and int(q[1]) >= 0]
            _own_now = [q for q in _ps_now if int(q[1]) == seat]
            if _enemy_now and _own_now:
                _de_now = lambda q: min(math.hypot(q[2] - e[2], q[3] - e[3]) for e in _enemy_now)
                if _pid in {q[0] for q in sorted(_own_now, key=_de_now)[:3]}:
                    _frontline = True
        safe_reinf_util[0 if _attacked else (1 if _frontline else 2)] += _ships
    end_planets = sum(1 for p in (last or []) if int(p[1]) == seat)
    reinf_to_saved = float(sum(reinf_into.values()))   # reinforce ships on planets still held at end
    reinf_recip, reinf_recip3_ph = _reinf_reciprocity(reinf_edges)
    out = {"captures": caps, "attack_launches": atk, "reinforce_launches": reinf,
           "reinf_recip": reinf_recip, "reinf_recip3_ph": reinf_recip3_ph,
           "reinf_n_ph": reinf_n_ph, "reinf_fwde_ph": reinf_fwde_ph,
           "reinf_cout_ph": reinf_cout_ph, "reinf_ftop3_ph": reinf_ftop3_ph,
           "reinf_n_sz": reinf_n_sz, "reinf_fwde_sz": reinf_fwde_sz,
           "attack_ships": atk_ships, "end_planets": end_planets,
           "redundant": redundant, "underkill": underkill, "atk_early": atk_early,
           "caps_early": caps_early, "atk_mid": atk_mid, "caps_mid": caps_mid,
           "reinf_early": reinf_early, "reinf_mid": reinf_mid,
           "reinf_fwd": reinf_fwd, "reinf_rear": reinf_rear, "reinf_dirn": reinf_dirn,
           "redundant_early": redundant_early, "underkill_early": underkill_early,
           "lost_caps": lost_caps, "hold_durations": hold_durations,
           "loss_mode": loss_mode, "loss_garr_cap": loss_garr_cap,
           "loss_garr_at": loss_garr_at, "loss_enemy_in": loss_enemy_in,
           "glen": len(steps), "reinf_bin": reinf_bin, "atk_bin": atk_bin,
           "launch_states": launch_states, "launch_count": launch_count,
           "fire_steps": fire_steps, "fire_frac_sum": fire_frac_sum,
           "launches_ph": launches_ph, "ship1_ph": ship1_ph, "ship_ph_sum": ship_ph_sum,
           "atkfar_n_ph": atkfar_n_ph, "atkfar_nearest_ph": atkfar_nearest_ph,
           "atkfar_ratio_sum_ph": atkfar_ratio_sum_ph, "atkdom_ph": atkdom_ph,
           "atkh_n_ph": atkh_n_ph, "atkh_best_ph": atkh_best_ph, "atkh_top3_ph": atkh_top3_ph,
           "atk_n_ph": atk_n_ph, "atkfail_n_ph": atkfail_n_ph,
           "atkfail_bucket_ph": atkfail_bucket_ph, "atkagg_reach_ph": atkagg_reach_ph,
           "dm_ratios_ph": dm_ratios_ph, "dm_neutral_ratios_ph": dm_neutral_ratios_ph,
           "hold_ph": hold_ph, "hold_age": hold_age,
           "reinf_class_ships": reinf_class_ships, "reinf_to_lost": reinf_to_lost,
           "reinf_to_saved": reinf_to_saved, "lost_by_class": lost_by_class,
           "lost_by_class_cap_nt": lost_by_class_cap_nt,
           "hopeless_lost_reinforced": hopeless_lost_reinforced,
           "hopeless_lost_abandoned": hopeless_lost_abandoned,
           "hopeless_lost_reinf_ships": hopeless_lost_reinf_ships,
           "reinf_lost_within": reinf_lost_within,
           "safe_reinf_util": safe_reinf_util,
           "cap_born_class": cap_born_class, "cap_born_lost_nt": cap_born_lost_nt}
    for ms in _CONV_MILESTONES:
        out[f"p{ms}"] = planets_at[ms]
        out[f"g{ms}"] = garrison_at[ms]
        out[f"if{ms}"] = inflight_at[ms]
    return out


def new_conversion_acc():
    acc = {"captures": 0, "attack_launches": 0, "reinforce_launches": 0,
           "attack_ships": 0, "end_planets": 0, "redundant": 0, "underkill": 0,
           "glen_sum": 0, "games": 0, "atk_early": 0, "caps_early": 0, "redundant_early": 0, "underkill_early": 0,
           "lost_caps": 0, "hold_durations": [],
           "loss_mode": [0, 0, 0, 0], "loss_garr_cap": [], "loss_garr_at": [], "loss_enemy_in": [],
           "launch_states": 0, "launch_count": 0, "fire_steps": 0, "fire_frac_sum": 0.0,
           # fire-rate split by game outcome — fire_frac inflates on losses (cornered to few
           # planets → firing from "many of few"), so the won-game value is the honest spray read.
           "launch_states_won": 0, "launch_count_won": 0, "fire_steps_won": 0, "fire_frac_sum_won": 0.0,
           "launch_states_lost": 0, "launch_count_lost": 0, "fire_steps_lost": 0, "fire_frac_sum_lost": 0.0,
           # ship0 (1-ship probe) by phase × outcome — the panic hypothesis
           "launches_ph": [0, 0, 0], "ship1_ph": [0, 0, 0], "ship_ph_sum": [0, 0, 0],
           "launches_ph_won": [0, 0, 0], "ship1_ph_won": [0, 0, 0], "ship_ph_sum_won": [0, 0, 0],
           "launches_ph_lost": [0, 0, 0], "ship1_ph_lost": [0, 0, 0], "ship_ph_sum_lost": [0, 0, 0],
           # attack target near-vs-far by phase × outcome — is far-targeting a losing-state artifact?
           "atkfar_n_ph": [0, 0, 0], "atkfar_nearest_ph": [0, 0, 0], "atkfar_ratio_sum_ph": [0.0, 0.0, 0.0],
           "atkfar_n_ph_won": [0, 0, 0], "atkfar_nearest_ph_won": [0, 0, 0], "atkfar_ratio_sum_ph_won": [0.0, 0.0, 0.0],
           "atkfar_n_ph_lost": [0, 0, 0], "atkfar_nearest_ph_lost": [0, 0, 0], "atkfar_ratio_sum_ph_lost": [0.0, 0.0, 0.0],
           "atkdom_ph": [0, 0, 0], "atkdom_ph_won": [0, 0, 0], "atkdom_ph_lost": [0, 0, 0],
           # holdable-ROI rank (reactive-aware target selection) by phase × outcome
           "atkh_n_ph": [0, 0, 0], "atkh_best_ph": [0, 0, 0], "atkh_top3_ph": [0, 0, 0],
           "atkh_n_ph_won": [0, 0, 0], "atkh_best_ph_won": [0, 0, 0], "atkh_top3_ph_won": [0, 0, 0],
           "atkh_n_ph_lost": [0, 0, 0], "atkh_best_ph_lost": [0, 0, 0], "atkh_top3_ph_lost": [0, 0, 0],
           # failed-attack decomposition by phase × outcome. Buckets are _ATK_FAIL_LABELS.
           "atk_n_ph": [0, 0, 0], "atkfail_n_ph": [0, 0, 0],
           "atkfail_bucket_ph": [[0, 0, 0] for _ in _ATK_FAIL_LABELS],
           "atkagg_reach_ph": [0, 0, 0],
           "atk_n_ph_won": [0, 0, 0], "atkfail_n_ph_won": [0, 0, 0],
           "atkfail_bucket_ph_won": [[0, 0, 0] for _ in _ATK_FAIL_LABELS],
           "atkagg_reach_ph_won": [0, 0, 0],
           "atk_n_ph_lost": [0, 0, 0], "atkfail_n_ph_lost": [0, 0, 0],
           "atkfail_bucket_ph_lost": [[0, 0, 0] for _ in _ATK_FAIL_LABELS],
           "atkagg_reach_ph_lost": [0, 0, 0],
           # retention split by outcome — lost-cap → 1 on elimination (lose every planet because you
           # LOST the game), so the won-game value is the honest "can we hold mid-game?" read.
           "captures_won": 0, "captures_lost": 0, "lost_caps_won": 0, "lost_caps_lost": 0,
           "hold_durations_won": [], "hold_durations_lost": [],
           # per-game LENGTHS split by outcome → median game length for wins (stall-and-win vs
           # quick wrap-up) and losses. glen_sum/n is the mean over ALL games; this is the median split.
           "game_len_won": [], "game_len_lost": [],
           # conversion + expansion split by outcome (planets@/open-cap aggregates are dominated by
           # the majority class — mostly losses vs a strong opp — so the won-game ramp is the real read)
           "attack_launches_won": 0, "attack_launches_lost": 0,
           "atk_early_won": 0, "atk_early_lost": 0, "caps_early_won": 0, "caps_early_lost": 0,
           "atk_mid": 0, "caps_mid": 0, "reinf_early": 0, "reinf_mid": 0,
           "reinf_fwd": 0, "reinf_rear": 0, "reinf_dirn": 0, "reinf_recip": [0, 0, 0],
           "reinf_recip3_ph": [0, 0, 0], "reinf_n_ph": [0, 0, 0], "reinf_fwde_ph": [0, 0, 0],
           "reinf_cout_ph": [0, 0, 0], "reinf_ftop3_ph": [0, 0, 0],
           "reinf_n_sz": [0, 0, 0], "reinf_fwde_sz": [0, 0, 0],
           "atk_mid_won": 0, "atk_mid_lost": 0, "caps_mid_won": 0, "caps_mid_lost": 0,
           "games_won": 0, "games_lost": 0,
           "reinf_bin": [0] * len(_REINF_BINS), "atk_bin": [0] * len(_REINF_BINS),
           # decisive-mass gap: pooled mass/floor ratios per phase (cross/gap/p50/overkill derive)
           "dm_ratios_ph": [[], [], []], "dm_neutral_ratios_ph": [[], [], []],
           # hold-floor (defensive): pooled hold ratios per phase + per age-after-capture bucket
           "hold_ph": [[], [], []], "hold_age": [[], [], []],
           # reinforce triage / save-efficiency. class_ships = reinforce ships by target class at
           # launch [safe,cheap,exp,hopeless]; to_lost/to_saved = reinforce ships on planets we then
           # lost vs kept; lost_by_class = the class of planets we LOST; hopeless_{reinf,aband} =
           # of hopeless losses, did we waste ships or recycle. _won/_lost = winner/loser split.
           "triage": {"class_ships": [0.0, 0.0, 0.0, 0.0],
                      "class_ships_won": [0.0, 0.0, 0.0, 0.0], "class_ships_lost": [0.0, 0.0, 0.0, 0.0],
                      "to_lost": 0.0, "to_saved": 0.0,
                      "to_lost_won": 0.0, "to_saved_won": 0.0, "to_lost_lost": 0.0, "to_saved_lost": 0.0,
                      "lost_by_class": [0, 0, 0, 0],
                      "lost_by_class_cap_nt": [0, 0, 0, 0],
                      "hopeless_reinf": 0, "hopeless_aband": 0,
                      "hopeless_reinf_ships": 0.0,
                      "lost_within": [0.0, 0.0, 0.0],
                      "safe_util": [0.0, 0.0, 0.0],
                      "cap_born": [0, 0, 0, 0], "cap_born_lost_nt": [0, 0, 0, 0]},
           # outmassed/WR conditioned on the OPENING-expansion bucket (planets@32): separates the
           # upstream economic-tempo lever ("we didn't expand fast/wide enough → less mass before the
           # peel") from the tactical aggregation/retention wall. Per bucket: lost-capture loss_mode
           # [abandoned,out-massed,too-late,other], games, wins. Keyed <=4 / 5 / >=6 (winners ~6+).
           "om32": {"<=4": {"lm": [0, 0, 0, 0], "games": 0, "wins": 0},
                    "5":   {"lm": [0, 0, 0, 0], "games": 0, "wins": 0},
                    ">=6": {"lm": [0, 0, 0, 0], "games": 0, "wins": 0}}}
    for ms in _CONV_MILESTONES:
        acc[f"p{ms}_sum"] = 0
        acc[f"p{ms}_n"] = 0
        acc[f"g{ms}_sum"] = 0    # garrison (parked) ships, summed over games reaching ms
        acc[f"if{ms}_sum"] = 0   # in-flight (deployed) ships, summed over games reaching ms
        acc[f"p{ms}_sum_won"] = 0; acc[f"p{ms}_n_won"] = 0
        acc[f"p{ms}_sum_lost"] = 0; acc[f"p{ms}_n_lost"] = 0
    return acc


def add_conversion(acc, conv, won=None):
    for k in ("captures", "attack_launches", "reinforce_launches", "attack_ships",
              "end_planets", "redundant", "underkill", "atk_early", "caps_early", "redundant_early",
              "underkill_early", "lost_caps", "launch_states", "launch_count",
              "fire_steps", "fire_frac_sum", "atk_mid", "caps_mid", "reinf_early", "reinf_mid",
              "reinf_fwd", "reinf_rear", "reinf_dirn"):
        acc[k] += conv[k]
    for _lk in ("reinf_recip", "reinf_recip3_ph", "reinf_n_ph", "reinf_fwde_ph",
                "reinf_cout_ph", "reinf_ftop3_ph", "reinf_n_sz", "reinf_fwde_sz"):
        for _k in range(3):                          # 3-element phase / size / step lists
            acc[_lk][_k] += conv[_lk][_k]
    # route the fire-rate fields into won/lost buckets so spray can be read free of the
    # losing-position confound (won=None from non-eval callers → overall only, no split)
    if won is not None:
        suf = "won" if won else "lost"
        for k in ("launch_states", "launch_count", "fire_steps", "fire_frac_sum",
                  "captures", "lost_caps", "attack_launches", "atk_early", "caps_early",
                  "atk_mid", "caps_mid"):
            acc[f"{k}_{suf}"] += conv[k]
        acc[f"hold_durations_{suf}"].extend(conv["hold_durations"])
        acc[f"game_len_{suf}"].append(conv["glen"])
        acc["games_won" if won else "games_lost"] += 1
    acc["hold_durations"].extend(conv["hold_durations"])
    for i in range(4):
        acc["loss_mode"][i] += conv["loss_mode"][i]
    for k in ("loss_garr_cap", "loss_garr_at", "loss_enemy_in"):
        acc[k].extend(conv[k])
    acc["glen_sum"] += conv["glen"]
    for i in range(len(_REINF_BINS)):
        acc["reinf_bin"][i] += conv["reinf_bin"][i]
        acc["atk_bin"][i] += conv["atk_bin"][i]
    for i in range(3):
        acc["dm_ratios_ph"][i].extend(conv["dm_ratios_ph"][i])
        acc["dm_neutral_ratios_ph"][i].extend(conv["dm_neutral_ratios_ph"][i])
        acc["hold_ph"][i].extend(conv["hold_ph"][i])
        acc["hold_age"][i].extend(conv["hold_age"][i])
    tr = acc["triage"]
    for i in range(4):
        tr["class_ships"][i] += conv["reinf_class_ships"][i]
        tr["lost_by_class"][i] += conv["lost_by_class"][i]
        tr["lost_by_class_cap_nt"][i] += conv["lost_by_class_cap_nt"][i]
        tr["cap_born"][i] += conv["cap_born_class"][i]
        tr["cap_born_lost_nt"][i] += conv["cap_born_lost_nt"][i]
    tr["to_lost"] += conv["reinf_to_lost"]
    tr["to_saved"] += conv["reinf_to_saved"]
    tr["hopeless_reinf"] += conv["hopeless_lost_reinforced"]
    tr["hopeless_aband"] += conv["hopeless_lost_abandoned"]
    tr["hopeless_reinf_ships"] += conv["hopeless_lost_reinf_ships"]
    for i in range(3):
        tr["lost_within"][i] += conv["reinf_lost_within"][i]
        tr["safe_util"][i] += conv["safe_reinf_util"][i]
    if won is not None:
        suf = "won" if won else "lost"
        for i in range(4):
            tr[f"class_ships_{suf}"][i] += conv["reinf_class_ships"][i]
        tr[f"to_lost_{suf}"] += conv["reinf_to_lost"]
        tr[f"to_saved_{suf}"] += conv["reinf_to_saved"]
    # outmassed/WR conditioned on opening expansion (planets@32). Needs the per-game outcome (won)
    # and a game that actually reached step 32 (p32 None = ended earlier → unbucketable, skipped).
    if won is not None and conv.get("p32") is not None:
        p32 = conv["p32"]
        b = "<=4" if p32 <= 4 else ("5" if p32 == 5 else ">=6")
        bk = acc["om32"][b]
        bk["games"] += 1
        bk["wins"] += 1 if won else 0
        for i in range(4):
            bk["lm"][i] += conv["loss_mode"][i]
    for i in range(3):
        acc["launches_ph"][i] += conv["launches_ph"][i]
        acc["ship1_ph"][i] += conv["ship1_ph"][i]
        acc["ship_ph_sum"][i] += conv["ship_ph_sum"][i]
        acc["atkfar_n_ph"][i] += conv["atkfar_n_ph"][i]
        acc["atkfar_nearest_ph"][i] += conv["atkfar_nearest_ph"][i]
        acc["atkfar_ratio_sum_ph"][i] += conv["atkfar_ratio_sum_ph"][i]
        acc["atkdom_ph"][i] += conv["atkdom_ph"][i]
        acc["atkh_n_ph"][i] += conv["atkh_n_ph"][i]
        acc["atkh_best_ph"][i] += conv["atkh_best_ph"][i]
        acc["atkh_top3_ph"][i] += conv["atkh_top3_ph"][i]
        acc["atk_n_ph"][i] += conv["atk_n_ph"][i]
        acc["atkfail_n_ph"][i] += conv["atkfail_n_ph"][i]
        acc["atkagg_reach_ph"][i] += conv["atkagg_reach_ph"][i]
        for j in range(len(_ATK_FAIL_LABELS)):
            acc["atkfail_bucket_ph"][j][i] += conv["atkfail_bucket_ph"][j][i]
        if won is not None:
            suf = "won" if won else "lost"
            acc[f"launches_ph_{suf}"][i] += conv["launches_ph"][i]
            acc[f"ship1_ph_{suf}"][i] += conv["ship1_ph"][i]
            acc[f"ship_ph_sum_{suf}"][i] += conv["ship_ph_sum"][i]
            acc[f"atkfar_n_ph_{suf}"][i] += conv["atkfar_n_ph"][i]
            acc[f"atkfar_nearest_ph_{suf}"][i] += conv["atkfar_nearest_ph"][i]
            acc[f"atkfar_ratio_sum_ph_{suf}"][i] += conv["atkfar_ratio_sum_ph"][i]
            acc[f"atkdom_ph_{suf}"][i] += conv["atkdom_ph"][i]
            acc[f"atkh_n_ph_{suf}"][i] += conv["atkh_n_ph"][i]
            acc[f"atkh_best_ph_{suf}"][i] += conv["atkh_best_ph"][i]
            acc[f"atkh_top3_ph_{suf}"][i] += conv["atkh_top3_ph"][i]
            acc[f"atk_n_ph_{suf}"][i] += conv["atk_n_ph"][i]
            acc[f"atkfail_n_ph_{suf}"][i] += conv["atkfail_n_ph"][i]
            acc[f"atkagg_reach_ph_{suf}"][i] += conv["atkagg_reach_ph"][i]
            for j in range(len(_ATK_FAIL_LABELS)):
                acc[f"atkfail_bucket_ph_{suf}"][j][i] += conv["atkfail_bucket_ph"][j][i]
    acc["games"] += 1
    for ms in _CONV_MILESTONES:
        v = conv[f"p{ms}"]
        if v is not None:
            acc[f"p{ms}_sum"] += v
            acc[f"p{ms}_n"] += 1
            acc[f"g{ms}_sum"] += conv[f"g{ms}"]
            acc[f"if{ms}_sum"] += conv[f"if{ms}"]
            if won is not None:
                suf = "won" if won else "lost"
                acc[f"p{ms}_sum_{suf}"] += v
                acc[f"p{ms}_n_{suf}"] += 1


def _fmt_conversion(acc):
    """Two-line conversion summary. cap/launch counts ATTACK launches only
    (reinforce can't capture). Reference = Isaiah (#1 player)."""
    n = max(acc["games"], 1)
    c, al, rl = acc["captures"], acc["attack_launches"], acc["reinforce_launches"]
    # reinforce SHARE by step-window (own-target launches ÷ all launches in that window) — shows
    # WHEN reinforcement kicks in. late = whole − early − mid (counts derived from totals).
    re_e, re_m = acc["reinf_early"], acc["reinf_mid"]
    re_l = rl - re_e - re_m
    at_e, at_m = acc["atk_early"], acc["atk_mid"]
    at_l = al - at_e - at_m
    rsh_e = re_e / max(re_e + at_e, 1)
    rsh_m = re_m / max(re_m + at_m, 1)
    rsh_l = re_l / max(re_l + at_l, 1)
    rdf = 100 * acc["reinf_fwd"] / max(acc["reinf_dirn"], 1)   # forward-staging % of reinforces
    rdr = 100 * acc["reinf_rear"] / max(acc["reinf_dirn"], 1)  # rear-defense % (forward-only blocks these)
    _rl = max(acc["reinforce_launches"], 1)                    # reinforce-ping-pong rate (of reinforces)
    rr1, rr2, rr3 = (acc["reinf_recip"][0] / _rl, acc["reinf_recip"][1] / _rl, acc["reinf_recip"][2] / _rl)
    _f3 = lambda num, den: tuple(num[i] / max(den[i], 1) for i in range(3))   # phase/size-safe fraction
    fwde = _f3(acc["reinf_fwde_ph"], acc["reinf_n_ph"])    # forward-to-nearest-enemy by phase
    cout = _f3(acc["reinf_cout_ph"], acc["reinf_n_ph"])    # centroid-outward by phase
    ftop3 = _f3(acc["reinf_ftop3_ph"], acc["reinf_n_ph"])  # target-frontline-top3 by phase
    fwsz = _f3(acc["reinf_fwde_sz"], acc["reinf_n_sz"])    # forward-to-enemy by ship size
    rc3p = acc["reinf_recip3_ph"]; rnp = acc["reinf_n_ph"]
    pl = lambda ms: (f"{acc[f'p{ms}_sum']/acc[f'p{ms}_n']:.0f}" if acc[f"p{ms}_n"] else "—")
    plw = lambda ms: (f"{acc[f'p{ms}_sum_won']/acc[f'p{ms}_n_won']:.0f}" if acc[f"p{ms}_n_won"] else "—")
    pll = lambda ms: (f"{acc[f'p{ms}_sum_lost']/acc[f'p{ms}_n_lost']:.0f}" if acc[f"p{ms}_n_lost"] else "—")
    # Hoard read at fixed milestones (not episode-averaged → no end-step skew):
    # garr_frac = parked / (parked + in-flight) ; ships/planet = parked / owned planets.
    gf = lambda ms: (f"{acc[f'g{ms}_sum']/(acc[f'g{ms}_sum']+acc[f'if{ms}_sum']):.2f}"
                     if (acc[f'g{ms}_sum'] + acc[f'if{ms}_sum']) > 0 else "—")
    spp = lambda ms: (f"{acc[f'g{ms}_sum']/acc[f'p{ms}_sum']:.0f}" if acc[f"p{ms}_sum"] else "—")
    # reinforce ramp by empire size: own-target share among launches made at that size,
    # with launch count in parens (low-count bins are noisy). Compare to the top-player ramp.
    def rb(i):
        r, a = acc["reinf_bin"][i], acc["atk_bin"][i]
        return f"{r/(r+a):.2f}({r+a})" if (r + a) else f"—(0)"
    ramp = "  ".join(f"{_REINF_BINS[i][2]}:{rb(i)}" for i in range(len(_REINF_BINS)))
    # churn = gross captures per planet held at end (capture-then-lose-then-recapture). ⚠️
    # LENGTH-CONFOUNDED: more steps → more gross re-captures, so a 500-step grind reads high
    # even when holding well (Isaiah 7.1 > Jake 3.5 purely on game length). Always read with
    # game length; `churn/100st` normalizes it (caps/end per 100 steps). The clean hold signal
    # is the planets@N trajectory turning over (peak then decline), not churn alone.
    # Launch waste, both vs cap_cost_at_arrival (== the roi-deflation's own condition) and
    # OPENING-windowed (<50) as the headline (whole-game inflated by benign end-game surplus in
    # long won games; `(WG x)` kept for context). redundant = OVERKILL (target already covered
    # before the launch); underkill = launch that still can't capture (e.g. 18 sent at a 23-ship
    # neutral). Top-player opening redundant ref ~0.12.
    glen = acc["glen_sum"] / n
    churn = c / max(acc["end_planets"], 1)
    churn_n = churn / max(glen / 100.0, 1e-6)
    # Retention (denominator-free, unlike churn): of planets we CAPTURE, the fraction we then lose,
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
    # lost-capture autopsy: WHY held planets fall. out-massed = enemy fleet > our garrison (the
    # force-concentration gap — planners mass a decisive strike; we hold a thin line everywhere).
    lmt = sum(acc["loss_mode"]) or 1
    lm = [x / lmt for x in acc["loss_mode"]]
    gcap_med, gloss_med, einb_med = _med(acc["loss_garr_cap"]), _med(acc["loss_garr_at"]), _med(acc["loss_enemy_in"])
    redf = acc["redundant_early"] / max(acc["atk_early"], 1)
    redf_wg = acc["redundant"] / max(al, 1)
    undf = acc["underkill_early"] / max(acc["atk_early"], 1)
    undf_wg = acc["underkill"] / max(al, 1)
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
    def _nf(suf):
        n, nr, rs, dm = (acc[f"atkfar_n_ph{suf}"], acc[f"atkfar_nearest_ph{suf}"],
                         acc[f"atkfar_ratio_sum_ph{suf}"], acc[f"atkdom_ph{suf}"])
        return "  ".join(
            (f"{nm} nearest {100*nr[i]/n[i]:.0f}% ratio {rs[i]/n[i]:.2f} dom {100*dm[i]/n[i]:.0f}%(n{n[i]})"
             if n[i] else f"{nm} —(n0)")
            for i, nm in enumerate(("early<50", "mid50-100", "late>=100")))
    nfwl = (f"\n     WON  {_nf('_won')}\n     LOST {_nf('_lost')}" if (gw + gl) > 0 else "")
    def _hr(suf):
        n, bst, t3 = acc[f"atkh_n_ph{suf}"], acc[f"atkh_best_ph{suf}"], acc[f"atkh_top3_ph{suf}"]
        return "  ".join(
            (f"{nm} best {100*bst[i]/n[i]:.0f}% top3 {100*t3[i]/n[i]:.0f}%(n{n[i]})" if n[i] else f"{nm} —(n0)")
            for i, nm in enumerate(("early<50", "mid50-100", "late>=100")))
    hrwl = (f"\n     WON  {_hr('_won')}\n     LOST {_hr('_lost')}" if (gw + gl) > 0 else "")
    def _af(suf):
        an = acc[f"atk_n_ph{suf}"]
        fn = acc[f"atkfail_n_ph{suf}"]
        buckets = acc[f"atkfail_bucket_ph{suf}"]
        names = ("single", "wsrc", "agg", "unaff", "suff", "red")
        out = []
        for i, nm in enumerate(("early<50", "mid50-100", "late>=100")):
            if an[i] == 0:
                out.append(f"{nm} —(n0)")
                continue
            if fn[i] == 0:
                out.append(f"{nm} fail 0%(n{an[i]})")
                continue
            mix = "/".join(f"{names[j]} {100*buckets[j][i]/fn[i]:.0f}%" for j in range(len(names)))
            out.append(f"{nm} fail {100*fn[i]/an[i]:.0f}%({fn[i]}/{an[i]}) [{mix}]")
        return "  ".join(out)
    afwl = (f"\n     WON  {_af('_won')}\n     LOST {_af('_lost')}" if (gw + gl) > 0 else "")
    def _agg2(suf):
        fn = acc[f"atkfail_n_ph{suf}"]
        raw = acc[f"atkfail_bucket_ph{suf}"][_ATK_FAIL_LABELS.index("aggregate")]
        reach = acc[f"atkagg_reach_ph{suf}"]
        out = []
        for i, nm in enumerate(("early<50", "mid50-100", "late>=100")):
            if raw[i] == 0:
                out.append(f"{nm} —(raw0)")
                continue
            out.append(
                f"{nm} {100*reach[i]/raw[i]:.0f}% raw-agg ({reach[i]}/{raw[i]}), "
                f"{100*reach[i]/max(fn[i],1):.0f}% failures"
            )
        return "  ".join(out)
    agg2wl = (f"\n     WON  {_agg2('_won')}\n     LOST {_agg2('_lost')}" if (gw + gl) > 0 else "")
    # decisive-mass GAP (force concentration toward producer_v2's capture floor; same floor as the
    # training reward). Per phase: gap = mean max(0,1-mass/floor) (DOWN=concentrating), cross =
    # mean(mass>=floor) (UP), p50 = median ratio. overkill = mean ratio on crossed (catch 3x dumb-
    # overkill); near-miss = fraction in [0.75,1) (approaching but not crossing). target-steps/game
    # = duration-weighted target observations per game (a long-ETA attack counts each step it is
    # inflight — a state-time signal, NOT unique targets/launches). project_force_concentration_wall.
    # hold-floor (DEFENSIVE): of threatened own planets, what fraction are under-defended (hold
    # ratio < 1), split by phase and by age-after-capture. High under% at 0-5 = we lose captures
    # immediately (can't route defensive mass in time → action-grammar/timing); high at 16+ =
    # later logistics/churn. The defensive mirror of the decisive-mass attack gap.
    def _hold_stats(rs):
        if not rs:
            return "—/— (n0)"
        under = sum(1 for r in rs if r < 1.0) / len(rs)
        p50 = sorted(rs)[len(rs) // 2]
        return f"{under:.0%}/{p50:.2f} (n{len(rs)})"
    hold_line = (
        f"  hold-floor (garr+friendly_in)/(enemy_in+β·reach+1) β={_DM_BETA_EVAL:.1f}, threatened own [under%(ratio<1)/p50]\n"
        f"     by phase <50/50-100/>=100  "
        + "  ".join(_hold_stats(acc["hold_ph"][i]) for i in range(3)) + "\n"
        f"     by age-after-capture 0-5/6-15/16+  "
        + "  ".join(_hold_stats(acc["hold_age"][i]) for i in range(3))
        + "   [under% high at 0-5 = fail immediately (route defense too slow); at 16+ = logistics/churn]\n")
    # reinforce TRIAGE / save-efficiency: are we reinforcing the WRONG planets (pouring mass into
    # hopeless/eventually-lost ones) instead of recycling it? Mirrors producer_v2 (safe_drain + roi).
    _tr = acc["triage"]
    def _share(part, whole):
        return part / whole if whole else 0.0
    _cs = _tr["class_ships"]; _cst = sum(_cs) or 1.0
    _lbc = _tr["lost_by_class"]; _lbt = sum(_lbc) or 1.0
    _lbc_cap = _tr["lost_by_class_cap_nt"]; _lbc_capt = sum(_lbc_cap) or 1.0
    _hl = _tr["hopeless_reinf"] + _tr["hopeless_aband"]
    _lw = _tr["lost_within"]; _reinf_mass = sum(_cs) or 1.0
    _su = _tr["safe_util"]; _sut = sum(_su) or 1.0
    _born = _tr["cap_born"]; _born_lost = _tr["cap_born_lost_nt"]
    def _hopeless_share(suf):                       # reinforce mass → hopeless targets, won/lost/overall
        cs = _tr[f"class_ships_{suf}"] if suf else _cs
        return _share(cs[3], sum(cs) or 1.0)
    def _tolost_share(suf):                         # reinforce mass on planets we then LOST
        tl = _tr[f"to_lost_{suf}"] if suf else _tr["to_lost"]
        ts = _tr[f"to_saved_{suf}"] if suf else _tr["to_saved"]
        return _share(tl, (tl + ts) or 1.0)
    triage_line = (
        f"  reinforce-triage  ships→ safe {_share(_cs[0],_cst):.0%} cheap-save {_share(_cs[1],_cst):.0%} "
        f"exp-save {_share(_cs[2],_cst):.0%} HOPELESS {_share(_cs[3],_cst):.0%}  (β={_DM_BETA_EVAL:.1f})\n"
        f"     reinforce mass on LOST planets {_tolost_share(''):.0%} (to-saved {1-_tolost_share(''):.0%})  ·  "
        f"of LOST planets: cheap-save-MISSED {_share(_lbc[1],_lbt):.0%} hopeless(ok-to-drop) {_share(_lbc[3],_lbt):.0%}\n"
        f"     captured-only nonterminal LOST: cheap-save-MISSED {_share(_lbc_cap[1],_lbc_capt):.0%} "
        f"hopeless(ok-to-drop) {_share(_lbc_cap[3],_lbc_capt):.0%} (n{sum(_lbc_cap)})\n"
        f"     hopeless-loss events: reinforced {_share(_tr['hopeless_reinf'],_hl or 1):.0%} abandoned {_share(_tr['hopeless_aband'],_hl or 1):.0%} (n{_hl}); "
        f"ship-waste {_tr['hopeless_reinf_ships']:.0f} ({_share(_tr['hopeless_reinf_ships'], _tr['to_lost'] or 1.0):.0%} of to-lost mass)\n"
        f"     lost≤{_TRIAGE_LOSS_WINDOWS[0]}/{_TRIAGE_LOSS_WINDOWS[1]}/{_TRIAGE_LOSS_WINDOWS[2]}st after reinforce "
        f"{_share(_lw[0],_reinf_mass):.0%}/{_share(_lw[1],_reinf_mass):.0%}/{_share(_lw[2],_reinf_mass):.0%} of reinforce mass  ·  "
        f"safe-reinf≤{_SAFE_REINF_LOOKAHEAD} utility attack/frontline/idle "
        f"{_share(_su[0],_sut):.0%}/{_share(_su[1],_sut):.0%}/{_share(_su[2],_sut):.0%}\n"
        f"     capture-born class safe/cheap/exp/hopeless "
        f"{_share(_born[0],sum(_born) or 1):.0%}/{_share(_born[1],sum(_born) or 1):.0%}/"
        f"{_share(_born[2],sum(_born) or 1):.0%}/{_share(_born[3],sum(_born) or 1):.0%}; "
        f"nonterminal lost-rate by birth "
        f"{_share(_born_lost[0],_born[0] or 1):.0%}/{_share(_born_lost[1],_born[1] or 1):.0%}/"
        f"{_share(_born_lost[2],_born[2] or 1):.0%}/{_share(_born_lost[3],_born[3] or 1):.0%}\n"
        f"     "
        f"WON/LOST hopeless-reinf-share {_hopeless_share('won'):.0%}/{_hopeless_share('lost'):.0%}  to-lost {_tolost_share('won'):.0%}/{_tolost_share('lost'):.0%}\n"
        f"   [born-hopeless high = capture quality/follow-up issue; safe-idle high = staging/churn, not useful defense]\n")
    # outmassed/WR conditioned on opening expansion (planets@32 bucket). The verdict is computed
    # from THIS panel (>=6 vs <=4), not asserted — if >=6 has materially lower out-massed / higher
    # WR, opening expansion looks upstream; if not, the wall is downstream (defensive/tactical).
    def _om32_vals(b):
        bk = acc["om32"][b]
        g = bk["games"]
        if g == 0:
            return None
        tot_lm = sum(bk["lm"])
        return (bk["lm"][1] / tot_lm if tot_lm else 0.0, bk["wins"] / g, g)
    def _om32(b):
        v = _om32_vals(b)
        return f"{b}: —(n0)" if v is None else f"{b}: outmassed {v[0]:.0%} WR {v[1]:.0%} (n{v[2]})"
    _lo, _hi = _om32_vals("<=4"), _om32_vals(">=6")
    if _lo is None or _hi is None or _lo[2] < 15 or _hi[2] < 15:
        _verdict = "low-n in a bucket → inconclusive"
    else:
        d_om = _lo[0] - _hi[0]            # +ve = >=6 is LESS out-massed (expansion helps)
        d_wr = _hi[1] - _lo[1]            # +ve = >=6 wins MORE
        if d_om >= 0.04 or d_wr >= 0.05:
            _verdict = (f">=6 better (outmassed {d_om:+.0%}, WR {d_wr:+.0%} vs <=4) "
                        "→ opening expansion looks UPSTREAM")
        else:
            _verdict = (f">=6 ~ <=4 (outmassed {d_om:+.0%}, WR {d_wr:+.0%}) "
                        "→ opening expansion NOT the lever; wall is downstream")
    om32_line = ("  outmassed by planets@32  "
                 + "  ".join(_om32(b) for b in ("<=4", "5", ">=6"))
                 + f"   [{_verdict}]\n")
    _dmp = acc["dm_ratios_ph"]
    _dm_all = _dmp[0] + _dmp[1] + _dmp[2]
    def _dm_ph(rs):
        if not rs:
            return "—/—/—"
        gap = sum(max(0.0, 1.0 - r) for r in rs) / len(rs)
        cross = sum(1 for r in rs if r >= 1.0) / len(rs)
        return f"{gap:.2f}/{cross:.2f}/{sorted(rs)[len(rs)//2]:.2f}"
    _cross_all = [r for r in _dm_all if r >= 1.0]
    dm_gap = sum(max(0.0, 1.0 - r) for r in _dm_all) / max(len(_dm_all), 1)
    dm_cross = len(_cross_all) / max(len(_dm_all), 1)
    dm_over = sum(_cross_all) / max(len(_cross_all), 1) if _cross_all else 0.0
    dm_nm = sum(1 for r in _dm_all if 0.75 <= r < 1.0) / max(len(_dm_all), 1)
    dm_tpg = len(_dm_all) / n
    dm_line = (f"  decisive-mass  gap {dm_gap:.2f}  cross {dm_cross:.2f}  overkill {dm_over:.2f}  "
               f"near-miss {dm_nm:.2f}  target-steps/game {dm_tpg:.1f}  (beta {_DM_BETA_EVAL:.1f})\n"
               f"     by phase <50/50-100/>=100 (gap/cross/p50)  "
               f"{_dm_ph(_dmp[0])}  {_dm_ph(_dmp[1])}  {_dm_ph(_dmp[2])}"
               f"   [enemy-target only; effectively the CONTESTED bucket. gap DOWN + cross UP = assembling to the capture floor]\n")
    dm_contested_line = (
        f"  decisive-mass CONTESTED  gap {dm_gap:.2f}  cross {dm_cross:.2f}  overkill {dm_over:.2f}  "
        f"near-miss {dm_nm:.2f}  target-steps/game {dm_tpg:.1f}   (enemy targets only; same floor as decmass reward)\n"
        f"     by phase <50/50-100/>=100 (gap/cross/p50)  "
        f"{_dm_ph(_dmp[0])}  {_dm_ph(_dmp[1])}  {_dm_ph(_dmp[2])}"
        f"   [this is the wall-facing bucket; compare against NEUTRAL to separate land-grab from contested conversion]\n"
    )
    _dmn = acc["dm_neutral_ratios_ph"]
    _dmn_all = _dmn[0] + _dmn[1] + _dmn[2]
    dmn_gap = sum(max(0.0, 1.0 - r) for r in _dmn_all) / max(len(_dmn_all), 1)
    dmn_cross = sum(1 for r in _dmn_all if r >= 1.0) / max(len(_dmn_all), 1)
    dmn_tpg = len(_dmn_all) / n
    dm_neutral_line = (f"  decisive-mass NEUTRAL  gap {dmn_gap:.2f}  cross {dmn_cross:.2f}  "
                       f"target-steps/game {dmn_tpg:.1f}   (floor = static garrison; the opening land-grab)\n"
                       f"     by phase <50/50-100/>=100 (gap/cross/p50)  "
                       f"{_dm_ph(_dmn[0])}  {_dm_ph(_dmn[1])}  {_dm_ph(_dmn[2])}"
                       f"   [cross = sent enough to TAKE the neutral; low <50 cross = undercommit in the land-grab]\n")
    failed_attack_line = (
        f"  failed-attack decomposition  {_af('')}{afwl}\n"
        f"     [single=chosen source had enough but undersent; wsrc=another source had enough; "
        f"agg=needed multiple sources; unaff=owned mass below floor; suff=sent enough but still failed; "
        f"red=already covered before launch]\n"
        f"  failed-attack agg2 reachable-drainable  {_agg2('')}{agg2wl}\n"
        f"     [agg2 narrows raw agg: drainable owned-source spare that can arrive by attack eta+10; "
        f"still an estimate, but closer to the AR-actionable coordination ceiling]\n")
    return (f"Conversion: caps/game {c/n:.1f}  atk-launch/game {al/n:.1f}  "
            f"cap/atk-launch {c/max(al,1):.3f} (open<50 {cap_open:.3f}  mid50-100 {cap_mid:.3f})  ships/cap {acc['attack_ships']/max(c,1):.0f}  "
            f"reinf_share {rl/max(al+rl,1):.2f}\n"
            f"  planets@16/32/50/100 {pl(16)}/{pl(32)}/{pl(50)}/{pl(100)}  end {acc['end_planets']/n:.1f}"
            f"   churn {churn:.2f} ({churn_n:.2f}/100st, mean-len {glen:.0f})\n"
            f"  game-len  median WON {medlen_w}st ({acc['games_won']}g)  ·  LOST {medlen_l}st ({acc['games_lost']}g)"
            f"   [long WON = stall-and-win/attrition; short = decisive snowball]\n"
            f"{pwl}"
            f"\n  retention  peel-rate {lost_rate:.2f} ({acc['lost_caps']}/{c} caps lost)  median-hold {med_hold}st\n"
            f"{rwl}"
            f"  hold-loss  out-massed {lm[1]:.0%} · abandoned {lm[0]:.0%} · too-late {lm[2]:.0%} · other {lm[3]:.0%}"
            f"   garr@cap {gcap_med:.0f}→@loss {gloss_med:.0f} vs enemy-inbound {einb_med:.0f}"
            f"   [out-massed = enemy fleet > our garrison = force-concentration gap]\n"
            f"{hold_line}"
            f"{triage_line}"
            f"{om32_line}"
            f"{dm_line}"
            f"{dm_contested_line}"
            f"{dm_neutral_line}"
            f"{failed_attack_line}"
            f"  launch-waste<50  redundant {redf:.2f} (WG {redf_wg:.2f})  underkill {undf:.2f} (WG {undf_wg:.2f})"
            f"   [⚠ underkill NON-discriminating: winners also ~0.40. THE signal = open<50 cap/atk above]\n"
            f"   [ref:winner  cap/atk whole 0.53 · open<50 0.51 · mid50-100 0.47 · planets 2/6/9/10 · reinf 0.30]\n"
            f"  fire-rate  launch_rate {lr:.3f}  fire_frac {ff:.2f}   [ref:Isaiah 0.036 / 0.17]\n"
            f"{wl}"
            f"  hoard  garr_frac@ {gf(16)}/{gf(32)}/{gf(50)}/{gf(100)}  "
            f"ships/planet@ {spp(16)}/{spp(32)}/{spp(50)}/{spp(100)}"
            f"   [ref:Isaiah {_ISAIAH_HOARD_REF}]\n"
            f"  reinf by empire size  {ramp}   [ref:ramp {_REINF_RAMP_REF}]\n"
            f"  reinf by step  <50:{rsh_e:.2f}  50-100:{rsh_m:.2f}  >100:{rsh_l:.2f}   [ref:winner {_REINF_STEP_REF}]\n"
            f"  reinf direction  fwd {rdf:.0f}%  rear {rdr:.0f}%  (n={acc['reinf_dirn']})   [ref:winner fwd ~57% · rear ~26%]\n"
            f"  reinf ping-pong  recip<=1/2/3st {rr1:.2f}/{rr2:.2f}/{rr3:.2f} of {acc['reinforce_launches']} reinf "
            f"({acc['reinf_recip'][2]} within 3st; by phase {rc3p[0]}/{rc3p[1]}/{rc3p[2]})   [A->B then B->A = waste loop; rank1 <0.01, NOT caught by same-turn mutex]\n"
            f"  reinf by phase  n {rnp[0]}/{rnp[1]}/{rnp[2]} (<50/50-100/>100)  fwd-enemy {fwde[0]:.2f}/{fwde[1]:.2f}/{fwde[2]:.2f} [rank1 0.77/0.50/0.48]\n"
            f"     centroid-out {cout[0]:.2f}/{cout[1]:.2f}/{cout[2]:.2f} [rank1 0.83/0.69/0.70]  target-front-top3 {ftop3[0]:.2f}/{ftop3[1]:.2f}/{ftop3[2]:.2f} [rank1 0.81/0.34/0.27]\n"
            f"  reinf fwd-enemy by size  <=20/21-50/51+ {fwsz[0]:.2f}/{fwsz[1]:.2f}/{fwsz[2]:.2f}  (n {acc['reinf_n_sz'][0]}/{acc['reinf_n_sz'][1]}/{acc['reinf_n_sz'][2]})   [rank1 <=20:0.42 51-100:0.71]\n"
            f"  ship0 1-ship-probe by phase  {_s0('')}{s0wl}\n"
            f"  attack target near-vs-far  {_nf('')}{nfwl}\n"
            f"     [nearest%=chose closest enemy/neutral; ratio=chosen/nearest dist; dom%=passed up a CLOSER-AND-RICHER "
            f"target (value-aware defect, no production excuse). WON≈LOST dom => systematic; LOST≫WON => losing-state artifact]\n"
            f"  attack target holdable-ROI rank  {_hr('')}{hrwl}\n"
            f"     [best%=chose top holdable-ROI target (value priced w/ reactive peel = decisive-mass floor); top3=within 3. "
            f"LOST best%≪WON => value-aware selection IS the lever; WON≈LOST => systematic but not outcome-relevant]")


def _fmt_tier_summary(acc):
    """⭐ TIERED SUMMARY — re-prints the highest-signal metrics in priority order so the
    decision-relevant numbers aren't buried in the ~30-line dump above. Values are DUPLICATED
    (not moved). Priority + confound notes per docs/metrics.md. Read top-down, stop when answered."""
    gw, gl = acc["games_won"], acc["games_lost"]
    wr = gw / max(gw + gl, 1)
    _med = lambda h: (sorted(h)[len(h) // 2] if h else 0)
    # T2 — diagnose our wall
    lmt = sum(acc["loss_mode"]) or 1
    outmassed = acc["loss_mode"][1] / lmt
    _dm = acc["dm_ratios_ph"][0] + acc["dm_ratios_ph"][1] + acc["dm_ratios_ph"][2]
    dm_gap = sum(max(0.0, 1.0 - r) for r in _dm) / max(len(_dm), 1)
    dm_cross = sum(1 for r in _dm if r >= 1.0) / max(len(_dm), 1)
    h05 = acc["hold_age"][0]
    h05_under = (sum(1 for r in h05 if r < 1.0) / len(h05)) if h05 else 0.0
    cap_open_w = acc["caps_early_won"] / max(acc["atk_early_won"], 1)
    rsh_e = acc["reinf_early"] / max(acc["reinf_early"] + acc["atk_early"], 1)  # opening reinforce/stage share
    p50w = (acc["p50_sum_won"] / acc["p50_n_won"]) if acc["p50_n_won"] else 0.0
    # T3 — degeneracy tripwires (binary; normal = ignore)
    lr = acc["launch_count"] / max(acc["launch_states"], 1)
    ff_w = acc["fire_frac_sum_won"] / max(acc["fire_steps_won"], 1)
    s0 = sum(acc["ship1_ph"]) / max(sum(acc["launches_ph"]), 1)
    medlen_w = _med(acc["game_len_won"])
    bar = "─" * 78
    return (
        f"\n{bar}\n"
        f"⭐ TIERED METRIC SUMMARY  (priority order; values duplicated from above — docs/metrics.md)\n"
        f"{bar}\n"
        f"  T1 ARBITER   win-rate {wr:.1%} ({gw}/{gw + gl})   ← only signal that sees absolute regression\n"
        f"  T2 THE WALL  out-massed {outmassed:.0%} (want ↓ = wall breaking)   "
        f"decisive-mass gap {dm_gap:.2f} / cross {dm_cross:.2f} (gap↓ cross↑ = concentrating)\n"
        f"               hold-floor @age0-5 under {h05_under:.0%} (↓; capture-then-lose wall)   "
        f"open<50 cap/atk WON {cap_open_w:.2f} (ref .51)   planets@50 WON {p50w:.0f} (ref 9)\n"
        f"               reinf@<50 {rsh_e:.2f} (ref .29 = stage the attack early)\n"
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
    target_sanity_penalty: float = 0.0,
    defensive_reinforce_k: int = 0,
    defensive_reinforce_beta: float = 2.2,
    defensive_reinforce_max_targets: int = 1,
    defensive_reinforce_value_margin: float | None = None,
    defensive_reinforce_overfill: float = 1.0,
    natural_head_audit: bool = False,
    natural_head_audit_beta: float = 2.2,
) -> dict:
    """Evaluate trained policy against a baseline using kaggle_environments.

    Args:
        opponent: "random" or path to a Python agent file (e.g. "main.py")
        num_players: 2 or 4
    """
    from kaggle_environments import make

    def_reinf_stats = {}
    natural_head_stats = {} if natural_head_audit else None
    agent_fn = build_agent_fn(model, device, fire_threshold=fire_threshold, sample=sample,
                              ship_bin_mode=ship_bin_mode, target_decode=target_decode,
                              target_sanity_penalty=target_sanity_penalty,
                              defensive_reinforce_k=defensive_reinforce_k,
                              defensive_reinforce_beta=defensive_reinforce_beta,
                              defensive_reinforce_max_targets=defensive_reinforce_max_targets,
                              defensive_reinforce_value_margin=defensive_reinforce_value_margin,
                              defensive_reinforce_overfill=defensive_reinforce_overfill,
                              defensive_reinforce_stats=def_reinf_stats,
                              natural_head_audit_stats=natural_head_stats,
                              natural_head_audit_beta=natural_head_audit_beta)
    opponents = [opponent] * (num_players - 1)
    agents = [agent_fn] + opponents

    wins = 0
    total_material = 0
    conv_tot = new_conversion_acc()
    results = []

    for seed in range(seed_start, seed_start + num_games):
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run(agents)
        final = env.steps[-1]
        rewards = [s.reward for s in final]

        obs = final[0].observation
        material = sum(p[5] for p in obs.planets if p[1] == 0)
        material += sum(f[6] for f in obs.fleets if f[1] == 0)

        # Rank by reward; player 0 wins if their reward is strictly highest
        my_reward = rewards[0] if rewards[0] is not None else 0.0
        best_opp = max((r for r in rewards[1:] if r is not None), default=0.0)
        is_win = my_reward > best_opp

        add_conversion(conv_tot, game_conversion(env.steps, 0), won=is_win)

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
        "defensive_reinforce": def_reinf_stats,
        "natural_head_audit": natural_head_stats or {},
        "results": results,
    }


def _accumulate_panel_records(records: list) -> dict:
    """Build the panel result dict from a list of per-game records.

    Each record is {archetype, my_seat, is_win, material, conv}. This is the SAME
    accumulation evaluate_panel does inline, factored out so sharded runs can collect
    records per-process and replay ALL of them here → numbers identical to a full panel
    (add_conversion is a pure additive accumulator, so partition-then-merge is exact).
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
        add_conversion(conv_tot, r["conv"], won=is_win)
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
    target_sanity_penalty: float = 0.0,
    defensive_reinforce_k: int = 0,
    defensive_reinforce_beta: float = 2.2,
    defensive_reinforce_max_targets: int = 1,
    defensive_reinforce_value_margin: float | None = None,
    defensive_reinforce_overfill: float = 1.0,
    natural_head_audit: bool = False,
    natural_head_audit_beta: float = 2.2,
    shard_idx: int = 0,
    shard_count: int = 1,
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

    def_reinf_stats = {}
    natural_head_stats = {} if natural_head_audit else None
    agent_fn = build_agent_fn(model, device, fire_threshold=fire_threshold, sample=sample,
                              ship_bin_mode=ship_bin_mode, target_decode=target_decode,
                              target_sanity_penalty=target_sanity_penalty,
                              defensive_reinforce_k=defensive_reinforce_k,
                              defensive_reinforce_beta=defensive_reinforce_beta,
                              defensive_reinforce_max_targets=defensive_reinforce_max_targets,
                              defensive_reinforce_value_margin=defensive_reinforce_value_margin,
                              defensive_reinforce_overfill=defensive_reinforce_overfill,
                              defensive_reinforce_stats=def_reinf_stats,
                              natural_head_audit_stats=natural_head_stats,
                              natural_head_audit_beta=natural_head_audit_beta)

    records: list = []
    total_games = sum(len(seeds) for seeds in BY_ARCHETYPE.values()) * 2
    shard_total = sum(1 for i in range(total_games) if i % shard_count == shard_idx)

    print(f"Panel eval START — opponent: {opponent} | {total_games} games "
          f"(128 seeds × 2 seats) | decode={'target' if target_decode else 'argmax'} "
          f"fire_thr={fire_threshold}"
          + (f" | SHARD {shard_idx}/{shard_count} ({shard_total} games)" if shard_count > 1 else ""),
          flush=True)

    # Global game index over the fixed (archetype, seed, seat) order. Shard i runs only
    # games where idx % shard_count == i — a deterministic partition of the SAME 256 games.
    gi = 0
    wins_running = 0
    for archetype, seeds in BY_ARCHETYPE.items():
        for seed in seeds:
            for my_seat in (0, 1):
                do_this = (gi % shard_count == shard_idx)
                gi += 1
                if not do_this:
                    continue
                agents = [agent_fn, opponent] if my_seat == 0 else [opponent, agent_fn]
                env = make("orbit_wars", configuration={"seed": seed}, debug=False)
                env.run(agents)
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
                if len(records) % 16 == 0 or len(records) == shard_total:
                    print(f"  panel progress: {len(records)}/{shard_total}  "
                          f"overall {wins_running}/{len(records)} "
                          f"({100*wins_running/max(len(records),1):.1f}%)",
                          flush=True)

    result = _accumulate_panel_records(records)
    result["defensive_reinforce"] = def_reinf_stats
    result["natural_head_audit"] = natural_head_stats or {}
    if collect_records:
        result["_records"] = records
    return result


def _fmt_defensive_reinforce(stats: dict) -> str:
    if not stats:
        return "Defensive reinforce overlay: no events recorded"
    forced = stats.get("forced_moves", 0.0)
    targets = stats.get("forced_targets", 0.0)
    threatened = stats.get("threatened_targets", 0.0)
    fillable = stats.get("fillable_targets", 0.0)
    ships = stats.get("forced_ships", 0.0)
    orig_total = max(forced, 1.0)
    same = stats.get("orig_same_target", 0.0) / orig_total
    nofire = stats.get("orig_no_fire", 0.0) / orig_total
    enemy = stats.get("orig_enemy", 0.0) / orig_total
    neutral = stats.get("orig_neutral", 0.0) / orig_total
    other_own = stats.get("orig_other_own", 0.0) / orig_total
    undersent = stats.get("orig_undersent", 0.0) / max(
        forced - stats.get("orig_no_fire", 0.0), 1.0)
    replaced = stats.get("policy_move_replaced", 0.0)
    capdrop = stats.get("policy_move_dropped_for_cap", 0.0)
    db = stats.get("deficit_before", 0.0)
    da = stats.get("deficit_after", 0.0)
    hn = max(stats.get("head_fire_n", 0.0), 1.0)
    tn = max(stats.get("head_target_rank_n", 0.0), 1.0)
    sn = max(stats.get("head_ship_rank_n", 0.0), 1.0)
    fire_mean = stats.get("head_fire_prob_sum", 0.0) / hn
    target_rank = stats.get("head_target_rank_sum", 0.0) / tn
    ship_rank = stats.get("head_ship_rank_sum", 0.0) / sn
    value_checked = stats.get("value_gate_checked", 0.0)
    value_line = ""
    if value_checked > 0:
        vg = max(value_checked, 1.0)
        value_line = (
            f"\n  value gate: checked {value_checked:.0f} · skipped "
            f"{stats.get('value_gate_skipped_targets', 0.0):.0f} "
            f"({stats.get('value_gate_skipped_targets', 0.0) / vg:.0%}) · "
            f"avg save/opportunity/net "
            f"{stats.get('value_gate_save_value', 0.0) / vg:.1f}/"
            f"{stats.get('value_gate_opportunity', 0.0) / vg:.1f}/"
            f"{stats.get('value_gate_net', 0.0) / vg:.1f}"
        )
    requested = stats.get("realized_fill_requested_sum", 0.0)
    realized_line = ""
    if requested > 0:
        fill = stats.get("realized_fill_forced_sum", 0.0)
        realized_line = (
            f"\n  realized fill: forced/requested {fill:.0f}/{requested:.0f} "
            f"({fill / requested:.2f}x) · full targets "
            f"{stats.get('realized_fill_full_targets', 0.0):.0f}/{max(targets, 1.0):.0f}"
        )
    return (
        "Defensive reinforce overlay:\n"
        f"  threatened {threatened:.0f} · fillable {fillable:.0f} · forced targets {targets:.0f} "
        f"moves {forced:.0f} ships {ships:.0f}\n"
        f"  deficit before/after {db:.0f}/{da:.0f} · hopeless {stats.get('hopeless_targets', 0.0):.0f} "
        f"· blocked cooldown/mask {stats.get('blocked_by_cooldown_or_mask', 0.0):.0f}"
        f"{value_line}{realized_line}\n"
        f"  original policy on forced sources: no-fire {nofire:.0%} · same-target {same:.0%} "
        f"· other-own {other_own:.0%} · enemy {enemy:.0%} · neutral {neutral:.0%} "
        f"· undersent(if fired) {undersent:.0%}\n"
        f"  head audit on forced sources: fire_p mean {fire_mean:.2f} "
        f"(<0.1/{stats.get('head_fire_lt_01',0.0)/hn:.0%}, <0.3/{stats.get('head_fire_lt_03',0.0)/hn:.0%}, "
        f"<0.5/{stats.get('head_fire_lt_05',0.0)/hn:.0%})\n"
        f"     target rank avg {target_rank:.1f} top1/top3/top5 "
        f"{stats.get('head_target_top1',0.0)/tn:.0%}/{stats.get('head_target_top3',0.0)/tn:.0%}/"
        f"{stats.get('head_target_top5',0.0)/tn:.0%} · ship sufficient rank avg {ship_rank:.1f} "
        f"top1/top3/top5 {stats.get('head_ship_top1_ge_send',0.0)/sn:.0%}/"
        f"{stats.get('head_ship_top3_ge_send',0.0)/sn:.0%}/{stats.get('head_ship_top5_ge_send',0.0)/sn:.0%}\n"
        f"     joint ready top1/top3 {stats.get('head_all_top1_ready',0.0)/orig_total:.0%}/"
        f"{stats.get('head_all_top3_ready',0.0)/orig_total:.0%}\n"
        f"  replaced policy moves {replaced:.0f} · dropped-for-cap {capdrop:.0f}"
    )


def _fmt_natural_head_audit(stats: dict) -> str:
    if not stats:
        return "Natural head audit: no events recorded"

    def row(label: str, prefix: str) -> str:
        slots = max(stats.get(f"{prefix}_slots", 0.0), 1.0)
        fire_mean = stats.get(f"{prefix}_fire_prob_sum", 0.0) / slots
        fired = stats.get(f"{prefix}_fired", 0.0) / slots
        veto = 1.0 - fired
        chosen_own = stats.get(f"{prefix}_chosen_own", 0.0) / max(stats.get(f"{prefix}_fired", 0.0), 1.0)
        chosen_enemy = stats.get(f"{prefix}_chosen_enemy", 0.0) / max(stats.get(f"{prefix}_fired", 0.0), 1.0)
        chosen_neutral = stats.get(f"{prefix}_chosen_neutral", 0.0) / max(stats.get(f"{prefix}_fired", 0.0), 1.0)
        atk_n = max(stats.get(f"{prefix}_attack_n", 0.0), 1.0)
        save_n = max(stats.get(f"{prefix}_save_n", 0.0), 1.0)
        atk_tr_n = max(stats.get(f"{prefix}_attack_target_rank_n", 0.0), 1.0)
        save_tr_n = max(stats.get(f"{prefix}_save_target_rank_n", 0.0), 1.0)
        atk_sr_n = max(stats.get(f"{prefix}_attack_ship_rank_n", 0.0), 1.0)
        save_sr_n = max(stats.get(f"{prefix}_save_ship_rank_n", 0.0), 1.0)
        atk_rank = stats.get(f"{prefix}_attack_target_rank_sum", 0.0) / atk_tr_n
        save_rank = stats.get(f"{prefix}_save_target_rank_sum", 0.0) / save_tr_n
        atk_ship = stats.get(f"{prefix}_attack_ship_rank_sum", 0.0) / atk_sr_n
        save_ship = stats.get(f"{prefix}_save_ship_rank_sum", 0.0) / save_sr_n
        return (
            f"  {label:<5s} slots {slots:.0f} fire_p {fire_mean:.2f} fired {fired:.0%} "
            f"veto {veto:.0%} (<0.5 {stats.get(f'{prefix}_fire_lt_05',0.0)/slots:.0%}; "
            f"fired own/enemy/neutral {chosen_own:.0%}/{chosen_enemy:.0%}/{chosen_neutral:.0%})\n"
            f"        attack-cand {stats.get(f'{prefix}_attack_n',0.0):.0f}: "
            f"fire>=.5 {stats.get(f'{prefix}_attack_fire_ready',0.0)/atk_n:.0%} · "
            f"target avg {atk_rank:.1f} top1/3/5 "
            f"{stats.get(f'{prefix}_attack_target_top1',0.0)/atk_tr_n:.0%}/"
            f"{stats.get(f'{prefix}_attack_target_top3',0.0)/atk_tr_n:.0%}/"
            f"{stats.get(f'{prefix}_attack_target_top5',0.0)/atk_tr_n:.0%} · "
            f"top1-veto {stats.get(f'{prefix}_attack_target_top1_veto',0.0)/atk_tr_n:.0%} · "
            f"ship>=req avg {atk_ship:.1f} top1/3 "
            f"{stats.get(f'{prefix}_attack_ship_top1',0.0)/atk_sr_n:.0%}/"
            f"{stats.get(f'{prefix}_attack_ship_top3',0.0)/atk_sr_n:.0%} · "
            f"joint top1/3 {stats.get(f'{prefix}_attack_joint_top1',0.0)/atk_n:.0%}/"
            f"{stats.get(f'{prefix}_attack_joint_top3',0.0)/atk_n:.0%} · "
            f"chosen-best {stats.get(f'{prefix}_attack_chosen',0.0)/atk_n:.0%}\n"
            f"        save-cand   {stats.get(f'{prefix}_save_n',0.0):.0f}: "
            f"fire>=.5 {stats.get(f'{prefix}_save_fire_ready',0.0)/save_n:.0%} · "
            f"target avg {save_rank:.1f} top1/3/5 "
            f"{stats.get(f'{prefix}_save_target_top1',0.0)/save_tr_n:.0%}/"
            f"{stats.get(f'{prefix}_save_target_top3',0.0)/save_tr_n:.0%}/"
            f"{stats.get(f'{prefix}_save_target_top5',0.0)/save_tr_n:.0%} · "
            f"top1-veto {stats.get(f'{prefix}_save_target_top1_veto',0.0)/save_tr_n:.0%} · "
            f"ship>=req avg {save_ship:.1f} top1/3 "
            f"{stats.get(f'{prefix}_save_ship_top1',0.0)/save_sr_n:.0%}/"
            f"{stats.get(f'{prefix}_save_ship_top3',0.0)/save_sr_n:.0%} · "
            f"joint top1/3 {stats.get(f'{prefix}_save_joint_top1',0.0)/save_n:.0%}/"
            f"{stats.get(f'{prefix}_save_joint_top3',0.0)/save_n:.0%} · "
            f"chosen-best {stats.get(f'{prefix}_save_chosen',0.0)/save_n:.0%}"
        )

    return "\n".join([
        "Natural head audit (passive; lightweight planner-like attack/save candidates):",
        row("all", "natural_all"),
        row("<50", "natural_open"),
        row("50-99", "natural_mid"),
        row("100+", "natural_late"),
    ])


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
    if result.get("defensive_reinforce"):
        print(_fmt_defensive_reinforce(result["defensive_reinforce"]))
    if result.get("natural_head_audit"):
        print(_fmt_natural_head_audit(result["natural_head_audit"]))
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
                        target_sanity_penalty: float = 0.0,
                        reinforce_gate_min_planets: int = None,
                        reinforce_forward_only: bool = None,
                        reinforce_garrison_floor: float = None,
                        sufficient_commit_factor: float = None,
                        defensive_reinforce_k: int = 0,
                        defensive_reinforce_beta: float = 2.2,
                        defensive_reinforce_max_targets: int = 1,
                        defensive_reinforce_value_margin: float | None = None,
                        defensive_reinforce_overfill: float = 1.0,
                        natural_head_audit: bool = False,
                        natural_head_audit_beta: float = 2.2,
                        shard_idx: int = 0,
                        shard_count: int = 1,
                        collect_records: bool = False):
    """Load a checkpoint and evaluate it."""
    device = torch.device(cfg.device)

    state_dict, ckpt_action_decode = load_checkpoint(params_path, cfg)
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

    model = EntityTransformer(cfg.model).to(device)
    # Carry the checkpoint's reinforcement setting onto the model so the agent's
    # target masking matches training (build_agent_fn reads it off the model).
    model.allow_reinforce = bool(getattr(cfg.model, "allow_reinforce", False))
    # Reinforce-discipline masks (gate / forward-staging / garrison floor) — MUST match
    # the training env, else the policy reinforces where it was masked and self-sabotages.
    # Not stored in the checkpoint config, so they come from CLI flags.
    model.reinforce_gate_min_planets = int(reinforce_gate_min_planets)
    model.reinforce_forward_only = bool(reinforce_forward_only)
    # Reverse-edge cooldown auto-loads from the checkpoint (persisted, stateful — eval keeps the
    # per-game edge history in build_agent_fn's closure). Parity with training.
    model.reverse_edge_cooldown = int(getattr(cfg.model, "reverse_edge_cooldown", 0))
    model.reinforce_garrison_floor = float(reinforce_garrison_floor)
    # Sufficient-commit mask (attacks) — also MUST match training. Independent of reinforce.
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
    if defensive_reinforce_k > 0:
        print(f"Defensive reinforce overlay: ON | nearest_k={defensive_reinforce_k} "
              f"beta={defensive_reinforce_beta} max_targets={defensive_reinforce_max_targets} "
              f"overfill={defensive_reinforce_overfill} "
              f"(eval-time hard override; training unchanged)")
        if defensive_reinforce_value_margin is not None:
            print(f"  value gate: save_value - foregone_attack_value >= "
                  f"{defensive_reinforce_value_margin}")
        if not model.allow_reinforce:
            print("  ⚠ overlay is inert unless checkpoint/eval has allow_reinforce=True")
    if natural_head_audit:
        print(f"Natural head audit: ON | beta={natural_head_audit_beta} "
              f"(passive logits/intent diagnostics; actions unchanged)")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"target_head.weight", "target_head.bias"} | PHASE4_COMPAT_MISSING_KEYS
    bad_missing = [k for k in missing if k not in allowed_missing]
    # VDN per-planet value head (Stage 2) is never used at eval — ignore it if the
    # checkpoint carries it but this (eval-time) model doesn't.
    bad_unexpected = [k for k in unexpected if not k.startswith("value_pp_")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch: missing={bad_missing}, unexpected={bad_unexpected}")
    model.eval()

    if panel:
        results = evaluate_panel(model, device, opponent=opponent,
                                 fire_threshold=fire_threshold, sample=sample,
                                 ship_bin_mode=cfg.model.ship_bin_mode,
                                 target_decode=target_decode,
                                 target_sanity_penalty=target_sanity_penalty,
                                 defensive_reinforce_k=defensive_reinforce_k,
                                 defensive_reinforce_beta=defensive_reinforce_beta,
                                 defensive_reinforce_max_targets=defensive_reinforce_max_targets,
                                 defensive_reinforce_value_margin=defensive_reinforce_value_margin,
                                 defensive_reinforce_overfill=defensive_reinforce_overfill,
                                 natural_head_audit=natural_head_audit,
                                 natural_head_audit_beta=natural_head_audit_beta,
                                 shard_idx=shard_idx, shard_count=shard_count,
                                 collect_records=collect_records)
        if not collect_records:
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
        target_sanity_penalty=target_sanity_penalty,
        defensive_reinforce_k=defensive_reinforce_k,
        defensive_reinforce_beta=defensive_reinforce_beta,
        defensive_reinforce_max_targets=defensive_reinforce_max_targets,
        defensive_reinforce_value_margin=defensive_reinforce_value_margin,
        defensive_reinforce_overfill=defensive_reinforce_overfill,
        natural_head_audit=natural_head_audit,
        natural_head_audit_beta=natural_head_audit_beta,
    )

    print(f"Win rate vs {opponent}: {results['win_rate']:.2%}  "
          f"({results['wins']}/{results['total_games']})")
    print(f"Fire threshold: {fire_threshold}")
    print(f"Target decode: {target_decode}")
    print(f"Target sanity penalty: {target_sanity_penalty}")
    print(f"Avg material: {results['avg_material']:.1f}")
    print(_fmt_conversion(results["conversion"]))
    if results.get("defensive_reinforce"):
        print(_fmt_defensive_reinforce(results["defensive_reinforce"]))
    if results.get("natural_head_audit"):
        print(_fmt_natural_head_audit(results["natural_head_audit"]))
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
    parser.add_argument("--ablate-roi", action="store_true",
                        help="DIAGNOSTIC: permute precomputed roi_20/roi_50 across targets per source slot "
                             "(destroys target<->roi mapping). If target-selection metrics/WR don't move vs "
                             "the control panel, the net is NOT using the precomputed ROI channel.")
    parser.add_argument("--ablate-sun", action="store_true",
                        help="PLACEBO control for --ablate-roi: permute the sun_safe channel (ch4) the same way. "
                             "If WR also drops, the roi drop is general brittleness to inconsistent inputs; "
                             "if WR holds, the roi effect is roi-specific.")
    parser.add_argument("--panel", action="store_true",
                        help="Use 128-seed community panel with both-seat eval "
                             "(256 games, per-archetype breakdown).")
    parser.add_argument("--sample", action="store_true",
                        help="Sample from policy distribution instead of argmax. "
                             "Use when the mode is degenerate but distribution mass "
                             "is on competent bins (1-ship-fleet trap).")
    parser.add_argument("--target-decode", action="store_true",
                        help="Aim with target_logits plus orbital intercept instead "
                             "of directly using the angle head.")
    parser.add_argument("--target-sanity-penalty", type=float, default=0.0,
                        help="Subtract this from dominated same-source target logits "
                             "before target decode.")
    parser.add_argument("--reinforce-gate-min-planets", type=int, default=None,
                        help="Reinforce-discipline parity: own targets legal only at "
                             ">= this many owned planets. Default=auto-load from checkpoint; "
                             "pass to override. MUST match training (p2rev1=3).")
    parser.add_argument("--reinforce-forward-only", action=argparse.BooleanOptionalAction, default=None,
                        help="Reinforce-discipline parity: own target legal only if closer "
                             "to the nearest enemy than the source. Default=auto-load from ckpt; "
                             "pass --reinforce-forward-only / --no-reinforce-forward-only to override.")
    parser.add_argument("--reinforce-garrison-floor", type=float, default=None,
                        help="Reinforce-discipline parity: veto a reinforce that drains the "
                             "source below this. Default=auto-load from ckpt (p2rev1=10).")
    parser.add_argument("--sufficient-commit-factor", type=float, default=None,
                        help="Sufficient-commit parity: veto an attack whose ships <= target "
                             "defense × this factor. Default=auto-load from ckpt (1.0 = strict).")
    parser.add_argument("--decisive-mass-beta", type=float, default=_DM_BETA,
                        help="Reactive-margin weight for the decisive-mass GAP diagnostic floor "
                             "(beta*rho(eta)*reachable_enemy_mass). Default 2.2 (= training default). "
                             "Pass the run's --decisive-mass-beta to match a non-default-beta decmass "
                             "run so the eval floor == the reward floor (beta isn't stored in the ckpt).")
    parser.add_argument("--defensive-reinforce-k", type=int, default=0,
                        help="Eval-time hard defensive overlay: for threatened own planets, force "
                             "up to K nearest reachable safe-drain sources to reinforce enough mass "
                             "to fill the hold-floor deficit. 0=off.")
    parser.add_argument("--defensive-reinforce-beta", type=float, default=None,
                        help="Reactive-margin beta for --defensive-reinforce-k. Default reuses "
                             "--decisive-mass-beta so the overlay and hold-floor diagnostic agree.")
    parser.add_argument("--defensive-reinforce-max-targets", type=int, default=1,
                        help="Max threatened own planets the eval-time defensive overlay may fill "
                             "per agent step.")
    parser.add_argument("--defensive-reinforce-value-margin", type=float, default=None,
                        help="Optional value/opportunity gate for --defensive-reinforce-k. "
                             "When set, force a save only if save_value - foregone_attack_value "
                             "is at least this margin. Default off preserves the original overlay.")
    parser.add_argument("--defensive-reinforce-overfill", type=float, default=1.0,
                        help="Multiplier applied to the selected defensive deficit after value "
                             "selection. 1.0 preserves current overlay; >1.0 tests aggregate "
                             "arrival sufficiency without changing target selection.")
    parser.add_argument("--retarget-top-roi", action="store_true",
                        help="SELECTION ISOLATION: leave fire/ship as-is; redirect each ATTACK the policy "
                             "launches to the top-holdable-ROI target from that source (keep source+ships). "
                             "No spray, no fire change. Tests if better target choice raises WR.")
    parser.add_argument("--retarget-resize", action="store_true",
                        help="With --retarget-top-roi: also re-size the redirected attack to capture its "
                             "NEW target (capped at garrison), removing the size<->target mismatch confound.")
    parser.add_argument("--force-fire-high-roi", action="store_true",
                        help="FIRE-HEAD ISOLATION: on sources the fire head vetoes (fire_prob<thr) that "
                             "have a high-holdable-ROI attack available, force fire toward the head's own "
                             "target+ship (fallback top-ROI). Tests if the fire veto costs winnable attacks.")
    parser.add_argument("--force-fire-roi-threshold", type=float, default=0.3,
                        help="Min holdable-ROI of the best available attack for --force-fire-high-roi to "
                             "force a vetoed source (avoids forcing spray on worthless targets).")
    parser.add_argument("--natural-head-audit", action="store_true",
                        help="Passive target-decode audit: log fire/target/ship agreement with "
                             "lightweight planner-like attack and save candidates. No action changes.")
    parser.add_argument("--natural-head-audit-beta", type=float, default=None,
                        help="Reactive-margin beta for --natural-head-audit save candidates. "
                             "Default reuses --decisive-mass-beta.")
    parser.add_argument("--panel-shards", type=int, default=1,
                        help="Split the --panel run into this many deterministic shards (by game "
                             "index). Run one process per shard with --panel-shard-idx + --shard-out, "
                             "then merge_panel_shards.py the pickles → identical numbers, parallel.")
    parser.add_argument("--panel-shard-idx", type=int, default=0,
                        help="Which shard this process runs (0..panel-shards-1).")
    parser.add_argument("--shard-out", type=str, default=None,
                        help="With --panel-shards>1: pickle this shard's per-game records here "
                             "(suppresses the report; merge_panel_shards.py prints the merged report).")
    args = parser.parse_args()
    _DM_BETA_EVAL = args.decisive_mass_beta   # module global → used by _decisive_gap_step
    if args.ablate_roi:
        set_ablate_roi(True)
        print("ABLATION: roi_20/roi_50 permuted across targets per slot (target<->roi mapping destroyed)")
    if args.ablate_sun:
        set_ablate_channels((4,))
        print("PLACEBO ABLATION: sun_safe (ch4) permuted across targets per slot")
    if args.retarget_top_roi:
        set_retarget_top_roi(True, resize=args.retarget_resize)
        print(f"SELECTION ISOLATION: retarget each attack to top-holdable-ROI target "
              f"(resize={'ON' if args.retarget_resize else 'OFF'})")
    if args.force_fire_high_roi:
        set_force_fire_high_roi(True, args.force_fire_roi_threshold)
        print(f"FIRE-HEAD ISOLATION: force-fire vetoed sources w/ best holdable-ROI >= {args.force_fire_roi_threshold}")

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
        target_sanity_penalty=args.target_sanity_penalty,
        reinforce_gate_min_planets=args.reinforce_gate_min_planets,
        reinforce_forward_only=args.reinforce_forward_only,
        reinforce_garrison_floor=args.reinforce_garrison_floor,
        sufficient_commit_factor=args.sufficient_commit_factor,
        defensive_reinforce_k=args.defensive_reinforce_k,
        defensive_reinforce_beta=(args.decisive_mass_beta if args.defensive_reinforce_beta is None
                                  else args.defensive_reinforce_beta),
        defensive_reinforce_max_targets=args.defensive_reinforce_max_targets,
        defensive_reinforce_value_margin=args.defensive_reinforce_value_margin,
        defensive_reinforce_overfill=args.defensive_reinforce_overfill,
        natural_head_audit=args.natural_head_audit,
        natural_head_audit_beta=(args.decisive_mass_beta if args.natural_head_audit_beta is None
                                 else args.natural_head_audit_beta),
        shard_idx=args.panel_shard_idx,
        shard_count=args.panel_shards,
        collect_records=bool(args.shard_out),
    )
    if args.shard_out and _eval_result is not None:
        import pickle
        with open(args.shard_out, "wb") as _f:
            pickle.dump({"records": _eval_result.get("_records", []),
                         "defensive_reinforce": _eval_result.get("defensive_reinforce", {}),
                         "natural_head_audit": _eval_result.get("natural_head_audit", {})}, _f)
        _o = _eval_result["overall"]
        print(f"SHARD {args.panel_shard_idx}/{args.panel_shards} → {args.shard_out}: "
              f"{len(_eval_result.get('_records', []))} games, {_o['wins']}/{_o['total']} wins",
              flush=True)
    if args.retarget_top_roi:
        rt = _RETARGET
        funnel = rt["uniq_sum"] / max(rt["turns"], 1)
        print(f"SELECTION ISOLATION: retargeted {rt['retargeted']}/{rt['attacks']} attacks "
              f"({rt['retargeted']/max(rt['attacks'],1):.0%}) to top-holdable-ROI target; "
              f"resize={'ON' if rt['resize'] else 'OFF'}; target-distinctness {funnel:.2f} "
              f"(1.0=all distinct, low=funneling to same targets)")
    if args.force_fire_high_roi:
        st = _FORCE_FIRE
        per_state = st["forced"] / max(st["states"], 1)
        print(f"FIRE-HEAD ISOLATION: forced {st['forced']} fires over {st['states']} states "
              f"({per_state:.2f}/state; {st['to_head_tgt']} to head's own target, "
              f"{st['forced'] - st['to_head_tgt']} to fallback top-ROI)")
