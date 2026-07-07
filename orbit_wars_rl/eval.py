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
from features import extract_features, _ETA_PROBE_SPEED, PAIRWISE_FEATURE_DIM
from action_mask import (compute_action_masks, actions_from_target_policy, _fleet_speed,
                         _ship_bin_to_count, _target_intercept_angle, MAX_OWNED_PLANETS)
# Decisive-mass floor constants — IMPORTED from torch_env so the eval dm_* gap diagnostic uses the
# EXACT same floor as the training reward/diag (they can never drift). project_force_concentration_wall.
from torch_env import (_DM_BETA, _DM_ETA_FREE, _DM_ETA_SCALE, _DM_HORIZON, _DM_OVERHEAD,
                       MAX_SHIP_SPEED as _DM_MAX_SPEED)
from kaggle_environments.envs.orbit_wars.orbit_wars import CENTER, ROTATION_RADIUS_LIMIT


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

    if "ship_bin_mode" in ckpt_cfg:
        cfg.model.ship_bin_mode = str(ckpt_cfg["ship_bin_mode"])

    # --- feature projection dims: always infer from weight shapes ---
    if "planet_proj.weight" in sd:
        cfg.model.planet_feature_dim = int(sd["planet_proj.weight"].shape[1])
    if "fleet_proj.weight" in sd:
        cfg.model.fleet_feature_dim = int(sd["fleet_proj.weight"].shape[1])
    if "global_proj.weight" in sd:
        cfg.model.global_feature_dim = int(sd["global_proj.weight"].shape[1])
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
    # Blessed feature config guard (2026-07 cleanup): feature semantics are hard-coded in
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
                   num_players: int = 2,
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
        features = extract_features(obs, player, num_players=num_players)
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
            moves = actions_from_target_policy(
                outputs["fire_logits"].cpu(),
                outputs["target_logits"].cpu(),
                outputs["ship_logits"].cpu(),
                {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()},
                obs, player,
                fire_threshold=fire_threshold,
                sample=sample,
                ship_bin_mode=ship_bin_mode,
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
    planets_at = {ms: None for ms in _CONV_MILESTONES}
    prev = {}
    last = None
    for t in range(1, len(steps)):
        if seat >= len(steps[t]) or seat >= len(steps[t - 1]):
            continue
        p0 = steps[t - 1][seat].observation.get("planets")
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
           "launches_ph": launches_ph, "ship1_ph": ship1_ph, "ship_ph_sum": ship_ph_sum}
    for ms in _CONV_MILESTONES:
        out[f"p{ms}"] = planets_at[ms]
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
    return acc


def add_conversion(acc, conv, won=None, material=None):
    if won is False and material is not None:
        acc["lost_material"].append(material)  # elimination-depth (graded loss signal)
    for k in ("captures", "attack_launches", "reinforce_launches", "attack_ships",
              "end_planets", "atk_early", "caps_early", "atk_mid", "caps_mid", "reinf_early",
              "lost_caps", "launch_states", "launch_count", "fire_steps", "fire_frac_sum"):
        acc[k] += conv[k]
    # route the fire-rate + conversion fields into won/lost buckets so spray + the opening ramp can
    # be read free of the losing-position confound (won=None from non-eval callers → overall only)
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
    # ── Trusted core only (aggressive prune 2026-07-06; git history has the full dump). ──
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
            f"  retention  peel-rate {lost_rate:.2f} ({acc['lost_caps']}/{c} caps lost)  median-hold {med_hold}st\n"
            f"{rwl}"
            f"{ldepth}"
            f"  fire-rate  launch_rate {lr:.3f}  fire_frac {ff:.2f}   [ref:Isaiah 0.036 / 0.17]\n"
            f"{wl}"
            f"  ship0 1-ship-probe by phase  {_s0('')}{s0wl}")


def _fmt_tier_summary(acc):
    """⭐ TIERED SUMMARY — re-prints the highest-signal metrics in priority order so the
    decision-relevant numbers aren't buried in the ~30-line dump above. Values are DUPLICATED
    (not moved). Priority + confound notes per docs/metrics.md. Read top-down, stop when answered."""
    gw, gl = acc["games_won"], acc["games_lost"]
    wr = gw / max(gw + gl, 1)
    _med = lambda h: (sorted(h)[len(h) // 2] if h else 0)
    # T2 — the wall, OUTCOME-grounded only. The model-based dm family (gap/take-rate/overkill/med,
    # take+hold/can't-hold/too-few, waste) and out-massed% were CULLED 2026-07: model-based, non-
    # discriminating in matched play, and take+hold was contradicted by observed retention. Keep
    # only what is grounded in the actual game outcome and tracks skill. [[project-ender-...]]
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
    bar = "─" * 78
    return (
        f"\n{bar}\n"
        f"⭐ TIERED METRIC SUMMARY  (priority order; values duplicated from above — docs/metrics.md)\n"
        f"{bar}\n"
        f"  T1 ARBITER   win-rate {wr:.1%} ({gw}/{gw + gl})   ← the only absolute-regression signal\n"
        f"  T2 THE WALL  loss-depth med-material-in-loss {lmed:.0f} · wiped-to-0 {wiped:.0f}%  (graded; want ↑ material)\n"
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
                              defensive_reinforce_k=defensive_reinforce_k,
                              defensive_reinforce_beta=defensive_reinforce_beta,
                              defensive_reinforce_max_targets=defensive_reinforce_max_targets,
                              defensive_reinforce_value_margin=defensive_reinforce_value_margin,
                              defensive_reinforce_overfill=defensive_reinforce_overfill,
                              defensive_reinforce_stats=def_reinf_stats,
                              natural_head_audit_stats=natural_head_stats,
                              natural_head_audit_beta=natural_head_audit_beta,
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
    bad_missing = [k for k in missing if k not in PHASE4_COMPAT_MISSING_KEYS]
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
                                 defensive_reinforce_k=defensive_reinforce_k,
                                 defensive_reinforce_beta=defensive_reinforce_beta,
                                 defensive_reinforce_max_targets=defensive_reinforce_max_targets,
                                 defensive_reinforce_value_margin=defensive_reinforce_value_margin,
                                 defensive_reinforce_overfill=defensive_reinforce_overfill,
                                 natural_head_audit=natural_head_audit,
                                 natural_head_audit_beta=natural_head_audit_beta,
                                 shard_idx=shard_idx, shard_count=shard_count,
                                 collect_records=collect_records)
        if shard_count <= 1:                 # real shards (>1) stay silent; --panel-out prints normally
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
    parser.add_argument("--panel-out", type=str, default=None,
                        help="Pickle the full --panel per-game records here (each game's conv dict "
                             "incl. dm_ratios), AND print the report normally. recompute_panel.py "
                             "re-derives any metric offline — so a later metric addition never needs a "
                             "panel re-run. No effect without --panel.")
    args = parser.parse_args()
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
        collect_records=bool(args.shard_out) or bool(args.panel_out),
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
    if args.panel_out and _eval_result is not None:
        import pickle
        with open(args.panel_out, "wb") as _f:
            pickle.dump({"records": _eval_result.get("_records", []),
                         "opponent": args.opponent}, _f)
        print(f"PANEL RECORDS → {args.panel_out}: "
              f"{len(_eval_result.get('_records', []))} games "
              f"(recompute any metric: python orbit_wars_rl/recompute_panel.py {args.panel_out})",
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
