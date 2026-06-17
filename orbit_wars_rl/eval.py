"""Evaluation: pit trained PyTorch policy against baselines."""

from __future__ import annotations

import argparse
import json
import math
import os
from statistics import mean

import torch
import numpy as np

from config import Config
from model import EntityTransformer, NUM_ANGLE_BINS, NUM_SHIP_BINS, ANGLE_BIN_WIDTH
from features import extract_features, _ETA_PROBE_SPEED
from action_mask import compute_action_masks, actions_from_policy, actions_from_target_policy


def _producer_candidate_overlay_moves(
    obs: dict,
    player: int,
    existing_moves: list,
    *,
    max_moves: int,
    score_min: float,
    target_owner: str,
    reranker: dict | None = None,
    trace: list | None = None,
    trace_top_k: int = 5,
) -> list:
    if max_moves <= 0:
        return []
    try:
        from orbit_wars_rl.analyze_producer_action_ranking import _enumerate_attack_candidates
        from orbit_wars_rl.action_mask import _target_intercept_angle
        from orbit_wars_rl.build_producer_reranker import _candidate_features
    except Exception:
        return []

    def owner_ok(candidate) -> bool:
        if target_owner == "any":
            return True
        if target_owner == "own":
            return bool(candidate.target_is_mine)
        if target_owner == "not-own":
            return not bool(candidate.target_is_mine)
        if target_owner == "neutral":
            return bool(candidate.target_is_neutral)
        if target_owner == "enemy":
            return (not bool(candidate.target_is_mine)) and (not bool(candidate.target_is_neutral))
        return False

    used_sources = {int(m[0]) for m in existing_moves if isinstance(m, list) and len(m) >= 1}
    planets = obs.get("planets") or []
    out = []
    try:
        candidates = _enumerate_attack_candidates(obs)["candidates"]
    except Exception:
        return []

    ranked = []
    filter_stats = {
        "raw_candidates": len(candidates),
        "kept": 0,
        "invalid": 0,
        "below_score": 0,
        "used_source": 0,
        "owner": 0,
        "bad_index": 0,
        "zero_ships": 0,
        "reranker_error": 0,
    }
    for raw_rank, candidate in enumerate(candidates):
        if not candidate.valid or candidate.score < score_min:
            if not candidate.valid:
                filter_stats["invalid"] += 1
            else:
                filter_stats["below_score"] += 1
            continue
        if int(candidate.source_id) in used_sources:
            filter_stats["used_source"] += 1
            continue
        if not owner_ok(candidate):
            filter_stats["owner"] += 1
            continue
        if int(candidate.source_idx) >= len(planets) or int(candidate.target_idx) >= len(planets):
            filter_stats["bad_index"] += 1
            continue
        if int(candidate.ships) <= 0:
            filter_stats["zero_ships"] += 1
            continue
        score = float(candidate.score)
        if reranker is not None:
            try:
                x = torch.tensor(_candidate_features(obs, candidate), dtype=torch.float32)
                n = int(reranker["weights"].numel())
                if x.numel() < n:
                    x = torch.cat([x, torch.zeros(n - x.numel(), dtype=x.dtype)])
                elif x.numel() > n:
                    x = x[:n]
                x = (x - reranker["mean"][:n]) / reranker["std"][:n].clamp(min=1e-6)
                score = float(torch.sigmoid((x * reranker["weights"]).sum() + reranker["bias"]).item())
            except Exception:
                filter_stats["reranker_error"] += 1
                continue
        ranked.append((score, raw_rank, candidate))
        filter_stats["kept"] += 1
    ranked.sort(key=lambda item: item[0], reverse=True)

    selected = []
    for score, raw_rank, candidate in ranked:
        if len(out) >= max_moves:
            break
        ships = int(candidate.ships)
        src = planets[int(candidate.source_idx)]
        target = planets[int(candidate.target_idx)]
        angle = _target_intercept_angle(src, target, ships, obs)
        move = [int(candidate.source_id), float(angle), ships]
        out.append(move)
        used_sources.add(int(candidate.source_id))
        selected.append({
            "rerank_score": score,
            "producer_rank": raw_rank,
            "move": move,
            "candidate": candidate.to_dict(),
        })
    if trace is not None and ranked:
        trace.append({
            "step": int(obs.get("step", 0)),
            "player": int(player),
            "existing_moves": existing_moves,
            "filter_stats": filter_stats,
            "selected": selected,
            "top": [
                {
                    "rerank_score": float(score),
                    "producer_rank": int(raw_rank),
                    "candidate": candidate.to_dict(),
                }
                for score, raw_rank, candidate in ranked[:max(0, trace_top_k)]
            ],
        })
    return out


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
        cfg.model.pairwise_feature_dim = int(sd["pair_kv.weight"].shape[1]) - D
    else:
        cfg.model.pairwise_feature_dim = 0

    # Detect value head version from fc1 input width (old=D, new=2D).
    if "value_fc1.weight" in sd:
        cfg.model.value_head_in = int(sd["value_fc1.weight"].shape[1])
    cfg.model.use_threat_head = "threat_head.weight" in sd

    action_decode = str(ckpt_cfg.get("action_decode", "angle"))
    # Reinforcement: eval must mask targets the SAME way the checkpoint was trained.
    cfg.model.allow_reinforce = bool(ckpt_cfg.get("allow_reinforce", False))
    return sd, action_decode


def build_agent_fn(model: EntityTransformer, device: torch.device,
                   fire_threshold: float = 0.5, sample: bool = False,
                   ship_bin_mode: str = "absolute",
                   target_decode: bool = False,
                   target_sanity_penalty: float = 0.0,
                   reserve_frac: float = 0.0,
                   allow_reinforce: bool = False,
                   threat_target_bias: float = 0.0,
                   reinforce_target_bias: float = 0.0,
                   defense_overlay: bool = False,
                   defense_overlay_recent_capture_window: int = 0,
                   defense_overlay_garrison_floor: int = 10,
                   defense_overlay_min_need: int = 5,
                   defense_overlay_max_moves: int = 1,
                   defense_overlay_selector: dict | None = None,
                   defense_overlay_selector_threshold: float = 0.5,
                   defense_overlay_selector_mode: str = "survive",
                   defense_overlay_multi_source_per_target: bool = False,
                   producer_overlay: bool = False,
                   producer_overlay_max_moves: int = 1,
                   producer_overlay_score_min: float = 1.5,
                   producer_overlay_target_owner: str = "any",
                   producer_overlay_late_step: int = 0,
                   producer_overlay_late_score_min: float | None = None,
                   producer_overlay_late_target_owner: str = "",
                   producer_reranker: dict | None = None,
                   producer_overlay_trace: list | None = None,
                   producer_overlay_trace_top_k: int = 5,
                   trace_context: dict | None = None,
                   veto_stats: dict = None):
    """Return a kaggle_environments-compatible agent function wrapping the model.

    sample=True uses Bernoulli/Categorical sampling instead of threshold/argmax —
    helps when the training-time distribution is multi-modal but the mode is
    degenerate (e.g. 1-ship-fleet trap).
    """
    model.eval()
    prev_owners: dict[int, int] = {}
    capture_steps: dict[int, int] = {}
    last_step = -1
    last_player = None

    def agent_fn(obs):
        nonlocal prev_owners, capture_steps, last_step, last_player
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
        step = int(obs.get("step", 0))
        if step <= last_step or last_player != player:
            prev_owners = {}
            capture_steps = {}
        for p in obs.get("planets") or []:
            pid = int(p[0])
            owner = int(p[1])
            was = prev_owners.get(pid)
            if was is not None and was != player and owner == player:
                capture_steps[pid] = step
            prev_owners[pid] = owner
        last_step = step
        last_player = player

        eligible_defense_targets = None
        defense_target_ages = None
        if defense_overlay and defense_overlay_recent_capture_window > 0:
            eligible_defense_targets = {
                pid for pid, cap_step in capture_steps.items()
                if 0 <= step - cap_step <= defense_overlay_recent_capture_window
            }
            defense_target_ages = {
                pid: step - cap_step
                for pid, cap_step in capture_steps.items()
                if 0 <= step - cap_step <= defense_overlay_recent_capture_window
            }
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
                threat_logits=outputs.get("threat_logits"),
                threat_target_bias=threat_target_bias,
                reinforce_target_bias=reinforce_target_bias,
                defense_overlay=defense_overlay,
                defense_overlay_garrison_floor=defense_overlay_garrison_floor,
                defense_overlay_min_need=defense_overlay_min_need,
                defense_overlay_max_moves=defense_overlay_max_moves,
                defense_overlay_eligible_target_pids=eligible_defense_targets,
                defense_overlay_target_ages=defense_target_ages,
                defense_overlay_selector=defense_overlay_selector,
                defense_overlay_selector_threshold=defense_overlay_selector_threshold,
                defense_overlay_selector_mode=defense_overlay_selector_mode,
                defense_overlay_multi_source_per_target=defense_overlay_multi_source_per_target,
                veto_stats=veto_stats,
            )
            if producer_overlay:
                before_overlay = [list(m) for m in moves]
                overlay_trace = [] if producer_overlay_trace is not None else None
                step_score_min = producer_overlay_score_min
                step_target_owner = producer_overlay_target_owner
                if producer_overlay_late_step > 0 and step >= producer_overlay_late_step:
                    if producer_overlay_late_score_min is not None:
                        step_score_min = producer_overlay_late_score_min
                    if producer_overlay_late_target_owner:
                        step_target_owner = producer_overlay_late_target_owner
                extra_moves = _producer_candidate_overlay_moves(
                    obs,
                    player,
                    moves,
                    max_moves=min(producer_overlay_max_moves, max(0, 8 - len(moves))),
                    score_min=step_score_min,
                    target_owner=step_target_owner,
                    reranker=producer_reranker,
                    trace=overlay_trace,
                    trace_top_k=producer_overlay_trace_top_k,
                )
                moves.extend(extra_moves)
                if producer_overlay_trace is not None and overlay_trace:
                    ctx = dict(trace_context or {})
                    for entry in overlay_trace:
                        entry.update(ctx)
                        entry["model_moves_before_overlay"] = before_overlay
                        entry["overlay_moves"] = extra_moves
                        entry["final_moves"] = [list(m) for m in moves]
                        entry["effective_score_min"] = step_score_min
                        entry["effective_target_owner"] = step_target_owner
                        producer_overlay_trace.append(entry)
            return moves

        raise NotImplementedError(
            "angle-decode path removed (angle head deleted); Phase 1 checkpoints "
            "use target-decode. Pass target_decode=True (--target-decode)."
        )

    return agent_fn


_CONV_MILESTONES = (16, 32, 50, 100)
# redundant/underkill are windowed to the OPENING: late-game surplus production re-fires at the
# last enemy planets (benign — we've already won), which inflates a whole-game fraction in long
# won games. The launch waste we care about is in the opening, where wasted ships feed the
# mid-game collapse. <50 isolates that phase. (phase2 / metrics.md)
_LAUNCH_WINDOW = 50
# Isaiah (#1 player) hoard reference at the same milestones. Contested phase (16-50)
# is the clean read: ~half the army deployed, ~11-22 ships/planet. The @100 jump
# (garr 0.87, 60 ships/planet) is won-game accumulation, not hoarding.
_ISAIAH_HOARD_REF = "garr_frac 0.50/0.51/0.54/0.87  ships/planet 11/15/22/60"

# reinforce-by-empire-size bins (owned planets AT LAUNCH TIME). The aggregate reinf_share
# is opponent/success-confounded (it co-moves with empire size — phase2 §6); bucketing by
# empire size makes it directly comparable to the top-player ramp (phase2 §2 / metrics.md):
# @1 ≈0.00, @2 ≈0.10, @9-12 ≈0.30, @13+ 0.34-0.61.
_REINF_BINS = [(1, 1, "1"), (2, 3, "2-3"), (4, 6, "4-6"),
               (7, 9, "7-9"), (10, 12, "10-12"), (13, 10**9, "13+")]
_REINF_RAMP_REF = "@1:0.00 @2:0.10 @9-12:0.30 @13+:0.34-0.61"


def _reinf_bin_idx(owned):
    for i, (lo, hi, _) in enumerate(_REINF_BINS):
        if lo <= owned <= hi:
            return i
    return 0  # owned 0 can't launch; guard


def _resolve_launch_target(planets, src, angle):
    """Planet a launch from `src` at `angle` is aimed at (direction match), or None.
    Mirrors fetch_analyze_top_replays._resolve_target so eval == replay analysis."""
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
    caps = atk = reinf = atk_ships = redundant = underkill = 0
    atk_early = redundant_early = underkill_early = 0      # opening window (t < _LAUNCH_WINDOW)
    # Retention: of the planets we CAPTURE, how many do we then lose, and how long did we hold
    # them? cap_step[pid] = step we (most recently) took pid; on a later loss we close the episode.
    # lost_caps/captures is the recapture/turnover rate — immune to the end->0 churn degeneracy.
    # Home/initial planets are excluded by construction (never entered cap_step).
    cap_step: dict = {}
    lost_caps = 0
    hold_durations: list = []   # steps held before losing (lost episodes only; held-to-end censored)
    reinf_bin = [0] * len(_REINF_BINS)   # own-target launches by empire size at launch
    atk_bin = [0] * len(_REINF_BINS)     # attack launches by empire size at launch
    planets_at = {ms: None for ms in _CONV_MILESTONES}
    garrison_at = {ms: None for ms in _CONV_MILESTONES}   # ships parked on owned planets
    inflight_at = {ms: None for ms in _CONV_MILESTONES}   # ships in owned fleets (deployed)
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
            owned_now = garrison_now = 0
            for p in p1:
                pid, own = p[0], int(p[1])
                if own == seat:
                    owned_now += 1
                    garrison_now += p[5]
                was = prev.get(pid)
                if was is not None and was != seat and own == seat:
                    caps += 1
                    cap_step[pid] = t                  # open a hold episode
                elif was == seat and own != seat and pid in cap_step:
                    hold_durations.append(t - cap_step[pid])   # lost what we took
                    lost_caps += 1
                    del cap_step[pid]
                prev[pid] = own
            last = p1
            if t in planets_at:
                fleets = steps[t][seat].observation.get("fleets") or []
                planets_at[t] = owned_now
                garrison_at[t] = garrison_now
                inflight_at[t] = sum(f[6] for f in fleets if int(f[1]) == seat)
        if not p0:
            continue
        byid = {p[0]: p for p in p0}
        f0 = steps[t - 1][seat].observation.get("fleets") or []   # in-flight at decision time
        bidx = _reinf_bin_idx(sum(1 for p in p0 if int(p[1]) == seat))  # empire size at decision
        for mv in acts:
            if not mv or len(mv) < 3:
                continue
            src = byid.get(int(mv[0]))
            if src is None:
                continue
            sent, ssh = int(mv[2]), float(src[5])
            if not (ssh > 0 and sent <= ssh):       # legal launches only
                continue
            tgt = _resolve_launch_target(p0, src, float(mv[1]))
            if tgt is None:
                continue                            # unclassifiable → skip (== analyzer)
            if int(tgt[1]) == seat:
                reinf += 1                          # reinforce: cannot capture
                reinf_bin[bidx] += 1
            else:
                atk += 1
                atk_ships += sent
                atk_bin[bidx] += 1
                early = t < _LAUNCH_WINDOW
                if early:
                    atk_early += 1
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
                if fin >= capcost > 0:
                    redundant += 1
                    if early:
                        redundant_early += 1
                else:
                    eta = max(1, int(math.ceil(
                        math.hypot(tgt[2] - src[2], tgt[3] - src[3]) / _ETA_PROBE_SPEED)))
                    pid = tgt[0]
                    if not any(pid in us_pids_at[s] for s in range(t + 1, min(t + eta + 11, T))):
                        underkill += 1
                        if early:
                            underkill_early += 1
    end_planets = sum(1 for p in (last or []) if int(p[1]) == seat)
    out = {"captures": caps, "attack_launches": atk, "reinforce_launches": reinf,
           "attack_ships": atk_ships, "end_planets": end_planets,
           "redundant": redundant, "underkill": underkill, "atk_early": atk_early,
           "redundant_early": redundant_early, "underkill_early": underkill_early,
           "lost_caps": lost_caps, "hold_durations": hold_durations,
           "glen": len(steps), "reinf_bin": reinf_bin, "atk_bin": atk_bin}
    for ms in _CONV_MILESTONES:
        out[f"p{ms}"] = planets_at[ms]
        out[f"g{ms}"] = garrison_at[ms]
        out[f"if{ms}"] = inflight_at[ms]
    return out


def new_conversion_acc():
    acc = {"captures": 0, "attack_launches": 0, "reinforce_launches": 0,
           "attack_ships": 0, "end_planets": 0, "redundant": 0, "underkill": 0,
           "glen_sum": 0, "games": 0, "atk_early": 0, "redundant_early": 0, "underkill_early": 0,
           "lost_caps": 0, "hold_durations": [],
           "reinf_bin": [0] * len(_REINF_BINS), "atk_bin": [0] * len(_REINF_BINS)}
    for ms in _CONV_MILESTONES:
        acc[f"p{ms}_sum"] = 0
        acc[f"p{ms}_n"] = 0
        acc[f"g{ms}_sum"] = 0    # garrison (parked) ships, summed over games reaching ms
        acc[f"if{ms}_sum"] = 0   # in-flight (deployed) ships, summed over games reaching ms
    return acc


def add_conversion(acc, conv):
    for k in ("captures", "attack_launches", "reinforce_launches", "attack_ships",
              "end_planets", "redundant", "underkill", "atk_early", "redundant_early",
              "underkill_early", "lost_caps"):
        acc[k] += conv[k]
    acc["hold_durations"].extend(conv["hold_durations"])
    acc["glen_sum"] += conv["glen"]
    for i in range(len(_REINF_BINS)):
        acc["reinf_bin"][i] += conv["reinf_bin"][i]
        acc["atk_bin"][i] += conv["atk_bin"][i]
    acc["games"] += 1
    for ms in _CONV_MILESTONES:
        v = conv[f"p{ms}"]
        if v is not None:
            acc[f"p{ms}_sum"] += v
            acc[f"p{ms}_n"] += 1
            acc[f"g{ms}_sum"] += conv[f"g{ms}"]
            acc[f"if{ms}_sum"] += conv[f"if{ms}"]


def _fmt_conversion(acc):
    """Two-line conversion summary. cap/launch counts ATTACK launches only
    (reinforce can't capture). Reference = Isaiah (#1 player)."""
    n = max(acc["games"], 1)
    c, al, rl = acc["captures"], acc["attack_launches"], acc["reinforce_launches"]
    pl = lambda ms: (f"{acc[f'p{ms}_sum']/acc[f'p{ms}_n']:.0f}" if acc[f"p{ms}_n"] else "—")
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
    redf = acc["redundant_early"] / max(acc["atk_early"], 1)
    redf_wg = acc["redundant"] / max(al, 1)
    undf = acc["underkill_early"] / max(acc["atk_early"], 1)
    undf_wg = acc["underkill"] / max(al, 1)
    return (f"Conversion: caps/game {c/n:.1f}  atk-launch/game {al/n:.1f}  "
            f"cap/atk-launch {c/max(al,1):.3f}  ships/cap {acc['attack_ships']/max(c,1):.0f}  "
            f"reinf_share {rl/max(al+rl,1):.2f}\n"
            f"  planets@16/32/50/100 {pl(16)}/{pl(32)}/{pl(50)}/{pl(100)}  end {acc['end_planets']/n:.1f}"
            f"   churn {churn:.2f} ({churn_n:.2f}/100st, len {glen:.0f})"
            f"\n  retention  lost-cap {lost_rate:.2f} ({acc['lost_caps']}/{c} caps)  median-hold {med_hold}st"
            f"\n  launch-waste<50  redundant {redf:.2f} (WG {redf_wg:.2f})  underkill {undf:.2f} (WG {undf_wg:.2f})"
            f"   [ref Isaiah: cap/atk-launch 0.59  planets 2/6/9/10  reinf 0.30]\n"
            f"  hoard  garr_frac@ {gf(16)}/{gf(32)}/{gf(50)}/{gf(100)}  "
            f"ships/planet@ {spp(16)}/{spp(32)}/{spp(50)}/{spp(100)}"
            f"   [ref Isaiah: {_ISAIAH_HOARD_REF}]\n"
            f"  reinf by empire size  {ramp}   [ref ramp {_REINF_RAMP_REF}]")


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
    threat_target_bias: float = 0.0,
    reinforce_target_bias: float = 0.0,
    defense_overlay: bool = False,
    defense_overlay_recent_capture_window: int = 0,
    defense_overlay_garrison_floor: int = 10,
    defense_overlay_min_need: int = 5,
    defense_overlay_max_moves: int = 1,
    defense_overlay_selector: dict | None = None,
    defense_overlay_selector_threshold: float = 0.5,
    defense_overlay_selector_mode: str = "survive",
    defense_overlay_multi_source_per_target: bool = False,
    producer_overlay: bool = False,
    producer_overlay_max_moves: int = 1,
    producer_overlay_score_min: float = 1.5,
    producer_overlay_target_owner: str = "any",
    producer_overlay_late_step: int = 0,
    producer_overlay_late_score_min: float | None = None,
    producer_overlay_late_target_owner: str = "",
    producer_reranker: dict | None = None,
    producer_overlay_trace: list | None = None,
    producer_overlay_trace_top_k: int = 5,
) -> dict:
    """Evaluate trained policy against a baseline using kaggle_environments.

    Args:
        opponent: "random" or path to a Python agent file (e.g. "main.py")
        num_players: 2 or 4
    """
    from kaggle_environments import make

    trace_context = {}
    agent_fn = build_agent_fn(model, device, fire_threshold=fire_threshold, sample=sample,
                              ship_bin_mode=ship_bin_mode, target_decode=target_decode,
                              target_sanity_penalty=target_sanity_penalty,
                              threat_target_bias=threat_target_bias,
                              reinforce_target_bias=reinforce_target_bias,
                              defense_overlay=defense_overlay,
                              defense_overlay_recent_capture_window=defense_overlay_recent_capture_window,
                              defense_overlay_garrison_floor=defense_overlay_garrison_floor,
                              defense_overlay_min_need=defense_overlay_min_need,
                              defense_overlay_max_moves=defense_overlay_max_moves,
                              defense_overlay_selector=defense_overlay_selector,
                              defense_overlay_selector_threshold=defense_overlay_selector_threshold,
                              defense_overlay_selector_mode=defense_overlay_selector_mode,
                              defense_overlay_multi_source_per_target=defense_overlay_multi_source_per_target,
                              producer_overlay=producer_overlay,
                              producer_overlay_max_moves=producer_overlay_max_moves,
                              producer_overlay_score_min=producer_overlay_score_min,
                              producer_overlay_target_owner=producer_overlay_target_owner,
                              producer_overlay_late_step=producer_overlay_late_step,
                              producer_overlay_late_score_min=producer_overlay_late_score_min,
                              producer_overlay_late_target_owner=producer_overlay_late_target_owner,
                              producer_reranker=producer_reranker,
                              producer_overlay_trace=producer_overlay_trace,
                              producer_overlay_trace_top_k=producer_overlay_trace_top_k,
                              trace_context=trace_context)
    opponents = [opponent] * (num_players - 1)
    agents = [agent_fn] + opponents

    wins = 0
    total_material = 0
    conv_tot = new_conversion_acc()
    results = []

    for seed in range(seed_start, seed_start + num_games):
        trace_context.clear()
        trace_context.update({"seed": seed, "mode": "quick"})
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run(agents)
        final = env.steps[-1]
        rewards = [s.reward for s in final]

        add_conversion(conv_tot, game_conversion(env.steps, 0))

        obs = final[0].observation
        material = sum(p[5] for p in obs.planets if p[1] == 0)
        material += sum(f[6] for f in obs.fleets if f[1] == 0)

        # Rank by reward; player 0 wins if their reward is strictly highest
        my_reward = rewards[0] if rewards[0] is not None else 0.0
        best_opp = max((r for r in rewards[1:] if r is not None), default=0.0)
        is_win = my_reward > best_opp

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


def evaluate_panel(
    model: EntityTransformer,
    device: torch.device,
    opponent: str,
    fire_threshold: float = 0.5,
    sample: bool = False,
    ship_bin_mode: str = "absolute",
    target_decode: bool = False,
    target_sanity_penalty: float = 0.0,
    threat_target_bias: float = 0.0,
    reinforce_target_bias: float = 0.0,
    defense_overlay: bool = False,
    defense_overlay_recent_capture_window: int = 0,
    defense_overlay_garrison_floor: int = 10,
    defense_overlay_min_need: int = 5,
    defense_overlay_max_moves: int = 1,
    defense_overlay_selector: dict | None = None,
    defense_overlay_selector_threshold: float = 0.5,
    defense_overlay_selector_mode: str = "survive",
    defense_overlay_multi_source_per_target: bool = False,
    producer_overlay: bool = False,
    producer_overlay_max_moves: int = 1,
    producer_overlay_score_min: float = 1.5,
    producer_overlay_target_owner: str = "any",
    producer_overlay_late_step: int = 0,
    producer_overlay_late_score_min: float | None = None,
    producer_overlay_late_target_owner: str = "",
    producer_reranker: dict | None = None,
    producer_overlay_trace: list | None = None,
    producer_overlay_trace_top_k: int = 5,
) -> dict:
    """Stratified eval over the 128-seed community panel, playing both seats.

    256 games per opponent (128 seeds × 2 seats). Aggregates wins per
    archetype (8 games per cell = 4 seeds × 2 seats) and per seat, so a
    +5pp overall regression hidden by an asymmetric or board-shape-specific
    weakness is visible.
    """
    from kaggle_environments import make
    from eval_panel import BY_ARCHETYPE

    trace_context = {}
    agent_fn = build_agent_fn(model, device, fire_threshold=fire_threshold, sample=sample,
                              ship_bin_mode=ship_bin_mode, target_decode=target_decode,
                              target_sanity_penalty=target_sanity_penalty,
                              threat_target_bias=threat_target_bias,
                              reinforce_target_bias=reinforce_target_bias,
                              defense_overlay=defense_overlay,
                              defense_overlay_recent_capture_window=defense_overlay_recent_capture_window,
                              defense_overlay_garrison_floor=defense_overlay_garrison_floor,
                              defense_overlay_min_need=defense_overlay_min_need,
                              defense_overlay_max_moves=defense_overlay_max_moves,
                              defense_overlay_selector=defense_overlay_selector,
                              defense_overlay_selector_threshold=defense_overlay_selector_threshold,
                              defense_overlay_selector_mode=defense_overlay_selector_mode,
                              defense_overlay_multi_source_per_target=defense_overlay_multi_source_per_target,
                              producer_overlay=producer_overlay,
                              producer_overlay_max_moves=producer_overlay_max_moves,
                              producer_overlay_score_min=producer_overlay_score_min,
                              producer_overlay_target_owner=producer_overlay_target_owner,
                              producer_overlay_late_step=producer_overlay_late_step,
                              producer_overlay_late_score_min=producer_overlay_late_score_min,
                              producer_overlay_late_target_owner=producer_overlay_late_target_owner,
                              producer_reranker=producer_reranker,
                              producer_overlay_trace=producer_overlay_trace,
                              producer_overlay_trace_top_k=producer_overlay_trace_top_k,
                              trace_context=trace_context)

    per_arch: dict[str, dict] = {arch: {"wins": 0, "total": 0,
                                        "wins_seat0": 0, "wins_seat1": 0,
                                        "total_seat0": 0, "total_seat1": 0,
                                        "material_sum": 0}
                                 for arch in BY_ARCHETYPE}
    overall = {"wins": 0, "total": 0, "wins_seat0": 0, "wins_seat1": 0,
               "total_seat0": 0, "total_seat1": 0}
    conv_tot = new_conversion_acc()
    game_idx = 0
    total_games = sum(len(seeds) for seeds in BY_ARCHETYPE.values()) * 2

    for archetype, seeds in BY_ARCHETYPE.items():
        for seed in seeds:
            for my_seat in (0, 1):
                trace_context.clear()
                trace_context.update({
                    "seed": seed,
                    "my_seat": my_seat,
                    "archetype": archetype,
                    "mode": "panel",
                })
                agents = [agent_fn, opponent] if my_seat == 0 else [opponent, agent_fn]
                env = make("orbit_wars", configuration={"seed": seed}, debug=False)
                env.run(agents)
                final = env.steps[-1]
                add_conversion(conv_tot, game_conversion(env.steps, my_seat))
                rewards = [s.reward if s.reward is not None else 0.0 for s in final]
                my_reward = rewards[my_seat]
                opp_reward = rewards[1 - my_seat]
                is_win = my_reward > opp_reward
                # Material on the model's side
                obs = final[0].observation
                material = sum(p[5] for p in obs.planets if p[1] == my_seat)
                material += sum(f[6] for f in obs.fleets if f[1] == my_seat)

                c = per_arch[archetype]
                c["wins"] += int(is_win); c["total"] += 1
                c[f"wins_seat{my_seat}"] += int(is_win)
                c[f"total_seat{my_seat}"] += 1
                c["material_sum"] += material
                overall["wins"] += int(is_win); overall["total"] += 1
                overall[f"wins_seat{my_seat}"] += int(is_win)
                overall[f"total_seat{my_seat}"] += 1
                game_idx += 1
                if game_idx % 16 == 0 or game_idx == total_games:
                    print(f"  panel progress: {game_idx}/{total_games}  "
                          f"overall {overall['wins']}/{overall['total']} "
                          f"({100*overall['wins']/max(overall['total'],1):.1f}%)",
                          flush=True)

    return {"overall": overall, "per_archetype": per_arch, "conversion": conv_tot}


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


def evaluate_checkpoint(params_path: str, cfg: Config, num_games: int = 32,
                        seed_start: int = 0,
                        opponent: str = "random", fire_threshold: float = 0.5,
                        panel: bool = False, sample: bool = False,
                        target_decode: bool = False,
                        target_sanity_penalty: float = 0.0,
                        threat_target_bias: float = 0.0,
                        reinforce_target_bias: float = 0.0,
                        reinforce_gate_min_planets: int = 0,
                        reinforce_forward_only: bool = False,
                        reinforce_garrison_floor: float = 0.0,
                        defense_overlay: bool = False,
                        defense_overlay_recent_capture_window: int = 0,
                        defense_overlay_garrison_floor: int = 10,
                        defense_overlay_min_need: int = 5,
                        defense_overlay_max_moves: int = 1,
                        defense_overlay_selector_checkpoint: str = "",
                        defense_overlay_selector_threshold: float = 0.5,
                        defense_overlay_selector_mode: str = "survive",
                        defense_overlay_multi_source_per_target: bool = False,
                        producer_overlay: bool = False,
                        producer_overlay_max_moves: int = 1,
                        producer_overlay_score_min: float = 1.5,
                        producer_overlay_target_owner: str = "any",
                        producer_overlay_late_step: int = 0,
                        producer_overlay_late_score_min: float | None = None,
                        producer_overlay_late_target_owner: str = "",
                        producer_reranker_checkpoint: str = "",
                        producer_overlay_trace_json: str = "",
                        producer_overlay_trace_top_k: int = 5):
    """Load a checkpoint and evaluate it."""
    device = torch.device(cfg.device)

    state_dict, ckpt_action_decode = load_checkpoint(params_path, cfg)
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
    model.reinforce_garrison_floor = float(reinforce_garrison_floor)
    if model.allow_reinforce:
        print(f"Reinforcement: ON (own planets are legal targets) | "
              f"gate>={model.reinforce_gate_min_planets} planets, "
              f"forward_only={model.reinforce_forward_only}, "
              f"garrison_floor={model.reinforce_garrison_floor}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"target_head.weight", "target_head.bias"}
    bad_missing = [k for k in missing if k not in allowed_missing]
    # VDN per-planet value head (Stage 2) is never used at eval — ignore it if the
    # checkpoint carries it but this (eval-time) model doesn't.
    bad_unexpected = [k for k in unexpected if not k.startswith("value_pp_")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch: missing={bad_missing}, unexpected={bad_unexpected}")
    model.eval()
    defense_overlay_selector = None
    if defense_overlay_selector_checkpoint:
        defense_overlay_selector = torch.load(defense_overlay_selector_checkpoint, map_location="cpu")
        print(f"Loaded defense overlay selector: {defense_overlay_selector_checkpoint}")
    producer_reranker = None
    if producer_reranker_checkpoint:
        producer_reranker = torch.load(producer_reranker_checkpoint, map_location="cpu")
        print(f"Loaded Producer reranker: {producer_reranker_checkpoint}")
    producer_overlay_trace = [] if producer_overlay_trace_json else None

    if panel:
        results = evaluate_panel(model, device, opponent=opponent,
                                 fire_threshold=fire_threshold, sample=sample,
                                 ship_bin_mode=cfg.model.ship_bin_mode,
                                 target_decode=target_decode,
                                 target_sanity_penalty=target_sanity_penalty,
                                 threat_target_bias=threat_target_bias,
                                 reinforce_target_bias=reinforce_target_bias,
                                 defense_overlay=defense_overlay,
                                 defense_overlay_recent_capture_window=defense_overlay_recent_capture_window,
                                 defense_overlay_garrison_floor=defense_overlay_garrison_floor,
                                 defense_overlay_min_need=defense_overlay_min_need,
                                 defense_overlay_max_moves=defense_overlay_max_moves,
                                 defense_overlay_selector=defense_overlay_selector,
                                 defense_overlay_selector_threshold=defense_overlay_selector_threshold,
                                 defense_overlay_selector_mode=defense_overlay_selector_mode,
                                 defense_overlay_multi_source_per_target=defense_overlay_multi_source_per_target,
                                 producer_overlay=producer_overlay,
                                 producer_overlay_max_moves=producer_overlay_max_moves,
                                 producer_overlay_score_min=producer_overlay_score_min,
                                 producer_overlay_target_owner=producer_overlay_target_owner,
                                 producer_overlay_late_step=producer_overlay_late_step,
                                 producer_overlay_late_score_min=producer_overlay_late_score_min,
                                 producer_overlay_late_target_owner=producer_overlay_late_target_owner,
                                 producer_reranker=producer_reranker,
                                 producer_overlay_trace=producer_overlay_trace,
                                 producer_overlay_trace_top_k=producer_overlay_trace_top_k)
        if producer_overlay_trace_json and producer_overlay_trace is not None:
            os.makedirs(os.path.dirname(producer_overlay_trace_json) or ".", exist_ok=True)
            with open(producer_overlay_trace_json, "w") as f:
                json.dump(producer_overlay_trace, f, indent=2)
            print(f"Producer overlay trace: {producer_overlay_trace_json} ({len(producer_overlay_trace)} entries)")
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
        threat_target_bias=threat_target_bias,
        reinforce_target_bias=reinforce_target_bias,
        defense_overlay=defense_overlay,
        defense_overlay_recent_capture_window=defense_overlay_recent_capture_window,
        defense_overlay_garrison_floor=defense_overlay_garrison_floor,
        defense_overlay_min_need=defense_overlay_min_need,
        defense_overlay_max_moves=defense_overlay_max_moves,
        defense_overlay_selector=defense_overlay_selector,
        defense_overlay_selector_threshold=defense_overlay_selector_threshold,
        defense_overlay_selector_mode=defense_overlay_selector_mode,
        defense_overlay_multi_source_per_target=defense_overlay_multi_source_per_target,
        producer_overlay=producer_overlay,
        producer_overlay_max_moves=producer_overlay_max_moves,
        producer_overlay_score_min=producer_overlay_score_min,
        producer_overlay_target_owner=producer_overlay_target_owner,
        producer_overlay_late_step=producer_overlay_late_step,
        producer_overlay_late_score_min=producer_overlay_late_score_min,
        producer_overlay_late_target_owner=producer_overlay_late_target_owner,
        producer_reranker=producer_reranker,
        producer_overlay_trace=producer_overlay_trace,
        producer_overlay_trace_top_k=producer_overlay_trace_top_k,
    )
    if producer_overlay_trace_json and producer_overlay_trace is not None:
        os.makedirs(os.path.dirname(producer_overlay_trace_json) or ".", exist_ok=True)
        with open(producer_overlay_trace_json, "w") as f:
            json.dump(producer_overlay_trace, f, indent=2)
        print(f"Producer overlay trace: {producer_overlay_trace_json} ({len(producer_overlay_trace)} entries)")

    print(f"Win rate vs {opponent}: {results['win_rate']:.2%}  "
          f"({results['wins']}/{results['total_games']})")
    print(f"Fire threshold: {fire_threshold}")
    print(f"Target decode: {target_decode}")
    print(f"Target sanity penalty: {target_sanity_penalty}")
    if threat_target_bias:
        print(f"Threat target bias: {threat_target_bias}")
    if reinforce_target_bias:
        print(f"Reinforce target bias: {reinforce_target_bias}")
    if defense_overlay:
        print(f"Defense overlay: ON recent_capture_window={defense_overlay_recent_capture_window} "
              f"garrison_floor={defense_overlay_garrison_floor} "
              f"min_need={defense_overlay_min_need} max_moves={defense_overlay_max_moves} "
              f"multi_source_per_target={defense_overlay_multi_source_per_target}")
        if defense_overlay_selector is not None:
            print(f"Defense overlay selector threshold: {defense_overlay_selector_threshold} "
                  f"mode={defense_overlay_selector_mode}")
    if producer_overlay:
        print(f"Producer overlay: ON max_moves={producer_overlay_max_moves} "
              f"score_min={producer_overlay_score_min} target_owner={producer_overlay_target_owner}")
        if producer_overlay_late_step > 0:
            late_score = producer_overlay_late_score_min
            late_owner = producer_overlay_late_target_owner or producer_overlay_target_owner
            print(f"Producer overlay late schedule: step>={producer_overlay_late_step} "
                  f"score_min={late_score if late_score is not None else producer_overlay_score_min} "
                  f"target_owner={late_owner}")
    print(f"Avg material: {results['avg_material']:.1f}")
    print(_fmt_conversion(results["conversion"]))
    for r in results["results"][:5]:
        print(f"  seed={r['seed']} win={r['win']} "
              f"material={r['material']} rewards={r['rewards']}")

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
    parser.add_argument("--target-sanity-penalty", type=float, default=0.0,
                        help="Subtract this from dominated same-source target logits "
                             "before target decode.")
    parser.add_argument("--threat-target-bias", type=float, default=0.0,
                        help="If checkpoint has a threat head, add this times predicted "
                             "P(owned target lost soon) to own-target logits before "
                             "target decode.")
    parser.add_argument("--reinforce-target-bias", type=float, default=0.0,
                        help="Add this fixed bias to all own-target logits before "
                             "target decode. Negative values suppress reinforcement "
                             "without making own targets illegal.")
    parser.add_argument("--reinforce-gate-min-planets", type=int, default=0,
                        help="Reinforce-discipline parity: own targets legal only at "
                             ">= this many owned planets. MUST match training (p2rev1=3).")
    parser.add_argument("--reinforce-forward-only", action="store_true",
                        help="Reinforce-discipline parity: own target legal only if closer "
                             "to the nearest enemy than the source. MUST match training.")
    parser.add_argument("--reinforce-garrison-floor", type=float, default=0.0,
                        help="Reinforce-discipline parity: veto a reinforce that drains the "
                             "source below this. MUST match training (p2rev1=10).")
    parser.add_argument("--defense-overlay", action="store_true",
                        help="After target decode, append conservative rear-source support "
                             "moves to threatened owned planets. This is an isolated "
                             "supervised/synthetic-defense inference ablation.")
    parser.add_argument("--defense-overlay-recent-capture-window", type=int, default=0,
                        help="If >0, defense overlay only targets planets captured by "
                             "the model within this many observed steps.")
    parser.add_argument("--defense-overlay-garrison-floor", type=int, default=10,
                        help="Minimum ships to leave on overlay support sources.")
    parser.add_argument("--defense-overlay-min-need", type=int, default=5,
                        help="Minimum projected defensive surplus for overlay support.")
    parser.add_argument("--defense-overlay-max-moves", type=int, default=1,
                        help="Maximum extra support moves appended per step.")
    parser.add_argument("--defense-overlay-multi-source-per-target", action="store_true",
                        help="Allow defense overlay to use multiple rear sources for the "
                             "same threatened target when max_moves permits.")
    parser.add_argument("--defense-overlay-selector-checkpoint", default="",
                        help="Optional selector checkpoint from build_defense_selector.py; "
                             "when set, overlay candidates below threshold are skipped.")
    parser.add_argument("--defense-overlay-selector-threshold", type=float, default=0.5,
                        help="Minimum selector survival score required to fire an overlay move.")
    parser.add_argument("--defense-overlay-selector-mode", choices=["survive", "risk"],
                        default="survive",
                        help="'survive': fire when predicted survival >= threshold. "
                             "'risk': fire when predicted survival <= threshold.")
    parser.add_argument("--producer-overlay", action="store_true",
                        help="Append high-confidence Producer planner candidates after model decode. "
                             "Diagnostic upper bound for supervised distillation.")
    parser.add_argument("--producer-overlay-max-moves", type=int, default=1)
    parser.add_argument("--producer-overlay-score-min", type=float, default=1.5)
    parser.add_argument("--producer-overlay-target-owner",
                        choices=["any", "own", "not-own", "neutral", "enemy"], default="any")
    parser.add_argument("--producer-overlay-late-step", type=int, default=0,
                        help="If >0, switch Producer overlay filters at this step.")
    parser.add_argument("--producer-overlay-late-score-min", type=float, default=None,
                        help="Optional score_min after --producer-overlay-late-step.")
    parser.add_argument("--producer-overlay-late-target-owner",
                        choices=["", "any", "own", "not-own", "neutral", "enemy"], default="",
                        help="Optional target-owner filter after --producer-overlay-late-step.")
    parser.add_argument("--producer-reranker-checkpoint", default="",
                        help="Optional supervised reranker checkpoint from build_producer_reranker.py. "
                             "When set, Producer overlay candidates are ordered by reranker score.")
    parser.add_argument("--producer-overlay-trace-json", default="",
                        help="If set, write per-step Producer overlay candidate choices to this JSON file.")
    parser.add_argument("--producer-overlay-trace-top-k", type=int, default=5,
                        help="Number of reranked candidates to keep per trace entry.")
    args = parser.parse_args()

    cfg = Config()
    cfg.env.num_players = args.num_players
    evaluate_checkpoint(
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
        threat_target_bias=args.threat_target_bias,
        reinforce_target_bias=args.reinforce_target_bias,
        reinforce_gate_min_planets=args.reinforce_gate_min_planets,
        reinforce_forward_only=args.reinforce_forward_only,
        reinforce_garrison_floor=args.reinforce_garrison_floor,
        defense_overlay=args.defense_overlay,
        defense_overlay_recent_capture_window=args.defense_overlay_recent_capture_window,
        defense_overlay_garrison_floor=args.defense_overlay_garrison_floor,
        defense_overlay_min_need=args.defense_overlay_min_need,
        defense_overlay_max_moves=args.defense_overlay_max_moves,
        defense_overlay_selector_checkpoint=args.defense_overlay_selector_checkpoint,
        defense_overlay_selector_threshold=args.defense_overlay_selector_threshold,
        defense_overlay_selector_mode=args.defense_overlay_selector_mode,
        defense_overlay_multi_source_per_target=args.defense_overlay_multi_source_per_target,
        producer_overlay=args.producer_overlay,
        producer_overlay_max_moves=args.producer_overlay_max_moves,
        producer_overlay_score_min=args.producer_overlay_score_min,
        producer_overlay_target_owner=args.producer_overlay_target_owner,
        producer_overlay_late_step=args.producer_overlay_late_step,
        producer_overlay_late_score_min=args.producer_overlay_late_score_min,
        producer_overlay_late_target_owner=args.producer_overlay_late_target_owner,
        producer_reranker_checkpoint=args.producer_reranker_checkpoint,
        producer_overlay_trace_json=args.producer_overlay_trace_json,
        producer_overlay_trace_top_k=args.producer_overlay_trace_top_k,
    )
