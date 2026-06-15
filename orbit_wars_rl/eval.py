"""Evaluation: pit trained PyTorch policy against baselines."""

from __future__ import annotations

import argparse
import math
import os
from statistics import mean

import torch
import numpy as np

from config import Config
from model import EntityTransformer, NUM_ANGLE_BINS, NUM_SHIP_BINS, ANGLE_BIN_WIDTH
from features import extract_features, _ETA_PROBE_SPEED, set_game_phase_features
from action_mask import compute_action_masks, actions_from_policy, actions_from_target_policy


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
    cfg.model.reinforce_garrison_floor = float(ckpt_cfg.get("reinforce_garrison_floor", 0.0))
    cfg.model.sufficient_commit_factor = float(ckpt_cfg.get("sufficient_commit_factor", 0.0))
    cfg.model._discipline_persisted = ("reinforce_gate_min_planets" in ckpt_cfg)
    # provenance (inspectable; eval always clamps regardless of how training handled overflow)
    cfg.model.ship_overflow_mode = str(ckpt_cfg.get("ship_overflow_mode", "drop"))
    return sd, action_decode


def build_agent_fn(model: EntityTransformer, device: torch.device,
                   fire_threshold: float = 0.5, sample: bool = False,
                   ship_bin_mode: str = "absolute",
                   target_decode: bool = False,
                   target_sanity_penalty: float = 0.0,
                   reserve_frac: float = 0.0,
                   allow_reinforce: bool = False,
                   veto_stats: dict = None):
    """Return a kaggle_environments-compatible agent function wrapping the model.

    sample=True uses Bernoulli/Categorical sampling instead of threshold/argmax —
    helps when the training-time distribution is multi-modal but the mode is
    degenerate (e.g. 1-ship-fleet trap).
    """
    model.eval()

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
            return action_fn(
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
                veto_stats=veto_stats,
            )

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
_MID_WINDOW = 100      # mid-game cap/atk window = [_LAUNCH_WINDOW, _MID_WINDOW) = steps 50-100
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
                    if t < _LAUNCH_WINDOW:
                        caps_early += 1                # opening captures (for opening cap/atk-launch)
                    elif t < _MID_WINDOW:
                        caps_mid += 1                  # mid-game (50-100) captures
                    cap_step[pid] = t                  # open a hold episode
                    cap_garr[pid] = p[5]               # garrison right after capture
                elif was == seat and own != seat and pid in cap_step:
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
            tgt = _resolve_launch_target(p0, src, float(mv[1]))
            if tgt is None:
                continue                            # unclassifiable → skip (== analyzer)
            if int(tgt[1]) == seat:
                reinf += 1                          # reinforce: cannot capture
                reinf_bin[bidx] += 1
                reinf_edges.append((t, int(src[0]), int(tgt[0])))   # for reciprocal ping-pong
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
        if fired_this_step > 0 and owned_dec > 0:
            fire_steps += 1
            fire_frac_sum += fired_this_step / owned_dec
        launch_count += fired_this_step
    end_planets = sum(1 for p in (last or []) if int(p[1]) == seat)
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
           "launches_ph": launches_ph, "ship1_ph": ship1_ph, "ship_ph_sum": ship_ph_sum}
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
           # retention split by outcome — lost-cap → 1 on elimination (lose every planet because you
           # LOST the game), so the won-game value is the honest "can we hold mid-game?" read.
           "captures_won": 0, "captures_lost": 0, "lost_caps_won": 0, "lost_caps_lost": 0,
           "hold_durations_won": [], "hold_durations_lost": [],
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
           "reinf_bin": [0] * len(_REINF_BINS), "atk_bin": [0] * len(_REINF_BINS)}
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
    return (f"Conversion: caps/game {c/n:.1f}  atk-launch/game {al/n:.1f}  "
            f"cap/atk-launch {c/max(al,1):.3f} (open<50 {cap_open:.3f}  mid50-100 {cap_mid:.3f})  ships/cap {acc['attack_ships']/max(c,1):.0f}  "
            f"reinf_share {rl/max(al+rl,1):.2f}\n"
            f"  planets@16/32/50/100 {pl(16)}/{pl(32)}/{pl(50)}/{pl(100)}  end {acc['end_planets']/n:.1f}"
            f"   churn {churn:.2f} ({churn_n:.2f}/100st, len {glen:.0f})\n"
            f"{pwl}"
            f"\n  retention  peel-rate {lost_rate:.2f} ({acc['lost_caps']}/{c} caps lost)  median-hold {med_hold}st\n"
            f"{rwl}"
            f"  hold-loss  out-massed {lm[1]:.0%} · abandoned {lm[0]:.0%} · too-late {lm[2]:.0%} · other {lm[3]:.0%}"
            f"   garr@cap {gcap_med:.0f}→@loss {gloss_med:.0f} vs enemy-inbound {einb_med:.0f}"
            f"   [out-massed = enemy fleet > our garrison = force-concentration gap]\n"
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
            f"  ship0 1-ship-probe by phase  {_s0('')}{s0wl}")


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
) -> dict:
    """Evaluate trained policy against a baseline using kaggle_environments.

    Args:
        opponent: "random" or path to a Python agent file (e.g. "main.py")
        num_players: 2 or 4
    """
    from kaggle_environments import make

    agent_fn = build_agent_fn(model, device, fire_threshold=fire_threshold, sample=sample,
                              ship_bin_mode=ship_bin_mode, target_decode=target_decode,
                              target_sanity_penalty=target_sanity_penalty)
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
) -> dict:
    """Stratified eval over the 128-seed community panel, playing both seats.

    256 games per opponent (128 seeds × 2 seats). Aggregates wins per
    archetype (8 games per cell = 4 seeds × 2 seats) and per seat, so a
    +5pp overall regression hidden by an asymmetric or board-shape-specific
    weakness is visible.
    """
    from kaggle_environments import make
    from eval_panel import BY_ARCHETYPE

    agent_fn = build_agent_fn(model, device, fire_threshold=fire_threshold, sample=sample,
                              ship_bin_mode=ship_bin_mode, target_decode=target_decode,
                              target_sanity_penalty=target_sanity_penalty)

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

    print(f"Panel eval START — opponent: {opponent} | {total_games} games "
          f"(128 seeds × 2 seats) | decode={'target' if target_decode else 'argmax'} "
          f"fire_thr={fire_threshold}", flush=True)

    for archetype, seeds in BY_ARCHETYPE.items():
        for seed in seeds:
            for my_seat in (0, 1):
                agents = [agent_fn, opponent] if my_seat == 0 else [opponent, agent_fn]
                env = make("orbit_wars", configuration={"seed": seed}, debug=False)
                env.run(agents)
                final = env.steps[-1]
                rewards = [s.reward if s.reward is not None else 0.0 for s in final]
                my_reward = rewards[my_seat]
                opp_reward = rewards[1 - my_seat]
                is_win = my_reward > opp_reward
                add_conversion(conv_tot, game_conversion(env.steps, my_seat), won=is_win)
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
                        reinforce_gate_min_planets: int = None,
                        reinforce_forward_only: bool = None,
                        reinforce_garrison_floor: float = None,
                        sufficient_commit_factor: float = None):
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
    model.reinforce_garrison_floor = float(reinforce_garrison_floor)
    # Sufficient-commit mask (attacks) — also MUST match training. Independent of reinforce.
    model.sufficient_commit_factor = float(sufficient_commit_factor)
    if model.allow_reinforce:
        print(f"Reinforcement: ON (own planets are legal targets) | "
              f"gate>={model.reinforce_gate_min_planets} planets, "
              f"forward_only={model.reinforce_forward_only}, "
              f"garrison_floor={model.reinforce_garrison_floor}")
    if model.sufficient_commit_factor > 0.0:
        print(f"Sufficient-commit mask: ON | veto attacks with ships <= "
              f"target_defense × {model.sufficient_commit_factor}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"target_head.weight", "target_head.bias"}
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
                                 target_sanity_penalty=target_sanity_penalty)
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
    )

    print(f"Win rate vs {opponent}: {results['win_rate']:.2%}  "
          f"({results['wins']}/{results['total_games']})")
    print(f"Fire threshold: {fire_threshold}")
    print(f"Target decode: {target_decode}")
    print(f"Target sanity penalty: {target_sanity_penalty}")
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
        reinforce_gate_min_planets=args.reinforce_gate_min_planets,
        reinforce_forward_only=args.reinforce_forward_only,
        reinforce_garrison_floor=args.reinforce_garrison_floor,
        sufficient_commit_factor=args.sufficient_commit_factor,
    )
