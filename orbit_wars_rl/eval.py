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
from features import extract_features
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
    return sd, action_decode


def build_agent_fn(model: EntityTransformer, device: torch.device,
                   fire_threshold: float = 0.5, sample: bool = False,
                   ship_bin_mode: str = "absolute",
                   target_decode: bool = False,
                   target_sanity_penalty: float = 0.0,
                   reserve_frac: float = 0.0,
                   allow_reinforce: bool = False):
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
            )

        raise NotImplementedError(
            "angle-decode path removed (angle head deleted); Phase 1 checkpoints "
            "use target-decode. Pass target_decode=True (--target-decode)."
        )

    return agent_fn


_CONV_MILESTONES = (16, 32, 50, 100)
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
    caps = atk = reinf = atk_ships = 0
    reinf_bin = [0] * len(_REINF_BINS)   # own-target launches by empire size at launch
    atk_bin = [0] * len(_REINF_BINS)     # attack launches by empire size at launch
    planets_at = {ms: None for ms in _CONV_MILESTONES}
    garrison_at = {ms: None for ms in _CONV_MILESTONES}   # ships parked on owned planets
    inflight_at = {ms: None for ms in _CONV_MILESTONES}   # ships in owned fleets (deployed)
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
                if pid in prev and prev[pid] != seat and own == seat:
                    caps += 1
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
    end_planets = sum(1 for p in (last or []) if int(p[1]) == seat)
    out = {"captures": caps, "attack_launches": atk, "reinforce_launches": reinf,
           "attack_ships": atk_ships, "end_planets": end_planets,
           "reinf_bin": reinf_bin, "atk_bin": atk_bin}
    for ms in _CONV_MILESTONES:
        out[f"p{ms}"] = planets_at[ms]
        out[f"g{ms}"] = garrison_at[ms]
        out[f"if{ms}"] = inflight_at[ms]
    return out


def new_conversion_acc():
    acc = {"captures": 0, "attack_launches": 0, "reinforce_launches": 0,
           "attack_ships": 0, "end_planets": 0, "games": 0,
           "reinf_bin": [0] * len(_REINF_BINS), "atk_bin": [0] * len(_REINF_BINS)}
    for ms in _CONV_MILESTONES:
        acc[f"p{ms}_sum"] = 0
        acc[f"p{ms}_n"] = 0
        acc[f"g{ms}_sum"] = 0    # garrison (parked) ships, summed over games reaching ms
        acc[f"if{ms}_sum"] = 0   # in-flight (deployed) ships, summed over games reaching ms
    return acc


def add_conversion(acc, conv):
    for k in ("captures", "attack_launches", "reinforce_launches", "attack_ships", "end_planets"):
        acc[k] += conv[k]
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
    return (f"Conversion: caps/game {c/n:.1f}  atk-launch/game {al/n:.1f}  "
            f"cap/atk-launch {c/max(al,1):.3f}  ships/cap {acc['attack_ships']/max(c,1):.0f}  "
            f"reinf_share {rl/max(al+rl,1):.2f}\n"
            f"  planets@16/32/50/100 {pl(16)}/{pl(32)}/{pl(50)}/{pl(100)}  end {acc['end_planets']/n:.1f}"
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

    for archetype, seeds in BY_ARCHETYPE.items():
        for seed in seeds:
            for my_seat in (0, 1):
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
                        reinforce_gate_min_planets: int = 0,
                        reinforce_forward_only: bool = False,
                        reinforce_garrison_floor: float = 0.0):
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
    parser.add_argument("--reinforce-gate-min-planets", type=int, default=0,
                        help="Reinforce-discipline parity: own targets legal only at "
                             ">= this many owned planets. MUST match training (p2rev1=3).")
    parser.add_argument("--reinforce-forward-only", action="store_true",
                        help="Reinforce-discipline parity: own target legal only if closer "
                             "to the nearest enemy than the source. MUST match training.")
    parser.add_argument("--reinforce-garrison-floor", type=float, default=0.0,
                        help="Reinforce-discipline parity: veto a reinforce that drains the "
                             "source below this. MUST match training (p2rev1=10).")
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
    )
