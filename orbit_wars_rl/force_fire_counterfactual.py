"""ENV-GROUNDED force-fire counterfactual (docs/q-head.md step 1, env-grounded) — the falsification
that needs no Q-head and no confounded data.

The three offline Q-head gates all came back ≤0/≈0 on the idle-spare counterfactual, but each is
confounded (on-policy data says idle→win) or ambiguous (expert data lacks the wrongly-held-spare case).
This test sidesteps both: it asks the ENVIRONMENT directly — if our policy ALWAYS fired its idle spares
onto contested neutrals (the exact af1+agg spare-fire opportunity set from value_spare_diagnostic, incl.
multi-source pooling to the capture floor), would it WIN MORE vs Ajay?

  baseline agent : the eval-faithful policy, plays normally (leaves spares idle)
  overlay  agent : same policy + ALWAYS fire idle spares onto contested neutrals
  paired by (seed, seat): same env seed = common random numbers → the only difference is the injected fires.

Win-rate(overlay) ≫ Win-rate(baseline)  ⇒  aggregation converts losses to wins ⇒ COMA's target is REAL,
the problem is purely bootstrapping the signal. Flat ⇒ "we fail to fire spares" is NOT the wall.

Run from repo root (slow — Ajay games):
  ORBIT_GAME_PHASE_FEATURES=1 <venv>/bin/python orbit_wars_rl/force_fire_counterfactual.py \
      --checkpoint gpu_run_artifacts/r32_stage_hlr/checkpoints/torch_step_2097152_*.pt \
      --opponent opponents/candidate_ajay_1200.py --seeds 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from kaggle_environments import make

import eval as E
from transition_autopsy import _load_model, _obs_to_dict
from features import extract_features
from action_mask import compute_action_masks, _target_intercept_angle, MAX_OWNED_PLANETS
from value_spare_diagnostic import _spare_sources_for_neutral, STEP_LO, STEP_HI


def _policy_moves(model, obs, device, ship_bin_mode, allow_reinforce):
    """The eval-faithful baseline moves — identical decode path to eval / the Q-head gate."""
    player = obs["player"]
    features = extract_features(obs, player, num_players=2)
    masks = compute_action_masks(obs, player)

    def f(name):
        return features[name].unsqueeze(0).to(device)
    slot_valid = masks["slot_valid"].to(device)
    if slot_valid.dim() == 1:
        slot_valid = slot_valid.unsqueeze(0)
    with torch.no_grad():
        fwd_kw = dict(
            fire_mask=masks["fire_mask"].to(device), angle_mask=masks["angle_mask"].to(device),
            slot_valid=slot_valid, owned_indices=masks["owned_indices"].to(device),
            owned_count=masks["owned_count"],
            pairwise_features=f("pairwise_features") if "pairwise_features" in features else None,
        )
        base = (f("planet_features"), f("fleet_features"), f("global_features"),
                f("planet_mask"), f("fleet_mask"))
        outputs = model(*base, **fwd_kw)
    return E.actions_from_target_policy(
        outputs["fire_logits"].cpu(), outputs["target_logits"].cpu(), outputs["ship_logits"].cpu(),
        {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()},
        obs, player, ship_bin_mode=ship_bin_mode, allow_reinforce=allow_reinforce,
        reinforce_gate_min_planets=getattr(model, "reinforce_gate_min_planets", 0),
        reinforce_forward_only=getattr(model, "reinforce_forward_only", False),
        reinforce_garrison_floor=getattr(model, "reinforce_garrison_floor", 0.0),
        sufficient_commit_factor=getattr(model, "sufficient_commit_factor", 0.0),
        reverse_edge_cooldown=int(getattr(model, "reverse_edge_cooldown", 0)),
    )


def _overlay_fires(obs, moves, step_lo, step_hi, stats, roi_min=0.0):
    """Append fire-spare→contested-neutral moves for IDLE spare sources — SELECTIVE: only HOLDABLE
    neutrals (E._holdable_roi ≥ roi_min; that ROI prices the reactive peel, so a capturable-but-unholdable
    'hopeless' target scores LOW and is skipped), best-ROI-first (so the best targets claim the spare),
    pooled cheapest-first to the capture floor (no overkill — sized to floor, sources used once).
    roi_min = -1e9 reproduces the indiscriminate 'maximal' overlay (no holdability filter)."""
    t = int(obs.get("step", 0))
    if t < step_lo or t > step_hi:
        return moves
    planets = obs["planets"]
    fleets = obs.get("fleets") or []
    player = int(obs["player"])
    fired_src = {int(m[0]) for m in moves}
    cands = []
    for _, neutral in [(i, p) for i, p in enumerate(planets) if int(p[1]) < 0]:
        sources = _spare_sources_for_neutral(neutral, planets, fleets, player)
        if not sources:
            continue
        total_spare = sum(s for _, s, _ in sources)
        cheapest = min(c for _, _, c in sources)
        solo = any(s >= c for _, s, c in sources)
        bucket = "af1" if solo else ("agg" if total_spare >= cheapest else "cap")
        if bucket == "cap":                                   # can't afford even pooled — not an opportunity
            continue
        idle = [(pidx, spare, cost) for (pidx, spare, cost) in sources
                if int(planets[pidx][0]) not in fired_src]    # only sources the policy left idle
        if not idle:
            continue
        stats["opps"] += 1
        roi = max((E._holdable_roi(planets[pidx], neutral, planets, fleets, player) or -1e9)
                  for pidx, _, _ in idle)                     # best holdable-ROI route to this neutral
        if roi < roi_min:                                     # HOPELESS / unholdable — skip (the USER fix)
            continue
        cands.append((roi, neutral, max(cheapest, 1.0), bucket))
    cands.sort(key=lambda c: -c[0])                           # best-ROI neutrals claim the spare first
    for roi, neutral, floor, bucket in cands:
        if len(moves) >= MAX_OWNED_PLANETS:
            break
        idle = [(pidx, spare, cost) for (pidx, spare, cost)
                in _spare_sources_for_neutral(neutral, planets, fleets, player)
                if int(planets[pidx][0]) not in fired_src]    # re-check: a better target may have claimed it
        idle.sort(key=lambda x: x[2])                         # cheapest route first
        sent = 0.0
        for pidx, spare, _cost in idle:
            if sent >= floor or len(moves) >= MAX_OWNED_PLANETS:
                break
            ships = int(min(spare, floor - sent))
            if ships <= 0:
                continue
            angle = _target_intercept_angle(planets[pidx], neutral, ships, obs)
            moves.append([int(planets[pidx][0]), float(angle), int(ships)])
            fired_src.add(int(planets[pidx][0]))
            sent += ships
            stats["fires"] += 1
        if sent > 0:
            stats["neutrals"] += 1
            stats[bucket] += 1
    return moves


def build_agent(model, device, ship_bin_mode, allow_reinforce, overlay, step_lo, step_hi, stats,
                rec=None, snap=50, roi_min=0.0):
    model.eval()

    def agent_fn(obs):
        obs = _obs_to_dict(obs)
        if rec is not None and "planets" not in rec and int(obs.get("step", 0)) >= snap:
            player = int(obs["player"])                       # graded wall metric at the snapshot step
            owned = [p for p in obs["planets"] if int(p[1]) == player]
            rec["planets"], rec["ships"] = len(owned), float(sum(p[5] for p in owned))
        moves = _policy_moves(model, obs, device, ship_bin_mode, allow_reinforce)
        if overlay:
            moves = _overlay_fires(obs, moves, step_lo, step_hi, stats, roi_min)
        return moves

    return agent_fn


def _play(agent, opponent, seed, our_seat):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([agent, opponent] if our_seat == 0 else [opponent, agent])
    final = env.steps[-1]
    r_us = final[our_seat].reward or 0.0
    r_opp = final[1 - our_seat].reward or 0.0
    return 1 if r_us >= r_opp else 0


def _report(results, stats, n_overlay_games, snap=50):
    bw = sum(r[2] for r in results)
    ow = sum(r[3] for r in results)
    n = len(results)
    both_win = sum(1 for r in results if r[2] and r[3])
    both_lose = sum(1 for r in results if not r[2] and not r[3])
    prize = sum(1 for r in results if not r[2] and r[3])      # baseline LOSS -> overlay WIN
    regress = sum(1 for r in results if r[2] and not r[3])    # baseline WIN  -> overlay LOSS

    print("\n" + "=" * 84)
    print("ENV-GROUNDED FORCE-FIRE OVERLAY — does systematic spare-aggregation win vs Ajay?")
    print("=" * 84)
    fpg = stats["fires"] / max(1, n_overlay_games)
    print(f"overlay activity: {stats['fires']} fires over {n_overlay_games} games "
          f"({fpg:.1f} fires/game; {stats['neutrals']} holdable-piled of {stats['opps']} idle-spare opps; "
          f"af1={stats['af1']} agg={stats['agg']})")
    if stats["fires"] == 0:
        print("  ⚠ overlay injected ZERO fires — test is null (no idle-spare opportunities in window). "
              "Widen --step-hi or check opportunity detection before reading anything into win-rate.")
    print(f"\npaired outcomes (n={n} games):")
    print(f"  both win:                              {both_win}")
    print(f"  both lose:                             {both_lose}")
    print(f"  baseline LOSS → overlay WIN  (PRIZE):  {prize}")
    print(f"  baseline WIN  → overlay LOSS (regress):{regress}")
    print(f"\nbaseline win-rate: {bw}/{n} = {100*bw/n:.1f}%")
    print(f"overlay  win-rate: {ow}/{n} = {100*ow/n:.1f}%   (Δ {100*(ow-bw)/n:+.1f} pp)")
    print(f"discordant pairs: prize {prize} vs regress {regress}")
    bp = [r[4]["planets"] for r in results if r[4].get("planets") is not None]
    op = [r[5]["planets"] for r in results if r[5].get("planets") is not None]
    bm = [r[4]["ships"] for r in results if r[4].get("ships") is not None]
    om = [r[5]["ships"] for r in results if r[5].get("ships") is not None]
    dp = None
    if bp and op:
        dp = sum(op) / len(op) - sum(bp) / len(bp)
        print(f"\ngraded wall metric @step {snap} (mean over {len(bp)} games reaching it):")
        print(f"  planets : baseline {sum(bp)/len(bp):.2f} → overlay {sum(op)/len(op):.2f}  (Δ {dp:+.2f})")
        print(f"  material: baseline {sum(bm)/len(bm):.1f} → overlay {sum(om)/len(om):.1f}  "
              f"(Δ {sum(om)/len(om)-sum(bm)/len(bm):+.1f})")
    if stats["fires"] == 0:
        verdict = "NULL — overlay did nothing (no idle-spare opportunities in window)"
    elif ow - bw >= 2 and prize > regress:
        verdict = ("aggregation CONVERTS losses→wins → the signal COMA needs is REAL in the env; "
                   "the offline ≤0 gates are the bootstrap/confound problem, not a mirage → COMA worth GPU")
    elif ow - bw <= -2 or (regress > prize and dp is not None and dp < 0):
        pdelta = f"planets@{snap} Δ {dp:+.2f}" if dp is not None else "planets@N n/a"
        verdict = (f"NET-NEGATIVE — firing idle spares HURTS (win {100*bw/n:.0f}%→{100*ow/n:.0f}%, {pdelta}); "
                   "spares are idle CORRECTLY → confirms the offline ≤0 reads as TRUE signal (not the on-policy "
                   "confound) → COMA's 'fire more spares' premise undercut (a correct Q would reinforce idling)")
    elif dp is not None and dp >= 0.75:
        verdict = (f"win-rate flat but planets@{snap} Δ {dp:+.2f} → overlay EXPANDS more mid-game but doesn't "
                   "convert → CAPTURE isn't the gap, RETENTION/conversion is (consistent with the held@+15 loop)")
    elif abs(ow - bw) <= 1:
        verdict = (f"FLAT — firing the idle spares changes neither win-rate nor planets@{snap} → 'we fail to "
                   "fire spares' is NOT the wall; aggregation doesn't even translate to expansion")
    else:
        verdict = "ambiguous — read prize vs regress + planets@N Δ, scale up seeds"
    print(f"\nREAD: {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", default="opponents/candidate_ajay_1200.py")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--step-lo", type=int, default=STEP_LO)
    ap.add_argument("--step-hi", type=int, default=STEP_HI)
    ap.add_argument("--snap", type=int, default=50, help="graded wall-metric snapshot step (planets/material)")
    ap.add_argument("--roi-min", type=float, default=0.0,
                    help="holdable-ROI floor for overlay targets (skips hopeless/unholdable neutrals)")
    ap.add_argument("--maximal", action="store_true",
                    help="no holdability filter — reproduce the indiscriminate maximal overlay")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    model, cfg, target_decode, ship_bin_mode = _load_model(args.checkpoint)
    allow_reinforce = bool(getattr(model, "allow_reinforce", False))
    device = torch.device("cpu")

    from expansion_autopsy import _seed_list
    seeds = _seed_list("random", args.seeds)
    print(f"playing {len(seeds)*2*2} games ({len(seeds)} seeds x 2 seats x baseline+overlay) "
          f"vs {args.opponent}, window [{args.step_lo},{args.step_hi}]", flush=True)
    roi_min = -1e9 if args.maximal else args.roi_min
    stats = {"fires": 0, "neutrals": 0, "af1": 0, "agg": 0, "opps": 0}
    results = []
    for sd in seeds:
        for seat in (0, 1):
            brec, orec = {}, {}
            base = build_agent(model, device, ship_bin_mode, allow_reinforce, False,
                               args.step_lo, args.step_hi, stats, brec, args.snap)
            bw = _play(base, args.opponent, sd, seat)
            ovl = build_agent(model, device, ship_bin_mode, allow_reinforce, True,
                              args.step_lo, args.step_hi, stats, orec, args.snap, roi_min)
            ow = _play(ovl, args.opponent, sd, seat)
            results.append((sd, seat, bw, ow, brec, orec))
            print(f"  seed={sd} seat={seat}: baseline {'W' if bw else 'L'} | overlay {'W' if ow else 'L'}"
                  f"  (p@{args.snap} {brec.get('planets','-')}→{orec.get('planets','-')})", flush=True)
    _report(results, stats, n_overlay_games=len(seeds) * 2, snap=args.snap)


if __name__ == "__main__":
    main()
