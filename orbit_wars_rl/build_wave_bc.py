"""Build a Phase 5 wave-planner BC dataset and run the §9 pre-training poisoning checks.

The wave planner (wave_planner.plan) self-plays via kaggle_environments; every visited state for
seat 0 is labeled with the planner's per-source (fire, target, ship-bin) decision. Before any
training we verify the labels are not poisoned (spec §9):

  positive_reactive_cross_rate   high     bundles size to the REACTIVE floor, not the static one
  positive_static_cross_rate     (info)   high-while-reactive-low => poisoned
  positive_arrival_spread_p90  <= 2*tol   bundles arrive as a wave, not a trickle
  ready_quota_error              small    rounding/quota residual under control
  overcommit_ratio (post-round)  bounded  launched / reactive floor
  held_when_holdable_rate        high     HOLDABLE planets actually get reinforced
  reinforce_into_DOOMED_rate     ~0       never feed a doomed planet

Run from repo ROOT:
  orbit_wars_rl/.venv/bin/python orbit_wars_rl/build_wave_bc.py --games 30 [--opponent random] [--out ds.pkl]
"""

from __future__ import annotations

import argparse
import pickle
import statistics

import torch
from kaggle_environments import make

import wave_planner as WPL
import wave_primitives as wp
from features import extract_features, MAX_OWNED_PLANETS
from action_mask import compute_action_masks


def plan_to_sample(obs, player, res, max_owned=MAX_OWNED_PLANETS, max_planets=48):
    """Convert a planner decision over one obs into BC sample tensors (explicit target, no angle
    round-trip — the planner knows the target planet directly)."""
    features = extract_features(obs, player, num_players=2)
    masks = compute_action_masks(obs, player)
    if int(masks["owned_count"]) == 0:
        return None
    planets = obs["planets"]
    owned_indices = masks["owned_indices"].numpy()
    pid_to_slot = {}
    for slot in range(int(masks["owned_count"])):
        pidx = int(owned_indices[slot])
        if pidx < len(planets):
            pid_to_slot[int(planets[pidx][0])] = slot
    pid_to_arrayidx = {int(p[0]): i for i, p in enumerate(planets[:max_planets])}

    fire_target = torch.zeros(max_owned, dtype=torch.long)
    ship_target = torch.zeros(max_owned, dtype=torch.long)
    target_target = torch.full((max_owned,), -1, dtype=torch.long)
    for d in res.decisions.values():
        if d.kind == "noop" or d.target_pid is None:
            continue
        slot = pid_to_slot.get(d.src_pid)
        tidx = pid_to_arrayidx.get(d.target_pid)
        if slot is None or tidx is None or d.ship_bin is None:
            continue
        fire_target[slot] = 1
        ship_target[slot] = int(d.ship_bin)
        target_target[slot] = int(tidx)

    return {
        "planet_features": features["planet_features"],
        "fleet_features": features["fleet_features"],
        "global_features": features["global_features"],
        "planet_mask": features["planet_mask"],
        "fleet_mask": features["fleet_mask"],
        "fire_mask": masks["fire_mask"][0],
        "angle_mask": masks["angle_mask"][0],
        "slot_valid": masks["slot_valid"][0],
        "owned_indices": masks["owned_indices"],
        "owned_count": int(masks["owned_count"]),
        "fire_target": fire_target,
        "ship_target": ship_target,
        "target_target": target_target,
        "pairwise_features": features["pairwise_features"],
    }


def _p90(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--opponent", default="random")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--episode-steps", type=int, default=500)
    ap.add_argument("--min-ship-bin", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional .pkl dataset path")
    args = ap.parse_args()

    def agent(obs, config=None):
        return WPL.plan_agent(obs, config, episode_steps=args.episode_steps,
                              min_ship_bin=args.min_ship_bin)

    samples = []
    atk_waves, def_waves = [], []
    holdable_total = holdable_defended = 0
    defense_moves = defense_into_doomed = 0
    bin_overshoot = []   # §9 bin-resolution (C): chosen_count / quota_target per fired source

    for s in range(args.seed0, args.seed0 + args.games):
        env = make("orbit_wars", configuration={"seed": s}, debug=False)
        env.run([agent, args.opponent])
        for t in range(len(env.steps)):
            ob = env.steps[t][0].observation
            if not ob.get("planets"):
                continue
            res = WPL.plan(ob, 0, episode_steps=args.episode_steps, min_ship_bin=args.min_ship_bin)
            samp = plan_to_sample(ob, 0, res)
            if samp is not None:
                samples.append(samp)
            atk_waves += [w for w in res.waves if w.kind == "attack"]
            def_waves += [w for w in res.waves if w.kind == "defense"]
            # defense coverage metrics
            doomed = {pid for pid, h in res.holds.items() if h.hold_class == wp.HOLD_DOOMED}
            holdable = {pid for pid, h in res.holds.items() if h.hold_class == wp.HOLD_HOLDABLE}
            defended = {d.target_pid for d in res.decisions.values() if d.kind == "defense"}
            holdable_total += len(holdable)
            holdable_defended += len(holdable & defended)
            for d in res.decisions.values():
                if d.kind == "defense":
                    defense_moves += 1
                    if d.target_pid in doomed:
                        defense_into_doomed += 1
                if d.kind != "noop" and d.quota_target > 1e-6:
                    bin_overshoot.append(d.ship_count / d.quota_target)

    # ---- §9 poisoning checks ----
    def rate(waves, pred):
        return (sum(1 for w in waves if pred(w)) / len(waves)) if waves else float("nan")

    # "bundle" = friendly mass already inbound in the window (prior steps' launches) + this step's
    # launch. A synchronized wave fills across steps, so the per-step LAUNCH alone undercounts.
    reactive_cross = rate(atk_waves, lambda w: w.cover_inflight + w.launched + 1e-6 >= w.floor)
    reactive_cross_launch_only = rate(atk_waves, lambda w: w.launched + 1e-6 >= w.floor)
    static_cross = rate(atk_waves, lambda w: w.cover_inflight + w.launched + 1e-6 >= w.static_floor)
    spreads = [max(w.arrival_taus) - min(w.arrival_taus) for w in atk_waves if w.arrival_taus]
    spread_p90 = _p90(spreads)
    quota_err = [abs(w.launched - w.remaining) for w in atk_waves]
    overcommit = [w.launched / max(w.floor, 1e-6) for w in atk_waves]
    held_when_holdable = (holdable_defended / holdable_total) if holdable_total else float("nan")
    into_doomed = (defense_into_doomed / defense_moves) if defense_moves else 0.0

    print(f"\n=== WAVE-BC DATASET + §9 POISONING CHECKS")
    print(f"    {args.games} games vs {args.opponent}  |  samples {len(samples)}  "
          f"attack-waves {len(atk_waves)}  defense-waves {len(def_waves)}")
    print(f"\n    ATTACK:")
    print(f"      positive_reactive_cross_rate   {reactive_cross:.3f}   (bundle=cover+launch; target HIGH)")
    print(f"        (launch-only, per-step)      {reactive_cross_launch_only:.3f}")
    print(f"      positive_static_cross_rate     {static_cross:.3f}   (info)")
    print(f"      arrival_spread_p90             {spread_p90:.2f}    (target <= 2*tol = {2*wp.WAVE_TOL_STEPS:.0f})")
    print(f"      ready_quota_error (median)     {statistics.median(quota_err) if quota_err else float('nan'):.2f}    (target SMALL)")
    print(f"      overcommit_ratio p50/p90       {statistics.median(overcommit) if overcommit else float('nan'):.2f}"
          f" / {_p90(overcommit):.2f}    (target BOUNDED)")
    print(f"      bin_overshoot p50/p90          {statistics.median(bin_overshoot) if bin_overshoot else float('nan'):.2f}"
          f" / {_p90(bin_overshoot):.2f}    (chosen/quota from ship-bin rounding; §3.0.1/C)")
    _med = lambda xs: (statistics.median(xs) if xs else float("nan"))
    print(f"      [diag medians] floor {_med([w.floor for w in atk_waves]):.0f}  "
          f"cover {_med([w.cover_inflight for w in atk_waves]):.0f}  "
          f"launched {_med([w.launched for w in atk_waves]):.0f}  "
          f"ready_safe {_med([w.ready_safe for w in atk_waves]):.0f}  "
          f"remaining {_med([w.remaining for w in atk_waves]):.0f}]")
    print(f"\n    DEFENSE:")
    print(f"      held_when_holdable_rate        {held_when_holdable:.3f}   (target HIGH)")
    print(f"      reinforce_into_DOOMED_rate     {into_doomed:.3f}   (target ~0)")

    # Synchronization (arrival_spread) was DROPPED by the 2026-06-19 audit — early arrival is now
    # allowed, so spread is INFO only. The gate is sufficiency (cross the reactive floor) + clean
    # defense triage (never feed a DOOMED planet).
    ok = (reactive_cross == reactive_cross and reactive_cross >= 0.9 and into_doomed <= 1e-9)
    print(f"\n    GATE: {'PASS' if ok else 'REVIEW'} "
          f"(reactive_cross>=0.9 AND no DOOMED-reinforce; arrival_spread now INFO-only)")

    if args.out:
        with open(args.out, "wb") as f:
            pickle.dump(samples, f)
        print(f"\n    dataset → {args.out} ({len(samples)} samples)")


if __name__ == "__main__":
    main()
