"""Expansion-phase autopsy (2026-06-21): is the out-mass wall SET before step 50?

transition_autopsy showed the 50-100 mid-game is the losing TAIL: at step 50 WON/LOST have a
similar planet COUNT (8 vs 7) but very different mass SHARE (0.61 vs 0.34), and matched-state
decisions don't differ. So the divergence is upstream. This asks WHERE in steps 0-50 it opens,
and disambiguates the fork that decides the next lever:

  - similar planets, LOWER ships/planet      -> OVER-EXTENSION (grab planets we can't garrison;
                                                "consolidate harder before expanding" is the lever)
  - FEWER planets                            -> out-EXPANDED (we need to expand MORE, not less;
                                                consolidate-harder would be the WRONG advice)
  - similar planets AND similar ships/planet,
    lower ABSOLUTE mass from ~step 10        -> out-PRODUCED / out-fought from the start
                                                (combat/production efficiency, not expansion shape)

For N seeds x 2 seats vs the opponent, split WON/LOST, report at steps [10,20,30,40,50]:
  our planets, our mass (garrison+inflight), ships/planet, mass-share, enemy planets,
  enemy ships/planet, cumulative OUR captures. Plus garrison surplus at OUR captures in
  steps<50 (garrison held vs enemy-inbound at the captured planet) WON vs LOST.

Reuses eval's threat geometry + transition_autopsy's play loop so numbers match the panel.
Run from repo ROOT:
  orbit_wars_rl/.venv/bin/python orbit_wars_rl/expansion_autopsy.py --checkpoint <ck> \
      --opponent opponents/candidate_ajay_1200.py --seeds 24
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict

import torch
from kaggle_environments import make

import eval as E
from transition_autopsy import _load_model, _obs_to_dict

SAMPLES = [10, 20, 30, 40, 50]


def _expandable_neutrals(planets, fleets, seat, max_eta=25):
    """Causal gate: of NEUTRAL planets still on the board, how many could we capture RIGHT NOW from a
    single owned planet's SPARE garrison (spare = ships - that planet's own enemy-inbound threat, so
    we don't strip a planet under attack) within max_eta steps? Uses eval's _cap_cost_at_arrival
    (ships needed at arrival incl. the neutral's growth) so 'affordable' matches the engine.
      high in LOST @20-40 = we leave catchable land on the table (HOARD by choice, lever real)
      near-zero          = board tapped / enemy got there first (opponent-speed-limited, no lever)
    Returns (n_expandable, n_neutral_total)."""
    owned = [p for p in planets if int(p[1]) == seat]
    neutrals = [p for p in planets if int(p[1]) < 0]
    cnt = 0
    for n in neutrals:
        for p in owned:
            dist = math.hypot(n[2] - p[2], n[3] - p[3])
            eta = max(1.0, math.ceil(dist / E._ETA_PROBE_SPEED))
            if eta > max_eta:
                continue
            threat, _ = E._enemy_threat(planets, fleets, p, seat)
            if (p[5] - threat) >= E._cap_cost_at_arrival(p, n, seat):
                cnt += 1
                break
    return cnt, len(neutrals)


def _classify_neutrals(planets, fleets, seat, max_eta=25):
    """4-way partition of NEUTRAL planets — resolves the reach-vs-aggregation fork. spare = max(0,
    ships - that planet's own enemy-inbound threat) = ships we can send without leaving it under attack.
      far  : no owned planet within eta<=max_eta             -> REACH-limited (no unilateral lever)
      af1  : some single reachable source spare >= its cost  -> affordable-skipped (free, shown small)
      agg  : no single source, but POOLED reachable spare >= cheapest cost
                                                             -> AGGREGATION/COMMITMENT lever (have the
                                                                ships across planets, won't pool/strip)
      cap  : reachable but even pooled spare < cheapest cost -> genuine CAPACITY (opponent-capability)
    Returns (far, af1, agg, cap)."""
    owned = [p for p in planets if int(p[1]) == seat]
    neutrals = [p for p in planets if int(p[1]) < 0]
    far = af1 = agg = cap = 0
    for n in neutrals:
        reach = []                                  # (cost_from_this_source, spare_here)
        for p in owned:
            dist = math.hypot(n[2] - p[2], n[3] - p[3])
            eta = max(1.0, math.ceil(dist / E._ETA_PROBE_SPEED))
            if eta > max_eta:
                continue
            threat, _ = E._enemy_threat(planets, fleets, p, seat)
            reach.append((E._cap_cost_at_arrival(p, n, seat), max(0.0, p[5] - threat)))
        if not reach:
            far += 1
        elif any(s >= c for c, s in reach):
            af1 += 1
        elif sum(s for _, s in reach) >= min(c for c, _ in reach):
            agg += 1
        else:
            cap += 1
    return far, af1, agg, cap


def _seed_list(seed_set, n_seeds):
    """random = seeds 0..n-1 (ALL rotating-rate boards). static/rotating = the panel's EXTREME
    archetype cells (mostly_X__big_X) — the only authoritative static-vs-rotating split, since the
    obs angular_velocity scalar is ~identical (~0.03-0.05) for both classes (rotation is positional,
    set by orbit radius, not the global rate)."""
    if seed_set == "random":
        return list(range(n_seeds))
    from eval_panel import BY_ARCHETYPE
    tag = "mostly_static__big_static" if seed_set == "static" else "mostly_rotating__big_rotating"
    return [s for k, seeds in BY_ARCHETYPE.items() if tag in k for s in seeds]


def _mass(planets, fleets, seat):
    """Total controllable mass for `seat`: garrison on owned planets + ships inflight."""
    g = sum(p[5] for p in planets if int(p[1]) == seat)
    f = sum(fl[6] for fl in fleets if int(fl[1]) == seat)
    return g + f


def _enemy_mass(planets, fleets, seat):
    g = sum(p[5] for p in planets if int(p[1]) >= 0 and int(p[1]) != seat)
    f = sum(fl[6] for fl in fleets if int(fl[1]) >= 0 and int(fl[1]) != seat)
    return g + f


def _new_acc():
    return {
        "n_games": 0,
        # per-sample-step lists (averaged in the report)
        "planets": defaultdict(list), "mass": defaultdict(list), "spp": defaultdict(list),
        "share": defaultdict(list), "e_planets": defaultdict(list), "e_spp": defaultdict(list),
        "caps_cum": defaultdict(list),
        "exp_neutral": defaultdict(list), "tot_neutral": defaultdict(list),
        "neu_far": defaultdict(list), "neu_af1": defaultdict(list),
        "neu_agg": defaultdict(list), "neu_cap": defaultdict(list),
        # capture-quality in steps<50
        "cap_garr": [], "cap_surplus": [], "cap_underfloor": 0, "cap_n": 0,
    }


def _scan_game(records, seat, A):
    A["n_games"] += 1
    by_step = {}
    for rec in records:
        od = rec["obs"]
        by_step[od["step"]] = od

    # cumulative OUR captures (territory gained): a planet not ours at t-1 that is ours at t.
    steps_sorted = sorted(by_step)
    prev_owner = {}
    caps_cum = 0
    caps_by_step = {}
    for t in steps_sorted:
        od = by_step[t]
        planets = od["planets"]; fleets = od["fleets"]
        for p in planets:
            pid = int(p[0]); owner = int(p[1])
            was = prev_owner.get(pid, owner)  # at t=0 treat as no-flip
            if owner == seat and was != seat:
                caps_cum += 1
                if t < 50:
                    garr = p[5]
                    enemy_in, _ = E._enemy_threat(planets, fleets, p, seat)
                    A["cap_garr"].append(garr)
                    A["cap_surplus"].append(garr - enemy_in)
                    A["cap_n"] += 1
                    if garr < enemy_in:
                        A["cap_underfloor"] += 1
            prev_owner[pid] = owner
        caps_by_step[t] = caps_cum

    for S in SAMPLES:
        if S not in by_step:
            continue
        od = by_step[S]
        planets = od["planets"]; fleets = od["fleets"]
        n_own = sum(1 for p in planets if int(p[1]) == seat)
        if n_own == 0:
            continue
        n_en = sum(1 for p in planets if int(p[1]) >= 0 and int(p[1]) != seat)
        om = _mass(planets, fleets, seat)
        em = _enemy_mass(planets, fleets, seat)
        A["planets"][S].append(n_own)
        A["mass"][S].append(om)
        A["spp"][S].append(om / max(n_own, 1))
        A["share"][S].append(om / (om + em + 1e-9))
        A["e_planets"][S].append(n_en)
        A["e_spp"][S].append(em / max(n_en, 1))
        A["caps_cum"][S].append(caps_by_step.get(S, 0))
        exp, tot = _expandable_neutrals(planets, fleets, seat)
        A["exp_neutral"][S].append(exp)
        A["tot_neutral"][S].append(tot)
        far, af1, ag, cp = _classify_neutrals(planets, fleets, seat)
        A["neu_far"][S].append(far); A["neu_af1"][S].append(af1)
        A["neu_agg"][S].append(ag); A["neu_cap"][S].append(cp)


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _report(acc):
    print("\n" + "=" * 84)
    print(f"EXPANSION AUTOPSY — WON ({acc['won']['n_games']}g) vs LOST ({acc['lost']['n_games']}g)")
    print("=" * 84)
    hdr = f"{'step':>4} | {'our_planets':>11} {'our_mass':>9} {'ships/plnt':>10} {'mass_share':>10} | {'en_planets':>10} {'en_s/p':>7} | {'our_caps':>8}"
    for key in ("won", "lost"):
        A = acc[key]
        print(f"\n[{key.upper()}]")
        print(hdr)
        for S in SAMPLES:
            n = len(A["planets"][S])
            print(f"{S:>4} | {_mean(A['planets'][S]):>11.1f} {_mean(A['mass'][S]):>9.0f} "
                  f"{_mean(A['spp'][S]):>10.1f} {_mean(A['share'][S]):>10.2f} | "
                  f"{_mean(A['e_planets'][S]):>10.1f} {_mean(A['e_spp'][S]):>7.1f} | "
                  f"{_mean(A['caps_cum'][S]):>8.1f}  (n={n})")
        cn = max(A["cap_n"], 1)
        print(f"  captures @<50: n={A['cap_n']}  garrison-at-capture {_mean(A['cap_garr']):.1f}  "
              f"surplus(garr-enemy_in) {_mean(A['cap_surplus']):+.1f}  "
              f"under-threat-at-capture {A['cap_underfloor']/cn:.0%}")
        # acquisition-vs-retention discriminator @50: caps_cum = territory GAINED; planets = territory HELD.
        # peeled = gained - held. Low caps_cum -> ACQUISITION fails (expansion/tempo). High peeled with
        # normal caps_cum -> RETENTION fails (defensive bleed = the separate non-expansion mechanism).
        caps50, pl50 = _mean(A["caps_cum"][50]), _mean(A["planets"][50])
        print(f"  @50 acquire-vs-retain: caps_gained {caps50:.1f}  planets_held {pl50:.1f}  "
              f"peeled {caps50 - pl50:+.1f}")
        print(f"  neutral 4-way (far / af1 / agg / cap)  [agg = pooled-reachable lever]:")
        for S in (20, 30, 40):
            print(f"     step {S}: far {_mean(A['neu_far'][S]):4.1f}  af1 {_mean(A['neu_af1'][S]):4.1f}  "
                  f"agg {_mean(A['neu_agg'][S]):4.1f}  cap {_mean(A['neu_cap'][S]):4.1f}  "
                  f"(tot {_mean(A['tot_neutral'][S]):.1f})")
    print("\n  FORK (read LOST @20-40): far dominates = REACH-limited, no unilateral lever.")
    print("        agg dominates = AGGREGATION/COMMITMENT lever (ships exist across planets, won't pool).")
    print("        cap dominates = genuine capacity / opponent-capability.")
    print("=" * 84)


def run(ckpt_path, opponent, n_seeds, seed_set):
    model, cfg, target_decode, ship_bin_mode = _load_model(ckpt_path)
    device = torch.device("cpu")
    base_agent = E.build_agent_fn(model, device, target_decode=target_decode,
                                  ship_bin_mode=ship_bin_mode,
                                  allow_reinforce=bool(model.allow_reinforce))
    records = []

    def rec_agent(obs):
        moves = base_agent(obs)
        records.append({"obs": _obs_to_dict(obs)})
        return moves

    seeds = _seed_list(seed_set, n_seeds)
    print(f"seed_set={seed_set}  {len(seeds)} seeds x2 seats = {len(seeds)*2} games  vs {opponent}", flush=True)
    acc = {"won": _new_acc(), "lost": _new_acc()}
    for seed in seeds:
        for our_seat in (0, 1):
            records.clear()
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run([rec_agent, opponent] if our_seat == 0 else [opponent, rec_agent])
            final = env.steps[-1]
            r_us = final[our_seat].reward or 0.0
            r_opp = final[1 - our_seat].reward or 0.0
            outcome = "won" if r_us >= r_opp else "lost"
            _scan_game(list(records), our_seat, acc[outcome])
    _report(acc)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", default="opponents/candidate_ajay_1200.py")
    ap.add_argument("--seeds", type=int, default=24, help="#seeds for --seed-set random (each BOTH seats)")
    ap.add_argument("--seed-set", choices=["random", "static", "rotating"], default="random",
                    help="random=seeds 0..n; static/rotating=panel extreme archetype cells")
    args = ap.parse_args()
    run(args.checkpoint, args.opponent, args.seeds, args.seed_set)


if __name__ == "__main__":
    main()
