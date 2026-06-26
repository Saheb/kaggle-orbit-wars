"""50->100 transition autopsy  (WON vs LOST).

The win/loss split is identical in the opening (planets 2/4, cap/atk discipline) and diverges
in the mid-game: winners 8->12 planets across step 50-100, losers 7->5. The eval panel says
losers pour ~98% of to-lost mass into HOPELESS planets. This script asks, at the decision level,
on BOTH won and lost games, at steps in [START,END]:

  (A) COUNTERFACTUAL (allocation-sufficiency, static, one-step):
      pool the mass actually sent to HOPELESS planets, greedily redistribute to defendable-but-
      under-floor planets (cheapest shortfall first), count extra hold-floor crossings. Proves the
      mass existed & was mis-routed -- NOT that we'd win (opponent static). Deficit -> not the lever.

  (B) HEADS vs FEATURES, split WON/LOST:
      - fire head: P(fire) when a defendable planet needs help (veto test) + #firing-sources/decision
        histogram (is multi-source launching even attempted?)
      - target head: aggregation ratio ON decisions where >=2 sources fire (do they concentrate on
        one target, or each pick a different one?) + entropy
      - feature adequacy: threat-deficit separability of HOPELESS vs DEFENDABLE.
      If features separate the classes but heads don't act -> head problem, not feature.

Reuses eval's decode + target-resolution + triage, so hopeless% matches the panel (validation).
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict

import torch
from kaggle_environments import make

from orbit_wars_rl.config import Config
from orbit_wars_rl.model import EntityTransformer
import orbit_wars_rl.eval as E

from orbit_wars_rl.features import extract_features  # noqa: F401  (kept for parity of import env)
from orbit_wars_rl.action_mask import compute_action_masks


def _obs_to_dict(obs):
    if isinstance(obs, dict):
        return obs
    return {
        "step": int(getattr(obs, "step", 0)),
        "player": int(getattr(obs, "player", 0)),
        "planets": [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production] for p in obs.planets],
        "fleets": [[f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships] for f in obs.fleets],
        "angular_velocity": float(getattr(obs, "angular_velocity", 0.0)),
        "initial_planets": [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
                            for p in getattr(obs, "initial_planets", obs.planets)],
        "comet_planet_ids": list(getattr(obs, "comet_planet_ids", [])),
    }


def _load_model(ckpt_path):
    cfg = Config()
    state_dict, ckpt_action_decode = E.load_checkpoint(ckpt_path, cfg)
    m = cfg.model
    model = EntityTransformer(m)
    for attr in ("allow_reinforce", "reinforce_gate_min_planets", "reinforce_forward_only",
                 "reinforce_garrison_floor", "sufficient_commit_factor", "reverse_edge_cooldown"):
        setattr(model, attr, getattr(m, attr, 0))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    target_decode = (ckpt_action_decode == "target")
    print(f"loaded {ckpt_path} target_decode={target_decode} allow_reinforce={model.allow_reinforce} "
          f"gate={model.reinforce_gate_min_planets} cd={model.reverse_edge_cooldown} "
          f"ship_bin_mode={m.ship_bin_mode}", flush=True)
    return model, cfg, target_decode, m.ship_bin_mode


def _hold_floor_ratio(planets, fleets, tgt, seat):
    enemy_in, threat_eta = E._enemy_threat(planets, fleets, tgt, seat)
    reach = E._reachable_enemy_mass(planets, tgt, seat)
    denom = enemy_in + E._DM_BETA_EVAL * reach + 1.0
    # ignores friendly_inbound -> conservative (slightly over-counts under-floor planets)
    return tgt[5] / denom if denom > 0 else 99.0, enemy_in, threat_eta


def _new_acc():
    return {
        "mass_by_class": [0.0, 0.0, 0.0, 0.0],
        "cf_extra": 0, "cf_dec": 0, "cf_hopeless": 0.0, "cf_recovered": 0.0,
        "tgt_entropy": [], "agg_ratio": [], "fire_p": [],
        "feat_hopeless": [], "feat_defendable": [],
        "nfire_hist": defaultdict(int), "multi_fire_dec": 0,
        "fire_p_even": [], "agg_even": [], "n_even": 0, "balance_sum": 0.0,
        "n_decisions": 0, "n_games": 0,
        "nfire_hist_peven": defaultdict(int), "multi_fire_dec_peven": 0,
        "nfire_hist_jeven": defaultdict(int), "multi_fire_dec_jeven": 0,
        "n_peven": 0, "n_jeven": 0,
        "planet_owned_sum": 0, "planet_enemy_sum": 0,
        "dec_by_owned": defaultdict(int), "mf_by_owned": defaultdict(int),
    }


def run(ckpt_path, opponent, n_seeds, start, end):
    model, cfg, target_decode, ship_bin_mode = _load_model(ckpt_path)
    device = torch.device("cpu")

    captured = []
    def _hook(_m, _inp, out):
        captured.append({k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()})
    model.register_forward_hook(_hook)
    base_agent = E.build_agent_fn(model, device, target_decode=target_decode,
                                  ship_bin_mode=ship_bin_mode,
                                  allow_reinforce=bool(model.allow_reinforce))

    records = []
    def rec_agent(obs):
        od = _obs_to_dict(obs)
        n0 = len(captured)
        moves = base_agent(obs)
        out = captured[-1] if len(captured) > n0 else None
        records.append({"obs": od, "moves": moves, "out": out})
        return moves

    acc = {"won": _new_acc(), "lost": _new_acc()}
    for seed in range(n_seeds):
        for our_seat in (0, 1):
            records.clear()
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run([rec_agent, opponent] if our_seat == 0 else [opponent, rec_agent])
            final = env.steps[-1]
            r_us = final[our_seat].reward or 0.0
            r_opp = final[1 - our_seat].reward or 0.0
            outcome = "won" if r_us >= r_opp else "lost"
            acc[outcome]["n_games"] += 1
            _autopsy_game(records, our_seat, start, end, acc[outcome])
    _report(acc)


def _autopsy_game(records, seat, start, end, A):
    for rec in records:
        od = rec["obs"]
        t = od["step"]
        if t < start or t >= end:
            continue
        planets = od["planets"]; fleets = od["fleets"]
        byid = {int(p[0]): p for p in planets}
        owned = [p for p in planets if int(p[1]) == seat]
        if len(owned) < 2:
            continue
        cls_by_id = {int(p[0]): E._threat_class(planets, fleets, p, seat) for p in owned}

        # state-match: our mass share at this decision. Winners are usually already AHEAD at 50-100,
        # so an unconditioned head delta is a consequence of winning, not a cause. Compare on EVEN boards.
        our_mass = (sum(p[5] for p in planets if int(p[1]) == seat)
                    + sum(f[6] for f in fleets if int(f[1]) == seat))
        enemy_mass = (sum(p[5] for p in planets if int(p[1]) >= 0 and int(p[1]) != seat)
                      + sum(f[6] for f in fleets if int(f[1]) >= 0 and int(f[1]) != seat))
        balance = our_mass / (our_mass + enemy_mass + 1e-9)
        even = 0.40 <= balance <= 0.60
        A["balance_sum"] += balance
        if even:
            A["n_even"] += 1
        # planet-count state-match: #launches scales with source (planet) count, not mass, so
        # mass-even leaves the planet gap intact. planet-even tests the snowball mechanism directly;
        # joint-even (mass AND planet) is the strongest control.
        n_enemy_planets = sum(1 for p in planets if int(p[1]) >= 0 and int(p[1]) != seat)
        n_owned_planets = len(owned)
        peven = abs(n_owned_planets - n_enemy_planets) <= 1
        jeven = even and peven
        A["planet_owned_sum"] += n_owned_planets
        A["planet_enemy_sum"] += n_enemy_planets
        if peven:
            A["n_peven"] += 1
        if jeven:
            A["n_jeven"] += 1

        # (A) parse the ACTUAL launches this step (ground truth, not pf>0.5)
        hopeless_mass = 0.0
        launch_targets = []                  # resolved target-id of every fleet launched this step
        for mv in rec["moves"]:
            src = byid.get(int(mv[0]))
            if src is None:
                continue
            tgt = E._resolve_launch_target(planets, src, mv[1])
            if tgt is None:
                continue
            launch_targets.append(int(tgt[0]))
            if int(tgt[1]) == seat:          # reinforce (own target)
                ships = float(mv[2])
                c = cls_by_id.get(int(tgt[0]), E._threat_class(planets, fleets, tgt, seat))
                A["mass_by_class"][c] += ships
                if c == 3:
                    hopeless_mass += ships
        # launches-per-step + aggregation from real moves
        nlaunch = len(launch_targets)
        A["nfire_hist"][min(nlaunch, 5)] += 1
        if peven:
            A["nfire_hist_peven"][min(nlaunch, 5)] += 1
        if jeven:
            A["nfire_hist_jeven"][min(nlaunch, 5)] += 1
        # stratify by absolute owned count: isolates source-count (composition) from policy.
        # at owned=k, equal WON/LOST >=2-rate -> difference is all composition (snowball).
        A["dec_by_owned"][n_owned_planets] += 1
        if nlaunch >= 2:
            A["mf_by_owned"][n_owned_planets] += 1
        if nlaunch >= 2:                      # aggregation only meaningful w/ >=2 launches
            A["multi_fire_dec"] += 1
            if peven:
                A["multi_fire_dec_peven"] += 1
            if jeven:
                A["multi_fire_dec_jeven"] += 1
            _agg = nlaunch / len(set(launch_targets))
            A["agg_ratio"].append(_agg)
            if even:
                A["agg_even"].append(_agg)

        # counterfactual redistribution
        shortfalls = []
        for p in owned:
            if cls_by_id[int(p[0])] in (1, 2):
                ratio, enemy_in, _ = _hold_floor_ratio(planets, fleets, p, seat)
                if ratio < 1.0:
                    need = (enemy_in + E._DM_BETA_EVAL * E._reachable_enemy_mass(planets, p, seat)
                            + E._DM_OVERHEAD - p[5])
                    if need > 0:
                        shortfalls.append(need)
        shortfalls.sort()
        pool = hopeless_mass; extra = 0
        for need in shortfalls:
            if pool >= need:
                pool -= need; extra += 1
            else:
                break
        A["cf_extra"] += extra; A["cf_dec"] += 1
        A["cf_hopeless"] += hopeless_mass; A["cf_recovered"] += hopeless_mass - pool
        A["n_decisions"] += 1

        # (B) heads
        out = rec["out"]
        if out is None or out.get("target_logits") is None:
            continue
        masks = compute_action_masks(od, od["player"])
        slot_valid = masks["slot_valid"].cpu().numpy().tolist()
        fire_l = out["fire_logits"][0]
        tgt_l = out["target_logits"][0]
        for slot, valid in enumerate(slot_valid):
            if not valid:
                continue
            fl = fire_l[slot]; fl = fl[torch.isfinite(fl)]
            if fl.numel() == 0:
                continue
            pf = torch.sigmoid(fl.max()).item()
            dist = torch.softmax(tgt_l[slot], dim=-1)
            A["tgt_entropy"].append(-(dist * (dist + 1e-9).log()).sum().item())
            if shortfalls:
                A["fire_p"].append(pf)
                if even:
                    A["fire_p_even"].append(pf)

        for p in owned:
            c = cls_by_id[int(p[0])]
            enemy_in, _ = E._enemy_threat(planets, fleets, p, seat)
            sep = enemy_in - p[5]
            if c == 3:
                A["feat_hopeless"].append(sep)
            elif c in (1, 2):
                A["feat_defendable"].append(sep)


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _report(acc):
    names = ["safe", "cheap", "exp", "HOPELESS"]
    print("\n" + "=" * 78)
    print(f"TRANSITION AUTOPSY — WON ({acc['won']['n_games']}g) vs LOST ({acc['lost']['n_games']}g)")
    print("=" * 78)
    for key in ("won", "lost"):
        A = acc[key]
        tot = sum(A["mass_by_class"]) or 1.0
        cls = "  ".join(f"{names[i]} {A['mass_by_class'][i]/tot:.0%}" for i in range(4))
        nd = max(A["cf_dec"], 1)
        nfire_total = sum(A["nfire_hist"].values()) or 1
        hist = "  ".join(f"{k}:{A['nfire_hist'][k]/nfire_total:.0%}" for k in sorted(A["nfire_hist"]))
        print(f"\n[{key.upper()}]  decisions={A['n_decisions']}")
        print(f"  reinforce class:  {cls}")
        print(f"  counterfactual:   hopeless={A['cf_hopeless']:.0f}  redistributable="
              f"{A['cf_recovered']/max(A['cf_hopeless'],1):.0%}  extra-floor-crossings="
              f"{A['cf_extra']} ({A['cf_extra']/nd:.3f}/dec)")
        bal = A["balance_sum"] / max(A["n_decisions"], 1)
        print(f"  mass balance (our share): {bal:.2f}   even-state decisions (0.4-0.6): {A['n_even']} "
              f"({A['n_even']/max(A['n_decisions'],1):.0%})")
        print(f"  fire P(fire)@defendable-need:  all {_mean(A['fire_p']):.2f}   EVEN-state {_mean(A['fire_p_even']):.2f}   "
              f"(target entropy {_mean(A['tgt_entropy']):.2f}/{math.log(48):.2f})")
        print(f"  #launches/decision: {hist}   (>=2 launches: {A['multi_fire_dec']/nfire_total:.0%} of decisions)")
        # planet-count confirmer: tests whether the WON>LOST multi-launch rate is the snowball
        # (more planets -> more sources) or a residual activity-level policy difference.
        nhist_pe = sum(A["nfire_hist_peven"].values()) or 1
        nhist_je = sum(A["nfire_hist_jeven"].values()) or 1
        mean_owned = A["planet_owned_sum"] / max(A["n_decisions"], 1)
        mean_enemy = A["planet_enemy_sum"] / max(A["n_decisions"], 1)
        hist_pe = "  ".join(f"{k}:{A['nfire_hist_peven'][k]/nhist_pe:.0%}" for k in sorted(A["nfire_hist_peven"]))
        hist_je = "  ".join(f"{k}:{A['nfire_hist_jeven'][k]/nhist_je:.0%}" for k in sorted(A["nfire_hist_jeven"]))
        print(f"  planets (mean/decision):      owned {mean_owned:.1f}  enemy {mean_enemy:.1f}   "
              f"planet-even(|d|<=1): {A['n_peven']} ({A['n_peven']/max(A['n_decisions'],1):.0%})   "
              f"joint-even: {A['n_jeven']}")
        print(f"  #launches PLANET-even:        {hist_pe or 'n/a'}   "
              f"(>=2: {A['multi_fire_dec_peven']/nhist_pe:.0%}, n={sum(A['nfire_hist_peven'].values())})")
        print(f"  #launches JOINT-even:         {hist_je or 'n/a'}   "
              f"(>=2: {A['multi_fire_dec_jeven']/nhist_je:.0%}, n={sum(A['nfire_hist_jeven'].values())})")
        print(f"  aggregation ratio (>=2-launch decisions): all {_mean(A['agg_ratio']):.2f}   "
              f"EVEN {_mean(A['agg_even']):.2f}   [1.0=distinct; >1=concentrate; n_multi={A['multi_fire_dec']}]")
        print(f"  feature sep (enemy_in-garr):  HOPELESS {_mean(A['feat_hopeless']):+.1f}  "
              f"DEFENDABLE {_mean(A['feat_defendable']):+.1f}")
    # confirmer (decisive): >=2-launch rate stratified by ABSOLUTE owned count. At owned=k an equal
    # WON/LOST rate means the unmatched gap is pure composition (snowball); a persistent WON>LOST gap
    # within the same k means a residual activity-level policy difference.
    buckets = [("2-3",2,3),("4-5",4,5),("6-7",6,7),("8-9",8,9),("10-11",10,11),("12+",12,99)]
    def _bkt(d_mf, d_dec, lo, hi):
        d = sum(d_dec[k] for k in range(lo, hi+1))
        m = sum(d_mf[k] for k in range(lo, hi+1))
        return (m/d if d else float("nan")), d
    print("\n  >=2-launch rate by OWNED count  [equal WON=LOST within bucket -> snowball/composition]:")
    for label, lo, hi in buckets:
        w_rate, w_n = _bkt(acc["won"]["mf_by_owned"], acc["won"]["dec_by_owned"], lo, hi)
        l_rate, l_n = _bkt(acc["lost"]["mf_by_owned"], acc["lost"]["dec_by_owned"], lo, hi)
        print(f"    owned {label:>5}:  WON {w_rate:3.0%} (n={w_n:<4})   LOST {l_rate:3.0%} (n={l_n})")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", default="opponents/candidate_ajay_1200.py")
    ap.add_argument("--seeds", type=int, default=24, help="each played BOTH seats")
    ap.add_argument("--start", type=int, default=50)
    ap.add_argument("--end", type=int, default=100)
    args = ap.parse_args()
    run(args.checkpoint, args.opponent, args.seeds, args.start, args.end)


if __name__ == "__main__":
    main()
