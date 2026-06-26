"""Per-launch audit for one game. Reconstructs every fleet (source=from_planet_id f[5],
ships=f[6], unique id=f[0]), follows it to arrival, and reports the target's garrison AT
LAUNCH vs AT ARRIVAL (garrison + production growth + co-arrivals) and the actual outcome.
Surfaces: undercommit (sent < defense-at-arrival), wasted ships, wrong-target fires."""
import argparse
import math
from kaggle_environments import make
from orbit_wars_rl.scripts.onegame_load import load_faithful_agent
from orbit_wars_rl.eval import _dm_fleet_target
from orbit_wars_rl.features import extract_features

WHO = {-1: "neutral", 0: "US", 1: "OPP"}


def _target_value_row(obs, player, src_pid, tgt_pid):
    """Return pairwise target-value channels for the source->target launch, if the source
    is represented in the policy's owned-source slots. Values are model inputs at launch:
    capture_value_40, reactive_roi_40, friendly_reachable_mass, keepability_margin."""
    feats = extract_features(obs, player, num_players=2, max_planets=48, max_fleets=128)
    planets = obs.get("planets") or []
    src_idx = next((i for i, p in enumerate(planets) if int(p[0]) == int(src_pid)), None)
    tgt_idx = next((i for i, p in enumerate(planets) if int(p[0]) == int(tgt_pid)), None)
    if src_idx is None or tgt_idx is None:
        return None
    owned = feats["owned_indices"].numpy()
    slots = [i for i, idx in enumerate(owned) if int(idx) == int(src_idx)]
    if not slots:
        return None
    row = feats["pairwise_features"].numpy()[slots[0], tgt_idx]
    return dict(cap_value=float(row[16]), reactive_roi=float(row[17]),
                friendly_reach=float(row[18]), keepability=float(row[19]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", default="opponents/candidate_ajay_1200.py")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gate", type=int, default=2)
    ap.add_argument("--side", default="0", help="0=us,1=opp,both")
    ap.add_argument("--maxstep", type=int, default=60)
    args = ap.parse_args()

    agent = load_faithful_agent(args.checkpoint)
    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.run([agent, args.opponent])
    steps = env.steps
    n = len(steps)

    def obs(t):
        o = steps[t][0].observation
        return (o.get("planets") or []), (o.get("fleets") or [])

    # planet state by id at each step
    pstate = []  # list over t of {pid: (owner, ships, prod)}
    for t in range(n):
        P, _ = obs(t)
        pstate.append({int(p[0]): (int(p[1]), float(p[5]), float(p[6])) for p in P})

    # track fleets by id: first-seen (launch) and last-seen (arrival = last+1)
    seen = {}     # fid -> dict(launch, owner, src, ships, last, tgt_at_launch)
    order = []
    for t in range(n):
        P, F = obs(t)
        present = set()
        for f in F:
            fid = int(f[0]); present.add(fid)
            if fid not in seen:
                tgt = _dm_fleet_target(P, f)
                seen[fid] = dict(launch=t, owner=int(f[1]), src=int(f[5]),
                                 ships=float(f[6]), last=t,
                                 tgt=int(tgt[0]) if tgt is not None else -1)
                order.append(fid)
            else:
                seen[fid]["last"] = t
                # refine target mid-flight (heading stable; take nearest each step)
                tgt = _dm_fleet_target(P, f)
                if tgt is not None:
                    seen[fid]["tgt"] = int(tgt[0])

    sides = [0, 1] if args.side == "both" else [int(args.side)]
    print(f"\n=== SEED {args.seed}  per-launch audit (side={args.side}, steps<= {args.maxstep}) ===")
    print("legend: sent | tgt@launch g0(prod) | eta | value roi keep | tgt@arrival g_arr owner | RESULT")
    print("        UNDERCOMMIT = sent <= defense at arrival (can't take); WASTE = arrived, no capture\n")

    summ = {0: dict(n=0, cap=0, waste=0, under=0, ships_sent=0.0, ships_waste=0.0),
            1: dict(n=0, cap=0, waste=0, under=0, ships_sent=0.0, ships_waste=0.0)}
    for d in summ.values():
        d.update(tv_n=0, cap_value=0.0, reactive_roi=0.0, keepability=0.0,
                 low_value=0, negative_keep=0)

    for fid in order:
        s = seen[fid]
        if s["owner"] not in sides:
            continue
        if s["launch"] > args.maxstep:
            continue
        t0, arr = s["launch"], s["last"] + 1
        eta = arr - t0
        tid = s["tgt"]
        g0 = pstate[t0].get(tid, (None, 0, 0))
        launch_obs = steps[t0][0].observation
        tv = _target_value_row(launch_obs, s["owner"], s["src"], tid)
        # target state just before impact (growth + reinforcement captured by replay)
        ti = min(arr, n - 1)
        g_pre = pstate[ti - 1].get(tid, (None, 0, 0)) if ti - 1 >= 0 else (None, 0, 0)
        g_post = pstate[ti].get(tid, (None, 0, 0))
        owner_before = g_pre[0]
        owner_after = g_post[0]
        sent = s["ships"]
        def_arr = g_pre[1]  # garrison just before impact (already grown)
        captured = (owner_before != s["owner"]) and (owner_after == s["owner"])
        # was it a real attack target (not ours at launch)?
        attack = g0[0] != s["owner"]
        under = attack and (not captured) and (sent <= def_arr + 1e-6)
        waste = attack and (not captured)
        d = summ[s["owner"]]
        d["n"] += 1; d["ships_sent"] += sent
        if captured:
            d["cap"] += 1
        if waste:
            d["waste"] += 1; d["ships_waste"] += sent
        if under:
            d["under"] += 1
        if tv is not None and attack:
            d["tv_n"] += 1
            d["cap_value"] += tv["cap_value"]
            d["reactive_roi"] += tv["reactive_roi"]
            d["keepability"] += tv["keepability"]
            d["low_value"] += int(tv["cap_value"] < 0.05 or tv["reactive_roi"] < 0.0)
            d["negative_keep"] += int(tv["keepability"] < 0.0)
        tv_str = "   n/a   " if tv is None else (
            f"{tv['cap_value']:>4.2f} {tv['reactive_roi']:>+5.2f} {tv['keepability']:>+5.2f}"
        )
        flag = "CAPTURED" if captured else ("UNDERCOMMIT/WASTE" if under else ("WASTE" if waste else "reinforce/ok"))
        print(f" t{t0:>3} {WHO[s['owner']]:<3} p{s['src']:<3}->p{tid:<3}({WHO[g0[0]]:<7}) "
              f"sent {sent:>5.0f} | g0 {g0[1]:>4.0f}(pr{g0[2]:>4.1f}) | eta {eta:>2} | "
              f"{tv_str} | "
              f"g_arr {def_arr:>5.0f} {WHO[owner_before] if owner_before is not None else '?':<7} -> {flag}")

    print("\n--- summary ---")
    for sd in sides:
        d = summ[sd]
        print(f" {WHO[sd]}: launches {d['n']}  captured {d['cap']}  waste(no-cap) {d['waste']}  "
              f"undercommit {d['under']}  | ships sent {d['ships_sent']:.0f} wasted {d['ships_waste']:.0f} "
              f"({100*d['ships_waste']/max(d['ships_sent'],1):.0f}%)")
        if d["tv_n"]:
            n = max(d["tv_n"], 1)
            print(f"      target-value attacks: avg value {d['cap_value']/n:.3f}  "
                  f"avg reactive_roi {d['reactive_roi']/n:+.3f}  avg keep {d['keepability']/n:+.3f}  "
                  f"low-value {100*d['low_value']/n:.0f}%  negative-keep {100*d['negative_keep']/n:.0f}%")


if __name__ == "__main__":
    main()
