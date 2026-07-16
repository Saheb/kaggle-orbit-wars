"""Why is peel-rate ~1.0 vs Ender? — per-capture retention forensics.

The Ender panel reports peel-rate 0.99 (5,277/5,309 captures lost). That number is close to
TAUTOLOGICAL: peel-rate = lost_caps/captures, and we are wiped to 0 material in 100% of games, so
every capture is eventually lost by construction. It restates "we lose"; it does not explain it.

This separates the two live hypotheses per capture:

  A. WE TAKE WHAT WE CANNOT HOLD (target selection). Captures die fast even while the game is
     still even; the planet is re-takeable the moment we land (thin post-capture garrison, enemy
     mass already inbound/reachable). → fixable by pricing the candidate's survival, which is
     exactly what the counterfactual + economy channels are for.
  B. WE HOLD FINE AND GET OUT-PRODUCED (macro). Captures survive a long time; losses cluster in
     the terminal collapse once the economy has already compounded away. → target selection is
     not the lever; budget/economy is.

Discriminators, all per-capture and split by whether the game was still competitive AT CAPTURE
TIME (so the terminal collapse cannot masquerade as a selection failure):
  - hold-duration distribution (censored = still held at end)
  - post-capture garrison, and enemy mass reachable within the next K steps
  - churn: distinct planets vs total captures (re-taking the same rock is a thrash loop)
  - whether we ever reinforced the capture before losing it

    CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/peel_diagnosis.py --seeds 6
"""
import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
os.chdir(_REPO)

import torch                              # noqa: E402
import eval as ev                         # noqa: E402
from config import Config                 # noqa: E402
from model import EntityTransformer       # noqa: E402
from kaggle_environments import make      # noqa: E402

CHAMPION = ("gpu_run_artifacts/binarymarg100m_l4_from25m/checkpoints/"
            "torch_step_40108032_binarymarg100m_l4_from25m_20260714_163936.pt")
ENDER = "opponents/candidate_ender.py"
REACH_K = 15          # steps ahead we ask "can enemy mass get here"
SHORT_HOLD = 20       # a capture lost within this many steps was arguably never held


def _material(obs, seat):
    return (sum(p[5] for p in obs.planets if p[1] == seat)
            + sum(f[6] for f in obs.fleets if f[1] == seat))


def _enemy_reachable_mass(planets, fleets, tgt, seat, k=REACH_K):
    """Enemy ships that could be on `tgt` within k steps: fleets already resolved there, plus
    garrisons on enemy planets close enough to launch and arrive."""
    total = 0.0
    for f in fleets:
        if int(f[1]) == seat:
            continue
        # resolve by lead-collision from the fleet's own position/heading
        t = ev._lead_collision_target(planets, float(f[2]), float(f[3]), float(f[4]), float(f[6]))
        if t is not None and t[0] == tgt[0]:
            total += float(f[6])
    for p in planets:
        if int(p[1]) == seat or int(p[1]) < 0 or p[0] == tgt[0]:
            continue
        d = ((float(p[2]) - float(tgt[2])) ** 2 + (float(p[3]) - float(tgt[3])) ** 2) ** 0.5
        if d / max(ev._ship_speed_py(float(p[5])), 1e-6) <= k:
            total += float(p[5])
    return total


def analyse(steps, seat, out):
    open_caps = {}      # pid -> dict(t_cap, garrison, enemy_reach, material_delta, reinforced)
    cap_count = Counter()
    for t in range(1, len(steps)):
        if seat >= len(steps[t]) or seat >= len(steps[t - 1]):
            continue
        obs0 = steps[t - 1][seat].observation
        obs1 = steps[t][seat].observation
        p0, p1 = obs0.get("planets"), obs1.get("planets")
        if not p0 or not p1:
            continue
        prev_own = {p[0]: int(p[1]) for p in p0}
        my_mat = _material(obs1, seat)
        op_mat = sum(_material(obs1, s) for s in (0, 1) if s != seat)
        mat_delta = my_mat - op_mat

        # reinforcement we sent to planets we currently hold
        for mv in (steps[t][seat].action or []):
            if not mv or len(mv) < 3:
                continue
            src = next((p for p in p0 if p[0] == int(mv[0])), None)
            if src is None or not (float(src[5]) > 0 and int(mv[2]) <= float(src[5])):
                continue
            tgt = ev._resolve_launch_target(p0, src, float(mv[1]), int(mv[2]))
            if tgt is not None and tgt[0] in open_caps:
                open_caps[tgt[0]]["reinforced"] += 1

        for p in p1:
            pid, own = p[0], int(p[1])
            was = prev_own.get(pid)
            if was is None:
                continue
            if was != seat and own == seat:                     # WE CAPTURED IT
                cap_count[pid] += 1
                open_caps[pid] = {
                    "t_cap": t,
                    "garrison": float(p[5]),
                    "enemy_reach": _enemy_reachable_mass(p1, obs1.get("fleets") or [], p, seat),
                    "mat_delta": mat_delta,
                    "reinforced": 0,
                }
            elif was == seat and own != seat and pid in open_caps:   # WE LOST IT
                c = open_caps.pop(pid)
                out["episodes"].append({**c, "t_lost": t, "hold": t - c["t_cap"],
                                        "censored": False, "game_len": len(steps)})
    for pid, c in open_caps.items():                              # still held at the end
        out["episodes"].append({**c, "t_lost": None, "hold": len(steps) - c["t_cap"],
                                "censored": True, "game_len": len(steps)})
    out["distinct"] += len(cap_count)
    out["captures"] += sum(cap_count.values())


def _pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "—"


def report(out):
    eps = out["episodes"]
    if not eps:
        print("no captures recorded")
        return
    lost = [e for e in eps if not e["censored"]]
    print(f"\n{'='*78}\nPEEL DIAGNOSIS — {out['games']} games vs Ender\n{'='*78}")
    print(f"captures {out['captures']}  distinct planets {out['distinct']}  "
          f"churn (captures/distinct) {out['captures']/max(out['distinct'],1):.2f}")
    print(f"episodes {len(eps)}  lost {len(lost)} ({_pct(len(lost), len(eps))})  "
          f"still-held-at-end {len(eps)-len(lost)}")

    holds = [e["hold"] for e in lost]
    if holds:
        print(f"\nhold duration (lost episodes): median {statistics.median(holds):.0f}st  "
              f"mean {statistics.mean(holds):.0f}st")
        for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 10**9)):
            n = sum(1 for h in holds if lo <= h < hi)
            lab = f"{lo}-{hi}st" if hi < 10**9 else f">={lo}st"
            print(f"    {lab:>10}  {_pct(n, len(holds)):>6}  {'#'*int(40*n/len(holds))}")

    # THE discriminator: captures made while the game was still even/ahead.
    comp = [e for e in eps if e["mat_delta"] > -50]
    comp_lost_fast = [e for e in comp if not e["censored"] and e["hold"] < SHORT_HOLD]
    print(f"\nCaptures made while still competitive (material delta > -50): {len(comp)}"
          f" of {len(eps)} ({_pct(len(comp), len(eps))})")
    print(f"  ...of those, lost within {SHORT_HOLD}st: {len(comp_lost_fast)} "
          f"({_pct(len(comp_lost_fast), len(comp))})   <-- hypothesis A")
    late = [e for e in lost if e["t_lost"] > 0.8 * e["game_len"]]
    print(f"  losses landing in the last 20% of the game: {len(late)} "
          f"({_pct(len(late), len(lost))})   <-- hypothesis B")

    print("\nAt capture time (all episodes):")
    g = [e["garrison"] for e in eps]
    er = [e["enemy_reach"] for e in eps]
    print(f"  post-capture garrison   median {statistics.median(g):.0f}")
    print(f"  enemy mass reachable within {REACH_K}st  median {statistics.median(er):.0f}")
    doomed = sum(1 for e in eps if e["enemy_reach"] > e["garrison"])
    print(f"  captures landing ALREADY OUT-MASSED (enemy reach > our garrison): "
          f"{doomed} ({_pct(doomed, len(eps))})")
    nrf = sum(1 for e in lost if e["reinforced"] == 0)
    print(f"  lost captures we NEVER reinforced: {nrf} ({_pct(nrf, len(lost))})")

    for lab, sel in (("competitive & out-massed at capture",
                      [e for e in comp if e["enemy_reach"] > e["garrison"]]),
                     ("competitive & NOT out-massed",
                      [e for e in comp if e["enemy_reach"] <= e["garrison"]])):
        ls = [e for e in sel if not e["censored"]]
        h = [e["hold"] for e in ls]
        med = f"{statistics.median(h):.0f}st" if h else "—"
        print(f"  {lab:>38}: n={len(sel):4d}  lost {_pct(len(ls), len(sel)):>6}  median hold {med}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--checkpoint", default=CHAMPION)
    ap.add_argument("--opponent", default=ENDER)
    ap.add_argument("--reinforce-gate-min-planets", type=int, default=2)
    ap.add_argument("--reinforce-garrison-floor", type=float, default=0.0)
    ap.add_argument("--out", default="gpu_run_artifacts/binarymarg100m_l4_from25m/"
                                     "replay_analysis/peel_diagnosis_ender.json")
    args = ap.parse_args()

    cfg = Config(); cfg.device = "cpu"
    device = torch.device("cpu")
    sd, action_decode = ev.load_checkpoint(args.checkpoint, cfg)
    model = EntityTransformer(cfg.model).to(device)
    model.load_state_dict(sd)
    model.eval()
    # ⚠ build_agent_fn reads these OFF THE MODEL (eval.py:391 / :1560), not from its kwargs.
    # Omitting them silently disables reinforcement — which makes the "was this capture ever
    # reinforced" question answer itself. Mirror the eval/watcher masks (gate2/floor0).
    model.allow_reinforce = bool(getattr(cfg.model, "allow_reinforce", False))
    model.reinforce_gate_min_planets = int(args.reinforce_gate_min_planets)
    model.reinforce_forward_only = bool(getattr(cfg.model, "reinforce_forward_only", False))
    model.reverse_edge_cooldown = int(getattr(cfg.model, "reverse_edge_cooldown", 0))
    model.reinforce_garrison_floor = float(args.reinforce_garrison_floor)
    model.sufficient_commit_factor = float(getattr(cfg.model, "sufficient_commit_factor", 0.0))
    if not model.allow_reinforce:
        raise SystemExit(
            "checkpoint has allow_reinforce=False — a reinforcement diagnosis on it is vacuous.")
    print(f"  allow_reinforce={model.allow_reinforce} gate>={model.reinforce_gate_min_planets} "
          f"floor={model.reinforce_garrison_floor} cooldown={model.reverse_edge_cooldown}")
    agent_fn = ev.build_agent_fn(
        model, device, fire_threshold=0.5, ship_bin_mode=cfg.model.ship_bin_mode,
        target_decode=(action_decode == "target"), num_players=2)

    out = {"episodes": [], "captures": 0, "distinct": 0, "games": 0}
    for seed in range(args.seeds):
        for my_seat in (0, 1):
            agents = ([agent_fn, args.opponent] if my_seat == 0
                      else [args.opponent, agent_fn])
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run(agents)
            analyse(env.steps, my_seat, out)
            out["games"] += 1
            print(f"  seed={seed} seat={my_seat} len={len(env.steps)} "
                  f"episodes={len(out['episodes'])}", flush=True)

    report(out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
