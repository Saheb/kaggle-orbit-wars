"""Recompute phase4e's TRAINING reward at every step of a replay (seat 0), decomposed by
component, to see what the policy was actually optimizing. phase4e reward (from train log):
  win_margin_coeff 0.5 | expansion_coef 0.03 | early_capture_coef 0.3 (exp decay) | shaping 0.0
  defense 0.0 | first_strike off | speed 0.0 | reinforce_cost 0 (not set)
Formulas mirror torch_env.step():
  expansion[t]   = 0.03 * ((prod0[t]-prod0[t-1]) - (prod1[t]-prod1[t-1]))     # production-LEAD delta
  earlycap[t]    = 0.3 * (exp(-2.5*t/500)+0.10) * clamp(owned0[t]-owned0[t-1], -1, +1)
  terminal[T]    = +1 + 0.5*(my_score/total) if win else -1                    # score = ships owned+inflight
  shaping/material = 0  (OFF)  -> ship loss during play is INVISIBLE to the reward
"""
import argparse
import json
import math

EXP_COEF = 0.03
EC_COEF = 0.3
WIN_MARGIN = 0.5
EPISODE_STEPS = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    R = json.load(open(args.json))
    F = R["frames"]
    n = len(F)

    def prod(fr, own):
        return sum(p["prod"] for p in fr["planets"] if p["own"] == own)

    rows = []
    cum = 0.0
    tot_exp = tot_ec_pos = tot_ec_neg = 0.0
    for t in range(1, n):
        f, f0 = F[t], F[t - 1]
        dprod0 = prod(f, 0) - prod(f0, 0)
        dprod1 = prod(f, 1) - prod(f0, 1)
        exp_r = EXP_COEF * (dprod0 - dprod1)
        decay = math.exp(-2.5 * t / EPISODE_STEPS) + 0.10
        d_owned = max(-1, min(1, f["our_p"] - f0["our_p"]))
        ec_r = EC_COEF * decay * d_owned
        step_r = exp_r + ec_r
        cum += step_r
        tot_exp += exp_r
        if ec_r > 0:
            tot_ec_pos += ec_r
        else:
            tot_ec_neg += ec_r
        rows.append((t, exp_r, ec_r, step_r, cum, f["our_p"], f["opp_p"]))

    # terminal
    meta = R["meta"]
    term = -1.0 if not meta["win"] else 1.0  # this game is a loss
    cum_with_term = cum + term

    print(f"\n=== phase4e per-step REWARD (seat0)  seed {meta['seed']}  {'WIN' if meta['win'] else 'LOSS'} ===")
    print(" components: expansion(0.03·Δprod-lead) + early-capture(0.3·decay·Δplanets); shaping/material OFF\n")
    print(f" {'t':>4} {'expans':>8} {'earlycap':>9} {'step':>8} {'cumul':>8} | ourP oppP  note")
    ev_by_t = {}
    for e in R["events"]:
        ev_by_t.setdefault(e["t"], []).append(e)
    for (t, exp_r, ec_r, step_r, c, op, pp) in rows:
        note = ""
        for e in ev_by_t.get(t, []):
            who = {-1: "neu", 0: "US", 1: "OPP"}
            tag = "CAP" if e["to"] == 0 else ("LOST" if e["frm"] == 0 else "flip")
            note += f" {tag} p{e['pid']}({who[e['frm']]}->{who[e['to']]})"
        # only print steps with an event or notable reward
        if note or abs(step_r) > 0.05 or t in (1, n - 1):
            print(f" {t:>4} {exp_r:>+8.3f} {ec_r:>+9.3f} {step_r:>+8.3f} {c:>+8.3f} | {op:>4} {pp:>4} {note}")

    print(f"\n terminal reward (loss): {term:+.2f}   ->  GAME TOTAL {cum_with_term:+.3f}")
    print("\n--- reward decomposition over the game ---")
    print(f"  expansion (net):        {tot_exp:+.3f}")
    print(f"  early-capture POSITIVE: {tot_ec_pos:+.3f}  (reward for capturing)")
    print(f"  early-capture NEGATIVE: {tot_ec_neg:+.3f}  (penalty for LOSING planets)")
    print(f"  pre-terminal subtotal:  {cum:+.3f}")
    print(f"  terminal (win/loss):    {term:+.3f}")
    print("\n  KEY: material/shaping coef = 0.0  ->  every ship launched-and-lost contributes")
    print("       EXACTLY 0 to reward. Failed attacks and idle are reward-identical.")


if __name__ == "__main__":
    main()
