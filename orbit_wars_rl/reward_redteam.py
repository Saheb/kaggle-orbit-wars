"""Tier-2 reward red-team: validate the proposed reward FUNCTION before any training.

NEW reward (what we designed):
  dense_t = C · decay(t_cap) · Δ(owned_production / total_board_prod)   # symmetric, capture-time-anchored
  terminal = +1 win / -1 loss          (win = more material, matching env score)
  decay(t) = exp(-2.5 t/500) + 0.10,  C = 0.5

Checks:
  (real replays)  I1 every LOSS total < 0   I2 every WIN total > 0   I3 max(loss) < min(win)  [no inversion]
                  + flag any old-reward LOSS that scored POSITIVE (the decay-bug we're fixing)
  (synthetic)     capture-then-lose = 0 · tennis ≤ 0 · hoard = 0 · initial loss < 0 ·
                  hold-to-end in (0, 1.1C] · win-small > lose-big
"""
import argparse
import math
from kaggle_environments import make
from onegame_load import load_faithful_agent

C = 0.5
def decay(t):
    return math.exp(-2.5 * t / 500.0) + 0.10


def material(stp, seat):
    ob = stp[0].observation
    P = ob.get("planets") or []
    F = ob.get("fleets") or []
    return (sum(float(p[5]) for p in P if int(p[1]) == seat)
            + sum(float(f[6]) for f in (F or []) if int(f[1]) == seat))


def new_reward(steps, seat):
    n = len(steps)
    P0 = steps[0][0].observation.get("planets") or []
    total = sum(float(p[6]) for p in P0) or 1.0
    owner, cap_t = {}, {}
    dense = 0.0
    for t in range(n):
        P = steps[t][0].observation.get("planets") or []
        for p in P:
            pid, own, prod = int(p[0]), int(p[1]), float(p[6])
            if pid not in owner:
                owner[pid], cap_t[pid] = own, t
            elif own != owner[pid]:
                prev = owner[pid]
                if prev == seat:                      # loss: anchored to OUR capture time
                    dense -= C * decay(cap_t[pid]) * prod / total
                if own == seat:                       # gain: anchored to now
                    dense += C * decay(t) * prod / total
                owner[pid], cap_t[pid] = own, t
    won = material(steps[-1], seat) > material(steps[-1], 1 - seat)
    term = 1.0 if won else -1.0
    return dense, term, dense + term, won


def old_reward(steps, seat):
    """phase4e reward: early_capture(count, UNanchored decay) + expansion(prod-lead) + terminal(±1+0.5margin)."""
    n = len(steps)
    def owned_count(stp, s):
        P = stp[0].observation.get("planets") or []
        return sum(1 for p in P if int(p[1]) == s)
    def prod_sum(stp, s):
        P = stp[0].observation.get("planets") or []
        return sum(float(p[6]) for p in P if int(p[1]) == s)
    dense = 0.0
    for t in range(1, n):
        d_own = max(-1, min(1, owned_count(steps[t], seat) - owned_count(steps[t-1], seat)))
        dense += 0.3 * decay(t) * d_own                                    # early_capture (unanchored)
        dp0 = prod_sum(steps[t], seat) - prod_sum(steps[t-1], seat)
        dp1 = prod_sum(steps[t], 1-seat) - prod_sum(steps[t-1], 1-seat)
        dense += 0.03 * (dp0 - dp1)                                        # expansion (relative)
    us, opp = material(steps[-1], seat), material(steps[-1], 1-seat)
    won = us > opp
    frac = us / max(us + opp, 1.0)
    term = (1.0 + 0.5 * frac) if won else -1.0
    return dense + term, won


# ---------- synthetic adversarial trajectories (pure formula, no env) ----------
def synth_dense(initial, events, seat, total):
    """initial: {pid: (owner, prod)} at t=0 (pre-existing ownership, no dense credit).
    events: list of (t, pid, new_owner, prod) transitions. Returns seat dense."""
    owner, cap_t, dense = {}, {}, 0.0
    for pid, (own, prod) in initial.items():
        owner[pid], cap_t[pid] = own, 0
        if own == seat:
            dense += C * decay(0) * prod / total
    for (t, pid, new, prod) in events:
        prev = owner[pid]
        if prev == seat:
            dense -= C * decay(cap_t[pid]) * prod / total
        if new == seat:
            dense += C * decay(t) * prod / total
        owner[pid], cap_t[pid] = new, t
    return dense


def synthetic_checks():
    print("\n=== SYNTHETIC adversarial checks (C=0.5, total_prod=80) ===")
    T = 80.0
    ok = True
    # capture-then-lose (anchored) = 0
    d = synth_dense({1: (-1, 4)}, [(2, 1, 0, 4), (400, 1, 1, 4)], 0, T)
    p = abs(d) < 1e-9; ok &= p
    print(f"  capture@2 then lose@400 (anchored)   dense={d:+.4f}  -> {'PASS (=0)' if p else 'FAIL'}")
    # tennis: capture/lose/recapture/lose = 0
    d = synth_dense({1: (-1, 4)}, [(2,1,0,4),(5,1,1,4),(8,1,0,4),(11,1,1,4)], 0, T)
    p = abs(d) < 1e-9; ok &= p
    print(f"  tennis cap/lose/cap/lose             dense={d:+.4f}  -> {'PASS (<=0)' if p else 'FAIL'}")
    # hoard: own home from t0, never act -> no per-step accrual and no initial credit
    d = synth_dense({12: (0, 4)}, [], 0, T)
    p = abs(d) < 1e-9; ok &= p
    print(f"  hoard (hold home, no action)         dense={d:+.4f}  -> {'PASS (=0)' if p else 'FAIL'}")
    # initial home loss: no prior gain, but the negative delta is anchored at t=0
    d = synth_dense({12: (0, 4)}, [(20, 12, 1, 4)], 0, T)
    p = abs(d + C*decay(0)*4/T) < 1e-9; ok &= p
    print(f"  lose initial home@20                 dense={d:+.4f}  -> {'PASS (<0, anchored t0)' if p else 'FAIL'}")
    # hold-to-end: capture@2 prod5, never lose -> in (0, 1.1C]
    d = synth_dense({5: (-1, 5)}, [(2, 5, 0, 5)], 0, T)
    p = 0 < d <= 1.1 * C + 1e-9; ok &= p
    print(f"  capture@2 hold to end (prod5)        dense={d:+.4f}  -> {'PASS (0<d<=0.55)' if p else 'FAIL'}")
    # win-small vs lose-big: win holding 2 planets vs lose holding 10
    win_small = synth_dense({1:(-1,4),2:(-1,4)}, [(2,1,0,4),(2,2,0,4)], 0, T) + 1.0
    lose_big = synth_dense({i:(-1,4) for i in range(10)}, [(2,i,0,4) for i in range(10)], 0, T) - 1.0
    p = win_small > lose_big; ok &= p
    print(f"  win-small({win_small:+.3f}) > lose-big({lose_big:+.3f})       -> {'PASS' if p else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", default="opponents/candidate_ajay_1200.py")
    ap.add_argument("--games", type=int, default=24)
    args = ap.parse_args()

    agent = load_faithful_agent(args.checkpoint)
    print(f"\n=== REAL replays: NEW reward across {args.games} games (seat0 vs Ajay) ===")
    print(f" {'seed':>5} {'res':>4} | {'NEW dense':>9} {'term':>5} {'NEW tot':>8} | {'OLD tot':>8} {'old<0?':>7}")
    wins, losses, old_loss_pos = [], [], 0
    for s in range(args.games):
        env = make("orbit_wars", configuration={"seed": s}, debug=False)
        env.run([agent, args.opponent])
        steps = env.steps
        dense, term, tot, won = new_reward(steps, 0)
        old_tot, old_won = old_reward(steps, 0)
        (wins if won else losses).append(tot)
        if not won and old_tot > 0:
            old_loss_pos += 1
        print(f" {s:>5} {'WIN' if won else 'LOSS':>4} | {dense:>+9.3f} {term:>+5.0f} {tot:>+8.3f} | "
              f"{old_tot:>+8.3f} {'POS!' if (not won and old_tot>0) else '':>7}")

    print("\n=== INVARIANT CHECKS (real data) ===")
    i1 = all(t < 0 for t in losses) if losses else True
    i2 = all(t > 0 for t in wins) if wins else True
    i3 = (max(losses) < min(wins)) if (losses and wins) else True
    print(f"  I1 every LOSS < 0 :  {'PASS' if i1 else 'FAIL'}  (n_loss={len(losses)}, worst {max(losses) if losses else float('nan'):+.3f})")
    print(f"  I2 every WIN  > 0 :  {'PASS' if i2 else 'FAIL'}  (n_win={len(wins)}, worst {min(wins) if wins else float('nan'):+.3f})")
    print(f"  I3 no inversion   :  {'PASS' if i3 else 'FAIL'}  (max loss {max(losses) if losses else float('nan'):+.3f} < min win {min(wins) if wins else float('nan'):+.3f})")
    print(f"  OLD reward: {old_loss_pos} LOST games scored POSITIVE (the decay bug; NEW should be 0)")

    syn_ok = synthetic_checks()
    print(f"\n=== OVERALL: {'ALL PASS' if (i1 and i2 and i3 and syn_ok) else 'FAILURES PRESENT'} ===")


if __name__ == "__main__":
    main()
