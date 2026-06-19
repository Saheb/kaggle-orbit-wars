"""Scan the 4 seeds of the worst archetype (med_high_prod__mostly_static__big_rotating),
our agent at seat 0 vs Ajay, to pick the most decisive loss for a single-game deep dive."""
import argparse
from kaggle_environments import make
from onegame_load import load_faithful_agent


def totals(planets, fleets, seat):
    g = sum(float(p[5]) for p in planets if int(p[1]) == seat)
    fl = sum(float(f[6]) for f in (fleets or []) if int(f[1]) == seat)
    return g + fl


def planets_at(steps, t, seat):
    if t >= len(steps):
        t = len(steps) - 1
    ps = steps[t][0].observation.get("planets") or []
    return sum(1 for p in ps if int(p[1]) == seat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", default="opponents/candidate_ajay_1200.py")
    ap.add_argument("--gate", type=int, default=2)
    args = ap.parse_args()

    seeds = [422, 4814, 4963, 7186]
    agent = load_faithful_agent(args.checkpoint)
    print(f"archetype med_high_prod__mostly_static__big_rotating  | our agent seat0 vs {args.opponent}\n")
    print(f"{'seed':>6} {'result':>6} {'len':>5} | {'our->opp planets':>18} | {'mass ratio @25/50/75/100':>26}")
    rows = []
    for s in seeds:
        env = make("orbit_wars", configuration={"seed": s}, debug=False)
        env.run([agent, args.opponent])
        steps = env.steps
        last = steps[-1]
        rew = [x.reward if x.reward is not None else 0.0 for x in last]
        win = rew[0] > rew[1]
        P = last[0].observation.get("planets") or []
        F = last[0].observation.get("fleets") or []
        us_p = sum(1 for p in P if int(p[1]) == 0)
        opp_p = sum(1 for p in P if int(p[1]) == 1)
        us_m = totals(P, F, 0)
        ratios = []
        for tk in (25, 50, 75, 100):
            if tk < len(steps):
                ob = steps[tk][0].observation
                pp, ff = ob.get("planets") or [], ob.get("fleets") or []
                ratios.append(totals(pp, ff, 0) / max(totals(pp, ff, 1), 1.0))
            else:
                ratios.append(float("nan"))
        rstr = "/".join(f"{r:.2f}" for r in ratios)
        print(f"{s:>6} {'WIN' if win else 'LOSS':>6} {len(steps):>5} | "
              f"{us_p:>2}->{opp_p:<2} (mat {us_m:>4.0f}) | {rstr:>26}")
        rows.append((s, win, len(steps), us_m))

    losses = [r for r in rows if not r[1]]
    if losses:
        worst = min(losses, key=lambda r: (r[3], r[2]))  # lowest material, then shortest
        print(f"\nWORST seat0 loss: seed {worst[0]}  (material {worst[3]:.0f}, len {worst[2]})")


if __name__ == "__main__":
    main()
