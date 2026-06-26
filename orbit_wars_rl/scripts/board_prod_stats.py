"""Measure initial-board production economics across panel seeds, to size the capture-reward
coefficient. For the unified term r = c·decay(t_cap)·Δ(normalized production), we want the
max cumulative dense (hold the whole board) to stay under the terminal gap.

normalize by TOTAL board production -> each capture = c·decay·(prod_i/total_prod);
holding the whole board = c·decay·1.0, so max episode dense <= 1.1·c (decay_max=1.1).
"""
import statistics as st
from kaggle_environments import make
from orbit_wars_rl.eval_panel import BY_ARCHETYPE


def board(seed):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset()
    P = env.steps[0][0].observation.get("planets") or []
    prods = [float(p[6]) for p in P]
    return len(P), prods


# sample: the worst archetype's seeds + a spread of low/med/high prod archetypes
sample = {}
for arch in ("med_high_prod__mostly_static__big_rotating",
             "high_prod__mostly_static__big_static",
             "low_prod__mostly_static__big_static",
             "med_low_prod__mostly_static__big_static"):
    if arch in BY_ARCHETYPE:
        sample[arch] = BY_ARCHETYPE[arch]

print(f"{'archetype':<46} {'seed':>5} {'#pl':>4} {'totProd':>8} {'avg':>5} {'min':>4} {'max':>4}")
all_tot, all_np, all_prods = [], [], []
for arch, seeds in sample.items():
    for s in seeds:
        n, prods = board(s)
        tot = sum(prods)
        all_tot.append(tot); all_np.append(n); all_prods += prods
        print(f"{arch:<46} {s:>5} {n:>4} {tot:>8.1f} {st.mean(prods):>5.1f} "
              f"{min(prods):>4.1f} {max(prods):>4.1f}")

print("\n--- across sampled boards ---")
print(f"  planets/board: {min(all_np)}–{max(all_np)} (median {int(st.median(all_np))})")
print(f"  total prod/board: {min(all_tot):.0f}–{max(all_tot):.0f} (median {st.median(all_tot):.0f})")
print(f"  per-planet prod: {min(all_prods):.1f}–{max(all_prods):.1f} (median {st.median(all_prods):.1f})")

print("\n--- coefficient (TOTAL-production normalization; max dense <= 1.1·c) ---")
for c in (0.3, 0.5, 0.9):
    print(f"  c={c}:  full-board dense <= {1.1*c:.2f}   "
          f"(loss-with-full-board total <= {1.1*c - 1:+.2f};  win/lose invert if > 2.0)")
