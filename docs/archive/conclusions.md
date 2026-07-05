# Conclusions — closed directions & why

Durable record of branches we explored and *deliberately stopped*, so we don't re-litigate them.

---

## Planet-centric credit assignment (per-slot PPO + VDN) — SHELVED (2026-06-09)

**The idea.** The model is already planet-factored (per-planet action heads). Make the *credit
assignment* planet-factored too: Stage 1 = per-slot policy surrogate (each owned planet clipped
independently); Stage 2 = VDN per-planet value head + per-planet advantages. Hypothesis: better
per-planet credit → better conversion/selectivity than the joint loss.

**What we ran (all: resume rev53b 10M + heuristic-ladder externals + LR 5e-5, one delta apart, evals
local 256-game panels):**

| variant | failure mode | Ajay +1/2/3M | `Vμ` |
|---|---|---|---|
| joint (baseline) | — (stable) | 9.4 / 7.0 / 8.2 | stays **positive** |
| per-slot (global adv) | ship-undercommitment | 6.6 / 7.4 / 5.9 | drifts neg |
| hybrid (per-slot ft + joint ship) | **over-firing** | 8.2 / 5.9 / **4.3** | +0.62 → **−0.35** |
| VDN (per-planet adv) | **hoarding-without-conversion** | 8.0 (+1M) → killed @2M | +0.69 → +0.05 → neg |

**Why we stopped — three reasons:**

1. **It's a structural instability, not a tuning bug.** The joint loss keeps `Vμ` **positive** under the
   *exact same* anchoring/LR. Only the factorized variants drift. The factorization *decouples* the
   planets' decisions, and the only "fix" for decoupling is coupling — which *is* the joint loss.
2. **Patching is whack-a-mole.** Three variants → three *different* degenerate Nash (under-commit →
   over-fire → hoard). Each fix (ship-joint, per-planet value) just revealed the next failure. The
   pattern is "this factorization finds *a* bad Nash regardless of the credit signal," not "one knob away."
3. **No demonstrated edge over joint, even at peak.** At their *best* moments the factorized variants
   were ≈ joint or *worse* (VDN +1M Ajay 8.0% vs joint 9.4%; hybrid's great +1M was 8.2% vs 9.4%). So even
   a perfectly-stabilized per-slot would land back at **joint-level** — a more fragile, more complex method
   that merely *ties* the simple stable one. Negative EV to chase.

**Control that makes this conclusion trustworthy.** A separate run (joint loss + `--reinit-critic`,
isolating the VDN cold-critic confound) stayed **rock-stable** (Ajay 8.2→9.4→9.0). So the collapses are
the **method**, not the resume or the fresh critic. (Cold critic re-warms EV→0.9 in <300K steps, no KL
shock — value-warmup unnecessary.)

**Useful things this produced (kept):**
- `Vμ`/`rewμ` drift is a proven **collapse canary** — visible in training metrics *before* eval panels.
- `srcs_multi` is empire-size-confounded and **removed from the code**; use `fire_fraction` +
  `owned_planets` + `fire_rate` + `avgfleet` + `Vμ`.
- **eval-not-on-training-box** (CPU contention), **resume-is-trustworthy** (control), and the
  **cross-eval cycling detector** + **diversity study** (our opponent set is a transitive ladder, no
  non-transitivity → need exploiters / distilled-Ajay).
- VDN code (model/GAE/ppo, `--vdn-value`, `--reinit-critic`, `test_vdn_gae.py`) is preserved on the
  `claude/strange-khorana-6f6141` branch if ever revisited (e.g. with a firing floor) — but reason #3
  above says don't, absent a new reason to expect an *edge*.

**Where the headroom is instead:** the **joint baseline** + levers that move *it* — shaping anneal,
`rollout32 + ppo_epochs 1`, diversity (exploiters / Ajay distillation). See `docs/next-steps.md`.
