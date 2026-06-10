# Orbit Wars — Training Metrics Reference

Which numbers in the training log mean what, what's normal, and **which to trust for
detecting collapse**. Written after a Rev54 false-alarm where we mis-called a collapse
on `Vμ` + `avgfleet` + `srcs_multi` — all of which turned out to be non-signals.

---

## TL;DR — trust order for "is the policy collapsing?"

1. **Held-out win-rate** (Ajay / 1166 / 1300 panels) — the only signal that sees *absolute*
   regression. This is our **exploitability** proxy and the real arbiter. Everything else is
   a leading indicator at best.
2. **Behavioral degeneracy** (unconfounded, specific to our action space):
   - `fire_rate → 0` — the fire=0 passive Nash.
   - `fire_frac → 1.0` — carpet-bomb (firing from every owned planet).
   - `ship0 → high` — 1-ship-probe collapse.
3. **EV, entropy, KL/clip** — *training health* (is the optimiser sane), **not** policy quality.
4. **Vμ, rewμ** — weak corroborators at best. **Ignore the sign.** Never kill a run on these alone.

**Why self-play needs held-out eval:** internal metrics are structurally blind to *mutual*
collapse — if both policy copies degrade together, value / KL / win-rate-vs-self all look fine
while the agent gets globally worse. This is why AlphaStar/OpenAI Five track Elo vs a fixed
league; the principled measure is **exploitability** (how badly a best-response beats you),
approximated by win-rate vs a fixed held-out panel. (Literature note: PPO implementations log
*explained variance* of the value head, never the *value mean*, for exactly this reason.)

---

## The `iter` line (every iteration)

| Field | Meaning | Normal | Signal |
|---|---|---|---|
| `SPS` | env steps/sec (throughput) | L4 ~600, H200 ~600–3400 (externals cap it) | ops only |
| `EV` | explained variance of the value head (critic accuracy) | ~0.85–0.95 | **EV dropping = critic collapse** (trust) |
| `KL` | approx KL between old/new policy this update | <0.02 healthy; target 0.05 | >0.05 sustained = unstable updates → halve LR |
| `clip` | clip_frac (fraction of ratios clipped); `fire:` = per-slot fire rate | **≤0.25 healthy** | **HARD THRESHOLD 0.25 → halve LR** (>0.28 actively degrades; do NOT wait for 0.32). Rev54 v2 sat at ~0.3+ most of its length → late checkpoints DEAD (14.68M ~0% vs Ajay, 0.4% vs 1300); only <0.25-clip early checkpoints healthy. Raising entropy 0.02→0.05 pushes clip ~0.20→~0.30 — pair higher entropy with lower LR. |
| `H_fire` | fire-head entropy | ~0.1–0.25 with entropy-coef 0.05 | <0.07 = deterministic fire collapse |
| `V_loss` | value loss | <1 settled; spikes on resume/re-warm | explosion = critic divergence |
| `r_p0 / r_p1` | mean reward of seat 0 / seat 1 this rollout | seat-asymmetric, oscillates | NOT a quality signal (zero-sum + shaping) |
| `LR` | learning rate | per schedule | ops |
| `estop` | KL early-stop triggered (1/0) | 0 | frequent 1 = updates too big |

## The `diag` line (every 10 iters — see `train_torch.py`)

| Field | Meaning | Normal | Signal |
|---|---|---|---|
| `fire[0]` | slot-0 fire probability | ~0.2–0.4 | slot-0-only firing (>0.8 while others 0) = degenerate |
| `rest_max` | max fire prob among non-0 slots | — | context for fire[0] |
| `fire_frac` | **on firing steps**, fraction of owned planets that fire | ?? (no champion baseline) | **→1.0 = carpet-bomb** (trust; but calibrate vs eval) |
| `owned` | mean planets owned (expansion) | ~6–9 | low = under-expanding |
| `ship0` | fraction of fires choosing bin 0 (=1 ship) | ~0 (with min-ship-bin) | **high = 1-ship-probe collapse** (trust) |
| `meanshipbin` | mean ship-size bin chosen when firing | ~15–20 | low = undersized launches (undercommitment) |
| `avgfleet` | **mean ships HOARDED on planets** (passivity proxy) — NOT ships sent | **champion ~108** (rev38 mean 107.8, range 50–147) | only meaningful vs the champion baseline; 110 is normal |
| `p90` | p90 of planet ship inventories | rev38 ~200–240 | as above |
| `H_ship` | ship-head entropy | ~3.4 | low = ship collapse |
| `Vμ` | **mean of V(s)** | rev38 swings −0.48..+1.18, mean +0.32 | **NOT a collapse signal — ignore sign** (see TL;DR) |
| `Rμ / Rσ` | mean / std of returns | ~0 in self-play (zero-sum) | Rμ→0 = equilibrium, NOT collapse |
| `Aσ` | advantage std | — | ~0 = no learning signal |
| `rewμ / rewNZ` | mean per-step reward / fraction nonzero | shaping-dependent | weak |
| `featσ p/f/g/pw` | feature std (planet/fleet/global/pairwise) | ~0.3–0.45 | →0 = representation collapse |
| `wnorm roi20/roi50/ec` | weight norms of the 3 new pairwise features | grows from ~0 if active | 0 = feature never activated |

## The `CKPT_METRICS` line (at each checkpoint — checkpoint-aligned, parsed by track.py)

`step EV KL clip fire_frac owned avgfleet fire_rate Hfire` — the checkpoint-aligned subset.
`fire_rate` = overall fraction of owned-planet-slots firing (unconditional, vs `fire_frac`
which conditions on firing steps).

---

## Deprecated / removed / misleading

- **`srcs_multi`** — REMOVED (2026-06-09). Empire-size-confounded (counts sources firing
  given ≥2 owned → rises with empire size), outlier-dominated. Optimising it (rev5–48) never
  moved wins. Do not reintroduce. (The `--srcs-multi-penalty` *shaping knob* is separate and
  deprecated — floor=0 → fire=0 Nash.)
- **Zach panel** — saturated ~88–89%, retired as a decision metric.
- **Ajay/1166 panel as the *objective*** — NOT LB-predictive (rev53b 10.9% Ajay → 933 LB <
  rev38 2.7% → 994). Use as a *guardrail/yardstick*, not the north star. The only honest LB
  signal is submitting. See `docs/submissions.md`.

## The honest hierarchy of "is it working"

```
LB submission           ← ground truth (but slow, rate-limited)
  └ held-out panel WR    ← exploitability proxy (Ajay/1166/1300) — trust for regression
      └ behavioral degeneracy (fire_rate→0, fire_frac→1, ship0) — concrete failure modes
          └ EV / entropy / KL — optimiser is sane
              └ Vμ / rewμ / avgfleet / srcs_multi — confounded; do NOT decide on these
```
