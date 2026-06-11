# Orbit Wars — Training & Eval Metrics Reference

Which numbers mean what, what's normal, and **which to trust**. Two overhauls baked in:
- **Rev54 false-alarm** — we mis-called a collapse on `Vμ` + `avgfleet` + `srcs_multi`, all non-signals.
- **2026-06-11 conversion/hoard overhaul** — replaced the end-skewed `avgfleet`/`p90` with
  controlled-time **milestone** metrics, and added **conversion** metrics (eval + training) with a
  top-player reference. See "Conversion & hoard" below.

---

## TL;DR — trust order

1. **Held-out win-rate** (Ajay / 1166 / 1300 / zach panels) — the only signal that sees *absolute*
   regression (our exploitability proxy, the real arbiter). Everything else is a leading indicator.
2. **Conversion & hoard** (eval, vs the top-player reference) — *why* we win/lose: are we capturing
   efficiently, expanding, and deploying the army, or churning/hoarding? Printed on every eval.
3. **Behavioral degeneracy** (unconfounded):
   - `fire_rate → 0` — passive fire=0 Nash. (But note: a *low* fire rate is normal — see fire note.)
   - `fire_frac → 1.0` — carpet-bomb (firing from every owned planet).
   - `ship0 → high` — 1-ship-probe collapse.
4. **EV, entropy, KL/clip** — *training health* (is the optimiser sane), **not** policy quality.
5. **Vμ, rewμ** — weak corroborators. **Ignore the sign.** Never kill a run on these alone.

**Why self-play needs held-out eval:** internal metrics are blind to *mutual* collapse — if both
policy copies degrade together, value / KL / win-rate-vs-self look fine while the agent gets globally
worse. The principled measure is **exploitability** (how badly a best-response beats you), approximated
by win-rate vs a fixed held-out panel. PPO logs *explained variance* of the value head, never the
*value mean*, for exactly this reason. Corollary: the conversion/hoard metrics in the **training diag**
are also self-play-confounded — read them as behavioral colour; the **eval** conversion numbers are the
trusted ones.

---

## The `iter` line (every iteration)

| Field | Meaning | Normal | Signal |
|---|---|---|---|
| `SPS` | env steps/sec | L4 ~600 (externals cap it; heuristic-worker pool ~250–300) | ops only |
| `EV` | explained variance of value head | ~0.85–0.95 | **EV dropping = critic collapse** (trust) |
| `KL` | approx KL old/new policy | <0.02 healthy; target 0.05 | >0.05 sustained = unstable → halve LR |
| `clip` | clip_frac; `fire:` = per-slot fire | **≤0.25 healthy** | **HARD 0.25 → halve LR** (>0.28 degrades; don't wait for 0.32). Higher entropy 0.02→0.05 pushes clip ~0.20→~0.30 — pair with lower LR. |
| `H_fire` | fire-head entropy | ~0.1–0.25 (entropy-coef 0.05); recovers after warmup | <0.07 = deterministic fire collapse |
| `V_loss` | value loss | <1 settled; spikes on resume | explosion = critic divergence |
| `r_p0 / r_p1` | mean reward seat 0 / 1 | seat-asymmetric, oscillates | NOT a quality signal (zero-sum + shaping) |
| `LR` / `estop` | learning rate / KL early-stop | per schedule / 0 | ops; frequent estop=1 = updates too big |

## The `diag` line (every 10 iters — `train_torch.py`)

| Field | Meaning | Normal | Signal |
|---|---|---|---|
| `fire[0]` | slot-0 fire *probability* (sigmoid) | ~0.06–0.19 (top-player-normal; see fire note) | **slot-0-ONLY** firing (fire[0]≫rest_max) = degenerate. A low fire[0] alone is fine. |
| `rest_max` | max fire prob among non-0 slots | tracks fire[0] | context for fire[0]; fire[0]≈rest_max = healthy spread |
| `fire_frac` | **on firing steps**, fraction of owned planets that fire | ~0.3–0.5 (top players ~0.17) | **→1.0 = carpet-bomb**; well above 0.5 = scattershot |
| `owned` | mean planets owned | ~6–10 | low = under-expanding |
| `ship0` | fraction of fires choosing bin 0 (=1 ship) | ~0 (with `--min-ship-bin`) | **high = 1-ship-probe collapse** (trust) |
| `meanshipbin` | mean ship-size bin when firing | ~15–20 | low = undersized launches (undercommitment) |
| `pl@16/50/100` | planets owned at episode-step 16/50/100 (player 0) | ramp toward ~6→9→10 | flat/low = expansion plateau (the deploy-and-hold gap) |
| `garrfrac@50` | garrison fraction at step 50 = parked / (parked+in-flight) | ~0.5 (top players) | **high (>0.65) = hoarding** (army parked, not deployed) |
| `shipspp@50` | ships per owned planet at step 50 (parked) | ~22 (Isaiah) | high (≫ref) = piling ships instead of spending |
| `H_ship` | ship-head entropy | ~3.4 | low = ship collapse |
| `Vμ` | mean of V(s) | swings; rev38 mean +0.32 | **NOT a collapse signal — ignore sign** |
| `Rμ / Rσ / Aσ` | mean/std returns, adv std | Rμ~0 (zero-sum) | Rμ→0 = equilibrium not collapse; Aσ→0 = no learning signal |
| `rewμ / rewNZ` | mean per-step reward / fraction nonzero | shaping-dependent | weak |
| `featσ p/f/g/pw` | feature std | ~0.3–0.45 | →0 = representation collapse |
| `reinf / tgt n/e` | reinforce rate / target-share neutral/enemy (if `--allow-reinforce`) | ramps with empire size | flood = high reinf **+ ballooning hoard** (read with garrfrac/shipspp) |

`pl@`, `garrfrac@`, `shipspp@` are player-0 snapshots taken **at the exact episode step** (16/32/50/100),
accumulated across the rollout — controlled-time, so not skewed by episode length or end-state. Full set
(`garr_frac@`, `ships_per_planet@`, `planets@` for all four milestones) goes to W&B under `hoard/*`.

## The `CKPT_METRICS` line (at each checkpoint — parsed by track.py)

`step EV KL clip fire_frac owned garrfrac@50 shipspp@50 fire_rate Hfire reinf`
(`fire_rate` = overall fraction of owned-planet-slots firing, unconditional; vs `fire_frac` which
conditions on firing steps.)

---

## Conversion & hoard metrics (the 2026-06-11 overhaul)

Printed on **every eval** (`eval.py`, both `--panel` and baseline) and computed identically from the
top-player replays, so eval is directly comparable to the human reference. The whole-game read that
win-rate alone can't give you: *are we capturing efficiently, expanding, and deploying — or churning
and hoarding?* Implementation: `eval.py:game_conversion()`; training side mirrors it for player 0.

```
Conversion: caps/game X  atk-launch/game X  cap/atk-launch X  ships/cap X  reinf_share X
  planets@16/32/50/100 a/b/c/d  end X   churn X (X/100st, len X)
  retention  lost-cap X (lost/total caps)  median-hold Xst
  launch-waste<50  redundant X (WG X)  underkill X (WG X)
  hoard  garr_frac@ a/b/c/d  ships/planet@ a/b/c/d
  reinf by empire size  1:r(n)  2-3:r(n)  4-6:r(n)  7-9:r(n)  10-12:r(n)  13+:r(n)   [ref ramp @1:0.00 @2:0.10 @9-12:0.30 @13+:0.34-0.61]
```
Printed on **every** eval (`--panel` and baseline). `churn`, `launch-waste<50 (redundant/underkill)` are
defined below; all compute identically from top-player replays via `conversion_from_replays.py`.

**Definitions** (a "launch" = a legal fire, `sent ≤ source ships`):
- **capture** = a planet whose owner transitions **to us**. Counts re-captures, so it's gross, not net.
- **attack-launch** = a launch whose aimed target is **not owned by us**. *Reinforce launches (target
  owned by us) CANNOT capture and are excluded from the cap/launch denominator* — counting them deflates
  the ratio (was the original bug). Launches whose target can't be resolved by angle are skipped
  (matches `fetch_analyze_top_replays._resolve_target`, so eval == replay analysis).
- **cap/atk-launch** = captures ÷ attack-launches — per-attack conversion *efficiency* (the skill axis).
- **ships/cap** = attack-ships ÷ captures — *force per capture*. ⚠️ **Deflated by churn**: re-captures
  of flip-flopping planets are cheap, so a low value can mean efficiency **or** churn. Always read with
  caps/game vs `end`: caps/game ≫ end_planets = churn.
- **churn** = gross captures ÷ end_planets (capture-then-lose-then-recapture the same flip-flopping
  planets — the "can't hold the lead" signal). ⚠️ **LENGTH-CONFOUNDED** like the removed `avgfleet`: more
  steps → more gross re-captures, so the longest games read highest *regardless of holding skill*. Top-2
  raw churn is **Isaiah 7.1 > Jake 3.5 purely because Isaiah's games run 447 vs 284 steps.** Read the
  printed **`churn/100st`** (caps/end normalized per 100 steps) instead — it collapses to a tight elite
  band: **Isaiah 1.59 · TonyK 1.16 · Jake 1.23 · 213tubo 1.48.** Even normalized it's a *secondary* read;
  the clean hold signal is the **planets@N trajectory turning over** (peak then decline) + the
  proximity crossover step (`deb_proximity.py`).
- **retention** (`retention  lost-cap X (lost/total caps)  median-hold Xst`, added 2026-06-11) — the
  **denominator-free** hold signal, built to replace churn's degeneracy. `lost-cap` = of the planets we
  CAPTURE, the fraction we then lose (`lost_caps ÷ captures`); `median-hold` = median steps from capture to
  loss over LOST episodes (held-to-end is censored, not counted). **Why it beats churn:** normalized by
  *captures* (≈stable ~14/game), not `end_planets` → it does NOT inflate as `end→0` on elimination. churn
  rising 16→26 across p2rev3 500k→2.6M was almost entirely `end` falling 0.9→0.5 (caps flat) — a denominator
  artifact, NOT worsening turnover; lost-cap/median-hold measure the turnover directly. Read: lost-cap→1 +
  short hold = capture-and-lose ("can't hold the lead"); lost-cap low + hold≈game-length = sticky. Home/
  initial planets excluded by construction (only planets that entered `cap_step` count). Diagnostic of the
  Phase-2 retention gap; expect high lost-cap/short hold vs a strong planner (deb), low/long vs weak (Zach).
- **launch-waste<50** (printed `launch-waste<50  redundant X (WG X)  underkill X (WG X)`) — the OPENING
  (step <50) launch-discipline pair, both keyed to `cap_cost_at_arrival` (the SAME quantity the roi-deflation
  uses, replicated in `eval.py _cap_cost_at_arrival`/`_eta` from `_ETA_PROBE_SPEED`). Windowed to the opening
  because a whole-game fraction is inflated by *benign end-game surplus re-fire* in long won games (a
  phase-composition confound, not a volume one — length-normalizing does NOT fix it); the `(WG x)`
  whole-game value is kept for context.
  - **redundant (OVERKILL)** = attack-launch at a target ALREADY covered to capture by own fleets inbound
    *before* the launch (`friendly_inbound ≥ cap_cost_at_arrival`) ÷ opening attack-launches → pure surplus,
    exactly what the deflation zeroes. ⚠️ **Top-player floor is ~0.02** (Isaiah 0.02 · TonyK 0.03 · Jake 0.02
    · 213tubo 0.02) — top players essentially never pile on. (An earlier current-ships threshold read ~0.12,
    but that was almost all *enemy multi-wave* the deflation rightly leaves alone — the aligned metric is the
    true one.) So our agent reading ≫0.02 = real over-fire the deflation should cut.
  - **underkill (INEFFECTIVE)** = FORWARD-looking: the target never becomes ours within ~eta+10 steps of the
    launch (id-keyed lookahead, no slot-reorder issue) ÷ opening attack-launches → ships that didn't lead to
    a capture (the seed1030 18-at-23 lone-undercommit case). A per-launch `sent+inbound<cost` threshold is
    WRONG here — it mis-flags legit multi-wave (each wave < cost) at ~0.86; forward-looking fixes that
    (a target a later wave captures reads effective for all waves). Top-player ref **~0.43** (Isaiah 0.40 ·
    Jake 0.43 · TonyK 0.50). Note it ≈ opening `1 − cap/atk-launch`, so it's the dominant component of the
    conversion gap: since redundant is tiny (~0.02), a low cap/atk-launch is mostly **undercommit/ineffective
    targeting, not overkill**. Use it to see if the next run tightens *which* opening launches we commit to.
  (Same-step double-fires at one fresh target aren't counted for redundant — neither fleet exists at decision
  time — matching the fix's reach.)
- **reinf_share** = reinforce-launches ÷ all launches. ⚠️ **Opponent/success-confounded**: it co-moves
  with empire size (own planets only become legal targets above the gate, and their fraction rises as the
  empire grows — phase2 §6), so the *same policy* reads ~0.08 vs debatreya (eliminated, stuck small/
  below-gate), ~0.23 vs a winnable opponent, ~0.5 in self-play training. Don't compare the aggregate
  across opponents — use the per-empire-size ramp below.
- **reinf by empire size** = own-target share among launches made *at that owned-planet count* (with the
  launch count in parens — low-count bins are noisy). This is the apples-to-apples comparison to the
  top-player ramp (phase2 §2): @1 ≈0.00, @2 ≈0.10, @9-12 ≈0.30, @13+ 0.34-0.61. Empire size is measured
  at decision time (obs@t-1). The aggregate `reinf_share` is empire-mix-weighted; the ramp decouples it.
- **planets@N** = owned planets at episode-step N — the expansion/retention trajectory.
- **garr_frac@N** = parked ÷ (parked + in-flight) at step N — scale-free deployment ratio; high = army
  parked. (Snapshot mid-game; at the terminal step everything lands → 1.0 trivially.)
- **ships/planet@N** = parked ships ÷ owned planets at step N — pile-up per planet.

**Why milestones, not `avgfleet`/`p90`:** an episode *average* (or p90) is dominated by the late-game
large-empire steps, where high ship counts are *normal* — a winner with 12 planets producing every turn
shows a huge average garrison that is *winning*, not hoarding. Fixed-step **snapshots** read the hoard at
controlled, comparable points (immune to game length / end-state), and the **ratios** (`garr_frac`,
`ships/planet`) decouple empire size. The `@100` column intentionally shows the won-game accumulation
(Isaiah garr_frac 0.87) so you can *see* it instead of letting it poison an average.

### Top-player reference (timing-corrected replays: action@t paired with obs@t-1)

Computed over **269 replays** (`/tmp/fresh_validate` 180 + `/tmp/snowball` 89) by running the eval's own
`game_conversion()` over the replays — driver `conversion_from_replays.py <dirs> [--player NAME]`, so the
top-player numbers and the eval numbers come from the *same function* (true eval == replay parity).
`n` = games each player appears in (all outcomes, not just wins); ratios are pooled over all those games.

| player | rank | n | cap/atk-launch | ships/cap | reinf_share | planets@16/32/50/100 | garr_frac@16/32/50/100 | ships/planet@16/32/50/100 |
|---|---|---|---|---|---|---|---|---|
| Isaiah @ Tufa Labs | #1 | 194 | **0.59** | 168 | 0.30 | 2/6/9/10 | 0.50/0.51/0.54/0.87 | 11/15/22/60 |
| Jake Will | #2 | 83 | 0.42 | 140 | 0.43 | 2/5/8/11 | 0.59/0.59/0.60/0.77 | 13/16/24/43 |
| TonyK | — | 83 | 0.53 | 181 | 0.40 | 2/6/9/9 | 0.46/0.46/0.50/0.80 | 10/14/20/63 |
| 213tubo | mid | 46 | 0.07 | 199 | 0.45 | 2/5/8/10 | 0.52/0.51/0.45/0.49 | 11/15/18/21 |

Read it: elite play = **high cap/atk-launch** (Isaiah 0.59 vs carpet-bomber 213tubo 0.07), steady
expansion to ~10 planets, and **~half the army deployed** mid-game (garr_frac ~0.5, ~11–22 ships/planet).
The contested phase (steps 16–50) is the clean window; @100 is endgame accumulation.

### Fire-rate reference (how often top players actually launch)

`launch_rate` = fraction of owned-planet-steps with a legal launch (the empirical analog of `fire[0]`).

| player | launch_rate | active_step_frac | fire_frac |
|---|---|---|---|
| Isaiah (#1) | **0.036** | 0.25 | 0.17 |
| Jake (#2) | 0.081 | 0.43 | 0.17 |
| 213tubo (mid) | 0.41 | 0.77 | 0.47 |

**The top players fire RARELY** (Isaiah 3.6%). A *high* fire rate (213tubo 41%) is a *losing* trait.
So our `fire[0] ~0.09` is top-player-normal, even slightly high — **firing more is the wrong lever**;
raising fire-entropy pushes toward 213tubo's carpet-bombing. The gap is *which* launches and *deploy/hold*,
not *how often*.

### Decode note (eval / export)

Eval and export use **`fire_threshold=0.5`** (the default), NOT `--sample`. On the 256-game zach panel,
threshold-0.5 = **45.3%** vs sample **35.9%** — firing only when confident is more selective and plays
better; sampling fires low-confidence slots and adds noise. (A noisy 4-game run suggested the opposite —
don't trust small-N win-rate; use the full panel.) The reinforce-discipline masks must also match training
at inference: `--reinforce-gate-min-planets 3 --reinforce-forward-only --reinforce-garrison-floor 10`
(see `action_mask.py`), else the policy reinforces where it was masked and self-sabotages.

---

## Deprecated / removed / misleading

- **`avgfleet` / `p90` (planet ship inventories)** — REMOVED from the diag (2026-06-11). Episode-average
  level metrics are **end-step-skewed**: late-game large empires inflate them for *winners* as much as
  hoarders, conflating "winning" with "hoarding". Replaced by milestone `garr_frac@`/`ships/planet@`
  (controlled-time, scale-free). Don't reintroduce a raw fleet-size average.
- **`srcs_multi`** — REMOVED (2026-06-09). Empire-size-confounded, outlier-dominated. Optimising it never
  moved wins. (The `--srcs-multi-penalty` shaping knob is separately deprecated — floor=0 → fire=0 Nash.)
- **Zach panel** — saturated ~88–89% as a *win-rate* metric **for the Phase-1 lineage only** (rev32b 88.7%,
  rev31/38 ~85%). ⚠️ **NOT saturated for Phase 2:** the reinforcement lineage (p2rev1/p2rev2) started from
  snowball-BC and trades general Zach-beating strength for reinforcement-skill acquisition vs the aggressive
  pool, so it sits far below ~88% (p2rev2 @1M was ~37% on the first 16 games). For Phase 2, Zach WR is an
  **informative, non-ceiling signal** — the gap to Phase-1's ~88% measures how much general strength the
  lineage has (re)gained; track it climbing as a real progress signal, don't dismiss it as saturated. The
  **conversion line vs zach** is useful for either lineage (long games expose the hoard). Use zach for
  conversion/hoard diagnostics + (Phase-2) general-strength tracking, not as a Phase-1 WR headroom ceiling.
- **Ajay/1166 panel as the *objective*** — NOT LB-predictive (rev53b 10.9% Ajay → 933 LB < rev38 2.7% →
  994). Guardrail/yardstick, not north star. Only honest LB signal is submitting (`docs/submissions.md`).
- **In-training pool `wr` / `ema_wr`** — NOT a performance metric. Measured inside `torch_env` with
  sampled actions; aiming-heavy heuristics play far weaker there than in the real env (p2rev1 8.25M:
  pool ema 0.48 vs kaggle 0/6% vs lb1152). It's a PFSP sampling signal only. For the comparable number
  use cross-eval. **Full analysis: `docs/train-eval.md`.**

## The honest hierarchy of "is it working"

```
LB submission                 ← ground truth (slow, rate-limited)
  └ held-out panel WR          ← exploitability proxy (Ajay/1166/1300/zach) — trust for regression
      └ conversion & hoard (eval, vs top-player ref) — WHY: cap/atk-launch, planets@, garr_frac@
          └ behavioral degeneracy (fire_rate→0, fire_frac→1, ship0) — concrete failure modes
              └ EV / entropy / KL — optimiser is sane
                  └ Vμ / rewμ — confounded; do NOT decide on these
```
