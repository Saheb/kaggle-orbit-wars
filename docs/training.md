# Training Dynamics & Self-Play Stabilization

Findings on making self-play training **scale to 100M+ steps**, which is the real goal
(throughput is solved — see `docs/perf.md`; the blocker is now *stable* long self-play).

The central constraint: **external opponents don't scale.** They run as CPU worker processes,
single-thread-bound, and tank SPS. So the throughput path to 100M+ steps is **pure self-play**
— and the top writeups confirm this works (Isaiah #1: *"I used pure self-play in order to
maximize throughput,"* trained ~15B steps). The problem is not self-play; it's *naive,
unanchored* self-play, which drifts.

---

## The collapse we observed (noopkl2, 2026-07-09)

Continued noopkl1 (0.5M, **51.6% vs Ajay** at 3.14M steps) with a self-snapshot pool
(`--pool-mode self`, no external opponents), mature LR 1e-4, all perf optimizations. The Ajay
win-rate **collapsed monotonically**:

| checkpoint | Ajay WR |
|---|---|
| noopkl1 (resume, 3.14M) | 51.6% |
| noopkl2 @ 1M | 37.5% |
| noopkl2 @ 2M | 6.25% |
| noopkl2 @ 4M | **0%** |

Internal metrics stayed **healthy the whole time** (EV 0.97, low KL, `estop 0`, positive
self-play rewards) — the policy didn't destabilize, it *re-specialized* to the wrong thing.

### What it is NOT
- **Not "we removed external opponents."** noopkl1 had **no external opponents either** and
  still reached 51.6%. Opponent composition is not the cause.
- **Not simple internal cycling.** Self-play cycles in WR-*vs-the-pool* (RPS among snapshots);
  that's a different axis from WR vs a *fixed external* opponent. A whole self-play cycle can
  drift away from the region that beats Ajay, so the external number falls monotonically while
  the internal dynamics happily oscillate. 0% is deeper than a cycle trough — it's a genuine
  drift into a region Ajay dominates.

### Root causes (revised)
1. **Unanchored self-play drift.** With nothing pulling the policy back toward a known-good
   reference, gradient descent on "beat past selves" wanders. WR-vs-external is non-monotonic,
   so **51.6% @ 3.14M was likely a transient peak**, and continuing rode the down-slope.
2. **Cold optimizer on resume (train_torch bug/limitation).** `--resume` loads model weights
   only; the Adam optimizer is created fresh (moments = 0). A mature policy through a cold Adam
   at LR 1e-4 takes large, undamped first steps → fast kick off the optimum (already 37.5% at
   1M). noopkl1's *own* run never hit this — it kept one warm optimizer throughout.
3. **Homogeneous pool → over-specialization.** Self-snapshots are all one lineage; the learner
   finds a narrow strategy that beats them, which needn't be robust — and Ajay counters it.

---

## The recipe to make pure self-play stable at scale (Isaiah #1)

Isaiah ran pure self-play to billions of steps **because he anchored it**:
1. **Play against the previous BEST checkpoint**, not a grab-bag snapshot pool.
2. **Anchor to the previous best in the loss:** policy-KL + value cross-entropy terms against
   the previous-best checkpoint. This is the missing piece — it *prevents the drift*: the
   policy can't wander far from the best, so external WR stays in a band near the best's
   instead of collapsing.
3. **Promotion gate:** only replace the best when the new policy wins **>70% head-to-head**.
   Turns cycling into monotonic improvement (you never adopt a regression).
4. **Warm optimizer + adequate capacity** (his was 200M; small models over-specialize faster).

**The mechanism:** anchoring (2) converts "collapse to 0%" into "bounded oscillation near the
best," and the promotion gate (3) ratchets the best upward. Neither existed in our run — noopkl1
had no anchor, which is *why it couldn't be safely continued*.

---

## Concrete next steps (to run 100M+ self-play safely)

1. ~~**Warm-optimizer resume**~~ **DONE 2026-07-10**: `--resume` now loads
   `ckpt["optimizer"]` (Adam moments) by default; `--cold-optimizer` restores the old
   fresh-Adam behaviour. LR still comes from this run's schedule (re-synced after load).
   Removes cause #2 outright.
2. **Best-checkpoint anchoring** (structural, the real fix): add policy-KL + value-CE loss
   terms vs a frozen previous-best; add best-promotion gated on >X% h2h. The pool machinery +
   PFSP already exist in `opponent_pool.py`; this adds the anchor + gate.
3. **Track external WR during the run** (the watcher's Ajay panel) — self-play internal metrics
   look healthy through a collapse, so the held-out panel is the only honest progress signal.
4. Then a long run with anchoring on; expect a bounded, ratcheting Ajay WR rather than drift.

---

## Learning rate — how the winners used it vs how we did (2026-07-10)

Every top writeup treats LR as a **planned decay across the whole run**, not a per-run
constant:

| Agent | LR usage |
|---|---|
| SimJeg (~top 5) | from-scratch stages of 3B/3.5B/3.5B steps at **1e-3 → 3e-4 → 1e-4** |
| Jake Will (#2) | **1e-4 start, "adjusted down several times"**; plateaus fixed by lowering LR/entropy; "better annealing schedules would likely result in shorter training runs" |
| kiyotah (~40) | **1e-4 → 1e-5** decay over 412M steps |
| Ender (top 10) | **3e-5 initially, periodically reduced** |

**Our historical usage was neither:** we pinned a flat LR per run (1e-4, or the "mature"
2.5e-5 arrived at by ad-hoc halvings) and never rode a schedule — the cosine infra existed
but every real launch overrode it flat. Worse, every `--resume` silently created a **cold
Adam** (see collapse root-cause #2 above) — so "lower the LR and continue" runs were also
taking undamped first steps.

**Tooling now in place (2026-07-10):**
- **Warm-optimizer resume** (default on): Adam moments carry across stages; `--cold-optimizer`
  opts out.
- **`--lr-offset-steps <ckpt steps>`**: with a full-horizon `--lr-schedule-steps`, a resumed
  stage CONTINUES the cosine mid-decay instead of restarting at peak — the winners' staged
  decay, expressible directly.

**Recipe for the from-scratch timeline run:** peak 3e-4 (config default; SimJeg's 1e-3 rode
a much larger batch), warmup on, `--lr-schedule-steps` = the FULL planned horizon across all
stages (so a 20M first stage of a planned 100M passes `--lr-schedule-steps 100000000`), and
each resume passes `--lr-offset-steps <checkpoint step count>`. Flat-LR overrides are for
diagnostics, not real runs.

---

## tl100m — first timeline run (VERDICT 2026-07-12: CONFIRMED — Ajay 0% → 74.6%)

**The lever under test: projected-future timeline features** (writeup lesson 1, planet token
20→116 — see docs/writeup_lessons.md §1 for the wiring). First run at 100M steps (~10–20×
our previous budgets) and the closest config to the winners' recipe we've ever run.

**Hypothesis:** the raw resolved timeline fixes conversion timing (launching too-early/too-late
— the diagnosed gap vs Ajay/Ender) without reward shaping, because bad launches become visible
to the critic the step they happen. Watch `us_first_cap`, underkill/conversion lines,
`planets@50` vs the pre-timeline lineage; Ajay panel (watcher default) is the verdict metric —
expect ~0% for a long while (from scratch), judge the back-half trend.

**Config (MINIMAL-CLEAN, user decision — lesson 5 "winners ran sparse"):**
- FROM SCRATCH, pure sparse ±1 reward: NO first-strike / early-capture / expansion /
  win-margin, NO sufficient-commit mask.
- Kept: noop-KL 0.3 → 0.10 launch rate (Jake's prior), reinforce legality masks
  (gate 2, reverse-edge cooldown 3, floor 0, no forward-only — matches eval defaults),
  target decode, self-pool 0.5 (snapshot 1M, PFSP min-games 30, max 20).
- LR recipe (above): peak 3e-4, warmup on, cosine over a 120M horizon → ~2.2e-5 at 100M.
- 512 envs × 64 rollout, mb 32, epochs 2, bf16 + compile + compile-features.
  **NO --gpu-storage on L4**: 22 GB OOMs (~7 GB storage @ 512 envs/116-dim + update
  activations + compile workspace = 21.5 GB); CPU storage is the L4 fallback.
- Launch: GCP L4 `orbit-wars-tl100m` us-west1-a (asia-south1-b/c + europe-west4-a/b/c were
  STOCKOUT), ~705 SPS steady (~40 h / ~$45), wandb run `gfiwzpf4`, checkpoints every 1M,
  watchers = controller sync + held-out Ajay. Script: `gpu_run_artifacts/tl100m/start_training.sh`.

**10M readout (256-game Ajay panels):** 0/0/0.4/2.0/5.5/—/9.0/12.5/6.6/**14.5%** at 1..10M —
from-scratch sparse already near corrpack3e's old-era 18% (which had lineage + full shaping);
pre-timeline from-scratch comparables were 0.8–3%. **Watch item — launch discipline:** eval
launch_rate 0.253 (Isaiah ref 0.036), 112 atk-launches/game at 0.21 cap/atk, ship0 11% —
expected consequence of dropping the commit mask (discipline must be LEARNED now); WON games
are far cleaner (fire_frac 0.21, cap/atk 0.58) so the disciplined mode exists. TRIPWIRE: if
launch_rate ~0.25 persists with ship0 >15% by 30–40M → raise noop_kl_coef toward 0.5 (Jake's
early value) / restore commit mask / jump to intent sizing (experiments.md #4).

**100M verdict (2026-07-12):** completed 100,007,936 steps in ~52 h (537 SPS settled).
**Ajay full panel: 74.6% final, best 77.7% @ 96.5M** — previous best was 57.4% (stgpr1 0.5M,
spray-inflated head-to-head; README cross-eval). And tl100m gets there with launch_rate 0.092,
not spray — the WR is not style-inflated. Pure sparse self-play held for the whole run — no
collapse (EV 0.98, KL 0.013, estop 0 throughout; the noopkl2 failure mode never appeared).
**Not plateaued at 100M:** 10M-window Ajay averages ~65% (75–85M) → ~68% (80–90M) → ~73%
(90–100M), still ~+5pp per 10M at the tail. Launch-discipline tripwire never fired: final
diag launch_rate 0.092, ship0 0.12, fire_frac 0.29 — discipline was learned, as hoped.
H_fire drifted to ~0.027 (low but stable), H_ship 3.28, H_tgt 1.55.

### tl100m_s2 — stage-2 continuation (+100M, LAUNCHED 2026-07-12)

**One change: nothing.** Same flags; only the resume + stage-2 LR. Hypothesis: Ajay WR keeps
ratcheting (~+5pp/10M decelerating); held-out panel is the collapse guard as always.
- **Resume from `torch_step_99549184`** (NOT `_final`): interval ckpts carry warm Adam
  moments AND have an adjacent `pool_step_*` file, so the 20-member PFSP pool carries over —
  the `_final` ckpt has optimizer state but NO pool file (pool would rebuild from scratch).
- **LR: stage-2 cosine continuing where stage 1 ended** — peak 2.4e-5 (stage-1 cosine's value
  at 100M), `--lr-schedule-steps 200000000`, NO `--lr-offset-steps` → decays 2.4e-5 → ~1.2e-5
  over the stage, headroom for a stage 3. (A plain re-resume of the stage-1 schedule clamps to
  LR 0 at 120M — 80% of the extension would learn nothing. The re-warm alternative, horizon
  240M + offset 100M, jumps LR back to ~1.9e-4 into a mature policy — rejected given the
  collapse history.)
- Same instance (`orbit-wars-tl100m` us-west1-a, code untouched since stage-1 launch),
  run-name `tl100m_s2`, script `gpu_run_artifacts/tl100m_s2/start_training_s2.sh`,
  watchers restarted for the new run.

---

## Levers tried

### No-op KL bias — anti-spray launch-rate prior (Jake Will, Rank 2) — commit `a6e099c`

**What it does.** Adds a KL term to the PPO loss that pulls the **batch-mean launch rate**
toward a low prior (`noop_target_launch_rate = 0.10`). Goal: stop the policy from
carpet-bombing / firing 1-ship probes from every planet — save ships, fire decisively.

**The mechanism (why mean, not per-sample).** Compute `p_bar` = mean fire probability over
valid owned slots (with grad), then `KL(Bern(p_bar) ‖ Bern(0.10))` pulls `p_bar` → 0.10.
Anchoring the **batch mean** (not each decision) lets individual turns fire at 100% when
correct, as long as the average stays ~10% — kills spray **without killing decisiveness** (a
per-sample penalty would suppress good launches too). Adds to, not replaces, the fire-entropy
bonus. Config: `noop_kl_coef` (0 = off), `noop_target_launch_rate` (in `PPOConfig`).

**Result (noopkl1 run):**
- **Mechanism verified:** mean launch rate dropped **0.26 → 0.10**, no degeneracy (didn't
  collapse to passivity), "winner-level fire discipline." Does exactly what Jake described.
- noopkl1 reached **51.6% vs Ajay** (best-Ajay checkpoint, 3.14M steps — the one we resume).
- **Improvement is CONFOUNDED, not a clean win:** ~+11pp vs presres1 (~40% current-eval) mixes
  in +3.5M extra steps *and* the Ajay metric we'd flagged as not fully trustworthy. Not a
  same-step A/B.

**Status: banked as a verified MECHANISM, not a proven improvement.** To settle it: same-step
A/B (`noop_kl_coef` on vs off, identical seed/steps, held-out Ajay) — and run it *with* the
self-play anchoring fix above, so the comparison isn't riding a drift.

## Bugs fixed along the way (2026-07-09)

- **Compiled-model checkpoints carried the `torch.compile` `_orig_mod.` key prefix** →
  unloadable by eval/export/resume (all use uncompiled models). Affected *any* `--compile`
  run. Fixed both ends: `ppo.state_dict` now unwraps `_orig_mod` before saving (future
  checkpoints canonical); `eval.py` strips the prefix on load (handles existing ones).

## ship-KL probe verdict + conversion diagnosis + #4 hypothesis (2026-07-13)

**Ship-size KL-to-prior (Ender lever, experiments.md-adjacent to #2/#4):** replace the
uniform-seeking ship-entropy bonus with a KL toward a full-send-biased prior
(`--ship-kl-coef 0.01 --ship-kl-prior-exp 1.0 --entropy-coef-ships 0`). Resumed tl100m_s2
@36.5M, warm Adam.

**Verdict — small, real, FRONT-LOADED, plateaued (not a breakthrough):**
- Behavior changed immediately: `ship0` 0.13→0.01, `mean_ship_bin` ~16→23 (training diag).
- **Ajay 76.7% → ~80%** (8×256 held-out, ~3σ vs the 33-ckpt flat baseline; `mid_capatk_WON`
  0.55→0.61-0.64). Clean but modest +3pp.
- **yijie ~6%** (8×256, dead flat 1M→8M). Up from a 1.6% point estimate, but that baseline was
  only 64 games (CI overlaps) — treat the yijie gain as suggestive, the FLATNESS as robust.
- Kept in the recipe (net positive, removes the 1-ship pathology), but does NOT close the gap.

**Why so small — the step-by-step conversion diagnosis (all data-backed vs yijie):**
1. **Losses are economic, not a holding failure.** Force-share is **0.44 at the first
   captured-planet loss** (5/5 losing games are "force-first": the material deficit precedes
   planet loss). Holding is downstream — only ~28% of retakes had reinforcing force nearby,
   because we're globally out-massed. So it's NOT wrong-target and NOT can't-decide-what-to-save.
2. **The force deficit comes from non-converting attacks.** `cap/atk` ~0.13; we throw ~most
   attack-launches at planets we don't take, bleeding ships.
3. **The non-converting attacks are UNDER-COMMITTED, not mis-timed.** Of attacks vs yijie:
   **83% send fewer ships than the target's defense AT LAUNCH** (89% of wasted ships); only
   **1% out-raced** by arrival-time reinforcement. Timing (#9) is ruled out; sizing (#4) confirmed.
4. **Root cause:** the ship head sizes relative to the SOURCE ("how much of my garrison"), not
   the TARGET ("enough to beat its defense"). It is target-CONDITIONED (gathers ship logits at
   the chosen target, weights `cap_gap` ch11) but NOT target-RELATIVE — it emits an absolute
   count it must LEARN to calibrate, and symmetric self-play never punishes under-commitment, so
   it never learns to. The ship-KL made source-commits fuller (small help) but can't fix a
   source-relative reference point.

**Next: #4 Intent ship sizing.** Hypothesis: a `capture / capture-defend / maintain` intent that
RESOLVES to an exact ship count via timeline.py's defense math (+ resolved-size table as
features) makes commitment target-relative BY CONSTRUCTION — removing the under-commit degree of
freedom rather than hoping self-play rewards good calibration (same principle as N<10th all-in).
Predicted chain: under-commit↓ → conversion (cap/atk)↑ → force bleed↓ → out-massed↓ → yijie WR↑.
Measure on the yijie held-out panel (Ajay is saturated/blind). Verdict to follow.

## Binary NOOP/COMMIT experiment (2026-07-14)

**Hypothesis:** the four intent choices still alias when a source cannot afford them, so the
policy can repeatedly drain one-ship sources. Collapse the effective action to the winners'
hard-commit pattern: the fire head chooses NOOP/COMMIT, the target head chooses only a feasible
target, and the resolver deterministically sends all available ships to a non-owned target or
the projected maintain amount to an owned target. Commits below five ships and attacks that a
single source cannot afford are masked before sampling. The legacy ship head remains in the
checkpoint only for weight-shape compatibility and receives no policy loss.

**Decoder counterfactual on the 30.1M intent checkpoint (64 fixed-panel games each):**
- Yijie: 2/64 (3.1%, baseline full panel 2.7%); attack launches 108.8→24.3/game and cap/attack
  0.162→0.798.
- Ajay: 33/64 (51.6%, baseline full panel 52.3%); attack launches 97.4→42.3/game and cap/attack
  0.351→0.894.

The immediate WR is flat because the untrained target/fire policy proposes many unaffordable
actions, but the conversion mechanism changes by ~2.5-5× without retraining. The first launch
resumed the intent checkpoint, but that was rejected as a confounded test; the decisive run below
started the binary policy and optimizer from scratch. **Primary verdict is Yijie**, with Ajay as the
regression guard. Tripwires: one-ship launches must remain exactly zero; actionable-source rate,
NOOP rate, mean commit ships, attack share, launch rate, and cap/attack must remain non-degenerate.

### From-scratch result — mechanism PASS, strategic hypothesis FAIL

Run `binary100m_scratch_rtxpro6000_20260714_073829`: 1,280 envs × 64 steps,
32 minibatches, PPO epochs 2, bf16 + compiled model/features + GPU storage, self-play pool 0.5,
no-op KL 0.3. It was stopped after the 50,790,400 checkpoint; the model and pool checkpoint were
checksum-verified locally and the Jarvis instance was destroyed.

| checkpoint | Yijie WR (256g) | Ajay WR (256g) |
|---:|---:|---:|
| 5.08M | 0.0% | 1.2% |
| 10.16M | 0.0% | 9.0% |
| 15.24M | 0.8% | 18.0% |
| 20.32M | 1.6% | 27.7% |
| 25.40M | 0.8% | 35.5% |
| 30.47M | **2.3%** | 47.7% |
| 35.55M | 1.2% | 48.8% |
| 40.63M | 1.2% | 43.4% |
| 45.71M | 2.0% | 48.4% |
| 50.79M | 1.2% | **49.2%** |

**Mechanism PASS:** one-ship launches stayed exactly zero; actionable-source rate stayed
0.72-0.87, NOOP 0.88-0.92, mean resolved commit ~44→90 ships, and neither head collapsed
(`H_fire` ~0.13-0.24, `H_tgt` ~1.2-1.7). The final Yijie panel confirms that the resolver
removed under-committed spray and brought local launch discipline close to winner references:

| metric | ship-KL reference vs Yijie | binary 50.79M vs Yijie | winner behavioural reference |
|---|---:|---:|---:|
| capture / attack launch | 0.204 | **0.665** | Jake 0.710 |
| attack launches / capture | 4.89 | **1.50** | Jake 1.41 |
| opening capture / attack launch | 0.512 | **0.668** | Jake 0.700 |
| launch rate | 0.232 | **0.054** | Isaiah 0.036; Jake 0.081 |
| ships / capture | **82** | 92 | Jake 83 |
| reinforce share | 0.35 | **0.21** | Jake 0.56 |
| planets at step 50 | 8 | **6** | Jake 8 |
| capture peel rate | 0.94 | **0.98** | Ender 0.41 |

The winner columns are behavioural references from different replay/eval populations, not a
controlled opponent-matched A/B. They establish scale, not causal equivalence. In the
opponent-matched Yijie comparison, capture/launch improved 3.26× and launches/capture fell 69%; the
binary result is within 6% of Jake on capture/launch. Its won-game fire fraction was also 0.20,
close to the 0.17 winner reference, so the result is not explained by firing every source.

**Strategic hypothesis FAIL:** Yijie was flat at 0-2.3% for ten full panels and ended the
trajectory at 1.2%, below the ship-KL continuation's stable ~5-7%. Better conversion did not
produce better empire economics: the final binary policy made only 11.7 captures/game
(ship-KL 18.8), reached only 6 planets at step 50 (8), reinforced less (0.21 vs 0.35), lost
98% of captures (94%), and ended 99% of losses at zero material. Ajay ended at 49.2%, so the
policy learned and the guardrail did not collapse; the primary held-out opponent did not improve.

**Evidence-bounded conclusion:** launch sizing/under-commit is no longer the demonstrated
bottleneck. The remaining evidence points to target allocation, reinforcement, post-capture
retention, and expansion. This run does **not** isolate all-in as their cause: it jointly changed
the action space, masks, ship-head loss, initialization, and training lineage. All-in remains a
plausible source-drain mechanism, but replacing it should first be justified by either (a) a
same-checkpoint, same-seed decoder counterfactual measuring source survival and capture retention,
or (b) a matched from-scratch binary-all-in versus binary-sufficient/holdable training A/B.

### Exact-marginal binary PPO — matched follow-up (experiment contract, 2026-07-14)

**Single delta:** make the binary policy optimize the action the environment actually executes.
The model remains target-first and target-conditioned, but rollout sampling, PPO likelihood,
entropy, no-op KL, launch diagnostics, and deterministic eval now use the collapsed distribution

`P(COMMIT(t)) = P(t) P(COMMIT | t)` and
`P(NOOP) = sum_t P(t) P(NOOP | t)`.

Previously, PPO included `log P(t)` only for COMMIT and treated NOOP as just
`log P(NOOP | sampled t)`. That assigns target credit on one branch while conditioning the other
on a latent target sample, so the likelihood is not the probability of the executed action. The
corrected sampler draws once from `{NOOP, COMMIT(t_1), ..., COMMIT(t_k)}` and PPO recomputes that
exact likelihood.

Everything else matches `binary100m_scratch_rtxpro6000`: fresh initialization and optimizer,
100M schedule, seed, 1,280 envs x 64, 32 minibatches, two PPO epochs, binary resolver and masks,
reward, reinforcement gates, self-play pool, no-op KL 0.3, and 5M checkpoint cadence. Primary
verdict remains the 256-game Yijie trajectory; Ajay is the regression guard. The experiment is
worth keeping only if it lifts the prior 0-2.3% Yijie floor without losing the already-verified
conversion discipline. Mechanism gates are rollout/PPO likelihood parity (unit-tested), finite
joint KL/clip, non-collapsed exact action entropy, launch rate, NOOP rate, capture/attack,
reinforcement share, planets at step 50, and capture retention.

Launch order is A100-80GB spot, A100-80GB on-demand, then RTX PRO 6000. Hardware is a throughput
choice rather than an algorithm delta; the training configuration remains matched.

### Replay diagnosis after fixing under-commit — production race and source drain (2026-07-15)

This is the diagnosis that motivated the counterfactual feature experiments below. It does not
contradict the earlier ship-KL finding: that finding isolated under-commit in the old policy. These
replays use the exact-marginal binary policy after all-in resolution had already lifted local
capture/attack conversion. The remaining question was why good local conversion still did not turn
into wins.

**Yijie, 30.081M checkpoint:** compare seed 647 (win, our seat 0) with seed 1843 (loss,
our seat 0). The discriminating state variable was opponent-relative production, not raw captures
or planet count:

| replay / step | planet delta | production delta | material delta |
|---|---:|---:|---:|
| win @32 | +2 | **+6** | +21 |
| win @50 | +2 | **+6** | +64 |
| win @75 | +4 | **+10** | +100 |
| win @100 | +6 | **+16** | +314 |
| loss @32 | -1 | -1 | +9 |
| loss @50 | 0 | 0 | -21 |
| loss @75 | -1 | **-9** | -48 |
| loss @100 | -6 | **-22** | -416 |

The win established a production lead by step 15 and sustained it. The loss never sustained one:
it was approximately even at step 50, then Yijie's production compounded away. This is why
`production_delta` is the north-star passive metric; planet count alone hides planet quality.

The loss also exposed a concrete capital-allocation failure. At steps 46-47 the policy sent
124 + 40 = **164 ships** from two sources toward production-1 neutral planet 14 (garrison 14).
At steps 55-56 it sent 41 + 50 + 18 = **109 ships** from three sources toward production-1
neutral planet 23 (garrison 14). The first wave made later waves already covered in the observed
state. The audit does **not** show that the model lacked a production input: production had a
non-zero learned projection, and the selected planets were cheap/close relative to the more
productive alternatives. It shows that static target value plus per-source decisions did not price
the fleet already committed, the opportunity cost of emptying each source, or the resulting global
production trade. Thus the safe conclusion is coordination/capital misallocation, not simply
"production feature ignored" or "routing/ETA is wrong."

**Ajay, 40.108M checkpoint:** two losses (seed 32, both seats) showed the same economic reversal;
a seed-2078 win retained its production lead. In the seat-0 loss we moved from +4 production and
+23 material at step 50 to -4/-19 at step 75, -6/-31 at step 125, and -20/-316 at step 150. The
win remained +4/+57 at step 50, +8/+97 at step 75, and +5/+340 at step 100. The second seed-32
loss similarly moved from +1 production at step 32 to -5 at step 50 and -10 at step 75.

A target audit of the first loss made the source-side mechanism visible. Five late all-in launches
sent 135, 168, 131, 67, and 160 ships from planets worth 4, 4, 4, 2, and 2 production. All five
destination planets were captured, so these were local tactical successes. Four of the emptied
sources were then lost within 0-12 steps; the fifth was lost later. This does not prove that an
arbitrary smaller fleet would have won, but it rules out "failed capture" as the full explanation:
the policy could win the destination while losing more valuable production capacity behind it.

**Actionable conclusion:** target-side counterfactuals should expose whether the candidate capture
survives and earns production; source-side counterfactuals should expose whether launching loses the
origin and its production. Keep paired production/material deltas at fixed steps as the outcome
read. Do not infer causality from feature-weight norms alone, and do not add a middle commitment
until a matched feature-only arm establishes whether merely exposing this tradeoff is sufficient.

Replay provenance is under
`gpu_run_artifacts/binarymarg100m_l4_from25m/replay_analysis/`: the Yijie seed-647/1843 replay and
analysis JSONs, `yijie_30m_seed1843_bad_target_audit.json`, both Ajay seed-32 losses, the seed-2078
win, and `ajay_40m_seed32_target_audit.json`.

### Candidate-conditioned counterfactual timeline — experiment contract (2026-07-15)

Run `binarycf100m_rtxpro6000_spot` starts from random model and optimizer initialization. The sole
training delta from exact-marginal binary PPO is six appended source-target features computed by
replaying the existing 24-step arrival timeline with that candidate's deterministic commit added:
mine-at-arrival, signed arrival margin, owned fraction after arrival, held-through-horizon,
production delta versus no action, and terminal signed-margin delta versus no action. Existing
in-flight fleets retain the exact timeline combat recurrence; no eval-time action override is used.

**Hypothesis:** the current target head sees static value/pressure and a no-new-launch planet
timeline, but not the consequence of its own candidate action. Direct candidate outcomes should
prefer captures that survive and earn production, improving production advantage and retention
against Yijie without sacrificing the exact-marginal binary launch discipline.

Everything else remains matched: binary NOOP/COMMIT, exact executed-action likelihood, all-in
non-owned commit sizing, maintain sizing for own targets, no-op KL 0.3, sparse reward, self-play
pool 0.5, reinforcement gate 2, reverse-edge cooldown 3, seed/default initialization, 64-step
rollouts, 32 minibatches, two PPO epochs, and 5M checkpoint cadence. Primary decision evidence is
the 256-game Yijie trajectory; Ajay is the regression guard. Mechanism reads are production delta
at 50/100, capture retention, planets at 50, capture/attack, reinforce share, launch/NOOP rate,
candidate-feature weight norms, action entropy, clip fraction, and explained variance.

Pre-launch gates: 27 focused feature/eval/action tests passed; a 64-step fresh CPU rollout completed
PPO and checkpointing; the generated export ran four games without agent errors. The full suite was
123 passed with one pre-existing random-policy symmetry smoke outside its broad threshold after the
input-width change altered random initialization; targeted train/eval feature parity passed.

The initial GCP L4 run (`10bccw3y`) failed during the compiled PPO update: the 384-env job had only
175 MiB free and could not allocate another 180 MiB. The instance was deleted.

**Active launch:** Jarvis RTX PRO 6000 96 GB spot, machine `447053`, managed run `r_a3356010`,
W&B `orrk0m50`. It retains the matched 1,280-env configuration and completed compiled PPO without
OOM. After compilation, iterations 5 and 10 took 16.5s and 17.7s for 81,920 environment steps,
or about 4.6-5.0k steady SPS. Managed checkpoint sync and Ajay eval are attached at 5M-step cadence.

### Source-conditioned counterfactual timeline — matched experiment contract (2026-07-15)

Run `binarycfsrc100m_rtxpro6000_spot` is a fresh, same-seed arm alongside the target-only
counterfactual run above. Its sole training delta is four additional source-side outcomes for
each deterministic binary candidate: source owned fraction over the 24-step horizon,
held-through-horizon, source production delta versus no action, and source terminal signed-margin
delta versus no action. They are computed by deducting the candidate fleet from the source and
replaying the same existing arrivals. The original six target-side outcomes remain unchanged.

**Hypothesis:** Ajay loss replays showed locally successful all-in captures draining production
sources before a later economic reversal. Target-only outcomes price the destination but omit that
source cost. If merely exposing the net tradeoff is sufficient, this arm should improve Ajay
retention/production trajectories and eventually Yijie without changing the binary `NOOP/COMMIT`
action space. Everything else, including seed, optimizer, 100M schedule, rollout/PPO configuration,
pool, decoder, reward, and checkpoint cadence, remains matched.

Primary comparison is target-only versus target+source at matched 5M checkpoints: full Ajay and
Yijie panels, production delta at 50/100, capture retention, attack conversion, planets at 50,
launch/NOOP rate, and functional ablation of the four source channels. Do not add intermediate
sizing in this arm. If the source channels are learned but do not improve outcomes, the next
experiment may add a deterministic middle commitment; combining both changes now would make the
result uninterpretable.

**Active launch:** Jarvis RTX PRO 6000 96 GB spot, machine `447117`, managed run `r_02d2b8ec`,
W&B `yh3lh8dr`. The matched 1,280-env configuration completed compilation and PPO without OOM.
At iteration 10 it processed 81,920 environment steps in 17.3s, or about 4.73k steady SPS, versus
about 4.96k SPS for the target-only arm at iteration 5 (roughly 5% source-projection overhead).
GPU memory was 78.7/97.9 GiB. Managed checkpoint sync plus full Ajay and Yijie panels are attached
at 5M-step cadence.

### Target+source L4 continuation — matched-budget contract (2026-07-16)

Resume the source-conditioned arm from global step 25,395,200 for 30M additional steps on GCP L4
(`binarycfsrc_l4_from25m`), reaching 55,395,200 cumulative. This is a budget-matching experiment,
not a feature change: warm optimizer and self-play pool resume, binary NOOP/COMMIT and all model,
reward, decoder, and PPO settings remain unchanged. Continue the original 100M cosine schedule with
`--lr-offset-steps 25395200`.

**Decision evidence:** at the closest completed checkpoints, target-only 30.474M versus
target+source 25.395M scored 64.1% versus 62.5% against Ajay and both scored 5.1% against Yijie.
The differences are below one panel's sampling resolution. Source conditioning improved loss-depth
at step 100: production deficit −18→−12 and material −340→−260 against Ajay; production −54→−46
and material −1215→−1059 against Yijie. Its source channels are functionally active at 25M, while
capital efficiency remains worse (more ships per capture and lower Ajay conversion). Continuing
source to the target arm's 45M budget tests whether the economic improvement survives matched
learning time and whether binary all-in is the remaining actuator bottleneck.

L4 uses 512 envs with CPU rollout storage: deliberately omit `--gpu-storage`, which caused the
earlier 384-env compiled update OOM. The local stage is 30M steps (about one overnight at the
historical ~700 SPS), with checkpoints every 5M local / cumulative 30.395M through 55.395M.

### Projected-hold decoder calibration — reject as execution contract (2026-07-16)

**Hypothesis:** all-in attacks waste source capital. Execute the smallest fleet explicitly
verified to capture the target and keep it for the existing 24-step counterfactual timeline,
including production, combat, and already in-flight fleets. Reject a middle fleet if its source
falls while the no-launch baseline source stays ours; fall back to all-in when no verified middle
exists. Keep the checkpoint's NOOP probability, target choice, and all-in feasibility unchanged.

The resolver was tested against the same source-conditioned checkpoint
`torch_step_25067520_binarycfsrc_l4_from25m_20260715_185338.pt` on the same 16 canonical Ajay panel
games (panel shard 0/16). The paired result was decisive:

| metric | all-in | projected hold |
|---|---:|---:|
| wins | **13/16** | 1/16 |
| capture / attack launch | **0.742** | 0.181 |
| capture peel rate | **0.488** | 0.898 |
| median production delta at step 50 | **+3.0** | -0.5 |
| median material delta at step 50 | **0** | -32 |

The resolver found a verified result for 1,443/1,483 executed attacks (97.3%), used a strictly
smaller fleet on 1,394/1,483 (94.0%), and sent only 17,344/100,735 ships (17.2% of all-in). That is
the failure mechanism, not noise: a no-new-launch projection verifies that today's known arrivals
cannot peel the capture, but does not price the opponent's response after seeing an under-sized
garrison. The policy then repeats cheap attacks, conversion collapses, and captures peel.

**Verdict: reject.** Keep projected-hold as an eval diagnostic, not a training/eval decoder and
not a deterministic third action. If intermediate capital remains worth testing, the policy must
choose it (NOOP / HOLD / ALL-IN) or its resolver must model a conservative opponent response; the
24-step no-new-launch minimum is not a strategically sufficient hold amount. No PPO run was
launched from this result.

### Submitted-checkpoint cross-eval integrity audit (2026-07-16)

The earlier claim that binary 30.081M beat both final submitted 2p agents 256/256 is invalid. Both
opponent wrappers derived their bundle path from `__file__`; `kaggle_environments` executes path
agents without defining that name. The opponents therefore errored before acting, while the older
eval loop accepted the terminal reward instead of requiring both statuses to be `DONE`. The two
panels consequently produced identical aggregate statistics despite different embedded model
hashes.

The archived tarballs and extracted standalone 2p payloads are byte-identical:

- `presres1` `neural_agent.py`: `981723dc125183c12e99b396900f790b0bfb37e1052ef09883a399691f17631e`
- `stgpr1` `neural_agent.py`: `18146322db85c09032b4f103a2059fe35ca83dcd8001869cafc039ce8d78b41e`

Direct smoke games completed `DONE/DONE` and emitted 41 and 60 non-empty action steps,
respectively. The corrected comparison uses exact-marginal binary 40.108M
(`torch_step_40108032_binarymarg100m_l4_from25m_20260714_163936.pt`, the current Ajay peak at
80.5%) against those exact standalone payloads. Initial independent 16-game canonical samples were
12/16 versus `presres1` and 14/16 versus `stgpr1`, already disproving the perfect-sweep claim.
Full 256-game panels are running; record the final rates here when complete.
