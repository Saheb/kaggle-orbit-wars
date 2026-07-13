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
