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

## tl100m — first timeline run (LAUNCHED 2026-07-10, verdict pending)

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
