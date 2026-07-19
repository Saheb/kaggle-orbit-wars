# Orbit Wars — Agent Operating Instructions

Rules every Claude Code agent must follow in this project. These are not suggestions.

---

## General Coding Behaviour

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## venv

orbit_wars_rl/.venv

## Repo Map

```
orbit_wars_rl/          ← ALL active RL code lives here
  train_torch.py        ← PRIMARY: GPU self-play training (always use this)
  eval.py               ← PRIMARY: full 256-game panel eval
  export_agent.py       ← export checkpoint → submission agent
  model.py              ← entity transformer model (encode_state() + forward())
  torch_env.py          ← vectorised GPU env (physics, blessed shaping, action decode)
  torch_env_fn.py       ← pure-functional env twin (compile/JAX-ready; timeline parity oracle)
  timeline.py           ← projected-future timeline features (planet dim 20→116; import-free,
                           inlined into exports; writeup lesson 1)
  features.py           ← feature extraction (blessed-2026-07 bundle + timeline channels)
  ppo.py                ← PPO learner
  opponent_pool.py      ← self-play pool + PFSP
  config.py             ← ModelConfig / PPOConfig
  action_mask.py        ← action masking for eval (inlined into exports)
  reinforce_cooldown.py ← canonical reverse-edge cooldown rule (live core, baked into subs)
  eval_panel.py         ← stratified community-panel eval (used by eval.py)
  tests/                ← unit tests
  (one-off scripts, bc.py/env.py BC pipeline, producer_* ranking chain: archived in
   archive/cleanup_2026-07/ during C2–C5; runnable at git tag pre-cleanup-2026-07)

opponents/                 ← eval + training opponents
  candidate_hellburner.py
  candidate_zach_public.py   ← primary Zach opponent
  candidate_ajay_1200.py     ← regression GUARD (saturated ~77-80%; NOT the objective)
  candidate_yijie.py         ← ⭐ PRIMARY eval metric (rank 13, 1640 Elo)
  candidate_ender.py         ← top-10 reference (we are 0/256)
  candidate_producer_1200.py
  candidate_suneet_lb1200.py
  orbit_lite/                ← dependency for Ajay/Producer (intercept aiming etc.)

seed_checkpoints/          ← resume points uploaded to training instances
                             ⚠ ALL predate the 2026-07 blessed feature config (11-global and/or
                             no resolver flag) — HEAD's guards refuse them for resume/eval/export.
                             Use git tag `pre-cleanup-2026-07` for anything in here.
  rev32b_6M_resume.pt      ← Rev32b 6M (best Zach: 88.7%, 20/21 loss seeds)
  rev35c_1M_resume.pt      ← Rev35c 1M (best Ajay: 3.1%)
  bc_isaiah_hober_pressure_5k.pt  ← BC warmstart (Isaiah+Hober openings)

setup/                     ← install orbit_wars kaggle env (run once per instance)
docs/                      ← runbooks and logs
  scoreboard.md            ← ⭐ living table: our agent vs Ajay/Yijie/presres1/stgpr1/yusa/Ender
  commands.md              ← ⭐ copy-paste command reference (start here)
  submissions.md           ← full submission log with Kaggle IDs and checkpoint paths
  GCP_RUNBOOK.md           ← GCP L4 launch, monitoring, terminate
  JARVIS_RUNBOOK.md        ← Jarvis H100 spot instances
  perf.md                  ← ⭐ SPS profile: loop is PPO/model-compute-bound, JAX won't hit 10k
gpu_run_artifacts/         ← training scripts, watchers, synced checkpoints (gitignored)
archive/                   ← dead code, old logs (ignore unless archaeology)
  docs/training-till-submission.md ← full run history + reward/mask deltas through first submission
```

---

## GPU Instances

**Primary:** GCP L4 (`g2-standard-8`, ~$1.13/hr) — use `bash gpu_run_artifacts/launch_gpu_gcp.sh`
**Secondary:** Jarvis H100/H200 spot (₹112/hr; ~1,300 fp32 / ~1,600 bf16 SPS at 0.5M-param — the old "~4250" figure was a pre-pairwise model, see docs/perf.md) — use `jl` CLI, see JARVIS_RUNBOOK.md

**Hard rules:**
- GCP: always DELETE (not stop) instances after training — `gcloud compute instances delete`
- Jarvis: always DESTROY (not pause) spot instances — data may not persist on preemption
- **After `launch_gpu_gcp.sh`**: verify sync with `ssh ... "ls ~/orbit_wars_rl/orbit_wars_rl/train_torch.py"` before starting training — rsync can drop mid-transfer
- **Eval on training instances**: always prefix `CUDA_VISIBLE_DEVICES="" python3 orbit_wars_rl/eval.py` — training occupies GPU, eval OOMs otherwise
- Rsync checkpoints: use `-L` flag to follow symlinks (`rsync -azL`)
- **Watchers: ONLY via the controller** `bash gpu_run_artifacts/run_watchers.sh start <run> <platform> <target>` (sync + held-out eval; platform = `jarvis` target=IP / `gcp` target=config-ssh alias / `custom` set `RSYNC_SSH`/`HOST`/`REMOTE_*_DIR` env). NEVER launch ad-hoc per-run `*_watch.sh`/`sync_watcher.sh` — they survive across runs and end up watching the *previous* run's folder. `start` tears down all existing watchers first, and each watcher self-terminates when `.active_run` changes, so stale watchers can't accumulate. `… status` shows the active run; `… stop` kills all. Launch scripts must end with a `run_watchers.sh start` call. To add a SECOND held-out opponent (e.g. Ajay alongside the default zach) WITHOUT churning the live primary watcher, use `run_watchers.sh add-eval <run> <opp.py> [from-latest]` — it launches an extra `_eval` loop under the same run/marker (self-terminating, opponent-specific elog + own `eval_<opp>.csv`); `from-latest` seeds all-but-newest checkpoints as done so a slow panel (Ajay) tracks the frontier instead of backfilling history. Eval masks default to the current design **gate2/floor0/no-forward-only** (2026-06-14: gate3→2, winner-faithful reinforce@2≈0.10; pass `GATE=3`/`REINFORCE_MASKS` to reproduce the old gate3) — override via `REINFORCE_MASKS` if a run trains different masks (eval MUST match training).
- One change per run; record hypothesis before launching

### Key training flags
| Flag | Purpose |
|------|---------|
| `--first-strike-steps 50` | Double capture reward for t<50 (was the LB record fix) |
| `--early-capture-coef 0.3` | Delta-capture shaping (exp decay + 10% floor; blessed runs used 0.3) |
| `--expansion-coef 0.03` / `--win-margin-coeff 0.5` | Blessed-run economy shaping + terminal margin bonus |
| `--staging-shaping-coef 0.2` | PBRS staging toward neutrals (stgpr1 "spray" arm only) |
| `--pool-pfsp-min-games 30` | Prevents PFSP death-spiral |
| `--pool-external-fraction` | Fraction of pool samples to external opponents |

C4 cleanup (2026-07-05): the training loop is pruned to the levers the blessed runs used.
Removed (recover from git tag `pre-cleanup-2026-07`): SSDR, min-ship-bin, decisive-mass +
dm_* diag, scenario curriculum, handicap/self-boost, neutral-garrison scale, prod-share,
consolidation, capture-utility, speed/rank/shaping/defense coefs, eliminate-to-win,
timeout-planet, redundant-target, path-obstruction.

---

## Eval

**Primary metric: Yijie** (`opponents/candidate_yijie.py`) — rank 13, 1640 Elo. The verdict.
**Guard: Ajay** (`opponents/candidate_ajay_1200.py`) — SATURATED ~77–80%; a regression guard, NOT
an objective (needs `orbit_lite/`). **Reference: Ender** (`candidate_ender.py`) — top-10; we are
0/256. **Retired: Zach** — saturated ~88-89%.

⚠ **Both instruments are broken in opposite directions.** Ajay saturates; Yijie floors (0–7%, and
a 256-game panel at 4% has a ±2.5pp band). Nothing reads the 1300–1500 Elo band. So for a graded
signal use paired **production/material delta @50/100** and **loss-depth** (docs/metrics.md), not
the Yijie WR alone. Calibration: an agent genuinely +240 Elo over Ajay still scores only ~10–15%
vs Yijie — single digits are NOT automatically a failure.

**Diagnostic probes** (cheap, run these before proposing a lever — see Key Lesson 13):
```bash
# What does a top-10 agent actually SEND? (ships_sent / source_garrison histogram)
CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/ender_sizing.py --seeds 6            # Ender vs Ajay
CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/ender_sizing.py --seeds 5 \
    --opponent opponents/candidate_ender.py                                        # strong-vs-strong
CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/ender_sizing.py --seeds 6 \
    --agent-checkpoint <ckpt.pt> --opponent opponents/candidate_ender.py           # OURS, like-for-like
# Why do we lose captures? (per-capture retention forensics)
CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/peel_diagnosis.py --seeds 6
# How much of the action space do the hardcoded gates delete?
python gpu_run_artifacts/ender_ref/probe_binary_gate_pressure.py
```

```bash
# Full panel (256 games, ~40 min locally)
CUDA_VISIBLE_DEVICES="" python3 orbit_wars_rl/eval.py \
  --checkpoint <path> \
  --opponent opponents/candidate_ajay_1200.py \
  --panel --target-decode

# Quick eval (16 games, ~2 min) — for trend tracking only
CUDA_VISIBLE_DEVICES="" python3 orbit_wars_rl/eval.py \
  --checkpoint <path> \
  --opponent opponents/candidate_ajay_1200.py \
  --games 16 --target-decode
```

Opponent paths relative to repo root. Always `--target-decode` for Phase 1.

**Always write eval output to a file — never pipe to bare `tail`/in-memory.** `eval.py` writes
no file itself (the Ajay CSVs come from the watcher, not eval.py). For one-off panels:
`PYTHONUNBUFFERED=1 … eval.py … 2>&1 | tee gpu_run_artifacts/<run>/eval_<opp>_<step>.log`
(unbuffered → live progress; tee → survives session death). Slow opponents (Ender ≈ 1h+/panel
on CPU) make a buffered pipe an hour of unrecoverable compute. Opponents tracked repeatedly
belong in the watcher: `run_watchers.sh add-eval <run> <opp.py> [from-latest]` (CSV + wandb).

---

## Current Baselines (full panel, 256 games)

**The competition is OVER — no leaderboard, no submissions.** This is post-competition
learning: apply what the winner writeups teach, one lesson at a time
(`docs/writeup_lessons.md`), measured on held-out panels.

**⚠ AJAY IS NOT THE OBJECTIVE — it saturates ~77–80% and is blind to what beats us.**
Use **Yijie** (rank 13, 1640 Elo) as the verdict metric and Ender as the top-10 reference.
Our champion (binarygates, **98.0% Ajay**) now scores **14.1% vs Yijie** but is still **0/32 vs
Ender** (quick), wiped 100%. Ajay and Yijie both rose sharply once the commit gates were released —
but the top-10 wall is unchanged, so a high Ajay still says little about strong play. See
docs/training.md.

| Checkpoint | Ajay | Yijie | Notes |
|---|---|---|---|
| stgpr1 0.5M (final submission) | 57.4% | — | Pre-timeline best (spray-inflated) |
| tl100m 100M (2026-07-12) | 74.6% | — | ⭐ Timeline features, from-scratch sparse self-play |
| **tl100m_s2 (+35M → ~135M)** | **~77% ±4 (PLATEAU)** | — | **Stage 2 gained ~+1pp/10M — the "still +5pp/10M at the tail" claim was the noise band of a saturating metric. Budget is NOT the free explanation.** |
| shipkl_probe (absolute + soft ship-KL, ~136M cum.) | ~80% | 5.9–7.0% | Prev Yijie bar — **superseded by binarygates100m_l4**. Plateaued ("dead flat 1M→8M") |
| **binarymarg 40.108M** (binary all-in + gates) | **80.5%** | 3.9% | Best Ajay (old) · **0/256 vs Ender**, wiped 100%, peel 0.99 |
| **⭐⭐ binarygates100m_l4 100.008M (2026-07-18)** | **98.0%** | **14.1%** | **NEW CHAMPION — best Ajay AND best Yijie ever.** Arm B: `--binary-commit-gates minimal`, from scratch, 100M. Yijie 2.7→14.1% (2× the shipkl bar, ~4× binarymarg's gated 3.9%); plateaus ~15% over the last 30M. Still **0/32 vs Ender** (quick), wiped 100%. |

**RESOLVED (2026-07-18, Arm B):** the Yijie regression was the **hardcoded commit gates, not the
binary action space**. `--binary-commit-gates minimal` (binarygates100m_l4, 100M from scratch) took
Yijie to **14.1%** — 2× the shipkl bar and ~4× binarymarg's gated 3.9% — while also hitting **98.0%
Ajay** (best ever). Confirms Key Lesson 12: the gates computed a verdict from features the model
already saw, then deleted what it might disagree with. Yijie plateaus ~15% over the last 30M — the
stage-2 200M-horizon resume tests whether more budget breaks the plateau. Ender wall unchanged:
0/32 (quick 32g). Curve: `gpu_run_artifacts/binarygates100m_l4/eval_{ajay_1200,yijie}.csv`.

Competition-era numbers (Rev31 "918.8 LB", Rev32b 88.7% Zach, corrpack3e "18% Ajay") are an
**older eval era and not comparable** to current panels — see README and archive docs for
that history; don't cite them as prior-best.

Note: checkpoints above that predate the blessed feature config (everything before the presres
lineage, incl. corrpack3e) are refused by HEAD's feature-semantics guards — resume/eval/export
them from git tag `pre-cleanup-2026-07`. Of the preserved final artifacts, only
`final_submissions/presres1_0.5M_backfilled_resolver.pt` and `stgpr1_0.5M.pt` load under HEAD.

Timeline features (2026-07-10, planet dim 20→116): NO pre-timeline checkpoint can **resume**
under HEAD (guard + shape mismatch) — training restarts from scratch (the plan anyway). presres1
/ stgpr1 remain **eval/export-able**: eval and export infer the width from `planet_proj` and
feed 20-dim features (`extract_features(timeline=False)`).

---

## Key Lessons

1. **Zach panel is saturated** (~88-89%) — use Ajay as the primary signal. Ajay uses `orbit_lite` for targeting; our agent also uses orbital intercept — the gap is conversion timing, not routing.
2. **SSDR** (asymmetric planet starts) improves Ajay from 0.8% → 3.1% but improvement is transient — self-play Nash reforms after ~2M steps regardless of pool mask gating. *(Lever removed in C4 — revive from `pre-cleanup-2026-07` if needed.)*
3. **ship0 collapse** — agent learns to send 1-ship probes when behind. Fix was `--min-ship-bin 4` *(removed in C4; the blessed lineage doesn't exhibit it)*. Does NOT fix the underlying SSDR regression.
4. **First Strike** (`--first-strike-steps 50 --first-strike-mult 2.0`) fixed opening paralysis and scored 918.8 LB. It's a reward shaping band-aid, not a structural fix.
5. **BC aux at bc-coef=0.05** — too small to move the needle but can disrupt conversion (Rev34: us_first_cap 14→136 after 1M). *(BC/IL machinery removed in C5 — pre-cleanup tag if needed.)*
6. **Pool mask gating** — SSDR should only apply to self-play envs, not pool envs (was `env.set_ssdr_mask`; removed in C4 with SSDR). Slows regression but doesn't stop it.
7. **PFSP death-spiral**: small N games → noisy wr → never sampled. Fix: `--pool-pfsp-min-games 30`.
8. **LR** *(competition-era rule of thumb: 2.5e-5 mature, 1e-4 warmstart)* — current recipe is a cosine from peak 3e-4 across the full multi-stage horizon, continued on resume with a stage-2 cosine or `--lr-offset-steps`; see docs/training.md "Learning rate". `--resume` warm-loads Adam by default (cold Adam kicked mature policies off their optimum — the noopkl2 collapse).
9. **BC warmstart from partial/diagnostic checkpoints** causes clip_frac=0 (frozen policy). Always use a strong PPO checkpoint as `--resume`. *(BC aux `--bc-samples` removed in C5.)*
10. **Export**: always use `--target-decode` for Phase 1 checkpoints. Run 10/10 vs random before trusting an export.
11. **Scale + observability beat reward shaping** (tl100m, 2026-07-12): 100M from-scratch pure self-play with timeline features, sparse ±1 reward and NO shaping went 0→74.6% vs Ajay — past the shaped lineage's 57.4% best. Launch discipline (rate 0.09) was *learned*, not masked in; the shaping levers of lessons 3–4 were 5M-budget band-aids. Resume from interval checkpoints (they carry Adam + `pool_step` files), not `_final` (no pool file).
    ⚠ **QUALIFIED 2026-07-16:** scale plateaus. tl100m_s2 added 35M for ~+1pp/10M (~77% ±4);
    binary100m_scratch sat at 48–49% Ajay from 30M–50M. **"Just train longer" is not a valid
    explanation for a flat verdict — check the curve before invoking budget.**
12. **⭐ Hardcoded masks delete the action space — MEASURE before believing a lever** (2026-07-16).
    Binary mode's `capture_required` + `maintain`/`defend_ok` gates removed **80.2%** of all commit
    options (62.2% of attacks — no multi-source pincers; 73.3% of reinforces — **pre-emptive
    consolidation literally inexpressible**, since reinforcing needed ≥4 enemy ships already ≤6
    steps out). They compute a verdict from features the model already sees (ch10/ch20/ch22-25),
    then delete what it might disagree with. Isaiah reported masking made his model WORSE; the
    winners use **soft KL priors**, not prohibitions (Ender's 1:1:1:1:10 → converges to 97.7%
    all-in *on its own*). `--binary-commit-gates minimal` restores 83.7%. docs/training.md.
13. **Measured, not argued** (2026-07-16) — three beliefs died to cheap probes:
    - **Ender all-ins 97.3% vs Ajay / 97.7% vs itself** (`ender_sizing.py`) ⇒ learned middle
      commitment is worth ≤3% of launches. Killed a ~30h experiment.
    - **peel-rate vs Ender is TAUTOLOGICAL** (we're wiped 100% ⇒ every capture is lost by
      construction; 235/235). Valid only vs opponents we sometimes beat (0.59 vs Ajay).
    - **"reinforce share 0.21 vs Jake 0.56" was opponent-confounded.** Like-for-like vs Ajay:
      us 0.49, Ender 0.434. Never compare a rate across different opponents.
14. **Probes must configure the model exactly like `evaluate_checkpoint`.** `build_agent_fn` reads
    `allow_reinforce` + discipline masks **off the model object** (eval.py:391/:1560), NOT its
    kwargs. Forgetting them silently disables reinforcement — a reinforcement diagnosis then
    measures your own config (this produced a bogus "92% never reinforced"; true value 63.9%).
    Corollary: **never infer an offered distribution from the executed one when the offer is
    gated** — "executed reinforces are 94.6% all-in" looked like evidence the maintain sizing
    didn't bind; it was the *signature* of the gate (only max-threat options were legal).
