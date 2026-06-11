# Next steps / idea backlog

Living doc — efforts and ideas, roughly prioritized. Discipline: **one delta per cloud run.**
Status tags: 🔵 in-flight · 🟢 ready-to-build · 🟡 idea · ✅ done · ⏸ parked.
Current focus: **Phase 2 — reinforcement** (docs/phase2.md). VDN/per-slot concluded (doesn't work; back to standard arch).

---

## ⚠️ LIVE INSTANCE — p2rev1 (2026-06-11)

- **GCP `orbit-wars-training` (zone `asia-south1-b`, IP 34.93.147.78) RUNNING + BILLING (~$1.13/hr).**
  Running **p2rev1** (Phase 2 Tier-1) in tmux `training`. Watcher → `gpu_run_artifacts/p2rev1/`.
- Config: snowball-BC warmstart + forward-staging mask + drop defense + gate(3)/garrison-floor(10) +
  pool (**3 heuristic hammers: lb1152 + lb1138 doom_evac + lb1084 4p_relative_gap @ external-frac 0.25**)
  + **num-envs 192, rollout 128, ppo-epochs 2, heuristic-workers 2** (6 worker procs), LR 1e-4.
  Script: `gpu_run_artifacts/p2rev1/run_remote_p2rev1.sh` (scp'd to `~/orbit_wars_rl/`; seed scp'd to
  `~/orbit_wars_rl/seed_checkpoints/`). SPS ~250-300 (heuristic-worker overhead), GPU ~44%, mem ~17GB.
- **2026-06-11 relaunch:** first launch STALLED at iter 6 — the in-process `candidate_debatreya_1300`
  planner adapter (`--planner-externals`) wedged the main thread in `batched_planner.py`'s per-env
  `single_obs_to_tensor` Python loop (py-spy: 78% one core, GPU idle 14%). Per-env planner loop is too
  slow to keep the rollout fed. FIX: dropped debatreya + `--planner-externals` entirely, swapped to the
  3 fast heuristic hammers above. Relaunched 20:22 UTC, clears iter 6→9 cleanly (EV 0.47→0.67, clip
  0.23→0.19). `batched_planner.py` no longer a dependency (its rsync/Trash ops risk is moot for this run).
- SSH: `gcloud compute ssh orbit-wars-training --zone=asia-south1-b` (or alias
  `orbit-wars-training.asia-south1-b.orbit-wars-rl`). Tail: `grep '^iter' <log> | tail`.
- **WATCH (early):** cold-critic warmup — EV should climb >0.5 and clip drop <0.25 by ~iter 10-15
  (ppo-1 + rollout-32 warms the critic slowly). If clip >0.4 + EV ~0 persists → halve LR / value warmup.
  Then watch `reinf` (gated 0 below 3 planets, ramp with empire size), `H_tgt`/`tgt n/e` (target-head),
  `p90` (no flood). Selection is PURE: held-out win-rate decides.
- **TERMINATE when done:** `gcloud compute instances delete orbit-wars-training --zone=asia-south1-b`.
- ⚠️ Latent ops bug hit this launch: `orbit_wars_rl/batched_planner.py` (debatreya dep, git-tracked) was
  in Trash during rsync → `ModuleNotFoundError` → had to scp manually. Ensure it exists before launch.

---

## ⭐ 2026-06-11 session — findings + next levers (prioritized)

**Biggest finding — the gap vs strong planners is the OPENING, not conversion/hoarding.** Replay
analysis of p2rev1 9.8M's wins vs debatreya (`/tmp/analyze_deb_wins.py`) shows competence is
**bimodal**: wins = decisive snowball (led 80–90% of the game, eliminate the opp by step ~220, end
19–24 planets) — real strategy, zero timeout-luck. Losses = we lose the opening and get eliminated
**fast** (~90–107 steps, led 1–6%). When ahead we convert dominantly; we just lose the early exchange
~99% of the time vs a 1300-planner. **Lever = early-game strength/survival (First-Strike-style opening
pressure), NOT hoarding/late conversion.** Won-game HTML replays: `/tmp/p2rev1_WIN_*_seed{1948,9013}.html`.
p2rev1 9.8M panels (masks-on): **Ajay 0.4% · debatreya 0.8% · Zach ~62%** — weak vs strong planners
(opening), strong vs Zach. (Ajay/debatreya NOT LB-predictive — yardstick, not objective. Best-ever
Ajay refs: rev53b 10.9%, rev54 1M 5.5%, rev38 3.1%.)

- 🟢 **`--pool-seed-rl` BUILT + TESTED — use next run.** Pins fixed RL champions into the pool via the
  GPU "self" path (fast, **sim-gap-immune** — unlike the heuristics). Never FIFO-evicted, survives
  save/load. `opponent_pool.py` (`add_pinned_rl` + `pinned` flag), `tests/test_pool_pinned.py`. Next run:
  `--pool-seed-rl gpu_run_artifacts/rev38/checkpoints/torch_step_5242880_rev38_20260605_181635.pt,gpu_run_artifacts/preseed_pool/torch_step_10485760_rev53b.pt`
  (rev38 = first-strike aggressor → opening pressure; rev53b = selective). **Keep rev31/rev32b
  HELD-OUT** (cross-eval forgetting detector). p2rev1 loses to all four (cross-eval 7.7M) → real
  pressure. Compat verified (pairwise15; rev38's stale `angle_head` keys auto-filtered).
- 🟡 **Opening-strength shaping** — if the pinned aggressors aren't enough, revisit First-Strike
  (`--first-strike-steps/--first-strike-mult`) for the from-scratch reward to win the early exchange
  more often. The 9.8M analysis says this is THE lever vs strong planners.
- ⏸ **Train/eval sim gap — documented + PARKED** (`docs/train-eval.md`, memory
  `project_train_eval_sim_gap`). Pool wr is torch_env fiction (cross-eval: all 3 hammers **2–4% in
  kaggle** vs training ema 0.40–0.48). Found+fixed the **144-bin angle quantization** of opponents
  (continuous-angle override via `torch_env.step(angle_overrides=…)`, `tests/test_ship_bin_decode.py`)
  BUT a hammer-vs-hammer A/B = 52.6% (noise) → it's a **minor** contributor. Real gap likely our agent
  **overfitting torch_env physics** + win/timeout resolution. Decisive probe (parked): same checkpoint
  vs a hammer in BOTH engines, then diff one shared-seed trajectory. Aim fix kept (correct + free).
- 🟡 **clip_frac creep on p2rev1** — 0.069 (8.0M) → 0.158 (9.4M), first KL/EV wobble @9.4M (the
  entropy-0.05 / LR-1e-4 drift). Standing authority: **halve LR at 0.25**. Wait 2–3 checkpoints
  (9.2/9.8/10.3M panels) before concluding; conversion dipped slightly (cap/atk 0.56→0.51), likely the
  same drift.
- ✅ **Eval hygiene fixed:** debatreya watcher now masks-on + newest-first (`sort -rV`) + `MIN_STEP`
  watermark (backlog cleared at 9.2M); conversion eval prints a **reinf-by-empire-size ramp** vs the
  top-player ramp (aggregate `reinf_share` is opponent/empire-confounded — read the ramp; `eval.py` +
  `docs/metrics.md`); `cross_eval` now includes the pool hammers (`pool_lb1152/1138/1084`) + masks.
- ⏸ **orbit_lite candidate opponents parked** — locutus + producerlite (Kaggle kernels) ≈ 11 ms/call
  (vs hammer 0.7, ~debatreya 9.3) → too slow for the pool, eval-only. Rule: imports `orbit_lite` ⇒
  slow, conclude from imports (memory `feedback_orbit_lite_is_slow`). At `/tmp/{locutus,producerlite}_x/`.

---

## 0. In-flight — Phase 2 reinforcement, Tier-1 (docs/phase2.md)

**rev58/58b resume probes both flooded — cost knob dead, root cause re-diagnosed.** rev58 (cost 0)
and rev58b (cost 0.001) both *drifted* from a healthy gated start into a flood (~330–400k: reinf
0.69–0.75, `p90` 357–408, `Vμ`→negative). Drift-from-healthy ⇒ a property of the reward **objective**
(recurs from scratch), and the pump is **`defense_coef` itself**: in a symmetric self-play mirror,
hold-everything-via-reinforce dominates risky attacking, so the term meant to *incentivise*
reinforcement *is* the flood. Full write-up: `docs/phase2.md` top Update.

**Tier-1 design (locked 2026-06-10) — "outcome-tied attribution":**
- 🟢 **Forward-staging mask (BUILT + unit-tested):** own reinforce target legal only if closer to the
  nearest enemy than the source (`--reinforce-forward-only`) → rear hoard impossible by construction.
  `torch_env.py` + `tests/test_reinforce_mask.py`.
- 🟢 **Drop `defense_coef`** — reinforcement gets no shaping reward; credited purely via terminal +
  early_capture through GAE. Drop `reinforce_cost` (dead). Keep gate(3), garrison-floor(10),
  expansion 0.03, speed 0.3, early_capture anneal, fire-entropy 0.005.
- 🟢 **Small aggressive pool (rev53b-proven)** — held-out LB archetypes so hoarding *loses games*
  (the asymmetry that makes reinforcement instrumentally valuable; a pure mirror gives flood OR
  passivity). (c)-attribution removes the bad incentive; the pool supplies the attack pressure — both needed.
- 🟢 **p2rev1 READY** (fresh Phase-2 numbering, not the rev5x lineage): snowball-BC warmstart
  (`bc_snowball_pairwise15.pt`, aggressive winners, 53% reinforce coverage) + forward mask + drop
  defense + pool (lb1152 hammer + debatreya_1300 @0.25). Target-head diagnostics (`H_tgt`, target
  own/neutral/enemy share) added. Script: `gpu_run_artifacts/p2rev1/`. Awaiting launch.

Selection stays PURE: win-rate/Elo vs the held-out ladder decides; `reinforce_rate`/game-length are
diagnostics. Tier-2 (full causal fleet attribution) held in reserve if Tier-1 underperforms.

## 1. Phase 2 fallbacks / follow-ups (conditional on rev58b)

- 🟢 **From-scratch (BC warmstart) run** — only if we conclude the *base* matters after all (unlikely per the drift
  finding). BC seed options analyzed: Isaiah (controlled) vs Jake/aggressive-cohort (snowball, full 0→0.61 reinforce
  ramp). Snowball selector to pull aggressive replays without player names: high `avg_score` + `size_bytes`<3.5MB.
- 🟢 **Eval/export gate parity** — `action_mask.py` has NO empire gate; before submitting a Phase-2 checkpoint, add the
  gate there so inference matches training (else the policy could reinforce <3 planets at inference). Do at export time.
- 🟡 **Reinforce-aware BC warmstart from top-player replays** (180 games in `/tmp/fresh_validate`, 89 snowball in
  `/tmp/snowball`; analyzer `orbit_wars_rl/fetch_analyze_top_replays.py`, timing-corrected).

## 2. Diversity / anti-cycling (still the structural lever, independent of Phase 2)

- 🟢 **Pool curation** — fold cross-run best selves as fast `self` members; keep self-pool small (8–12 diverse) so PFSP
  gets ≥30 games/opponent. `--preseed-pool`.
- 🟢 **Distill Ajay → fast neural clone (DAgger)** — adds the selective-targeting style we lack; orbit_lite can't be
  GPU-batched. Spec: `docs/ajay_distillation_spec.md`; tool to build: `dagger_collect.py`.
- 🟡 **Neural exploiters / mini-league** — principled anti-cycling; all-neural avoids the slow-opponent problem. Endgame.
- ✅ **Cross-checkpoint eval panel** (`gpu_run_artifacts/cross_eval/run_cross_eval.sh`) — use every ~1M during training.

## 3. Hyperparameter / dynamics probes

- 🟡 **gamma 0.999** (not 1.0) — longer horizon propagates the win signal earlier → less shaping dependence.
- 🟡 **rollout 32 + ppo_epochs 1** — top-LB suggestion: short rollout = local credit; ppo_epochs 1 = more on-policy.

## 4. Measurement / instrumentation

- 🟢 **Always-on held-out ladder during training** — eval vs a fixed diverse set every ~1M; self-play WR + `Vμ`/EV are
  blind to absolute regression. Selection is PURE: win-rate decides, not shaped reward / `Vμ` / Ajay panel alone.

## 5. Parked

- ⏸ **FFA / 4p** — above par on wins, no validated lever, no faithful local eval. Don't spend GPU. (`project_ffa_not_the_gap`)
- ⏸ **VDN / per-slot credit** — concluded: per-slot ship-credit → undercommitment / collapse. Back to joint-credit standard arch.

## 6. Ops / tech debt

- 🟢 **Bake auto-terminate-on-completion into launch scripts** (GCP `--terminate-on-done` not wired; manual delete required).
- 🟡 **Run scripts + seed_checkpoints are rsync-EXCLUDED** by `launch_gpu_gcp.sh` → must scp them after launch (caught us on rev58).
- 🟡 **Pool-opponent `state_dict` reload** fires ~123×/iter — rollout-throughput inefficiency; optimize only if SPS bottlenecks.
