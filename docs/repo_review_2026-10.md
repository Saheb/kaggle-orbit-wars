# Repo review: bloat, dead code, hygiene (2026-10-06)

Purpose: get a clean base before strategy work resumes. This is a **review**. Nothing has been
deleted. Each item states its evidence so you can accept or reject it on facts. The proposed
execution order is in §7.

Scope: `orbit_wars_rl/` (13.1k source lines + 4.5k test lines), `opponents/`, `scripts/`, docs,
and on-disk artifacts. Evidence sources: every `train_torch.py` / `eval.py` flag mapped to the most
recent launch script that used it, docs verdicts (`experiments.md`, `training.md`, `docs/archive/`),
`vulture`, caller greps, persisted configs of the retained checkpoints, and the test suite.

---

## 0. Bug found during the review: probes evaluate the champion under the wrong gates

`ender_sizing._checkpoint_agent` (also used by `coord_overkill_probe.py`) and `peel_diagnosis.py`
hand-copy the model setup that `evaluate_checkpoint` does. They set `allow_reinforce` and the
discipline masks, but **not `model.binary_commit_gates`**. `build_agent_fn` reads that attribute off
the model with default `"full"` (eval.py:337). So any `minimal`-gates checkpoint runs under the
legacy gates, which delete about 80% of the action space it learned to use.

**Measured:** binarygates100m_l4 @95.26M, seed 0 vs Ajay. The probe agent reports
`_binary_gates=full` and **loses** (67 launches, 199 steps). The correctly-gated agent (`minimal`,
as persisted in the checkpoint) **wins** (37 launches, 108 steps).

**Affected results:** every `gpu_run_artifacts/ender_panels/coord_*.log` from 2026-07-19. All of
them ran the champion through this path. That includes the numbers behind experiments.md **#4b,
the recommended next run**: "us 42.6% redundant" and "same-turn multi-source us 7.4%". It also
includes the friendly-contest / candidate-delta ablation verdict in commit 843500b /
feature_audit.md. **Not affected:** watcher panels and `eval.py` (both go through
`evaluate_checkpoint`), the Ender-sizing numbers (Ender is a path agent), and the binarymarg
peel/sizing forensics (that checkpoint *was* `full`, so the default matched by luck).

This is the second instance of Key Lesson 14 (the earlier one was `allow_reinforce`). The fix is
structural: add one `eval.load_eval_model(path, cfg)` helper that both `evaluate_checkpoint` and
every probe call, so no caller can forget an attribute again. After the fix, re-run
`coord_overkill_probe.py` before deciding on #4b.

---

## 1. Headline numbers

| | |
|---|---|
| Removable with high confidence (Tier 1, §2) | **≈2,400 lines** (≈14% of `orbit_wars_rl/`) |
| Your call (Tier 2, §3) | ≈1,500 more lines |
| Dead parameters in the champion | **96.5k of 541k (18%)**: COMA Q-head 49.7k + ship head 46.8k (binary mode never uses either) |
| Training flags | 95 total; the champion recipe uses 25, and **all timeline-era runs (≥07-10) combined used ~35** |
| Tracked `archive/` | **854 MB** (`archive/replays/` = 793 MB, referenced nowhere); every git worktree copies it |
| Tests at HEAD | 138 pass, **2 fail** (both are test-design problems, §6); one test is 91% of suite runtime |

---

## 2. Tier 1: remove (dead, verdict recorded, nothing on the experiments.md queue needs it)

| # | Item | Where (≈ lines) | Evidence |
|---|---|---|---|
| T1 | **COMA counterfactual Q-head** (`q_*` params, `q_counterfactual`, `_q_slot_tokens`, `--dump-rollout-and-exit`) | model.py ~85, train_torch ~15, tests/test_q_head.py 150 | Called only by its own unit test. q-head.md is in `docs/archive/`. 49.7k params sit in every checkpoint and in the Adam param list. |
| T2 | **Eval-only probe overlays**: defensive-reinforce overlay, natural-head audit, `--retarget-top-roi`, `--force-fire-high-roi`, `veto_stats`, `reserve_frac`, `_SHIP_AUDIT`, `--decisive-mass-beta`, binary `capture-defend`/`projected-hold` sizing | action_mask.py ~650 (lines 134–697 plus hooks in `actions_from_target_policy`), eval.py ~450, timeline.projected_hold_sizes ~60, features ~15, tests/test_defensive_reinforce_overlay.py 233 | The flags are documented **only in `docs/archive/`** (head-audit, targeting-vs-sufficiency, train-eval). "Forced projected hold: Rejected" (experiments.md). No script uses them. `actions_from_target_policy` has **33 parameters**. Because export inlines action_mask.py, **every exported agent ships this code**. Removing it takes action_mask.py from 1,185 to ~500 lines. |
| T3 | **`intent` ship-bin mode** (keep `resolve_intent_sizes`: it produces live pairwise features ch22–25) | train_torch ~80 (decode branch, `intent_rollout_metrics`, logging), model ~13, torch_env ~25, action_mask ~12, config ~5, tests/test_intent_telemetry.py 91 | intent100m reached 63.7% Ajay / 2.7% Yijie and was superseded by binary (training.md "Binary NOOP/COMMIT experiment"). |
| T4 | **`fraction` ship-bin mode** (`FRACTION_BIN_VALUES`, decode branches) | ~60 across 5 files + part of test_ship_bin_decode.py | Never used in the timeline era. No retained checkpoint uses it. |
| T5 | **Ship-size KL** (`--ship-kl-coef/-prior-exp`) | ppo ~25, train ~15 | shipkl_probe superseded by binarygates. A learned middle commitment was rejected by measurement (Ender all-ins 97.7%). Binary mode already disables it (ppo.py:308). |
| T6 | **Critic warmup** (`--critic-warmup-ev/-max-updates`, `value_warmup_update`, `value_only`) and `--reinit-critic` | ppo ~40, train ~45 | Built for BC warmstarts (BC archived in C5) and the VDN control (vdn1, pre-timeline). Last used 06-19. |
| T7 | **Reinforce curriculum** (`--reinforce-bias-init/-anneal-frac`, `model.reinforce_logit_bias`) and **`--reinforce-cost`** | train ~35, model ~10, torch_env ~25 | Last used 06-10 / 06-11 (rev58, p2rev1). Lesson 11 superseded the shaped/costed lineage. |
| T8 | **`--phase4-residual-lr-mult`, `--phase4-residual-init-std`** (keep the zero-init behaviour) | ppo ~30, train ~20 | Last used 06-19 (h14feat). The 2-param-group optimizer path exists only for this flag. |
| T9 | **`--ship-overflow-mode drop`** | torch_env ~8, train ~10 | Described as "only to reproduce the legacy bug". Clamp is correct and is the default. |
| T10 | **Measured-dead perf probes**: `--lean-metrics`, `--compile-env`, `--disable-comets` | train ~25 | perf.md: lean-metrics "SPS barely moves… do not strip diagnostics"; compile-env "MEASURED DEAD END (+0%)". disable-comets was never used. |
| T11 | **Compat shims for checkpoints HEAD can't load anyway**: `angle_head` sniffing, `value_pp_` (VDN) ignore, `value_head_in` mean-pool path, ckpt `action_decode` default `"angle"` | eval ~15, model ~6, ppo ~1 | Pre-pairwise / pre-blessed checkpoints are already refused by the HEAD guards. Use the `pre-cleanup-2026-07` tag for those. |
| T12 | **Misc dead symbols**: `SelfPlayConfig` and unused `EnvConfig` fields, `Config.wandb_*`, `model.ship_bin_to_count`, `ppo.get_lr`, `_PER_ENV_KEYS = set()`, the `il_kl`/`il_coef` print (IL removed in C5), unused `Fleet`/`Planet` imports in features.py, the "matches make_batch in self_play.py" comment, `torch_env_fn.{make_board_pool, reset_masked, step_full_core}` | ~120 | vulture plus caller greps. torch_env_fn only needs `physics_step` / `state_from_torch_env` / `apply_actions_core` as the timeline test oracle (the JAX plan was retired in perf.md). |
| T13 | **`orbit_wars_rl/run_eval.sh`** | 70 | Broken: calls `orbit_wars_rl/ckpt_info.py` (missing), a `.codex/worktrees/...` opponent path, and reads `$4` as the game count. Superseded by the watchers. |
| T14 | **`eval.py --panel-shards/--panel-shard-idx/--shard-out`** | ~25 | Needs `merge_panel_shards.py`, which was archived in C5. Half a feature. (Keep `--panel-out` + `recompute_panel.py`, which still work.) |

**Checkpoint-compat note for T1:** existing checkpoints carry `q_*` keys and include those params
in the Adam `param_groups`. The `q_*` params are registered last, so the loader should (a) drop
`q_*` keys from the model state_dict and (b) trim the trailing 10 indices from the saved optimizer
group. Otherwise warm-Adam resume silently falls back to a cold optimizer (train_torch.py:836),
which is the noopkl2 pathology. Add a resume test.

---

## 3. Tier 2: your call (dormant, not on the experiments.md queue, but defensible to keep)

| # | Item | ≈ lines | The question |
|---|---|---|---|
| D1 | **Reward shaping**: `--win-margin-coeff`, `--expansion-coef`, `--early-capture-*`, `--first-strike-*`, `--staging-shaping-*`, plus `_decisive_mass_fields` / `_staging_potential` | torch_env ~190, train ~60, tests 277 (staging + early-capture) | Every run since 07-10 used sparse ±1 reward (Lesson 11). CLAUDE.md's "Key training flags" table still advertises these. **Recommend removing** (the git tag preserves them, and Lesson 11 records the outcome). Keep only if you want them as a live teaching example of shaping pathologies. |
| D2 | **External-heuristic pool** (`--pool-mode mixed`, `HeuristicWorkerPool`, `_heuristic_moves_to_action_tensor`, `to_legacy_obs_batch`, angle-bin decode + `angle_overrides`, `--pool-external-fraction`, `--pfsp-externals`, mastery eviction) | ~600 incl. 3 tests | Last used 06-19. Keep only if training against a fixed scripted/learned opponent (e.g. Yijie) is a real future lever. Keep `to_legacy_obs`: two parity tests use it as the torch_env↔features bridge. |
| D3 | **Legacy discipline masks**: `--sufficient-commit-factor`, `--reinforce-forward-only`, `--reinforce-garrison-floor` (train side ~110, eval side ~60) | ~170 | No timeline-era run used them. presres1/stgpr1 have `sufficient_commit_factor=1.0` persisted. **Do you still need to evaluate presres1/stgpr1 *as the subject* under HEAD `eval.py`?** If they are only used as **opponents**, their frozen tarballs carry their own code, so the eval side can go too. If yes, keep the eval side (and `features.timeline=False`) and drop only the train side. |
| D4 | **`binary_commit_gates="full"`** | ~30 | It is still the **default** (config.py, `--binary-commit-gates` default `None`), so a new run that forgets the flag silently gets the gates that Lesson 12 showed delete 80% of the action space. At minimum flip the training default to `minimal`. Keep the `full` code path only if you will re-evaluate binarymarg/binarycf checkpoints. |
| D5 | **`--global-econ`** | ~100 | econblock was stopped below baseline, but it was confounded with γ=0.999, and "econ-alone is untested" (experiments.md). Keep until you decide whether that clean test is worth a run. |
| D6 | **`mode_proj`** | 4 + fold-on-load | `global_proj(g) + mode_proj(g)` is two linear maps of the **same input** added together, which is exactly one linear map (W₁+W₂, b₁+b₂). It adds no representational capacity. You could fold it at load time. Cosmetic; it touches the checkpoint format. |

---

## 4. Keep: looks unused, isn't

- **Anchor + promotion gate** (`--anchor-*`). Built and unrun, but back-pocketed for 200M+ (experiments.md #1). Fix its test first (§6).
- **Pinned-RL pool, `--pool-pinned-fraction`, `--pool-hard-ramp-steps`, `--pool-seed-rl`**. The anchor needs them, and so do exploiters (#8): a from-scratch exploiter against a pinned main policy is exactly the case the ramp was built for.
- **4p path** (`--num-players 4`). Parked in experiments.md ("after 2p is competitive").
- **`absolute` ship mode and the binary-mode ship head params**. shipkl_probe and presres1/stgpr1 are absolute-mode.
- **`resolve_intent_sizes`** (live features ch22–25) and **`_THREAT_ETA_WINDOW` / `_REACH_HORIZON` / `_VALUE_HORIZON`**. Experiment #3b is the planned way to remove those.
- **`torch_env_fn.physics_step`** (timeline parity oracle), **`to_legacy_obs`** (parity bridge), **`recompute_panel.py` / `--panel-out`**.
- **Probes**: `ender_sizing.py`, `peel_diagnosis.py`, `coord_overkill_probe.py`, `ender_ref.py`, and the `_ABLATE_*` hooks (2 `if`s). Live per CLAUDE.md, but fix §0 first.

---

## 5. Repo hygiene (non-code)

**Disk** (untracked unless noted):

| Path | Size | Status |
|---|---:|---|
| `gpu_run_artifacts/` | 42 GB | ~170 run dirs; most are pre-timeline checkpoints HEAD can't resume |
| `orbit_wars_rl/{bc_from_replays,bc_top_agents,snowball_bc_15g}.pkl` | 1.7 GB | BC datasets; BC pipeline archived in C5 |
| `orbit_wars_rl/episode_data/` + `replays/` | 2.0 GB | BC-era replay data |
| `orbit_wars_rl/checkpoints/` | 217 MB | local scratch checkpoints |
| `archive/replays/` (**tracked**) | 793 MB | added in the Phase-1 commit, referenced nowhere |
| `archive/cleanup_2026-07/old_self_opponents` (**tracked**) | 38 MB | |
| git worktrees | 8 besides main | 3 `.codex`, 2 `.claude`, 1 `.kilo`, `orbit-audit`, `orbit-prephantom`. The `.kilo` and `.claude/strange-khorana` copies are **850–870 MB each, mostly the tracked archive** |

Removing `archive/replays` from HEAD shrinks every future checkout and worktree by 793 MB. `.git`
stays 190 MB unless you rewrite history, which is not recommended.

**Version control gaps:**
- `gpu_run_artifacts/` is gitignored wholesale, so **`run_watchers.sh` (26 KB) and
  `launch_gpu_gcp.sh` are not in git**, even though CLAUDE.md makes them the only sanctioned
  tooling. Move the live infra scripts to `scripts/` (tracked) and leave run outputs ignored.
- `AGENTS.md` is a stale copy of CLAUDE.md (last synced at 989a1a4; 158 lines now differ). Codex
  agents (you have three `.codex` worktrees) read the stale rules. Replace it with a symlink or a
  one-line pointer.

**Duplicated opponents:** `candidate_producer_1200.py` is byte-identical to
`candidate_ajay_1200.py`, and `candidate_jek.py` is byte-identical to `candidate_debatreya_1300.py`.
`candidate_producer_h{4,10,12,14}.py` are Ajay with one constant changed (`horizon`). That makes
six files for one agent, one of which is the "Ajay" every panel trusts. Never-referenced:
`candidate_carbon_v2`, `candidate_early_aggressor`, `candidate_flowdiff`,
`candidate_peeler_t1_sticky`, `candidate_rank1_bc` (2.2 MB), `ourbest/ajay_clone_v0` (2.2 MB).

**Stale docs:**
- `docs/commands.md` ("start here") still has the EC2 sections (§4–5), the producer-ranking / BC
  dataset / Kaggle-submission sections, and references to 12 files that no longer exist (`bc.py`,
  `env.py`, `quick_eval.py`, `producer_*`, `scripts/*`).
- `eval_box.md` references 3 missing scripts. `docs/archive/*` cross-links point at pre-archive
  paths.
- `submissions.md` and the Kaggle sections are moot now that the competition is over (archive them).
- CLAUDE.md "Key training flags" lists mostly shaping and external-pool levers that no current run
  uses (only `--pool-pfsp-min-games` is still live), and omits the ones the champion actually uses (`--ship-bin-mode binary --binary-commit-gates minimal
  --noop-kl-coef 0.3 --allow-reinforce --reinforce-gate-min-planets 2 --reverse-edge-cooldown 3`).

---

## 6. Test health

- **`test_anchor::test_kl_gradient_pulls_toward_anchor` fails.** It takes 5 SGD steps at
  **lr=1.0**. Probed KL trajectory: lr 1.0 → 0.08, 0.36, 0.35, 1.06, 0.37 (overshoot); lr 0.03 →
  0.080, 0.082, 0.065, 0.064, 0.062, 0.061 (descends after step 1). This is a step-size problem in
  the test, not evidence against the anchor KL. Use a small lr, or assert a first-order directional
  derivative instead.
- **`test_train_validation::test_env_symmetry` fails.** It is ill-posed. It asserts that *each*
  pairing of two different random models gives ≈50% P0 wins, which conflates model strength with
  seat bias. Observed: m1 wins 34.4% from seat 0 and 45.3% from seat 1, i.e. m1 is simply weaker.
  The right check is P(m1 wins | seat0) ≈ P(m1 wins | seat1), or mirror-match one model against
  itself. It also takes **169 s of the suite's 185 s**. Move it behind a `slow` marker.
- Tests for Tier-1 items go with them (test_q_head, test_defensive_reinforce_overlay,
  test_intent_telemetry, the fraction half of test_ship_bin_decode, and if D1/D2 are accepted,
  test_staging_shaping, test_early_capture_reward, test_sufficient_commit, and the three
  external-pool tests).

---

## 7. Suggested execution (one stage per commit, like C1–C6)

**Verification for every stage:** a *golden-output* check, plus the unit tests. Before touching
anything, record the champion's exact move lists over ~8 fixed seeds vs Ajay. Eval decode is
argmax and deterministic, so after each stage those moves must be **bit-identical**. Any diff means
the cleanup changed behaviour. This is far stronger than a win-rate check.

1. **D0: shared `load_eval_model`** (§0). Route evaluate_checkpoint and all probes through it. Add a
   test that a `minimal` checkpoint loaded by a probe reports `minimal`. Then re-run the coord probe.
2. **D1: eval/action_mask probe overlays (T2, T14)**. Largest win, eval-only, zero training risk.
3. **D2: Q-head (T1)** with the state-dict and Adam trimming plus a warm-resume test.
4. **D3: intent + fraction modes (T3, T4)**.
5. **D4: abandoned levers (T5–T12)**, plus fix the 2 tests and mark the slow test.
6. **D5: Tier-2 items you accept** (D1–D6).
7. **D6: hygiene**. Track the infra scripts, AGENTS.md pointer, dedupe opponents, remove
   `archive/replays` from HEAD, prune worktrees, refresh commands.md and the CLAUDE.md flags
   table. Delete untracked BC data only after you confirm it.

Tag `pre-cleanup-2026-10` before D1 so everything stays recoverable, same as `pre-cleanup-2026-07`.
