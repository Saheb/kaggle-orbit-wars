# Current State — 2026-06-23

Snapshot of live work. Companion to [`current_problem.md`](current_problem.md) (the standing
diagnosis) and [`eval_box.md`](eval_box.md) (the GCP eval box). Newest at top.

---

## ⭐ ACTIVE: phantom-neutral-production fix — 3 parallel Jarvis runs (baseline + roi-deflate + staging)

**Root cause (this session):** replays `submission_analysis/81498718.json` (+ `81501661`) — we
barely launched (1/218 steps in 81498718), idling a growing garrison while small **rotating**
neutrals (16/18 ships, prod-1) sat untaken. Diagnosis = **phantom neutral production**: the
pairwise capture-cost/ROI features projected `ships_at_arrival = ships + prod·eta` for *every*
target, but the engine applies production only to `owner != -1` — **neutrals don't regrow**. So a
16-ship prod-1 neutral was priced as ~24–43 effective ships (worse the longer the intercept — i.e.
exactly the slow/far rotating neutrals), making cheap neutrals read as bad ROI → the target head
deprioritized them and/or under-shipped them into the sufficient-commit veto.

**Fix (committed-pending, 2 files, parity-matched):** zero `prod·eta` for neutral targets in
- `features.py` `compute_pairwise_features` (inference, ch10/11/12/13/17), and
- `torch_env.py` `_compute_pairwise` (training, same channels) **and** the decisive-mass/staging
  `floor` (`_decisive_mass_fields`) — the staging potential is neutral-only, so the phantom also
  under-credited staging toward cheap neutrals in the **reward**.

`test_torch_env_features.py` parity 4/4 (train↔eval byte-identical preserved). **Counterfactual**
(old presres1 1.5M + fix, no retrain): Ajay 54.3% → **43.4%** with launch_rate flat — i.e. the
frozen policy was *fit to* the phantom encoding, so the fix only pays off via **retraining**
(confirms the 3 runs below, not evidence against the fix).

**3 runs — all on the phantom-fixed tree, A100-80GB spot, fresh pool, 10M:**

| run | instance | resume | delta vs ckpt | role | destroy |
|---|---|---|---|---|---|
| `presfix1` | 432608 | presres1 @1.5M (54.3%) | phantom-fix **only** | **clean baseline / control** | `jl destroy 432608 --yes` |
| `roidef1` | 432600 | presres1 @1.5M (54.3%) | phantom-fix **+ `--roi-enemy-deflate`** | roi-deflate arm | `jl destroy 432600 --yes` |
| `stgpr2` | 432594 | stgpr1 @0.5M (57.4%, best-WR) | phantom-fix (staging lineage) | staging arm | `jl destroy 432594 --yes` |

All: resolver (already in ckpt), gate2, sufficient-commit 1.0, reverse-edge 3, first-strike 2×,
game-phase, externals producer_v2+deb. presfix1/roidef1 LR 1e-4 (presres1 config, NO staging);
stgpr2 LR 5e-5 + `--staging-shaping-coef 0.2 --staging-topk 2`. **Reads:** `presfix1` = does the
fix alone beat 54.3% once retrained; `roidef1 − presfix1` = roi-deflate's marginal effect (clean,
identical base/config); `stgpr2` = the fix on the best-WR staging lineage (separate base, not
directly comparable to the presres1 pair). Scripts in `gpu_run_artifacts/{presfix1,roidef1,stgpr2}/`.

### Superseded: roi-deflation + pressure-resolver A/B (`roidefl2` / `roideflpr1`) — DEAD

Prior thread: `--roi-enemy-deflate` / `--zero-roi-channels` to fix the target head chasing
cheap-but-contested neutrals (replay `81454632`, ch12/13 deflated by friendly inbound only). The
`roidefl2`/`roideflpr1` A/B instances (432364/432365) are gone (preempted/destroyed). The
roi-deflate flag survives and is now tested **on the phantom-fixed baseline** as `roidef1` above;
`--zero-roi-channels` remains in-codebase (config/ppo/train/eval), untested on the fixed tree.

---

## Feature-accuracy audit (2026-06-23) — train/eval consistency

Swept the feature surface (features.py numpy ↔ torch_env.py torch). **All 22 pairwise channels
verified byte-identical across both paths** (no numpy↔torch divergence); the bug class is
"feature models the world wrong, but *consistently*" (so the model trained on it and the parity
test stayed green).

| # | finding | status |
|---|---|---|
| 1 | **Pressure channels (ch14 enemy_contest / ch19 keepability / ch20 enemy_mass_soon / ch21 threat_imminence) used the deprecated ~85% loose corridor** (perp<r+1.5, launch heading, no orbital lead, double-counts a fleet onto every planet in its path) vs the 98.4% `_fleet_target_idx` resolver the veto/reward already use | **FIXED** behind `--pressure-precise-resolver` (live in roideflpr1). Removes ~42% phantom over-count in enemy_contest; parity 0.0000 (numpy `_resolve_fleet_targets` == torch). |
| 1b | **Planet-level pressure features ch12/13** (`friendly_pressure` / `enemy_pressure`) STILL use the corridor — left out of #1 to keep the blast radius tight (they're train/eval-consistent, both paths corridor, so no parity break). For full consistency, fold them under the SAME `--pressure-precise-resolver` flag. | **FIXED** — ch12/13 now route through `incoming_pw` (torch) / `_resolve_fleet_targets` (numpy) under the existing flag; no new flag/wiring/dim change. Measured: corridor→resolver drops ch12/13 abs-sum ~42%/25% (same over-count as #1); planet parity 0.0000 (now hard-asserted in `run_pressure_resolver_parity`). |
| 2 | **Phantom neutral production** — roi/cap-cost adds `prod·eta` to neutrals, which don't regrow (confirmed: neutrals flat for 18 steps in the replay; small *rotating* neutrals priced ~24–43 ships vs 16/18 actual) | **FIXED** (2026-06-23) — `prod·eta` zeroed for neutrals in `features.py` (ch10/11/12/13/17) + `torch_env.py` `_compute_pairwise` AND the decisive-mass/staging `floor`. Parity 4/4. Live in `presfix1`/`roidef1`/`stgpr2`. Counterfactual on the frozen presres1 ckpt = 54.3→43.4% (OOD; needs retrain). |
| 3 | **Threat-ETA uses planet center not surface** (omits `−pr` that the resolver uses) → threat reads ~½ step under-urgent | **FIXED** behind `--threat-eta-surface` (own default-off flag, NOT folded into `pressure_precise_resolver` so the live presres1 — resolver-on, center-ETA-trained — stays clean). ch20/21 ETA = `(dist−radius)` both paths; parity 0.0000, flag moves ch21 +39% (surface 3.76 vs center 2.70). Plumbed config/features/torch_env/eval/train/export + `test_threat_eta_surface_parity`. |
| 4 | **`SHIP_COUNTS`/`NUM_SHIP_BINS`/`FRACTION_BIN_VALUES` hand-triplicated** across action_mask/model/torch_env (identical now, silent landmine) | **FIXED** — single source in action_mask, derived elsewhere |
| 5 | **Pairwise parity test only asserted *planet* feats** — the 22 pairwise channels (where #1/#2 live) were untested | **FIXED** — all 22 channels + roi flags + resolver now hard-asserted in `tests/test_torch_env_features.py` (pytest-collected) |

#1/#4/#5 committed (`72b20ba`, `80244cb`). **#2 FIXED + live (the 3 runs above; commit-pending).**
#3 (`--threat-eta-surface`) still default-off, to stack on whichever arm wins. All flags
checkpoint-compatible (no dim change), default-off, persisted + eval/export-synced. NOTE: #2 is an
*unconditional* correctness fix (no flag — neutrals never regrow), unlike #1/#3.

**#1b — DONE (gated under the existing `pressure_precise_resolver` flag, no new flag):**
- **torch** `torch_env.py` get_features: moved the `incoming_pw` block above `friend`/`enemy` and
  derived `friendly_pressure`/`enemy_pressure` (planet ch12/13) from it; `friendly_pressure_pw`
  now just aliases `friendly_pressure` (same attribution). Corridor preserved when flag off.
- **numpy** `features.py`: precompute `_resolve_fleet_targets(...)` ONCE before the planet loop →
  per-fleet target array-index; in the loop `incoming = (pp_resolved_tgt == i)` when the flag is
  on, else the corridor. (Planet array index `i` == resolver target index since both index
  `planets[:max_planets]`.)
- **parity** `tests/test_torch_env_features.py`: `run_pressure_resolver_parity` now also diffs the
  full **planet** feature vector under the flag (`max_planet_diff`, hard-asserted) — ch12/13 locked
  alongside the pairwise channels. `3 passed`; planet diff 0.0000.
- **wiring**: none — reuses the persisted `pressure_precise_resolver` config + eval/export sync.
- Verified: corridor→resolver drops ch12/13 abs-sum ~42%/25% (the same phantom over-count as #1);
  checkpoint-compatible, default-off. Lands consistency: no more two-ways-to-measure-the-same-thing
  in one feature vector. (Not yet committed — local edits.)

### Prior thread (concluded): redundant-target veto A/B

`phase4fs2` (432127, no veto) / `phase4fs_rv` (432208, `--redundant-target-factor 1.0`) — both
dead (preempted). Decode-only test was −3.5pp vs Ajay but +3pp h2h (train/eval mismatch → folded
into training). `--redundant-target-factor` is in the codebase (config/ppo/torch_env/train/eval +
`tests/test_redundant_target.py`). Superseded by the roi-deflation thread above.

---

## Eval infrastructure

- **GCP box** `orbit-wars-eval` (n2-standard-32, asia-south1-a) — sharded Ajay panels ~4 min.
  Bills ~$1.55/hr: `gcloud compute instances delete orbit-wars-eval --zone=asia-south1-a` when done.
- **`auto_eval_loop` daemon** (local) now tracks `stgpr2` + `roidef1` + `presfix1`, evals each new
  checkpoint on the box, appends `gpu_run_artifacts/<run>/eval_ajay_1200.csv`. **Box code re-synced
  2026-06-23 with the phantom-neutral-production fix** (`features.py` + `torch_env.py`) — required
  so the box evals the fixed-tree checkpoints with code matching their training (else the 54→43
  OOD swing). Box `checkpoints/` dirs pre-created for all three. Start/stop: [`eval_box.md`](eval_box.md) §7.
  **Local `_sync` watchers** kept (land checkpoints); local `_eval` watchers **killed** (slow).
- **Gotcha:** box needs `gpu_run_artifacts/<run>/checkpoints/` pre-created or the daemon's rsync
  fails (mkdir won't make nested parents) — done for stgpr2/roidef1/presfix1. **Persist gotcha:** a
  run launched before its flag's persist line lands won't round-trip the flag → eval runs the wrong
  feature regime (bit roidefl1). The current 3 runs all verified `PERSIST_FIX_PRESENT` (ppo.py
  cfg_blob fix), so resolver/roi flags round-trip and the box auto-loads them per checkpoint.
- **One box = one code regime:** the box now runs the *fixed* tree, so it can only correctly eval
  fixed-tree runs. Pre-fix runs (stgpr1/presres3) were dropped from tracking — their instances are
  already gone.

---

## Session findings (2026-06-23) feeding the above

1. **`cu` overfired = 1-ship spray (confirmed).** phase4fs_cu 51.6% Ajay but `ship0=49%`,
   launch_rate 0.36 (vs base 0.095), worse concentration+retention → spray artifact, won't
   transfer. **Keep base 4fs 7.34M** as the clean 2p agent. stg = same pathology, milder (39%).
2. **2p LB loss analysis** (44 real LB games, `/tmp/sub_replays/53958174`): the loss is
   **mid-game (step 40–75), NOT the opening** (launch rate 0–25 identical won/lost). At step 40
   games are tied; the winner pulls away 50→75. Mechanism = **retention** (mid-game hold% 47%
   lost vs 58% won) at *equal launch effort*. Steer by mid-game hold% / planets@50, not opening
   aggression or firing volume. Confirms `current_problem.md` v3 on real LB games.
3. **Capture floor** (`eval._attack_capture_floor` = `torch_env._decisive_mass_fields`):
   `garrison + production·eta + enemy_inbound + β·ρ(eta)·reachable_enemy_mass + overhead`
   (β=2.2, ρ=clamp((eta−3)/12,0,1), horizon 18, overhead=1 = the strict-capture +1 margin).
   The **`sufficient_commit` decode/train mask** uses a simpler static floor (garrison +
   enemy_inbound). Only `sufficient_commit` (and now `redundant_target`) are in **training**;
   the reactive decisive-mass floor is computed for **diagnostics only** (decmass reward off).

---

## Submission state (Kaggle)

`submission_2p4p/` dispatches by player count: **2p → neural** (phase4fs 7.34M), **4p →
`producer_v4.py`** (Producer Hybrid v4, FFA leader-guard 0.035 + target-prod 0.08). Last two
subs: 53958174 (2p 61% / 4p 29%), 53958510 (2p 57% / 4p 31%). 2p above par (50), 4p barely
above par (25). The 4p RL self-play (`ffa4p`, 15M, done) plateaued ~28% and **collapsed 2p
Ajay to ~2%** — abandoned; producer dominates 4p (won 100% vs the RL agents in a panel).

---

## Cleanup checklist (when done)

```bash
jl destroy 432608 --yes     # presfix1 (phantom-fix baseline)
jl destroy 432600 --yes     # roidef1  (phantom-fix + roi-deflate)
jl destroy 432594 --yes     # stgpr2   (phantom-fix, staging lineage)
pkill -f auto_eval_loop.sh                                  # stop daemon
bash gpu_run_artifacts/run_watchers.sh stop                 # stop _sync watchers
gcloud compute instances delete orbit-wars-eval --zone=asia-south1-a   # eval box
```
