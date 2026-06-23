# Current State — 2026-06-24

Snapshot of live work. Companion to [`current_problem.md`](current_problem.md) (the standing
diagnosis) and [`eval_box.md`](eval_box.md) (the GCP eval box). Newest at top.

---

## ✅ FINAL EVAL + SUBMIT 2026-06-24 — 2 subs uploaded, 4p switched to ajay

Final eval-and-submit session (separate from the bug-fix batch below). **The two bug fixes
(phantom-neutral-production + path-obstruction) are banked in code but did NOT make it into a
competition agent** — every retrain (`presfix1`/`pathobs1`/`stgpr2`) drifted back into the spray
Nash within ~2M (batch note below). So the two final submissions are from the **pre-bugfix
lineage** (presres1 carries the resolver fix #1/#1b only; stgpr1 is the staging lineage).

**Submitted — both daily slots, both 4p → ajay (NOT producer_v4):**

| Sub | 2p agent | 2p vs Ajay / Deb-1300 (256g, 0 draws) | 4p |
|---|---|---|---|
| 53991660 | presres1 0.5M "decisive" | 53.9% / 51.6% | ajay |
| 53991662 | stgpr1 0.5M "spray" | 57.4% / 59.4% | ajay |

stgpr1's higher WR is spray/churn-inflated (play-quality caveat, not a cert defect). Build dirs
`submission_build/{submission_presres05,submission_stgpr1}/`; isolated-tarball (2p 6/6 + 4p 6/6 vs
no-op) + direct routing test (2p→neural / 4p→ffa / step-reset→neural) pass. Tables +
raw logs: `results/eval_results.md` + `results/{cert,ffa}_results.log`. Log: `docs/submissions.md`.

**4p slot = ajay, validated against BOTH producers** (256-game FFA panels, `run_ffa_panel.py`,
seat-rotated, win-rate = 1st-place share = the LB metric):

| field | ajay | producer_v2 | producer_v4 | deb | stgpr1_05 |
|---|---|---|---|---|---|
| v2 panel | **39.1%** | 28.1% | — | 14.1% | 18.8% |
| v4 panel | **39.5%** | — | 23.0% | 24.2% | 13.3% |

ajay wins decisively in both (stable ~39%); **producer_v4 (23.0%) ≈ deb, no better than v2** — so
the prior 4p choice (producer_v4) was beatable. ajay ≡ producer_v2 **minus** its only code diff,
the ETA-aware reinforcement-risk gate → ajay grabs contestable captures the producers veto, and
the producers' caution costs them the FFA territory race. **2p strength ≠ 4p** (deb strongest in
2p, near-worst in 4p). Our 2p neural agent runs in 4p but places 3rd–4th → heuristic is the right
4p choice.

**Eval box `orbit-wars-eval` — DELETED 2026-06-24** after this session (it had been reused/recreated
for the certs + FFA panels above; the cleanup-✓ checklist at the bottom referred to an earlier
teardown). All results synced to `results/` before deletion. $0 ongoing spend.

### ⛔ Inference-only bugfix test: path-obstruction mask on the FROZEN agent = −2pp (do NOT ship without retrain)

Tested whether the path-obstruction fix (the one fix that's an *action mask*, not an observation
change, so in principle shippable onto a frozen agent) helps the **submitted presres1 0.5M** with
**no retrain**. Rigorous paired A/B on a fresh box: **prephantom code** (cherry-picked path-obstruction
ONTO `8f78555`, phantom fix deliberately OUT so the ckpt sees its trained features) + the
**resolver-correct backfilled ckpt**. Same 256 seeds, only `--path-obstruction-mask` differs.

Paired A/B vs Ajay (same 256 seeds, baseline validates by exactly reproducing each submitted agent;
only `--path-obstruction-mask` differs). **Tested BOTH submitted 2p agents:**

| agent (0.5M) vs Ajay | baseline | + `--path-obstruction-mask` | delta | signature |
|---|---|---|---|---|
| **presres1** "decisive" | **138/256 (53.9%)** ✓exact | **133/256 (52.0%)** | **−5 games (−1.9pp)** | seat 0 identical (70=70), all loss seat 1 (68→63) |
| **stgpr1** "spray" | **147/256 (57.4%)** ✓exact | **143/256 (55.9%)** | **−4 games (−1.5pp)** | seat 0 identical (68=68), all loss seat 1 (79→75) |

**Both net-NEGATIVE, ~−1.5 to −2pp, same signature** (seat 0 byte-identical → damage is exactly the
screened boards the mask retargets). **Conclusion: applying the bugfix at inference only is
net-negative on both lineages** — even an action-mask fix is OOD when the agent never trained with
the veto; the policy chose those "doomed" launches for learned reasons and force-retargeting them
loses more than the doomed launches cost. The doc's earlier **+1.5pp was on the 1.5M ckpt stacked on
the phantom fix; it did NOT transfer.** So **both bugfixes genuinely required a retrain to cash in —
there is no inference-time shortcut** (now measured on both agents, not assumed). Did not resubmit;
the two locked-in subs stand.

---

## ⛔ BATCH CONCLUDED 2026-06-24 — all runs down, $0 spend (only the local stgpr1 panel still running)

The phantom-fix-tree batch (stgpr2 / presfix1 / pathobs1 / roidef1) is **over**, with a clean
negative: **self-play drives EVERY run into the same spray-and-churn attractor regardless of the
lever** — it's an equilibrium, not a perception/reward bug. Tracked live via launch_rate
(Isaiah ref 0.036):

| run | launch_rate trajectory | verdict |
|---|---|---|
| `presfix1` (phantom + min-ship-bin) | 0.148 → 0.131 → 0.250 → **0.391** @2.1M (peel 0.65→0.72, WR flat ~45%) | started clean, self-play retrained it into spray by 2M |
| `pathobs1` (+ path-obstruction veto) | 0.118 → 0.126 → **0.284** @1.5M (faster than presfix1) | veto did NOT change the attractor — same drift |
| `stgpr2` (staging lineage) | already sprayed (peel 0.73–0.76, like stgpr1 0.38/0.71) | win-by-churn, not progress |

So **phantom-fix / `--min-ship-bin 4` / path-obstruction-veto are all the WRONG lever class**: they
clean the opening, then the self-play Nash retrains the agent back into flooding + non-holding within
~1.5–2M. This is [[force-concentration wall]] seen as a launch_rate trajectory. **Stop steering by
Ajay WR** (spray-inflated, doesn't transfer to LB); steer by **launch_rate→~0.04 / peel↓ / hold↑**.
Full write-up: `memory/project_stgpr_spray_inflated_wr.md`.

**Teardown (2026-06-24):** `stgpr2` (432654), `presfix1` (432650), `pathobs1` (432670) all destroyed;
`roidef1` (432600) earlier removed. Local watchers + `auto_eval_loop` daemon stopped. **GCP eval box
`orbit-wars-eval` DELETED.** Artifacts retained: `gpu_run_artifacts/{presfix1,pathobs1,stgpr2}/`
(incl. `_old_*` archives) + eval CSVs/logs. The **local stgpr1 0.5M re-eval panel is still running**
(the 57.4% recheck on current eval code — tracking ~45%; phantom fix changed eval).

**NEXT MOVE = STRUCTURAL, not another knob.** An opponent/curriculum that punishes spray and forces
holding (e.g. [[h14 win-gradient]] vs a forward-projector, or a league per the Toad Brigade recipe) —
decide fresh. Below is the (now-historical) lever detail from the concluded batch.

### Lever 1 — phantom-neutral-production (FIXED, in all runs)
Replays `81498718`/`81501661`: barely launched (1/218), idling while cheap rotating neutrals sat
untaken. Cause: pairwise capture-cost/ROI projected `ships_at_arrival = ships + prod·eta` for
*every* target, but **neutrals don't regrow** (engine applies prod only to owner≠−1) → a 16-ship
prod-1 neutral priced as ~24–43 ships, read as bad ROI. **Fix** (2 files, parity 4/4): zero
`prod·eta` for neutrals in `features.py compute_pairwise_features` (ch10/11/12/13/17) + `torch_env
_compute_pairwise` **and** the decisive-mass/staging `floor` (neutral-only staging reward).
Counterfactual (frozen presres1 1.5M + fix): 54.3%→43.4% (OOD; fix pays off via retrain). Committed
`1117944`.

### Lever 2 — path obstruction (screened targets) — INFERENCE fix shipped, training in `pathobs1`
Replay `81509243` (lost): home is a low-prod corner planet; its one cheap neutral (P19, 11 ships)
is **screened by an 86-ship neutral (P11) on the straight path**. Aim is correct, engine is correct
— the fleet hits P11 (swept-collision) and dies; P19 never taken; never expand; eliminated step 99.
**Not a math bug — a missing feature** (no planet-in-path signal; only the sun-cross flag exists).
**Fix:** `action_mask._path_obstruction_blocked` — pre-argmax mask of (source,target) pairs whose
path is screened by an uncapturable planet (cost > source garrison) → eval RETARGETS to a reachable
planet; training VETOES (torch_env `_apply_actions`), same split as redundant_target. Flag
`--path-obstruction-mask` (default off; eval/export auto-load from ckpt). Decode-mask eval on frozen
presres1 1.5M: 43.4%→**44.9%**, launch_rate 0.222→0.237 (small + safe; effect diffuse since screened
boards are a minority). `tests/test_path_obstruction.py` (6) + smoke pass. Training-internalised
version live in `pathobs1` (432670), which also carries `--min-ship-bin 4` so its only delta vs the
presfix1 v2 control is the veto.

### Lever 3 — late-game ship0 collapse (1-ship spray) → `--min-ship-bin 4` in presfix1 v2 + pathobs1 (⚠ symptom not disease)
stgpr2 1.5M eval: tight efficient opening (`<50` cap/atk 0.55, planets@50 9, ship0 0%) then
**81% of launches >100st are 1-ship probes** (ship_bin 0, n=384k) — incl. WON games. WON drag to
**332st median** by attrition instead of closing; LOST end early (131st) by snowball. So the spray
is *downstream* of the 50% ceiling (the ceiling is the early out-massing wall, `outmassed 93%`),
but it caps decisiveness/retention (`peel-rate WON 0.62`). `min_ship_bin` was wired but unused
(all runs =0). `pathobs1` (and the now-down `presfix1`) carry `--min-ship-bin 4` (bans 1–4 ship launches),
confirmed biting at iter 1 (`ship0 0.00 / meanshipbin ~16`). **⚠ Update 2026-06-24: it's a symptom
fix, not the disease.** stgpr2 carried it and peel stayed 0.73–0.76 — the ban resizes probes 1→5
but the spray is the *overall* launch_rate (0.38, ~10× Isaiah) + non-holding, not probe size.
ship0 banning alone won't lower peel. Real metric: **peel-rate↓ / launch_rate→~0.04 / hold↑** (see
NOTE at top + `memory/project_stgpr_spray_inflated_wr.md`).

### Superseded: roi-deflation + pressure-resolver A/B (`roidefl2` / `roideflpr1`) — DEAD

Prior thread: `--roi-enemy-deflate` / `--zero-roi-channels` to fix the target head chasing
cheap-but-contested neutrals (replay `81454632`, ch12/13 deflated by friendly inbound only). The
`roidefl2`/`roideflpr1` A/B instances (432364/432365) are gone (preempted/destroyed). The
roi-deflate flag was then tested **on the phantom-fixed baseline** as `roidef1` (432600), but that
instance has since been **removed** — no live roi-deflate run remains. The flag survives in-codebase
(config/ppo/train/eval), as does `--zero-roi-channels`, both untested on the fixed tree going forward.

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
| 2 | **Phantom neutral production** — roi/cap-cost adds `prod·eta` to neutrals, which don't regrow (confirmed: neutrals flat for 18 steps in the replay; small *rotating* neutrals priced ~24–43 ships vs 16/18 actual) | **FIXED** (2026-06-23) — `prod·eta` zeroed for neutrals in `features.py` (ch10/11/12/13/17) + `torch_env.py` `_compute_pairwise` AND the decisive-mass/staging `floor`. Parity 4/4. Live in `pathobs1` (and was in the now-down `presfix1`/killed `stgpr2`/removed `roidef1`). Counterfactual on the frozen presres1 ckpt = 54.3→43.4% (OOD; needs retrain). |
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
- **`auto_eval_loop` daemon** (local) tracks **1 run** every 500k checkpoint
  (`RUNS=("pathobs1:pathobs1")`), evals it on the box, appends
  `gpu_run_artifacts/pathobs1/eval_ajay_1200.csv`. History: 4→2 (roidef1 removed) → 3 (stgpr2 @1M
  stride) → 2 (stgpr2 killed) → 1 on 2026-06-24 (presfix1 taken down). The `run:short:stride_steps`
  entry format remains in the script (coarser cadence to cap box load); currently unused. Box is
  way under-utilised at 1 run — candidate to delete if no new runs land soon.
  **Box code re-synced 2026-06-23 with the phantom-fix AND the path-obstruction code** (`features.py`
  + `torch_env.py` + `action_mask.py` + `eval.py` path_obstruction auto-load) — required so the box
  evals fixed-tree + `path_obstruction_mask=True` checkpoints with code matching training (the
  path-obstruction code is default-off/backward-compatible, so it doesn't disturb the other runs).
  Box `checkpoints/` dirs pre-created. Start/stop: [`eval_box.md`](eval_box.md) §7.
  **Local `_sync` watchers** kept; local `_eval` watchers **killed** (slow).
- **Gotcha:** box needs `gpu_run_artifacts/<run>/checkpoints/` pre-created or the daemon's rsync
  fails (mkdir won't make nested parents). **Persist gotcha:** a run launched before its flag's
  persist line lands won't round-trip the flag → eval runs the wrong feature regime (bit roidefl1).
  All 4 runs verified `PERSIST_FIX_PRESENT` (ppo.py cfg_blob fix); flags (resolver/roi/path-
  obstruction/min-ship-bin) round-trip and the box auto-loads them per checkpoint.
- **One box = one code regime:** the box runs the *fixed* tree (now incl. path-obstruction), so it
  can only correctly eval that regime. Pre-fix runs (stgpr1/presres3) dropped — instances gone.

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

## Submission state (Kaggle) — ⚠️ SUPERSEDED by the FINAL EVAL + SUBMIT section at top (4p is now ajay, not producer_v4)

`submission_2p4p/` dispatches by player count: **2p → neural** (phase4fs 7.34M), **4p →
`producer_v4.py`** (Producer Hybrid v4, FFA leader-guard 0.035 + target-prod 0.08). Last two
subs: 53958174 (2p 61% / 4p 29%), 53958510 (2p 57% / 4p 31%). 2p above par (50), 4p barely
above par (25). The 4p RL self-play (`ffa4p`, 15M, done) plateaued ~28% and **collapsed 2p
Ajay to ~2%** — abandoned; producer dominates 4p (won 100% vs the RL agents in a panel).

---

## Cleanup checklist — ✅ ALL DONE 2026-06-24 ($0 spend)

```bash
# jl destroy 432670/432654/432650/432600  — pathobs1/stgpr2/presfix1/roidef1 all destroyed ✓
# pkill -f auto_eval_loop.sh              — box daemon stopped ✓
# run_watchers.sh stop                    — all _sync watchers stopped (0 left) ✓
# gcloud compute instances delete orbit-wars-eval --zone=asia-south1-a  — eval box deleted ✓
```
Only the **local stgpr1 0.5M panel** is still running (intentional — the 57.4% recheck); it self-exits.
