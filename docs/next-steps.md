# Next steps / idea backlog

Living doc — efforts and ideas, roughly prioritized. Discipline: **one delta per cloud run.**
Status tags: 🔵 in-flight · 🟢 ready-to-build · 🟡 idea · ✅ done · ⏸ parked.
Current focus: **🟡 NO LIVE RUN — both 2026-06-15 levers (HANDICAP + CONSOL) CONCLUDED NEGATIVE on the force-concentration wall; all boxes destroyed.** Next lever must supply a STRUCTURAL concentration signal (not another reward/pool knob). See the SESSION 2026-06-15 EVE2 block immediately below.

> **⭐⭐⭐⭐⭐ SESSION 2026-06-15 EVE2 — HANDICAP + CONSOL BOTH CONCLUDED NEGATIVE: reward/pool knobs do NOT crack the force-concentration wall (read FIRST).**
> Two single-delta runs this session, both resumed from stageb4 1.5M, both attacking the holding-under-peel wall, both run to their decision point and torn down ($0, all Jarvis boxes destroyed). **Neither moved the wall.** The wall metric is `out-massed %` of losses (enemy fleet > our garrison at the lost capture) + the absolute `garrison@loss vs enemy-inbound` gap ([[project_force_concentration_wall]]).
> - **HANDICAP (win-gradient lever) — crutch REFUTED, wall UNMOVED.** Inverse-SSDR: grant OUR seat +3 planets in pool envs vs deb-dominant pool, tapering to 0 over 4M (`--self-boost-planets 3 --self-boost-ramp-steps 4000000`). Ran the boost to 0 @4M. **Ajay-WR taper 4.3→4.3→5.9→5.9% @boost0 = HELD** (didn't collapse → the head-start was NOT a fake prop). BUT structural FLAT: **out-massed 99/99/98%**, garr@loss ~40 vs enemy-inbound ~90 (~2× out-massed, unchanged), peel-rate WON 0.64/0.59/0.64 (winner 0.43), planets@100 WON 11/9/11. **Reached rev38 parity (~6%) by REPRODUCING rev38's exact failure mode (out-massed 98% is rev38's own signature), not fixing it.** A matched-difficulty win-gradient vs the REAL planner, even with a tapering head-start, did NOT teach force concentration. Box 427065 destroyed @4M.
> - **CONSOL (reward-shaping lever) — wall UNMOVED, no gain over 6M.** `--consolidation-coef 0.02 --consolidation-steps 40`, deb-dominant pool (ext 0.6), rev38-KL λ0.05, gate2, NO self-boost, full 6M. **Ajay 1M→6M = 6.2/5.1/2.0/3.9/3.5/3.9%** — flat-noisy ~4%, ENDED BELOW its own 1M point AND below rev38 6.2% (another Stage-B run that doesn't beat rev38). Structural FLAT end-to-end: **out-massed 99/98/97/96/97/98%** (the 96@4M was noise — the early "slope" was a composition artifact, abandonment rose 1→3% so fewer losses classified out-massed, NOT real holding), garr@loss ~40 vs enemy-inbound ~92-96 (~2.3× out-massed, unchanged), **peel-rate WON drifted WORSE 0.60→0.73 then 0.59** (no improvement; hint the reward may mildly encourage HOARDING a few over-garrisoned planets the peeler picks apart). zach held-out stayed healthy 80-88% throughout (the low-N Ajay dips were noise, NOT collapse); clip crept 0.20→0.30 (benign: KL low/EV stable/estop 0, [[feedback_clipfrac_lowkl_benign]]). Box 427092 destroyed @6M (clean completion, final ckpt `gpu_run_artifacts/consol/checkpoints/torch_step_6012928_consol_..._final.pt`).
> - **⭐ THE VERDICT (both together):** a win-gradient lever AND a reward-shaping lever both left out-massed pinned ~98% and the garrison-vs-inbound gap at ~2×. **Reward/pool knobs at these magnitudes do NOT induce force concentration — the fix must be a STRUCTURAL training SIGNAL.** We SEE the incoming fleet (features ch13/pairwise14) but never learn to act on it; self-play never priced concentration ([[feedback_win_starvation]]).
> - **🟢 NEXT-LEVER CANDIDATES (supply the concentration signal directly — pick ONE, user to steer):** (A) a **mass-to-floor gate/reward** — credit/require massing to ≥ the forward-projected defender count at arrival BEFORE a launch counts (mirror deb's `capture_floor` in `opponents/orbit_lite/planner_core.py`); (B) **imitate deb's `_plan_regroup`** (multi-source aggregation into one strike) via a targeted BC/DAgger on regroup decisions; (C) an **action/arch change** enabling multi-planet aggregation (we currently fire per-planet reactively → can't muster 94 from one planet). Avoid another scalar reward knob — that's the disproven class.
> - **Eval/diagnostic tooling (kept):** Ajay full-panel logs carry the `hold-loss out-massed`/`garr@loss vs enemy-inbound`/`peel-rate WON` lines per ckpt — read those, don't re-probe ([[feedback_read_full_eval_not_reprobe]]). Local watcher/teardown/heartbeat scripts under `gpu_run_artifacts/` (`teardown.sh`, `heartbeat.sh`, `ajay_1m_eval.sh`) — bash-3.2/BSD-safe ([[feedback_mac_bash_monitor_scripts]]).
>
> **⭐⭐⭐⭐ SESSION 2026-06-15 — HANDICAP run (SUPERSEDED by EVE2 above — CONCLUDED; kept for config/mechanism detail).**
> The lever the compass kept pointing at ([[project_phase3_compass_wall]] / [[feedback_win_starvation]]): nothing in self-play+pool punishes
> a turtle/churner, so the policy settles at planets@50=6 sub-winner optima. **HANDICAP = make under-expansion LOSE vs the REAL planner.**
> - **Mechanism (inverse-SSDR, built + unit-tested):** `--self-boost-planets 3 --self-boost-ramp-steps 4000000` grants OUR seat +3
>   planets in POOL envs, tapering linearly to 0 over 4M steps (the inverse of SSDR, which boosts the OPPONENT). Hooks both reset
>   paths; per-env mask; test confirmed seat-0 1→4 planets at k=3. Tests in `orbit_wars_rl/tests/` (self-boost).
> - **Config:** resume **stageb4 1.5M** (best Stage-B mechanics) · teacher-KL anchor **rev38_5M_15g** (λ=0.05, the strong baseline, NOT weak
>   BC) · **deb DOMINANT in pool** (`--pool-external-fraction 0.6`, ×8 workers) · NO rev38 pin (`--pool-pinned-fraction 0`) · gate2 ·
>   win_margin 0.5 · speed 0 · early_capture 0.3 · fire-ent 0.02 · LR 5e-5 · 128 envs · total 6M. Script:
>   `gpu_run_artifacts/handicap/run_remote_handicap_jarvis.sh` (GCP-L4 twin `..._gcp.sh` kept for reference).
> - **Why it beats the stageb2r "learned-an-exploit-of-a-proxy" objection:** deb is the REAL planner, dominant in the pool, and we
>   train ON the target with only a tapering head-start — so the win-gradient is against the actual wall, not a beatable surrogate.
> - **⭐ DECISIVE READ:** as the boost tapers 3→0, does **WR-vs-deb HOLD/climb + peel-rate↓ + planets@100↑** — or collapse back to ~0?
>   Collapse ⇒ the holding-under-peel wall is STRUCTURAL even with a real win-gradient (reward/pool can't fix it; the ceiling is the
>   architecture/sim). Hold/climb ⇒ the handicap curriculum is a viable path and we lengthen/repeat it.
> - **🔵 LIVE — Jarvis A100-80GB SPOT, machine `427065` @ `217.18.55.39`, 28 cores, ₹84/hr.** ⚠️ SPOT — `jl destroy 427065 --yes`
>   when done (sync-verify first). **SPS ~360–400** (full 6M ≈ 3.7hr; partial-taper read ~1.5M ≈ 55min). iter-12 healthy: EV 0.83–0.93,
>   KL ~0.01, estop 0, **il_kl bounded 0.35–0.78** (pulling toward rev38 as designed), **clip peaked 0.34 then settling 0.29** = benign
>   teacher-swap adaptation ([[feedback_clipfrac_lowkl_benign]]), NOT divergence (KL low + EV stable + estop 0 → don't touch LR). deb
>   loaded ×8 confirmed. Watcher LIVE (controller, zach held-out gate2) → `gpu_run_artifacts/handicap/` (spot box backed up).
> - **⭐ BOX-MIGRATION LESSON:** first launched on the n2-standard-32 CPU eval box (deb-heavy "ideal for planners" reasoning) → **SPS 54**,
>   load only 12.7/32. Root cause: **`torch_env` is the GPU-vectorised env — a CPU-only box starves its single-threaded main loop; the
>   deb workers were NOT the bottleneck** (idle). Fix = a GPU box (A100 → 54→~370, ~7× + cheaper than the CPU box). L4 (8 vCPU) was also
>   wrong for deb @0.6 (planner-bound). **Rule: torch_env training ALWAYS wants a GPU box; the CPU box is for EVAL panels only.**
> - **EVAL workflow:** the 32-core **`orbit-wars-eval`** GCP box (n2-standard-32, asia-south1-a) is KEPT as the milestone evaluator
>   (`run_eval_batch.sh`, ~12min/panel; the controller only does slow local zach). Run deb + Ajay + zach panels at the taper milestones
>   (~1.5M / 3M / 4M(boost→0) / 6M). ⚠️ billing ~$1.55/hr — DELETE when the run concludes (`gcloud compute instances delete orbit-wars-eval --zone=asia-south1-a`).
> - **✅ Fire-weighted BC (the pos_weight fix) — CONCLUDED:** retraining the snowball BC with fire-head class-imbalance `pos_weight=6`
>   raised **planets@50 3.9→6.7** (WON@100 hit 9) — the passivity diagnosis was correct + fixable. BUT WR stayed **1/16 vs zach** —
>   expansion improved, winning didn't. Confirms the predicted split: pos_weight cures the FIRING/expansion deficit, but the HOLDING wall
>   is separate and BC doesn't touch it. Another data point that holding, not expansion, is the blocker.
> - **⭐⭐⭐ ROOT DIAGNOSIS — WHY we can't hold = we get OUT-MASSED (force concentration), never learned to act on it.**
>   Lost-capture autopsy (`orbit_wars_rl/hold_autopsy.py` + new eval `hold-loss` line: out-massed/abandoned/too-late % +
>   garr@cap→loss vs enemy-inbound). 16-game autopsies vs deb, gate2:
>   - We do NOT abandon caps — we garrison + reinforce them (handicap_500k 42→59, rev38 23→43) — but get **OUT-MASSED
>     100%** of losses (rev38 98%): deb arrives ~96 vs our ~59 (peel 0.97). **UNIVERSAL wall, NOT a regression** — rev38
>     (strongest, 967.6 LB) out-massed 98% too → no agent in our lineage ever learned force concentration. Reinforcing
>     HARDER fails (handicap_500k garrisons 2× rev38, still 100%) = throwing good ships after bad into a visibly-lost contest.
>   - **How deb does it** (`opponents/orbit_lite/planner_core.py`): `capture_floor` = ceil(defenders forward-projected to
>     ARRIVAL + production + overhead), then `_plan_regroup` aggregates MULTIPLE source planets into one strike. We fire
>     **per-planet, reactively, sized to CURRENT defense** → systematically under-send, can't muster 94 from one planet.
>   - **⭐ DO WE SEE IT? YES** — `features.py` planet ch13 (`enemy_pressure`) + pairwise feat 14 (`enemy_contest`) = incoming
>     enemy ships per planet. The 96 is in the obs; policy doesn't act on it. **→ fix is a training SIGNAL, not a feature/
>     restart** (self-play never priced concentration → win-starvation). firew6 BC = **0/256 vs deb AND Ajay** (expansion
>     w/o concentration = zero traction). **LEVER SEARCH reframed around DECISIVE ENGAGEMENT / CONCENTRATION** (watch eval
>     `hold-loss out-massed%` ↓). 500k handicap baseline (boost~2.6): deb 2.0% / ajay 1.6% / zach 61.7%. Memory:
>     `project_force_concentration_wall`.
>
> **⭐⭐⭐⭐ SESSION 2026-06-14 EVE — EVAL SWEEP: STAGE B IS ≤ rev38 BASELINE (sobering; read FIRST).**
> Ran a full ladder sweep on a GCP CPU eval box (n2-standard-32, see infra below). **THE HEADLINE: rev38_5M — our
> existing 967.6 LB record (Phase 1, NO reinforce/comets/game-phase) — MATCHES OR BEATS stageb3 on EVERY rung,** in
> the comet-faithful env, gate-agnostic (rev38 has no reinforce so gate is ignored):
>
> | opponent | **rev38_5M** | stageb3 best | stageb3 @13.6M |
> |---|---|---|---|
> | zach | **95.7%** | 86.3% (6.29M) | ~83% |
> | h12  | **56.2%** | 55.5% (9.96/11M) | 48.0% |
> | h14  | **22.3%** | 23.0% (9.96M) | 21.9% |
> | Ajay | **6.2%**  | 5.5% (11.0M)  | 2.3% |
>
> **So all of Stage B — from-scratch + comets + game-phase + teacher-KL + ramped pool + gate2 + early_capture — has NOT
> beaten the checkpoint we already had on the board.** rev38 ties on h12/h14, EDGES on Ajay, and CRUSHES zach (95.7 vs
> 86 — Stage B traded ~10pp of general strength for nothing). ⚠️ Caveat: Ajay panel isn't LB-predictive and we have
> ZERO Stage-B LB submissions, so "worse on LB" isn't proven — but on every PANEL metric rev38 ≥ stageb3. **This is the
> central problem: the bar is "beat rev38," and nothing in Stage B clears it.**
>
> **⭐ THE WALL READ (first-ever stageb3-vs-Ajay, full run):** 6.29→13.63M = 1.6/0.8/0.8/3.1/**5.5(peak@11.0M)**/4.3/1.2/2.3.
> Peak-then-fall, best ~11.0M (5.5%) — same wall as all of Phase 2 (rev38 2.7-6.2, rev53b 10.9). NOT cracked. h12 trend:
> 51/38/41/55.5/55.5/52/48 (peak ~10-11M). **stageb3 best ckpt = ~11.0M.**
>
> **⭐ BEHAVIOR vs Ajay (WON-subset, the honest "are we learning winner play"):** opening cap/atk WON ~0.47-0.50 ≈ winner
> 0.51 ✅ and reinf ramp ≈ winner ✅ — BUT **planets@50 = 6-7 EVEN IN WINS** (winner 9) ❌ and **peel-rate 0.71-0.87 EVEN
> IN WINS** (winner 0.43) ❌. **⚠️ CORRECTS the earlier "peel→winner 0.43" read — that was vs ZACH (a non-peeler); vs the
> real peeler (Ajay) we don't hold at all.** Damning: our rare Ajay wins are achieved WITHOUT winner expansion/holding →
> easy-board flukes → no win-gradient for the skills we need. The wall = **can't hold captures vs a strong peeler (peel
> 0.7+) → planets@50 can't accumulate.** Holding-under-peel is THE bottleneck; it caps expansion.
>
> **⭐ DECISIVENESS / game length (your speed-coef question):** speed_coef IS **0.3** (not 0). vs Ajay WON games grind to
> **~320 steps** (n=4), LOST ~116, vs zach ~362; winners eliminate ~100-150. So even our WINS aren't decisive — we win by
> attrition, never snowball into a fast kill. BUT it reads as a **symptom of the expansion/holding root** (can't dominate
> → can't close), not an independent lever. ⚠️ Raising speed_coef has a landmine: Rev26 speed_coef=0.5 → ship-bin-0
> collapse (drained ships for fake-fast wins). Don't bribe decisiveness; fix expansion/holding and it falls out.
>
> **⭐ BEATABLE-PEELER evidence (stageb2r, h12-in-pool, gate3):** h12 WR CLIMBED over its life **25→28.5→43.4%** (1.05/3.15/
> 5.77M) — the beatable peeler in the pool DID improve the in-pool-opponent WR. BUT Ajay stayed **0→0.4→2.0%** (no transfer
> to the wall) and it turtled on zach (hollow ~58%). So a beatable peeler teaches you to beat THAT peeler, doesn't transfer
> up to the strong peeler, and risks turtling. Mixed — not a clean win for lever B as run (confounded: stageb2r also had
> 5 other deltas vs stageb3).
>
> **✅ lever A (loosen teacher-KL, stageb3lo) REFUTED + killed** (see PM block): planets@50 flat 7 + zach drift 84→76 →
> anchor is NOT the expansion cap.
>
> **🟢 DECISIONS / NEXT (open — user to steer):**
> 1. **Submit rev38_5M to the LB** for a ground-truth anchor (we have ZERO Stage-B submissions; rev38 is the strongest
>    panel agent we have). And/or submit stageb3 11.0M to see if comets/reinforce moved LB despite ~equal panels.
> 2. **Reckon with "Stage B ≤ rev38":** is the from-scratch reinforce/teacher-KL direction worth continuing, or return to
>    the rev38 lineage + ONE targeted lever? The bar is now explicit: beat rev38 (zach 95.7 / h12 56 / Ajay 6.2).
> 3. If continuing the expansion attack: the bottleneck is **holding-under-peel** (not raw expansion) — lever B (beatable
>    peeler, but cleaner one-delta) and/or lever C (reward that makes HOLDING captures pay), judged vs rev38 + the wall.
> 4. **GCP eval box `orbit-wars-eval` (n2-standard-32)** — ⚠️ SUPERSEDED by the 2026-06-15 block: now KEPT as the handicap run's
>    milestone evaluator (delete only when that run concludes). Recipe: memory `reference_gcp_eval_box`. Reusable runner on box:
>    `run_eval_batch.sh <jobs> [P]` (per-job gate via 4th field). Logs → `gpu_run_artifacts/eval_box/`.
> 5. ⚠️ SUPERSEDED: a training box IS now running (handicap, Jarvis — see 2026-06-15 block). Process note: read metrics from full panel
>    logs, don't re-probe ([[feedback_read_full_eval_not_reprobe]]).

> **⭐⭐⭐ SESSION 2026-06-14 PM — COMPASS + DIRECTION FINDINGS (read this first; reframes everything below).**
> The big shift this session: **stop steering by ladder/held-out WR — it's a guardrail, NOT the compass.** Beating our
> own synthetic h-ladder (detuned Producers) ≠ progress (docs already say panel WR isn't LB-predictive). The metrics
> that matter are **winner-referenced + opponent-agnostic** (Isaiah/top-player ref): **planets@50→9 · open<50 cap/atk
> →0.51 · peel-rate→0.43 · reinf<50→0.29**. Only an **LB submission** is ground truth (we have ZERO this whole phase).
> - **THE WALL: planets@50 = 6 in BOTH live runs** (winner 9), unbroken — the master expand-and-hold metric. Neither
>   run improves it; both converge to **sub-winner equilibria** that survive the pool:
>   - **stageb3 = CHURN failure.** Mediocre opening (open<50 0.41 vs 0.51), can't hold (peel-rate WON 0.49→0.66,
>     degrades vs harder opp), grinds planets back by recapture, **floods reinforce at scale (13+ = 0.78** vs winner
>     0.34-0.61). Its high ladder WR (zach 86 / h10 85 / h12 51 / h14 21 @ 6.29M, gate2) is **volume/churn vs weak
>     planners, NOT skill** — mechanics are opponent-INVARIANT (same 0.41/6/0.20 regardless of WR). fire-rate/hoard are
>     actually ~winner (corrected: NOT a sprayer).
>   - **stageb2r = TURTLE/HOARD failure.** Near-winner opening (0.48) + holding (peel 0.38 ✅), no reinforce-flood
>     (13+ = 0.51 ✅), BUT **over-garrisons (ships/planet@50 = 39 vs 22), under-launches (launch_rate 0.026 vs 0.036),
>     under-expands (planets@50=6).** It hoards the army instead of spending it to expand.
>   - **stageb2r mechanics TREND (0.5M→2.1M, `mechanics_trend.py` vs zach) = NOT improving, REGRESSING on the disease:**
>     planets@50 flat 6 · garr_frac@50 ↑ 0.68→0.77 · ships/planet@50 ↑ 44→51 · launch_rate ↓ 0.022→0.014 · reinf<50
>     flat ~0.13. **The stable ~58% zach WR is HOLLOW** — it's getting better at *turtling* vs a weak opp, not at
>     winner-style expansion. (Corrects the earlier "WR holding = relaunch working" read — WR-hold masked a hoard drift.)
> - **✅ ship0 panic hypothesis RESOLVED = artifact.** Eval ship0 (deployed/argmax) is ~0% in EVERY phase × outcome,
>   even in LOST/late games (mean ships 32-80 throughout). Training-diag ship0 0.11 was a **self-play + sampling** tail
>   (argmax rarely picks bin0); deployed play is clean. min-ship-bin worry is moot. (New eval metric: ship0-by-phase ×
>   won/lost in `eval.py`.)
> - **✅ reinf<50 is opponent-ROBUST** (stageb3: 0.19-0.22 across a 21→86% WR swing) → trustworthy compass axis. Late
>   reinforce (>100) is LENGTH-confounded (shrinks vs harder opp via shorter games) — trust `<50`, discount `>100`.
> - **✅ DECISION: adopt GATE2 going forward** (supersedes the gate3 "locked design"). Clean evidence: gate2 → reinforce
>   @2-planets = 0.09 ≈ **winner 0.10**; gate3 BANS it (0.01). gate3 was stricter than winners actually play, and the
>   gate2-vs-gate3 A/B that locked gate3 was *inconclusive*. ⚠️ It's a winner-faithful **standardization, NOT the fix**
>   for planets@50. **✅ WIRED 2026-06-14:** `GATE=3→2` default flipped in `run_stageb_jarvis.sh` (GATE:-2),
>   `run_watchers.sh` REINFORCE_MASKS default, `export_agent.py` (both the fn default + CLI `default=2`), + CLAUDE.md
>   note. Applies to the NEXT build; the live stageb3 run stays gate3 (started pre-flip). Pass `GATE=3`/`REINFORCE_MASKS`
>   to reproduce the old design.
> - **⭐ THE DIRECTION VERDICT / NEXT LEVER:** the current setup (self-play + pool + teacher-KL) finds turtle/churn
>   local optima that *survive* the pool — there's no pressure to expand because turtling/churning still wins vs the
>   pool. The lever must make **UNDER-EXPANSION LOSE**: a pool opponent (or reward) that punishes a turtle/churner by
>   **out-expanding** it. `--expansion-coef 0.03` is far too weak. (Ties to win-starvation / pool-seed-RL pivot.)
> - **⭐⭐ EXPANSION PROBE (2026-06-14 PM, `orbit_wars_rl/expansion_probe.py`) — localizes the wall to the MID-GAME
>   SNOWBALL + indicts the ANCHOR.** Two diagnostics, both on eval's own `game_conversion` (apples-to-apples w/ winner
>   ref 2/6/9/10):
>   - **BC baseline (`bc_snowball_15global.pt` vs zach, 16 seeds, gate2):** planets@50 = **3.9 all / 3.0 won**, WR 1/16.
>     The warmstart + teacher-KL anchor (ratchet → own-best ≈6) is rooted in an agent that expands to only ~3 and loses
>     94% vs zach. stageb3's il_kl 0.55 = the anchor pulling HARD toward this under-expander. **The anchor is plausibly
>     the expansion CEILING** (PPO pushes expansion up to ~6, the KL drags it back toward ~3 → equilibrium at 6).
>   - **Same-seed us-vs-Ajay-vs-deb (stageb3 13.6M, identical board+opponent, 3 seeds):** the gap is the **step ~50→130
>     breakout**, not the opening or the economy. seed0 (all win): planners 2→5→11 eliminate @103; we 2→3→8 grind to
>     500. seed8526 (we LOSE): we hit 6 @step65 and **STALL 6/6/6 through step100** while planners break 6→11→14 and
>     eliminate @147. We are NOT ship-starved (g50=22, inflight50=85) — we **churn** (15 caps, keep 8) and hoard the army
>     instead of pressing NEW planets mid-game. Ajay≡deb trajectories (both orbit_lite planners, deterministic). Logs:
>     `gpu_run_artifacts/expansion_probe/`.
>   - **Reframed wall:** "planets@50=6" is a **mid-game snowball failure** — we stop acquiring planets right when planners
>     accelerate. Fix must produce the 6→12 breakout in steps 50-130 (redirect the parked/in-flight army at NEW planets),
>     AND/OR stop the anchor from capping expansion at the ~3-6 it's rooted in. **Candidate single-delta experiments (pick
>     one, see below):** (A) teacher-KL ablation/loosen (il-lambda 0.05→~0.01) — decisive cheap test of "anchor = ceiling"
>     (risk: drift returns); (B) snowball/out-expander pool opp harder than h12 (h12 alone didn't break stageb2r) so
>     under-expansion LOSES; (C) a NON-count expansion reward targeting NEW-planet captures in steps 50-130 (avoid the
>     rev48/49 count-based carpet-bomb trap). DECISION PENDING — user to pick the first delta.
> - **W&B is OFF** (`--wandb` never passed) → built **`orbit_wars_rl/plot_train_log.py`** (smoothed trend plots from the
>   text logs, retroactive) + **`orbit_wars_rl/mechanics_trend.py`** (winner-ref conversion-mechanics trend from per-ckpt
>   eval logs). Watcher CSV gained **`reinf_step_early`**. 🟡 OPEN: wire `garr_frac@50`/`ships_per_planet@50`/`launch_rate`
>   into the watcher CSV so the hoard trend is captured live (now re-parsed from logs).
> - **Live runs — UPDATED 2026-06-14 PM:** **stageb2r KILLED** (machine 426650/.108 DESTROYED — it was REGRESSING on the
>   turtle/hoard disease while burning spot $; best-effort harvest of the 5.77M ckpt+log under `gpu_run_artifacts/stageb2r/`).
>   **stageb3 REPURPOSED → stageb3lo** (machine 426674/.92): stageb3's 14M run KILLED (walled at planets@50=6 for 13M
>   steps = the established CONTROL); relaunched from its own **13.63M ckpt** as **stageb3lo** with the ONE delta
>   **il-lambda 0.05→0.01** (loosen the teacher-KL = the anchor=ceiling test). SAME teacher (il-ref still
>   bc_snowball_15global), same pool (resumed 28-member mature pool, ramp 0 = full immediately), LR 1e-4, gate2, deb-ext
>   0.10, early_capture 0.3 — all identical to stageb3. iter-1 = expected critic-warmup (EV 0.66<0.7, policy frozen, will
>   exit). **WATCH (THE test):** does planets@50 climb OFF 6 over the next 1-3M steps (anchor was the ceiling) AND does
>   drift return (held-out zach peak-then-fall)? Controller watcher LIVE on .92 (zach held-out, gate2 masks) →
>   `gpu_run_artifacts/stageb3lo/`. 13.63M resume ckpt also synced locally under `gpu_run_artifacts/stageb3/checkpoints/`.
>   ⚠️ spot — `jl destroy 426674 --yes` when done. Beatable-planner ladder (for option-B later): h10 44% · h12 28% · h14 0%.
> - **✅ stageb3lo (loosen teacher-KL, lever A) — CONCLUDED + REFUTED + KILLED 2026-06-14.** il-lambda 0.05→0.01 from
>   stageb3 13.6M; il_kl rose 0.55→2.07 (un-anchored as intended). Over 0.5→3.15M: **zach WR drifted DOWN 84→76**
>   (both seats) AND **planets@50 FLAT at 7** (full-panel zach — same as the stageb3 6.29M/9.96M base; winner 9). →
>   **drift with ZERO expansion payoff → anchor=ceiling REFUTED; the cap is the pool/reward dynamics, not the anchor.**
>   Box 426674 DESTROYED, 3.15M harvested to `gpu_run_artifacts/stageb3lo/`. (planets@50 read from the FULL panel logs,
>   not a re-probe — [[feedback_read_full_eval_not_reprobe]].) **NEXT lever = B (out-EXPANDER pool opp, not a peeler)
>   and/or C (bounded mid-game expansion reward), pending the stageb3-vs-Ajay wall batch on the GCP eval box.**
> **⭐ TWO CONCURRENT SPOT RUNS (2026-06-14) — both ⚠️ DESTROY when done (`jl destroy <id> --yes`):**
> - **stageb2 = PRIMARY** (machine `426650` @ `217.18.55.108`): re-anchor from 6.29M, has the controller watcher (zach
>   evals + sync). Details below. ⚠️ ship0 was creeping (0.14) — watch.
> - **stageb3 = "let it be"** (machine `426674` @ `217.18.55.92`, A100-40GB spot): **FROM-SCRATCH** BC warmstart +
>   **early-capture-coef 0.3** (always-on) + **gate2** + **deb-external 0.10** (rev38 pinned 0.267) + LR 0.0001 +
>   critic-warmup 0.7 + ramp ON 6M + no min-ship-bin. Tests: does early_capture give the from-scratch run the
>   expansion+commitment gradient (planets@50→9, self-punishes 1-ship probes) that phase-obs-alone didn't. Launch:
>   `RUN=stageb3 RESUME=IL_REF=orbit_wars_rl/seed_checkpoints/bc_snowball_15global.pt LR=0.0001 CRITIC_WARMUP_EV=0.7
>   GATE=2 POOL_EXTERNAL_FRAC=0.10 EARLY_CAPTURE_COEF=0.3 HEURISTIC_WORKERS=4 bash gpu_run_artifacts/run_stageb_jarvis.sh`.
>   iter-1 healthy (critic warmup, EV→, policy frozen). **Monitoring = SYNC-ONLY** (controller is single-run, on stageb2):
>   nohup loop PID in `gpu_run_artifacts/stageb3/.sync_pid` syncs /home/checkpoints+log every 300s → `gpu_run_artifacts/stageb3/`
>   (⚠️ kill that PID when stageb3 ends). NO auto-eval — for stageb3 zach reads, run manually with **gate2** masks
>   (`--reinforce-gate-min-planets 2 --reinforce-garrison-floor 0`) into `gpu_run_artifacts/stageb3/eval_logs/`.
>
> **⭐⭐ ACTIVE RUN — stageb2r (RE-ANCHOR from stageb2's 1M peak + deb→h12 swap, LAUNCHED 09:07 UTC 2026-06-14):**
> - **Why:** stageb2 (re-anchor from 6.29M, LR 2×, deb pool) showed a clean **peak-then-fall** on zach held-out
>   (524k→1M→1.5M→2M→2.62M = 46.9/**64.5**/49.6/41.4/31.2 — monotonic decline past the 1M peak) + diag ship0/meanshipbin
>   undercommit creep. Root-cause read: **β=0.05 anchor too loose under 2× LR** — il_kl plateaued ~0.26 (vs Stage A's
>   ~0.06 that HELD). So: re-anchor from the **1M peak**, **drop LR 0.0001→0.00005**, and swap the win-starving deb (0%)
>   for the **beatable planner h12** (28% vs 6.29M — real win-gradient). Old stageb2 KILLED; ckpts/CSV kept as history
>   under `gpu_run_artifacts/stageb2/`.
> - **Box:** same Jarvis spot @ `217.18.55.108`, IN2, **A100-PCIE-40GB**, 16 cores → 4 workers. is_spot=TRUE ⚠️ **DESTROY
>   when done:** `jl destroy <id> --yes` (`jl list` for id); spot data may not persist → watcher sync mandatory.
> - **Config:** resume+il-ref = **1M** (copied to `/home/seed_b2r_resume_1M.pt` — a path with NO sibling `pool_*.pt` so
>   the pool starts FRESH = rev38-pinned + h12-external only, NOT the resumed deb pool), **LR 0.00005**, critic-warmup OFF
>   (`CRITIC_WARMUP_EV=0`, trained critic), pool ramp OFF (`POOL_RAMP_STEPS=0`), `EXT_OPP=candidate_producer_h12.py`
>   (launch script now param'd, line 71), il-lambda 0.05 constant, gate3/floor0, game-phase, no early_capture, min-ship-bin
>   OFF (⚠️ ship0 unguarded — watch). iter-1 healthy: EV 0.794, clip 0.090 (≠0, not frozen), KL 0.008, **il_kl 0.009**
>   (≈0 at start, policy==teacher; THE thing to watch — should plateau LOW ~0.05-0.06, not 0.26), LR 5e-5, estop 0.
>   Pool log confirmed: members=3, "external loaded: candidate_producer_h12", NO deb, no "Pool resumed". Launch:
>   `RUN=stageb2r RESUME=$SEED IL_REF=$SEED LR=0.00005 GATE=3 CRITIC_WARMUP_EV=0 POOL_RAMP_STEPS=0 HEURISTIC_WORKERS=4
>   EXT_OPP=opponents/candidate_producer_h12.py nohup bash gpu_run_artifacts/run_stageb_jarvis.sh …`
> - **Watcher LIVE:** `run_watchers.sh start stageb2r jarvis 217.18.55.108 opponents/candidate_zach_public.py` → zach panel
>   per ckpt → `gpu_run_artifacts/stageb2r/eval_zach_public.csv`. Ratchet calc init'd (anchor=1M seed, WR bootstraps).
>   **READ:** does the LR-drop + tighter-anchor + h12 hold the held-out line (no peak-then-fall) AND does the h12
>   win-gradient teach anti-peel that transfers? Local trend plot: `plot_train_log.py <log> --eval <csv>`.
> - **(prior stageb2 block, superseded:)** resume 6.29M, LR 0.0001, deb pool, watcher → `gpu_run_artifacts/stageb2/`.
> - **RE-ANCHOR = MANUAL (no auto-ratchet).** Decision CALCULATOR (dry-run, executes nothing):
>   `orbit_wars_rl/.venv/bin/python gpu_run_artifacts/ratchet.py --run stageb2 --jarvis-ip 217.18.55.108 --opp zach_public check`
>   → prints rolling-3 / anchor_wr / threshold(+2pp) / step-gap / best-ckpt / HOLD-or-REANCHOR. Rule: rolling-3 ≥ anchor+2pp
>   AND ≥1M-step gap, never down (anchor_wr bootstraps from the first 3 evals ≈ the ~48% start).
> - **⚠️ MANUAL re-anchor relaunch — `check`'s printed command OMITS this run's env (would revert to script defaults:
>   LR 5e-5 / warmup-on / ramp-6M / RUN=stageb → WRONG + dir collision).** Use this instead (CK = best ckpt from `check`,
>   path under `/home/checkpoints/`): `ssh -i ~/.ssh/jarvis-labs-key root@217.18.55.108 "cd /home && pgrep -f
>   '[t]rain_torch.*--run-name stageb2'|xargs -r kill; sleep 5; RUN=stageb2 RESUME=<CK> IL_REF=<CK> LR=0.0001
>   CRITIC_WARMUP_EV=0 POOL_RAMP_STEPS=0 HEURISTIC_WORKERS=4 nohup bash gpu_run_artifacts/run_stageb_jarvis.sh
>   >/home/stageb2_reanchor.out 2>&1 </dev/null &"` then re-run `ratchet.py … init --anchor <CK>` to reset the calculator.
> - **GCP stageb run — ✅ KILLED + instance DELETED by user 2026-06-14 (no longer billing).** It was degrading (peak
>   6.29M, then ship0↑ / zach 48→32). Verdict it gave us: zach 6→48% but **0% vs Ajay** (transfer failed; same
>   strong-peeler wall). Next lever after stageb2/stageb3 = steepen peeler gradient (the beatable-planner ladder below).
>
> **⭐⭐ STRATEGY STATE + BEATABLE-PLANNER LADDER (2026-06-14 — the NEXT lever after stageb2/stageb3):**
> - **THE WALL (verdict from the GCP stageb run):** zach climbed 6%→48% but **0% vs Ajay** (6.29M 0/256, 9.44M 0/256,
>   8.91M 1/256 = one lucky econ board). Transfer to the hard opponent FAILED — same place as all of Phase 2. Opponent
>   ladder by our 6.29M panel WR: **zach ~48% (weak) · rev38 ~20% (beatable RL champion) · deb/Ajay ~0% (orbit_lite
>   PLANNERS = the wall).** So the gap is specifically **planner-style forward-sim PEELING**, not RL strength in general —
>   self-play + rev38 won't teach the counter. **Top-10 LB is >1500** (1153 = top-100; our record ~918) — long road.
> - **NEXT LEVER = a BEATABLE planner-peeler in the pool** — an opponent we win **30–60%** against (vs deb/Ajay's ~0%),
>   so there's a real win-gradient to LEARN the anti-peel/hold skill (win-starvation: you can't learn from a 0%-opponent).
> - **Kaggle access WORKS** (CLI + `~/.kaggle/kaggle.json`; `kaggle kernels list/pull --competition orbit-wars`). Pulled
>   ProducerLite kernels (pilkwang/sohaib = logistics-limited Producers) but **base64-packed self-contained kernels are
>   FRAGILE to extract** — the wrapper played random-level (43.8% vs random = broken integration). DROPPED. The bug was
>   **importlib-indirection loading** (also broke a first detune attempt); flat files loaded DIRECTLY by env.run are fine.
> - **✅ WORKING APPROACH = DETUNE the known-working `candidate_producer_1200.py`** (flat `cp` + `sed` the dataclass
>   `horizon: int = 18` default down; uses the shared `opponents/orbit_lite/`, loaded directly = no indirection).
>   **`opponents/candidate_producer_h14.py` + `candidate_producer_h10.py` (horizon 14/10) both CRUSH random 100%** =
>   genuine weaker planners. `horizon` = clean strength dial (max_offensive_targets/max_regroup_time are backup knobs).
> - **✅ FOUND IT + LADDER COMPLETE — `candidate_producer_h10.py` is the beatable planner.** Full horizon ladder
>   (32g vs 6.29M, gate3/floor0; random control alongside):
>
>   | horizon | vs random | vs 6.29M | verdict |
>   |---|---|---|---|
>   | h18 (full) | 100% | **0%** | too strong |
>   | h14 | 100% | **0%** | too strong (win-starvation) |
>   | **h12** | (val skipped — same detune path) | **28.12%** (9/32) | **🎯 THE PICK — harder, still safe** |
>   | h10 | 100% | **43.75%** (14/32) | beatable but near-matched (6.29M already ~masters it) |
>   | h4 | 94% (15/16) | **100%** (32/32) | too weak (trivial — teaches nothing) |
>
>   The cliff is **h12→h14** (28%→0%); usable beatable band = **h10–h12** (h13 ≈ the edge, untested). h10/h12 still
>   crush random 100% = REAL planners. **DECISION (2026-06-14, evidence-backed): pick h12, not h10.** 6.29M already beats
>   h10 at 44% (near-mastery) → a fixed h10 gets solved early and decays to weak-external self-play; h12's 28% gives real
>   anti-peel headroom that *stays* challenging (training only makes it easier → 28% is a rising floor, never starves).
>   Resulting pool spread is clean: rev38 ~20% + h12 ~28% + self ~50%. Other data: 6.29M vs rev38_5M = **19.9%**.
>   **→ NEXT: swap deb→`candidate_producer_h12.py` in the pool** (one delta, anchor stays 6.29M, no early_capture).
>   CURRICULUM ramp target = **h13** at the next re-anchor (skip h10 — agent's past it). Eval/export must keep it external.
> - **VALIDATE-FIRST rule (user, 2026-06-14, [[feedback_validate_kernel_before_screen]]):** any pulled/wrapped opponent
>   must ~crush random (≈100%) BEFORE screening vs our ckpt — control: producer_1200 + deb both 100% vs random; a broken
>   integration plays random-level and gives a FALSE screen.
> - **Once the beatable planner is picked:** add it to the next run's pool (alongside/replacing some deb), one delta.
>   **✅ Cleanup DONE:** broken `opponents/candidate_plite_sohaib.py` + `opponents/producerlite_sohaib/` removed.
>
> **🟢 8-HR MONITORING (⚠️ ALL STALE — GCP box `orbit-wars-training` is DELETED; goal elapsed 2026-06-14; superseded by stageb2/stageb3 above. Do NOT run any command in this block — kept for config reference only):**
> - **Box:** GCP L4 `orbit-wars-training` @ `34.14.144.22` zone `asia-south1-b`. Anchor (il-ref) = `bc_snowball_15global.pt`.
> - **CHECK each tick** (`gcloud compute ssh orbit-wars-training --zone=asia-south1-b -- "grep '^iter' \$(ls -t ~/orbit_wars_rl/train_gpu_phase1_stageb_*.log|head -1)|tail -5"` + `cat gpu_run_artifacts/stageb/eval_zach_public.csv`): clip, KL, EV, H_fire, il_kl, estop, SPS, ship0, reward, + zach-WR trend.
> - **ACT:** (a) **clip>0.25 sustained WITH KL↑/EV↓/estop>0** → halve LR (benign if KL low+EV stable+estop0 → DON'T halve, [[feedback_clipfrac_lowkl_benign]]). (b) **zach-WR sustained new best** (rolling-3 ≥ anchor+2pp, ≥1M-step gap) → re-anchor. (c) EV collapse / entropy floored / ship0 spike / box dead → investigate.
> - **RELAUNCH one-liners (box-side; LR now env-param):** latest ckpt = `CK=$(ls -t ~/orbit_wars_rl/checkpoints/torch_step_*stageb*.pt|head -1)`. **Halve LR:** `cd ~/orbit_wars_rl && pgrep -f '[t]rain_torch.*--run-name stageb'|xargs -r kill; sleep 3; RESUME=$CK IL_REF=<anchor> LR=0.000025 nohup bash gpu_run_artifacts/stageb_gcp/run_remote_stageb_gcp.sh >~/orbit_wars_rl/stageb_nohup.out 2>&1 </dev/null &`. **Re-anchor:** same but `RESUME=$CK IL_REF=$CK` (keep LR). After ANY relaunch the ramp resets (total_env_steps→0, expected) — re-verify iter-1 health + that the watcher still syncs (run-name unchanged so no new watcher needed). **⚠️ DELETE box when 8hr done if user not continuing.**
> - **Baseline @ iter50 (~1.64M):** clip 0.093, KL 0.012, EV 0.84, H_fire 0.17, il_kl 0.276 (plateauing), estop 0, SPS 556 (deb ramping in), reward +. HEALTHY — no action.
> **⭐⭐ STAGE B STATE (read first if resuming):**
> - **🔵 Box: GCP L4 `orbit-wars-training` @ `34.14.144.22`, zone `asia-south1-b`, SSH alias
>   `orbit-wars-training.asia-south1-b.orbit-wars-rl`, ~$1.13/hr.** ⚠️ **DELETE (not stop) when done:**
>   `gcloud compute instances delete orbit-wars-training --zone=asia-south1-b` ([[feedback_gcp_instance_cleanup]]).
>   (Jarvis `426481` DESTROYED 2026-06-14 — budget out.) **iter 1-3 HEALTHY: SPS ~880** (the ramp keeps deb≈0 early →
>   pure GPU-fast self-play, so the feared L4 CPU-bound ~191 SPS only bites in the back half), EV climbing 0.001→0.185
>   (critic warmup, policy frozen = KL/clip/H_fire 0 BY DESIGN until EV≥0.7, NOT the frozen-BC bug). Pool=3 (rev38 pin +
>   deb ext + self), ramp flags confirmed in argv. **✅ BC-warmstart PRE-FLIGHT PASSED** (iter 28 @917k: critic warmup
>   exited EV 0.762, `clip 0.054≠0` + `H_fire 0.178` not floored → policy exploring, not frozen; il_kl growing 0.08→0.17
>   = anchor engaged; KL ~0.016, estop 0). **Watcher LIVE** (controller, gcp) — **held-out = ZACH** (`eval_zach_public.csv`)
>   for EARLY progress signal (BC starts 6% vs zach; **Ajay is too hard early — 0.0 at 500k/1M — so it can't gauge early
>   progress; switch the held-out back to Ajay ~8M as the guardrail**: re-run `run_watchers.sh start stageb gcp <alias>
>   opponents/candidate_ajay_1200.py`). Monitor: `gcloud compute ssh orbit-wars-training --zone=asia-south1-b -- "grep
>   '^iter' ~/orbit_wars_rl/train_gpu_phase1_stageb_*.log | tail"`. **LR = 5e-5 (HELD, not 1e-4):** the low post-warmup
>   `clip ~0.046` was a TRANSIENT — clip rose 0.041→0.063 over iters 25-30 as PPO accelerated out of the frozen warmup
>   (entropy 0.16→0.20 too). Lesson #8's 1e-4-for-fresh-BC predates the critic-warmup+teacher-KL (which deliberately
>   suppress early clip) → not a direct precedent. **Fallback: if clip PLATEAUS <0.05 over 1-2M with no zach-WR progress,
>   restart from the current ckpt with LR 1e-4** (cheap, critic already warmed). Stop-all: `run_watchers.sh stop` + delete.
>   Decision: **manual re-anchors** (ratchet is Jarvis-hardcoded; not adapting it — re-anchor by hand on a sustained
>   held-out new-best per `gpu_run_artifacts/ratchet.py` semantics: rolling-mean-of-3 ≥ anchor+2pp, ≥1M steps gap).
> - **✅ DONE prior session:** (1) gate A/B inconclusive → **GATE=3** (locked design). (2) **rev38 pin 11→15-global
>   crash FIXED** → `seed_checkpoints/rev38_5M_15g.pt` (pre-flight MUST load every pin into the run's model). (3)
>   **CRITIC WARMUP built + validated** — `--critic-warmup-ev` freezes trunk+policy, trains ONLY the value head until
>   EV≥thresh; frozen trunk plateaus ~0.75 → default **0.7**; cap-exits to PPO at 30 rollouts. Off by default.
> - **✅ DONE this session (2026-06-14):**
>   1. **OPPONENT-DIFFICULTY RAMP — BUILT + UNIT-TESTED.** `OpponentPool.sample(external_fraction=, pinned_fraction=)`
>      now does the 3-way split (peeler / pinned-RL / PFSP-over-ORGANIC-non-pinned-selves), pulling rev38 OUT of PFSP
>      into its own ramped slice (PFSP would up-sample what we lose to = backwards). `pinned_fraction=None` preserves
>      legacy exactly. `train_torch` ramps both 0→target over `--pool-hard-ramp-steps` (keys off local total_env_steps),
>      with `pool_opp=None`→self-play fallback for the true-zero start. New flags `--pool-pinned-fraction` /
>      `--pool-hard-ramp-steps`. 5 tests in `orbit_wars_rl/tests/test_pool_ramp.py` (run via `.venv/bin/python` — no
>      pytest; has a `__main__` runner). At full ramp 0.267 of the 0.75 pool slice ⇒ **0.20 of TOTAL games each** for
>      rev38+deb (self-play 0.60 = 0.25 current-mirror + 0.35 organic snapshots). **⚠️ RAMP RESETS on each ratchet
>      re-anchor** (total_env_steps restarts at 0 on resume, like every schedule here) — consistent with "lengthen the
>      ramp at the next re-anchor"; raise `POOL_RAMP_STEPS` on a relaunch to extend. **OPEN (user decision):** if ramp
>      continuity across re-anchors is wanted instead, add a `--pool-ramp-start-step` offset the ratchet passes.
>   2. **BC step-0 matchup VALIDATED (local, 32 games, gate3/floor0): `bc_snowball_15global.pt` is 0.00% vs BOTH
>      rev38 AND deb** (both lose by elimination; planets@50=3 vs winner 9). Confirms the ramp is essential — zero
>      win-gradient at full strength = pure win-starvation. (The old `/home/bc_vs_*.log` on the destroyed box never
>      finished; superseded by this local run.)
>   3. **GCP L4 LAUNCH ARTIFACTS PREPARED:** `gpu_run_artifacts/stageb_gcp/run_remote_stageb_gcp.sh` (GCP twin of the
>      Jarvis script — same config; `cd ~/orbit_wars_rl`, seed_checkpoints repo-root-relative, `--heuristic-workers 4`,
>      ramp wired). Watcher controller already supports `gcp`; eval restores `game_phase_features` from the ckpt
>      (15-global panels load cleanly, no extra flag). `run_stageb_jarvis.sh` also ramp-wired (kept for A100 reuse).
> - **🟢 THE LAUNCH (ready — sequence):**
>   1. `bash gpu_run_artifacts/launch_gpu_gcp.sh --run stageb` (creates L4, syncs code, installs env; default zone
>      asia-south1-b — try asia-south1-c / europe-west4-a if stockout). Note the instance name + zone + SSH alias.
>   2. scp the two ckpts (launch script EXCLUDES seed_checkpoints/):
>      `gcloud compute scp seed_checkpoints/{bc_snowball_15global.pt,rev38_5M_15g.pt} <inst>:~/orbit_wars_rl/seed_checkpoints/ --zone=<z>`
>   3. start training: `gcloud compute ssh <inst> --zone=<z> -- 'cd ~/orbit_wars_rl && nohup bash gpu_run_artifacts/stageb_gcp/run_remote_stageb_gcp.sh >/dev/null 2>&1 &'`
>   4. watcher: `bash gpu_run_artifacts/run_watchers.sh start stageb gcp <inst>.<zone>.orbit-wars-rl`
>   5. PRE-FLIGHT (docs/phase3.md §8): watch clip_frac≠0 + entropy not floored in first ckpts; iter-1 EV/KL healthy.
> - **🔴 DEFERRED GAP — ratchet is Jarvis-hardcoded** (`~/.ssh/jarvis-labs-key`, `root@ip`, `/home/checkpoints`,
>   `run_stageb_jarvis.sh`). NOT blocking launch (first launch + watcher are manual; re-anchors only fire hours in on a
>   sustained +2pp best). **Before the first re-anchor is due, parameterize `ratchet.py` for GCP** (SSH cmd/user/host/
>   ckpt-dir/launch-script) OR do the first re-anchor manually. Stop-all: `run_watchers.sh stop` +
>   `gcloud compute instances delete <inst> --zone=<z>`.
>
> (Prior gate-A/B + ratchet-build LIVE STATE retained in the STAGE B PRE-FLIGHT section below for reference.)

Foundations all
GREEN: comet PHYSICS+FEATURES ✅, game-phase features ✅ (both parity-clean), **Stage A ✅ PASSED** (fixed teacher-KL
β=0.05 held the held-out band — see CONCLUDED block below), **closed-loop fidelity ✅ PASSED** (rev38 + deb FAITHFUL,
rev53b dropped → torch_env is a trustworthy selection signal; Stage-B pool = rev38 + deb; `docs/phase3.md` §9,
[[project_train_eval_sim_gap]]). **Next = the Stage B pre-flight + launch** — see the "STAGE B PRE-FLIGHT" section
immediately below. Lineage context: comet fix was the root blocker (train/eval sim gap = missing comets, now
byte-faithful); win-starvation finding ([[feedback_win_starvation]]) → the pool/selection-signal pivot; VDN/per-slot
concluded (standard arch).

---

## ✅ CONCLUDED — p3stageA (Phase 3 Stage A: teacher-KL anti-cycling anchor) — PASSED, box destroyed 2026-06-13

- **✅ VERDICT: PASS — the fixed teacher-KL anchor (β=0.05) held the held-out band.** 14-pt held-out Ajay trend
  over ~4.5M→10.3M cumulative oscillated **3.5–5.9% (mean ~5.1), FLAT — no peak-then-fall** (the un-anchored p2rev5
  control's signature was peak 5.9@4M → lower band; it did NOT appear). The one 3.5 dip @8.2M was noise (recovered
  to 5.1/5.9/5.1/5.5). Canaries stayed green: il_kl plateaued ~0.05–0.06 (bounded drift), clip ~0.19→0.24 (never→0,
  KL low/estop 0 = benign), entropy stable. **Carried forward = the validated KNOB (β=0.05 damps drift), NOT a
  checkpoint** (Stage B is from-scratch for the 15-global feature dim). Two leaderboard leaders (vkhydras, Billy
  Bradley) independently confirm a fixed anchor plateaus at its level — so this hold IS the success, not a stall.
  Box `426301` DESTROYED (billing stopped), watcher stopped, all ckpts+log harvested to `gpu_run_artifacts/p3stageA/`.
  **Next: Stage B** (ratchet v1 + BC-warmstart pre-flight + from-scratch run with comets ✅ + game-phase ✅ features,
  pool = rev38 + deb). Original live block kept below for config reference.

- **MIGRATED off GCP L4 (2026-06-13):** the L4 (`cap-probe-93160`) ran the SAME config at only **~191 SPS** (deb
  @0.25 is CPU-bound on 8 vCPU / 4 workers). Moved to a **Jarvis A100 80GB SPOT** for the 28-core CPU throughput.
  GCP box DELETED at 360k steps (below its first 500k ckpt — nothing harvested, no result lost).
- **Jarvis A100-80GB SPOT, machine id `426301`, IP `217.18.55.161`, region IN2 (Noida), 28 cores, ~₹84/hr.
  ⚠️ SPOT — DESTROY, never pause (preemption may lose data; the sync watcher below is mandatory).**
- **DESTROY:** `jl destroy 426301 --yes` (from the launch box; needs venv + `JL_API_KEY` + key — JARVIS_RUNBOOK §Auth).
- **Run = Phase 3 Stage A** (`docs/phase3.md` §5): resume **p2rev5 4M** + ONE delta = a CONSTANT self-anchored
  teacher-KL (`--il-lambda 0.05 --il-ref seed_checkpoints/p2rev5_4M.pt --il-decay-frac 100`). Else identical to
  p2rev5 AND the GCP p3stageA run (deb pool @0.25, gate3/floor0, early_capture off, LR 5e-5, gae 0.99,
  **256 envs**/128 rollout/**32 mb**/ppo-2). **CLEAN MIGRATION — num-envs stays 256 (NOT 512)** to preserve the
  single-delta comparison vs the 256-env p2rev5 control (a 2x batch would confound the anti-drift read); speedup
  is purely the A100's cores → **`--heuristic-workers 4→8`**. mb=32 (teacher's extra frozen forward fits 80GB
  trivially — uses only 13.4GB). Script: `gpu_run_artifacts/p3stageA/run_remote_p3stageA_jarvis.sh`. Comet-faithful engine.
- **iter 1-4 healthy (2026-06-13):** SPS ramping 465→644 (~600 steady, **3x the L4**), EV 0.69→0.90, KL ~0.015,
  clip ~0.22 (NOT→0), estop 0, **il_kl 0.004→0.006 / il_coef 0.050** (anchor active + constant; il_kl≈0 at iter1
  since policy==teacher, growing as it drifts). GPU 13.4/80GB, util 22% (CPU/deb-bound, as expected).
- **WATCH (THE hypothesis):** held-out **Ajay** WR HOLDS past the ~4-5M band instead of peak-then-fall (un-anchored
  p2rev5 control: peak 5.9%@4M → ~4.5% band). CANARIES: `clip_frac` must NOT→0 (anchor too strong/frozen → lower
  il-lambda); `il_kl` should grow then plateau (anchor working); entropy stable. KILL/RETUNE read: held-out still
  peaks-then-falls = il-lambda too weak (raise next run); clip→0 = too strong.
- **Watcher (controller):** `bash gpu_run_artifacts/run_watchers.sh start p3stageA jarvis 217.18.55.161`
  → sync + held-out Ajay full panel per ckpt (masks gate3/floor0/no-forward, match training) →
  `gpu_run_artifacts/p3stageA/`. First eval = step ~4.5M (500k after the 4M resume).
- Monitor: `ssh -i ~/.ssh/jarvis-labs-key root@217.18.55.161 "grep '^iter' /home/train_gpu_phase1_p3stageA_*.log | tail"`.

---

## 🟢 STAGE B PRE-FLIGHT & LAUNCH PLAN (the from-scratch ratcheted-teacher run, set 2026-06-13)

The from-scratch run (`docs/phase3.md` §5 Stage B). Foundations done: Stage A proved the anchor (β=0.05), fidelity
gate passed (pool = rev38 + deb), feature set ready (comets ✅ + game-phase ✅). Remaining = three coupled pre-flight
items, then the launch. **One open arch decision (#2) is answered by a free probe during the BC rebuild.**

> **⭐ LIVE STATE (2026-06-13 cont. — read first if resuming):**
> - **✅ BC REBUILD DONE (#1) + CAPACITY DECIDED (#2): keep entity_dim 96.** Rebuilt the 15-global dataset
>   (`orbit_wars_rl/snowball_bc_15g.pkl`, 15410 samples, reinforce_share 0.53) and trained both arms on the A100.
>   **96 vs 128 ≈ tied** (val_loss 3.90 vs 3.86 within noise; 96 EDGES target fidelity top1 0.28 vs 0.26) → no
>   capacity benefit, **keep 96**. **Stage-B warmstart = `seed_checkpoints/bc_snowball_15global.pt`** (96-dim,
>   `global_proj (96,15)`, `game_phase_features:True`, synced LOCAL + on box; d128 also synced, unused). Gate-fail
>   (top1 0.28 < 0.30) is the expected diverse-snowball ceiling → behavioral prior, PPO refines; **watch clip_frac at
>   launch** (#3). Feature audit fully CLOSED (comets + enemy_mass rejected, 15-global FINAL).
> - **🔵 GPU box LIVE:** Jarvis **A100-80GB SPOT, machine `426442`, IP `217.18.55.104`, region IN2, ₹84/hr.**
>   ⚠️ SPOT — `jl destroy 426442 --yes` when done (auth: JARVIS_RUNBOOK §Auth). Env installed (kaggle orbit_wars),
>   code synced, dataset + both BC ckpts on box at `/home/checkpoints/`.
> - **🔵 GATE=2-vs-3 A/B — LAUNCHED 2026-06-13 on box 426442 (both arms parallel, healthy).** Matched pair from
>   p2rev5 4M, single delta = `--reinforce-gate-min-planets 2` (run-name `gate2`) vs `3` (`gate3`), comet-fixed engine,
>   deb pool @0.25, clean p2rev5 config (NO sufficient-commit), 4 heuristic-workers each (parallel on 28 cores, ~386
>   SPS/arm, GPU 45/80GB). Script `gpu_run_artifacts/gate_ab/run_gate_ab_jarvis.sh <2|3>` (on box `/home/`). iter-1
>   healthy both (EV 0.69, KL<0.013, clip<0.13 NOT→0, estop 0, deb loaded, early_capture 0). **Watcher LIVE +
>   FIXED 2026-06-13** — re-launch as
>   **`MATCH=gate EVAL_GATE_FROM_RUNNAME=1 run_watchers.sh start gate_ab jarvis 217.18.55.104`** (NOT the bare
>   `start gate_ab …`: the umbrella name `gate_ab` matches NEITHER arm's filenames `gate2`/`gate3`, so the original
>   watcher synced 0 files). The controller now takes two new envs: `MATCH` = the filename-substring token (separate
>   from the folder/run label; `MATCH=gate` prefix-matches both arms into `gpu_run_artifacts/gate_ab/`), and
>   `EVAL_GATE_FROM_RUNNAME=1` = **per-arm eval mask** (each ckpt is auto-evaled with its OWN
>   `--reinforce-gate-min-planets`, parsed from the `gate2`/`gate3` token → eval matches training **automatically**;
>   the old manual-gate2-re-eval caveat is GONE). Verified end-to-end: both arms' logs + ckpts syncing, and the live
>   gate2 524288 panel ran with `--reinforce-gate-min-planets 2`. **READ:** two-sided canary — gate2 peel-rate /
>   planets@50→100 hold ↑ WITHOUT planets@50 regressing (the passivity-crutch failure); plus reinf@2 (split in eval)
>   should show gate2 actually reinforces at 2 planets. First ckpts landed ~500k (gate2 first). Monitor:
>   `ssh … "grep '^iter' /home/train_gpu_phase1_gate{2,3}_*.log | tail"`.
> - **✅ RATCHET v1 controller (#4) — BUILT 2026-06-13** (`gpu_run_artifacts/ratchet.py` + parameterized launch
>   `gpu_run_artifacts/run_stageb_jarvis.sh`). Reads the watcher's held-out Ajay CSV; on a GENUINE sustained new-best
>   it kills + relaunches the Stage B arm resuming from the new-best ckpt with `--il-ref` swapped to it (FULL AUTO,
>   user-chosen). Noise guards (the heart of it — held-out is a 256-game panel, SE ~1.4pp@5%): re-anchor only when the
>   **rolling mean of the trailing 3 evals** beats the anchor by a **2.0pp margin** AND ≥1M steps since the last anchor;
>   NEVER ratchets down. Validated: HOLDS on the p3stageA flat-noise fixture (3.5–5.9), RE-ANCHORS on a sustained climb,
>   rejects single +4pp peaks. Kill uses the bracket-trick pgrep scoped to `--run-name stageb` (SSH self-kill safe);
>   relaunch is nohup-detached (survives disconnect). The FIRST launch (#5) + watcher start stay manual; the ratchet
>   manages re-anchors only. Usage in `run_stageb_jarvis.sh` header. **Remaining before launch:** #3 BC-warmstart
>   pre-flight (entropy/critic canaries at first ckpts) + set GATE to the gate2-vs-gate3 winner.

1. ✅ **DONE — NEW 15-global BC REBUILD (2026-06-13).** Rebuilt + trained, keep 96-dim (see LIVE STATE above). Code:
   `bc.py --game-phase-features/--entity-dim/--device/--batch-size` + `_save_bc_checkpoint` saves `game_phase_features`;
   `build_snowball_bc.py --game-phase-features` (+ fixed a dual-`features`-module bug). Original spec retained below.
   The latest
   BC `seed_checkpoints/bc_snowball_pairwise15.pt` is pairwise-15 (current arch ✓) but **11-global** (`global_proj
   (96,11)`, no `game_phase_features` in config, dated Jun 11 → pre-comet-fix) → can't init the 15-global Stage B
   model. **To rebuild:** add `--game-phase-features` to `bc.py` + the snowball dataset builder (set
   `cfg.model.game_phase_features` + `global_feature_dim=15` + `features.set_game_phase_features(True)`; **`bc.py`
   `_save_bc_checkpoint` MUST save `game_phase_features` in the ckpt config — it currently omits it**), re-fetch the
   snowball replays, **REBUILD the dataset** (BC `--samples` pickles store PRE-EXTRACTED tensors → re-running bc.py on
   the old pickle won't upgrade the dim; must rebuild via the builder, which re-extracts), retrain → `bc_snowball_15global.pt`.
   Comets corrected automatically by re-extraction. Runs LOCALLY (bc.py = supervised on static tensors, no VecTorchEnv
   rollout → no Mac-CPU segfault; builder uses kaggle_env).
2. ✅ **DONE — CAPACITY PROBE: keep 96-dim** (96 vs 128 tied on the A100, 96 edges target fidelity — see LIVE STATE).
   Original rationale below. 🟡 Current net
   = `entity_dim 96 / 4 heads / 3 layers / mlp_exp 3` ≈ **391K params**, constant all project. **No capacity signal**
   (EV ~0.85–0.90, BC fits teachers) — failures are DYNAMICS (drift/signal), not fitting; bigger = more drift surface +
   a Stage-B confound. BUT from-scratch = the free moment to change arch, and GPU is IDLE (Stage A 22% util, CPU/deb-
   bound) so bigger is ~free on SPS. **Plan: train the new BC at 96-dim AND ~128-dim (or layers 3→4), compare val loss +
   imitation fidelity.** Meaningfully better fit → scale Stage B; ~same → keep 96 (default). Evidence-gated, clean A/B.
3. 🟢 **BC-warmstart pre-flight (Billy Bradley, `docs/phase3.md` §8).** (a) entropy collapsed in IL → can't explore:
   check `clip_frac≠0` / entropy not floored early; lever `--entropy-coef-*`. (b) trained policy + UNTRAINED critic →
   "unlearning" before critic catches up — CONFIRMED `bc.py` is policy-only (no value head). Cushion: the teacher-KL
   anchor (to the BC self) resists noisy-advantage unlearning; extra lever `--with-warmup` (low early LR); canary = low
   early EV.
4. ✅ **RATCHET v1 — BUILT 2026-06-13** (`gpu_run_artifacts/ratchet.py`, full-auto per user). Watcher computes held-out
   WR per ckpt → on a GENUINE sustained new-best (rolling-mean-of-3 ≥ anchor + 2.0pp margin, ≥1M steps since last
   anchor, never ratchet-down) it kills + relaunches `run_stageb_jarvis.sh` resuming from the new-best with `--il-ref`
   swapped to it (β CONSTANT 0.05 via `--il-decay-frac 100`). Validated (HOLDS on p3stageA noise, RE-ANCHORS on a
   climb, rejects single peaks). v2 (in-process reload) only if v1 works. See the LIVE STATE block above for usage.
5. 🟢 **THE LAUNCH (after 1–4).** From-scratch resume `bc_snowball_15global.pt` + `--game-phase-features` + `--il-lambda
   0.05` (Stage-A-validated) ratcheted + pool **rev38 + deb** + `--il-decay-frac 100` (no decay), comet-faithful engine,
   on Jarvis A100 spot (clean migration recipe in the CONCLUDED block above). ⭐ Falsifiable sub-exp: does phase-as-
   OBSERVATION let us RETIRE the time-scheduled shaping (first_strike t<50 / early_capture)? WATCH: held-out WR CLIMBS
   and HOLDS past 2M (ratchet rising), il_kl bounded, **all-env** planets@50 moves off 6 (NOT the WON-subset, which is
   survivorship — see Stage A analysis).

**Code landed this session (game-phase features, all gated/off-by-default):** `config.game_phase_features`,
`features.game_phase_channels`/`set_game_phase_features`, `torch_env` vectorized mirror + `VecTorchEnv(game_phase_features=)`,
`train_torch --game-phase-features` (sets dim 15 + saves flag via `ppo.state_dict`), `eval.load_checkpoint` restore +
flag, `export_agent` bake. Parity probe `feature_parity_gamephase_probe.py` CLEAN. **Fixed latent export bug:**
`export_agent` never patched `global_feature_dim` from weights (any dim change would've crashed export).

---

## 🟢 FROM-SCRATCH FEATURE SET + RL recipe (for the next from-scratch run, 2026-06-13)

The next from-scratch run rides on the faithful (comet) sim. Bundle ALL feature/observation changes here —
they change the model input dim, so they MUST land together in one from-scratch run (existing ckpts won't load),
and the cost of adding them is free *now* (we're going from-scratch anyway) and expensive later (retrofit = restart).

### ✅ DONE — comet features (path-aware, train/eval/export parity)
`is_comet` populated + comet slots OVERLOAD the orbital channels (no model-dim change on their own): feat 7=1,
feat 10/11 = comet PATH position +5 steps, feat 9 = normalized steps-to-departure. Pairwise treats comets
non-orbiting. Identical branch in `torch_env.get_features` + `features.extract_features` (kaggle obs exposes
`observation.comets`=paths+path_index → eval/export parity automatic). `to_legacy_obs` now surfaces comets (+ fixed
a pre-existing id-0 `initial_planets` collision). Regression test: `orbit_wars_rl/feature_parity_comet_probe.py`
(CLEAN). Physics fidelity tests still pass. Full write-up: `docs/training.md` Current State.

**✅ COMET-ENGAGEMENT AUDIT — CONCLUDED 2026-06-13 (no feature, no mask, no lever; input set complete).** Asked: is a
comet-collision-risk INPUT CHANNEL needed (the last restart-forcing candidate)? Measured exposure with p2rev5 4M
self-play (256 games, torch_env+comets, exact swept-collision labels via a throwaway diagnostic, since reverted) +
89 snowball winner replays. (1) **Motion is faithfully passed** — comets aren't approximated circular; we index the
engine's own sampled elliptical path (`observation.comets`), feeding current pos + a +5 path point. (2) **Collision is
conditional combat, not the sun's instant-kill** — a fleet whose swept path hits a comet's swept path enters combat
(capture if it out-ships, annihilated if not); faithful in BOTH engines. (3) **Captures are economically ~useless**
(user insight, confirmed): at expiry the comet is dropped from `planets` WITH its ships (`orbit_wars.py:415`) — parked
ships evaporate unless evacuated. Top players DO evacuate: own a comet 7.55% of steps, fire FROM comets 2.14% of
launches (capture→relay before expiry, 41/89 games); **our agent under-relays (0.55%)** and over-targets (fires AT
comets 3.18% vs winners 1.78%). (4) **Exposure is tiny & DELIBERATE** — total comet collisions 1.15% of launches
(78% capture/22% annihilate); deliberate targeting 3.18% >> any accidental crossing → policy/value, not a missing
input (the agent already sees `is_comet` + steps-to-departure + comet is a legal fire source). (5) **A comet-target
veto A/B** (p2rev5 vs p2rev5, one seat forbidden comet targets, 256 games): veto-seat **53.3%** — within ±6% noise
(seat asym 62/44.5 dwarfs it) → **a wash.** **Verdict:** no channel (no restart), no mask, no lever; the BC clone
seeds reasonable comet behavior (capture→relay is in the replays). The aimer-not-leading-comets refinement
(features.py:476) is a no-restart, low-value option, NOT pursued. **The Stage B input set (15-global + comets +
game-phase) is COMPLETE — nothing in the comet/feature audit forces a wider input or a different restart.**

**✅ PRODUCER-v2 `enemy_mass` FEATURE — INVESTIGATED & REJECTED 2026-06-13 (kept 15-global; do NOT rebuild to 16).**
Producer's author released v2 (`kaggle: slawekbiel/the-producer-v2`) citing one improvement: a `β·ρ(eta)·enemy_mass`
term, where `enemy_mass` = distance-decayed enemy GARRISON reachable to a target (`cheap_enemy_pressure`:
`Σ_{enemy s} ships[s]·(1−d(s,t)/(speed[s]·H))₊`), ρ = a flight-time ramp. The v1↔v2 code diff is LITERALLY just the
three `reinforce_*` knobs → this term is the whole delta. **We genuinely lack it** — our pairwise models only the
target's OWN growth (`ships_at_arrival`/`roi`) + in-flight enemy FLEETS (`enemy_contest` feat 14), i.e. exactly
Producer-v1's "opponents do-nothing" model; no reachable-garrison term. A full sweep of BOTH producers (offense
selection/sizing + defensive regroup + flow scorer) found `enemy_mass` is the ONLY observable we lack — everything
else maps to our features or is the value-head's job. **BUT three replay validations (89 snowball winners, target
resolved via `_find_target_planet_index`, enemy_mass ported w/ self-exclusion) ALL contradicted the feature's use:**
(1) **offense selection** — winners do NOT avoid high-mass targets (chosen/avail median 0.85 ≈ losers); (2) **offense
sizing** — winners' oversize is FLAT vs enemy_mass (1.74→1.78→1.66 across mass terciles); it's LOSERS who over-feed
high-mass targets (1.64→1.71→1.86) and lose; (3) **defense** — winners reinforce their LOW-mass REAR planets
(reinforced-planet mass 29 vs un-reinforced 47, ratio 0.61), the INVERSE of Producer's "reinforce the threatened
front" regroup (fits our bucket-brigade/rear-staging picture). **Unifying conclusion:** `enemy_mass` is load-bearing
for Producer's HEURISTIC PLANNER (patches its static-opponent flow scorer) but is NOT how the top RL agents we emulate
play — and an RL value head learns reactive reinforcement IMPLICITLY from self-play vs reactive opponents (it has the
raw enemy positions/ships). Adding it risks biasing toward the LOSER pattern (over-feeding contested targets). **The
path to beating Producer is a strong RL policy (Stage B), not importing its heuristic feature. Decision: 15-global is
FINAL.** Harness: `/tmp/enemy_mass_validate.py`; kernels at `/tmp/producer_v{1,2}/code.py`.

### ✅ GAME-PHASE features — BUILT + VALIDATED 2026-06-13 (Stage B feature set; off by default, opt-in)
> **✅ WEIGHT + USAGE VERIFIED ON STAGEB3 13.6M (2026-06-14).** Asked: are the 4 phase channels getting real weight /
> being used, or dead like rev38's zero-padded features? (1) **Weight = ALIVE.** `global_proj` AND `mode_proj` weight
> phase cols 11-14 at **0.91-1.01× the orig-11-globals mean** — identical scale, stable from BC warmstart (~1.0)
> through 13M PPO steps (~0.92-1.00). Opposite of rev38 (those <0.09 vs orig 0.8-1.4). BC warmstart gave real gradient
> from iter 1, as designed. (2) **Used = YES but MODEST effect.** Flipping ONLY the phase one-hot on a fixed board
> (sensitivity probe, inline): target-dist shifts L1 0.11-0.22 (~5-11% of mass moves), fire-prob barely moves in abs
> (up to ~50% rel per-slot late-game). So the channel is read + conditions behavior (NOT inert) — but the agent isn't
> *dramatically* re-strategizing by phase (cf. Isaiah). Targeting head is the phase-sensitive part; fire-rate isn't.
> Caveat: flipping phase vs an informative board is partly OOD + board/step-scalar(col1) already encode progress →
> one-hot's MARGINAL effect is redundancy-bounded. **Verdict: plumbing healthy — phase features are NOT where Phase 3
> is broken; but the modest phase-conditioning matches the planets@50=6 wall (the "retire shaping via phase-obs"
> hypothesis is only partially realized — wired in, not yet the expansion lever).**
**DONE:** `--game-phase-features` appends 4 global channels (11→15): a 3-way phase one-hot (early<50 / mid50-100 /
late>=100, the `<50` boundary deliberately = the first_strike/early_capture shaping window for the retire-shaping
test) + normalized steps-to-next-comet-spawn. Single source of truth `features.game_phase_channels`; vectorized
mirror in `torch_env.get_features`; **parity probe `feature_parity_gamephase_probe.py` CLEAN (0.00e+00 across phase
+ comet-spawn boundaries)**. Gated by `cfg.model.game_phase_features` (default OFF → every 11-global ckpt loads
unchanged); round-trips train→ckpt(`ppo.state_dict`)→`load_checkpoint`→eval→`export_agent` (exported 15-global agent
runs in real kaggle env). **Fixed a latent export bug found en route:** `export_agent` never patched
`global_feature_dim` from the ckpt weights (any dim change would've crashed export — now patches it). ⭐ Falsifiable
sub-experiment for Stage B: does phase-as-OBSERVATION let us retire the time-scheduled shaping (first_strike t<50 /
early_capture exp-decay)? Original rationale below.

**The gap:** the agent's ENTIRE temporal sense is ONE scalar — global feat 1 = `clip(step/500,0,1)`. No global
game-phase signal at all. Everything else temporal is entity-local (fleet ETAs, comet expiry, 5-turn orbit pred).
Isaiah added a discrete game-phase embedding (turn//40) + day-night cycle embedding and "the network quickly
developed dramatically different behaviors during beginning/middle/end game ... a crucial part of its success."
A scalar forces the net to manufacture sharp nonlinear thresholds; a bucketed/one-hot phase gives near-orthogonal
per-phase codes cheaply. Our architecture projects continuous globals via Linear → **one-hot phase channels are the
natural fit** (no new embedding layer). All parity-safe (computable from `step`, `angular_velocity`, constant spawn
steps). Ranked, add the top two:
1. 🟢 **Game-phase one-hot** (early/mid/late, or finer buckets aligned to our `planets@16/32/50/100` analysis
   windows). The direct Isaiah analog. We MEASURE behavior per phase but never tell the agent which phase it's in.
2. 🟢 **Comet-cycle phase** = normalized steps-to-next-comet-spawn (+ steps-since-last). The periodic day-night
   analog; complements the per-comet expiry feature (rhythm vs entity). Spawns at constant 50/150/250/350/450.
3. ⏸ **Endgame / time-to-hard-cap** — fold into the phase one-hot's last bucket (weak: games usually end by
   elimination well before the 500 cap, so a dedicated channel is low-value).
4. ⏸ **Orbital phase** (`sin/cos(angular_velocity·step)`) — PARK; per-planet positions + pairwise arrival already
   expose the geometry locally; a global orbital phase is mostly redundant.
**⭐ Hypothesis to test (phase-as-observation vs phase-as-reward-schedule):** we currently encode "phase matters"
through hand-tuned time-DEPENDENT REWARD shaping (`early_capture` exp-decay, `first_strike` t<50) — fragile and
Nash-eaten. Isaiah gives phase as an OBSERVATION and lets the policy self-condition. So the clean falsifiable bundle:
add the phase one-hot and test whether it lets the agent learn the opening aggression we've been BRIBING it into,
allowing us to RETIRE the time-scheduled shaping. Restraint: top-two only — more temporal channels risk the policy
leaning on the clock instead of the board.

### 🟡 RL-recipe borrows from the same writeup (bigger bets, after the feature run)
> **⭐ Now a concrete staged plan: `docs/phase3.md`** (ratcheted teacher-KL + league, on the faithful sim). The
> machinery already exists (`ppo.py` `frozen_il_model`/`_il_kl_penalty`, `--il-ref`/`--il-lambda`); the new build is
> the RATCHET (re-anchor to rising held-out-best; current schedule wrongly decays to 0). Summary bullets kept below.
- **⭐ Frozen-teacher KL as an ANTI-CYCLING STABILIZER (not imitation).** Isaiah: a KL loss to a frozen teacher
  "helped to stabilize behavior and prevent strategic cycles — both of which plague a pure self-play setup." That is
  our #1 documented failure (improve-then-degrade / Nash reforms ~2M, [[feedback_selfplay_collapse_metrics]]). We
  only ever used IL/BC as a *small imitation aux* (bc-coef 0.05 = "too weak"). The teacher-KL is a CORE stabilizer
  whose job is to keep self-play from wandering off the strong attractor — NOT to copy a style. Pair it with the
  from-scratch run: keep a frozen strong checkpoint as the KL anchor. Highest-leverage / lowest-risk borrow (machinery
  we already have). **Frame explicitly as stabilizer to avoid conflation with the dead bc-coef-0.05 aux.**
- **Shape-then-sparse with teacher distillation.** Isaiah: ~20M steps of dense shaping to bootstrap, then DROP shaping
  and train on pure ±1 terminal, with the previous (shaped) net as KL teacher. Maybe the missing bridge — every
  shaping term we anneal gets eaten by the Nash; the teacher is what carries behavior through the un-shaping.
- **Diverse opponent LEAGUE for resilience when behind** — Isaiah hit our EXACT "agent gives up when losing" failure
  and prescribed "a league of more diverse opponents." Independent corroboration of the pool-seed-RL pivot +
  win-starvation finding. (Already queued; see Pool levers below.)
- **Test-time 180° board-symmetry augmentation** (average action probs over the rotation-symmetric board) — near-free
  eval regularizer/boost we don't use. Greedy decode at eval = our threshold-decode (already do). Validated: joint
  action over all units + masking-to-−inf-with-no-op-escape = our arch + [[feedback_veto_mask_removes_not_teaches]].
- ⏸ **IMPALA/UPGO algorithm switch** — PARK (big bet); UPGO is a sparse-reward self-imitation helper but switching
  off PPO is not a quick lever. Network-size distillation curriculum also parked.
Full source: `writeups/Toad Brigade's Approach - Deep Reinforcement Learning  Kaggle.md` (Isaiah Pressman, Lux AI
2021 winner — very likely the SAME "Isaiah" we profile as #1 Orbit Wars → our top-player "style" replays are likely
another self-play RL agent's emergent behavior, not human heuristics; [[project_isaiah_style_profile]]).

---

## 🔴 CURRENT PROBLEMS & CONCLUSIONS (2026-06-12 — VALIDATED, read before any run)

Localized with honest, **outcome-split** metrics on **p2rev3 0.5M** (our best vs deb: 3.9%) — full
256 panels, correct masks. The dashboard rebuild (below) is the session's big win; prior diagnoses
were made on confounded numbers.

**THE GAP:** beat weak opponents (zach 87–89%), crushed by strong **peelers** (deb 2–4%, ajay ~0%).

**TWO PROBLEMS, TWO WINDOWS** (deb, p2rev3 0.5M; `ref:winner` open 0.51 / mid 0.47):
- **P-OPEN (secondary) — opening conversion weak.** `open<50 cap/atk` **0.29 lost / 0.43 won** vs
  winner **0.51** — below even in our wins. Mechanism: fragments fired *under* the target's defense
  (6 ships at a 43-ship neutral → annihilated). Real, but we still win 87% vs zach with it, so not
  the thing that loses to deb.
- **P-HOLD (DOMINANT) — mid-game peel.** The opening ramp is **identical won/lost through @50**
  (2/4/6); divergence is **@50→100**: won **6→9/10**, lost **6→4** (→0.9, eliminated). And mid-game
  *conversion is FINE* — `mid50-100 cap/atk` **0.47 = winner 0.47**. We convert mid-game well but
  **can't hold**: churn 16, **peel-rate WON 0.61** vs winner **0.43**. Holding, not conversion, is the
  mid-game failure, and it degrades *specifically vs a peeler* (zach WON peel-rate 0.31; deb 0.61).

**REINFORCEMENT (the hold mechanism) — validated against winner replays:**
- **`garrison_floor=0` is CORRECT — keep it.** Winners hard-commit: send 100% of source (median),
  frac-left mean ~7–8%, **no absolute or fractional reserve** (even 30+ ship sources drain; "keep 20%"
  is NOT a winner habit). `floor=10` blocked 86–91% of winner-style reinforces → it was the wrong default.
- **`--reinforce-forward-only` ≈ FINE — not the blocker (corrected).** Our direction **71% fwd / 20%
  rear** ≈ winner **57/26** (centroid). The mask uses *nearest-enemy*, which doesn't force
  centroid-100%-forward, so we already pull back ~20%. Low priority to change.
- **The real reinforce gap = RATE/TIMING, back-loaded.** reinf-by-step `<50:0.05 · 50-100:0.19 ·
  >100:0.42` vs winner `0.29 · 0.41 · 0.31` — too little early/mid, too much late.
- **Quantity ≠ the fix.** floor 10→0 (p2rev4) raised reinforce (mid 0.11→0.19) but peel did NOT drop
  (0.61→0.64), churn got WORSE (16→23). More reinforce fed the churn war, not holding ⇒ we reinforce
  **incorrectly** (wrong target/timing), not too little.

**THE VALIDATED LEVER → next run: a PEELER (deb-class) in the EXTERNAL pool.** The collapse happens
*only* vs opponents that peel (zach can't, deb does). Self-play + fast hammers never punish bad holding,
so holding-under-pressure is never learned. A peeler in the pool makes it a learned skill. Other reinforce
knobs (floor/forward-only) are already right; the missing ingredient is *training pressure*.

**RULED OUT (do NOT re-chase):** spray/`fire_frac` (empire-size + win/loss confound; won-game ≈ Isaiah
0.17 — [[feedback_firefrac_winloss_confound]]); ship-head credit (joint AND per-slot both undercommit —
user-confirmed); reward-shaping fire/target (fire-tax→fire=0 Nash, target-value→rev49 carpet-bomb,
target-KL→rev54 crater); early_capture as the collapse cause (was OFF in p2rev3; eval doesn't use the
reward; collapse is opponent-specific so it can't be a reward schedule); floor / forward-only as the
reinforce fix (floor correct, forward-only ≈ fine).

**METRICS REBUILT (trustworthy now — `eval.py game_conversion`/`_fmt_conversion`, docs/metrics.md):**
all conversion metrics **outcome-split (WON/LOST)** — `planets@`, `cap/atk` by phase
(**open<50 / mid50-100 / whole**), **peel-rate** (renamed from lost-cap), `fire_frac`/`launch_rate`;
**reinf by step** (<50/50-100/>100), **by empire size**, and **direction** (fwd/rear vs centroid);
`underkill` flagged NON-discriminating (winners ~0.40 too); `ref:winner` convention (refs prefixed
`ref:`, our values bare). See [[feedback_opening_capatk_discriminator]].

---

## ✅ SUPERSEDED (historical) — was "NEXT RUN: SUFFICIENT-COMMIT MASK"

> This plan ran as **p2rev6 and FAILED** (planets@50 flat at 6 — a veto removes fragments but doesn't teach
> concentration). Then defense_coef (p2rev7) + early_capture (p2rev8) also failed → all four levers left
> `planets@50=6` → the win-starvation finding → **pivot to pool-seed-RL (p2rev9 LIVE)**. Kept below for history.

### original plan (set 2026-06-12 from the held-out panel trend)

p2rev5's 12-ckpt held-out panel trend (live-instance block below) shows deb-in-pool did NOT move the two
outcome levers: **opening conversion `opnWON` flat ~0.38 vs winner 0.51**, and `peelW` ~0.6 vs 0.43.
Reinforce *direction/rate* are already winner-like vs Ajay (fwd 58%, rf>100 0.24) ⇒ **threat head is NOT
indicated.** The dominant unmoved gap is the OPENING → build the **sufficient-commit mask** (P-OPEN lever
#2 below): veto an attack launch whose `ships ≤ target's current defense` (exact for neutrals, the opening
targets) → fragments impossible by construction, forces concentration. Structural mask, NO reward tax
(avoids the rev41–45 fire=0 Nash). Resume p2rev5's best ckpt; ONE delta = the mask. Secondary: arrest the
early-reinforce decline (rf<50 0.15→0.09 vs winner 0.29). Threat head → parked (revisit only if a later
held-out panel shows reinforce *mis-targeting* vs a real opponent, which 6M did not).

### ~~peeler-in-pool (deb)~~ — DONE as p2rev5 (concluding; verdict above)

- **From-scratch (snowball-BC warmstart)** so holding-under-peeling is learned natively.
- **ONE delta vs the p2rev4 config: add debatreya_1300 to the external pool** (the peeler).
- **Keep validated defaults:** `garrison_floor 0`, `gate 3`, `floor=0` masks at eval/export.
  **DROP `--reinforce-forward-only` for the peeler run** (decided 2026-06-12): the run's whole point is
  learning to *hold* a peeled rear planet, and forward-only blocks pulling ships *back* to defend it
  (the "secondary blocker", 23%→42% of intents). Eval/export masks must match → also no forward-only.
- Selection: held-out WR vs a DIFFERENT strong opp (ajay/producer) — deb is now in training, so it's
  no longer the clean held-out yardstick. Watch `peel-rate WON` ↓ and `planets@50→100` stops collapsing.
- Open (decide at launch): whether to *also* turn on `early_capture` for P-OPEN (2nd delta) — or keep
  this run clean (deb-only) and address the opening separately.

### P-OPEN levers (opening under-commitment) — ranked

Disease: in the opening we fire **fragments *under* the target's defense** (replay: 6 ships at a 43-ship
neutral → annihilated) and **split forces** across several neutrals → each too small to capture →
under-expansion. Winners hard-commit (frac-sent 1.00, send everything at *one* target). Already ruled
out as an architecture bug (ship-head credit, joint AND per-slot, both undercommit) → it's **learned**.

**⭐ P-OPEN ↔ P-HOLD coupling (2026-06-12, user insight) — NOT universal, two regimes.** `planets@50 = 6`
is **invariant** across p2rev5/6/7 in our wins AND losses (winner 9). So: **(regime A) same 6-planet board,
holding alone decides** — won 6→10 vs lost 6→3 @50→100; here a pure holding fix (defense_coef) gets us far,
P-OPEN is NOT the blocker. **(regime B) opening gates holding** — a 6-planet empire is intrinsically harder to
hold than a 9-planet one (less depth to stage from, less production to *fund* reinforcement, peeler concentrates
on a smaller frontier, no margin for loss); here holding has a **ceiling set by the opening** — you can't hold
what you didn't expand to. **Implication:** holding (p2rev7) is a real lever, but its payoff plateaus once
regime-B dominates → **falsifiable read: if p2rev7's `peel-rate WON` improves then plateaus while `planets@50`
stays 6, that's the ceiling** → opening becomes the priority. The gap is **QUANTITY (6→9), not commitment** (the
sufficient-commit/commitment lever FAILED, #2) → revisit the quantity gradient (#3).

1. 🟢 **deb-in-pool may fix it for free (check FIRST).** deb is a strong *expander*, not just a peeler —
   it out-expands a fragment-dribbler (replay: 9 vs our 5 by step 48). So the peeler run punishes opening
   under-expansion *and* mid-game peeling with the same gradient. **Read `open<50 cap/atk` (WON) after the
   deb run before adding a dedicated opening lever** — it may already lift toward 0.51.
2. ❌ **Sufficient-commit MASK — TESTED & FAILED (p2rev6, CONCLUDED 2026-06-12).** `open<50 cap/atk WON` FLAT
   ~0.34 over all 9 ckpts, `planets@50` stuck at 6 → the veto removed fragments but the agent fired *less*, did
   NOT concentrate ([[feedback_veto_mask_removes_not_teaches]]) ⇒ **commitment was not the bottleneck; quantity
   is** (see #3). Relaxed 0.6 / neutrals-only is a low-priority remaining variant. Original spec kept below.
   Veto an attack launch whose
   `ships ≤ target's current defense × factor` → fragments impossible by construction, forces concentration
   (attack only a target you can actually take, else accumulate first). Structural like the reinforce
   masks — **no reward tax** (avoids the rev41–45 fire=0 Nash). **Aligns with measured winner behavior**
   (hard-commit). Training-only, internalized, eval/export parity like floor/forward-only.
   **Impl:** `--sufficient-commit-factor` (1.0 strict / 0.6 relaxed / 0 off) — post-decode veto in
   `torch_env.py _apply_actions` (alongside garrison_floor); parity in `action_mask.py`, `eval.py`,
   `export_agent.py` (bakes `_SUFFICIENT_COMMIT_FACTOR`); test `tests/test_sufficient_commit.py` (veto/fire/off,
   passing). Applies to BOTH self-play seats (near-no-op for strong opponents, who hard-commit). **Launch ready:**
   `gpu_run_artifacts/p2rev6/{run_remote,launch}_p2rev6_jarvis.sh` — resume p2rev5 best + ONE delta (factor 1.0),
   else identical to p2rev5 (deb pool, gate3/floor0/no-forward). ⚠️ update RESUME_CKPT to p2rev5's FINAL best
   after it concludes. Watch: `open<50 cap/atk` (WON) should lift toward 0.51; guard against forced-passive
   opening (if no neutral is takeable from the start garrison → relax to 0.6 or neutrals-only).
   Risks: (a) strict — blocks sub-threshold *softening* waves, but winners barely use those; (b) enemy
   planets reinforce in transit so "sufficient at launch" can fall short on arrival — **exact for neutrals**
   (the opening expansion targets, which don't regrow), approximate for enemy planets; (c) untested —
   masks have surprised us before. A relaxed variant (`ships > defense × 0.6`, or neutrals-only) is the
   fallback if it over-constrains.
3. 🟡 **early_capture / first_strike — REVISIT as the QUANTITY lever (reframed 2026-06-12), conditional.** Was
   parked as "rewards count not commitment" — but the sufficient-commit (commitment) lever FAILED and the data
   says **the gap IS count: `planets@50` 6 vs winner 9.** `early_capture 0.30` was literally the rev28 breakthrough
   that took the agent passive→expanding; the p2rev5/6/7 base runs it at **0** (no opening-expansion gradient at
   all). So re-introduce it as a dedicated **P-OPEN delta** (consider **always-on, not annealed**) with success =
   `planets@50` → 9, NOT as a holding lever. **TRIGGER (user, 2026-06-12): try this IF the opening doesn't improve**
   (i.e. p2rev7/holding plateaus against the regime-B ceiling above). Watch it doesn't regress to carpet-bomb
   (rev49) or 1-ship spray — pair with `--min-ship-bin`. The eventual winning config likely needs BOTH a 9-planet
   opening AND good holding, solved as separate clean deltas; opening arguably first since it *gates* regime B.
4. ⏸ **Capture-efficiency reward — graveyard.** Penalizing wasted/inefficient fires = a conditional
   fire-tax → passivity / fire=0 Nash. Do not. The mask (#2) is the non-tax way to encode the same intent.

### Discipline masks (training-only, NO input-dim change → addable to ANY run, no from-scratch restart)

1. 🟡 **Sun-blocking angle mask — DOWNGRADED 2026-06-13 (the prior audit's "NO signal" claim was WRONG).** The real game
   **destroys any fleet whose path crosses the sun** (`orbit_wars.py:607`, `point_to_segment_distance < SUN_RADIUS=10`).
   The earlier audit logged "the agent has no feature/mask for it" — **but it does: pairwise channel 4 = `sun_safe`**
   (`features.py:533`/`567` ↔ `torch_env.py:1052`, parity-clean), a per-(source,target) flag = "does the straight-line
   path clear the sun." So the agent **is aware** of sun-blocking; it just isn't hard-*masked* from it. That reframes
   this from a fill-the-void fix to a **discipline lever** (harden an existing signal, like forward-only) — much lower
   value than the audit implied. Still a no-restart mask if ever wanted; **not a priority.** (`sun_safe` uses the
   *current* target position, not the orbital-intercept arrival segment — a cheap no-restart refinement, also low value.)

### Reinforce-targeting levers

**⭐ FRAMING — the reinforce credit-assignment problem & how we "simplify" it (2026-06-13, read before any holding/reinforce lever).**
Reinforce earns **no immediate reward** (an attack captures → instant credit; a reinforce just moves ships, value =
"avoided a future loss / enabled a future push"). So its credit is **weak + counterfactual + diluted**, NOT merely
long-horizon (the horizon is ~10-30 steps; at γ=0.995 that's only ~0.86-0.95 discount — the killer is counterfactual
noise + GAE credit-dilution, not the discount). **The obvious simplification — reward holding directly — is a known
graveyard here:** `defense_coef` (p2rev7) FLOODED; in a symmetric self-play mirror a dense per-step holding reward IS
the flood pump, same family as fire-tax→fire=0 / target-value→carpet-bomb ([[feedback_selfplay_collapse_metrics]],
[[feedback_win_starvation]]). PBRS isn't a free pass either — `expansion_coef` is potential-based/telescoping and STILL
gets Nash-eaten; a "holding potential" is hard to design AND washes in a symmetric mirror. **So we do NOT solve the
credit assignment — Phase 3 ROUTES AROUND it three ways (this IS the simplification):** (1) **imitate** it — snowball
BC seeds the winner reinforce ramp so the agent starts doing it, not discovering it from a weak gradient; (2) **anchor
it terminally** — the peeler-in-pool makes bad holding actually LOSE games → converts the diffuse counterfactual into a
real terminal win/loss gradient GAE can propagate (the validated "fix the SIGNAL" lever); (3) **preserve** it —
teacher-KL stops self-play Nash from eroding the seeded behavior. **Recommendation: don't invent a new credit-assignment
mechanism — run Stage B and MEASURE whether reinforce-holding emerges (read the planets@50→100 hold + peel-rate
trend); the K-nearest mask (#1 below) is the structural CONTINGENCY** (it doesn't teach value, it SHRINKS the problem:
short rear→front hops → short transits, ships stay defensible, tiny per-hop horizon). **Guardrail from the enemy_mass
validation: PRESERVE rear-staging, do NOT force threatened-front reinforce** — winners reinforce their LOW-threat rear
(staging depth); pouring ships into high-threat contested planets was the LOSER pattern
([[feedback_heuristic_feature_not_rl_feature]]). A naive holding reward would push toward that loser pattern.

1. 🟡 **K-nearest own-target mask (NEW idea 2026-06-12) — structural "bucket-brigade" staging.** An own
   (reinforce) target is legal only if it is among the **K nearest** owned planets to the source (K=2–3);
   ships then cascade rear→front in short hops over successive turns instead of any one planet shipping far.
   Same structural family as `--reinforce-forward-only` / `--sufficient-commit-factor` — a legality **MASK,
   NO reward / NO target-ranking shaping** (biasing the target distribution via reward/KL = rev49 carpet-bomb /
   rev54 crater; the mask is the non-shaping way to encode "prefer near"). Motivation: (a) replay winners
   target by **distance rank ~0.3** (prefer near, NOT by production value); (b) short hops keep ships ON
   planets (defensible) vs long transits (idle/vulnerable); (c) **it sidesteps the reinforce credit-assignment
   void** — reinforce earns no reward at arrival (value is delayed+counterfactual: only "avoided a future
   loss"), so the agent can't efficiently *learn* the cascade is good; a mask just *makes* it stage in short
   forward hops. **Distinct from forward-only:** "nearest" does NOT force forward, so it still permits pulling
   ships BACK to defend a peeled rear planet (the exact behavior forward-only blocked → why we dropped it).
   Impl: slots next to the gate/forward block in `torch_env.py` (~line 636, the `allow_reinforce` target-mask
   branch) — per source slot, keep own-targets only among the K-smallest src→tgt distances; parity in
   `action_mask.py` / `eval.py` / `export_agent.py` like the other masks; test in `tests/`. Caution: real
   boards are scattered (not a clean front line) so K=1 is too rigid → start **K=2–3**. ⚠️ Risk per
   [[feedback_veto_mask_removes_not_teaches]]: banning far-reinforce may just make it reinforce *less*, not
   cascade — pair with a reason the cascade pays off. Sequencing: a single delta AFTER p2rev6 concludes.

2. 🟢 **Empire-gate threshold (`reinforce_gate_min_planets`) — UN-PARKED 2026-06-13: replay evidence weakens "gate=3
   is right"; now an evidence-backed single-delta A/B candidate.** The gate is **3** ("expand first"); a "felt logical"
   pick, never swept. **NEW replay analysis** (89 snowball winners, launches binned by empire size, target resolved via
   `bc._find_target_planet_index`; threat = any enemy fleet present):
   | owned | reinf% | expand% | attack% | reinf threat/none |
   |--:|--:|--:|--:|--:|
   | 1 | **0.7%** | 97.3 | 2.1 | 1/0 |
   | 2 | **10.1%** | 85.5 | 4.3 | **15/6** |
   | 3 | 18.7 | 75.3 | 6.0 | 28/0 |
   | 4–8 | 24→40 | 66→30 | 10→30 | ~all threat |
   **Two findings:** (a) at **1 planet winners essentially never reinforce (0.7%)** → banning reinforce@1 is right (both
   gate=2/3 do). (b) at **2 planets winners reinforce 10.1%, and ~71% of it is DEFENSIVE** (15 under-threat / 6 not) —
   **gate=3 bans this entirely.** So gate=3 contradicts a real, mostly-defensive winner behavior, and it's exactly the
   behavior tied to our **dominant P-HOLD/peel gap**: gate=3 forces us to expand off planet #2 when we should reinforce
   to *hold* it under threat. **Counter still stands** (don't just switch to 2): 85% of winner k=2 launches are still
   expansion, we under-expand (planets@50=6 vs 9), so gate=2 risks a **passivity crutch** (reinforce-not-expand when
   smallest) — the winner's 10% is affordable *because* they expand fast; we don't. **Can't be settled at inference**
   (the gate shapes LEARNED behavior — a gate=3-trained policy has no reinforce-at-2 to express; lowering the eval gate =
   untrained noise) → needs a **single-delta training A/B (gate=2 vs gate=3)**, no-restart/one-knob, off a stable resume
   (p2rev5 4M + gate=2), NOT confounded into Stage B's big changes. **Two-sided canary:** peel-rate / planets@50→100 hold
   must IMPROVE (the hoped gain) while planets@50 must NOT regress (the passivity-crutch failure). **DO FIRST regardless:**
   the metric-split — with gate=3, eval's "reinf@2-3" bin is diluted by gated 2s (true reinf@3 ≈ 2× reported); split the
   @2/@3 bins in `eval.py game_conversion` so reinf@3 reads cleanly (needed to read the A/B).

### Pool / anti-drift levers (training pressure, NOT reward)

1. 🔵 **pool-seed-RL + deb — LIVE as p2rev9 (2026-06-13).** Cross-eval validated the pins are winnable (p2rev5 4M
   beats rev38 27% / rev53b 37.5%, vs deb ~5%) → real win-gradient. Config: pins rev38+rev53b, deb 0.25→0.10, 35%
   self, clean reward (early_capture/defense OFF). See the LIVE block above. Original rationale below.
   **Motivation:** the improve-then-degrade pattern (held-out metric peaks ~500k–1M then drifts down) is the
   self-play **Nash reforming around a misaligned objective** (the documented "SSDR transient / Nash reforms
   ~2M" — [[feedback_selfplay_collapse_metrics]], [[project_phase2_reinforcement]]). The fix is MORE
   *asymmetric non-self* pressure so the degenerate behavior actually loses games. BUT you can't just crank the
   external fraction: deb/orbit_lite is CPU-slow (~11 ms/call, [[feedback_orbit_lite_is_slow]]) so SPS already
   654→~250 at just ~0.19 deb; pure-external also over-fits one **sim-gap archetype** ([[project_train_eval_sim_gap]])
   AND starves the win-signal (we beat deb ~4% → high-deb = nearly all-loss = no win-gradient + passivity
   collapse). **The throughput-viable version:** add **`--pool-seed-rl`** (GPU-fast, sim-gap-IMMUNE, diverse RL
   champions — same fast path as self-play) **+ keep deb as the peeler external.** Pin **rev38** (first-strike
   aggressor → opening pressure) + **rev53b** (selective) [± more]; **KEEP rev31/rev32b HELD-OUT** as the
   cross-eval forgetting detector (do NOT pin them). Optionally bump external **0.25→~0.4** but keep **~30–40%
   self-play** (the matched-difficulty auto-curriculum + winnable games — do NOT go pure-external). Both halves
   needed: pinned-RL = cheap diverse strong pressure; deb = the specific peeler that punishes bad holding.
   **BUILT + TESTED** (`opponent_pool.py add_pinned_rl`, `tests/test_pool_pinned.py`; the p2rev4 GCP script
   already wired `--pool-seed-rl rev38_5M.pt,rev53b_10M.pt`). **Read:** held-out trend PAST 2M — if it stops
   drifting (holds/climbs past the early peak) we changed the attractor (the rare durable-win signal); if it
   still peaks-then-falls, the pressure still isn't enough. Sequencing: a single composite "pool" delta off the
   p2rev5 4M base (or whichever base wins the p2rev6-vs-p2rev7 A/B).

---

## ✅ CONCLUDED — p2rev6 on Jarvis A100 (sufficient-commit mask DID NOT lift opening conversion; box destroyed 2026-06-12)

- **VERDICT (ended at 7.8M, box 425850 destroyed):** the sufficient-commit mask (factor 1.0) **failed to move
  its own target metric.** `open<50 cap/atk WON` was **FLAT ~0.33–0.35 across all 9 held-out Ajay checkpoints**
  (500k→6.8M; winner 0.51) — at or *below* the p2rev5 4M base (~0.40). WR flat-to-declining (5.5→1.6). **Why:**
  `planets@50` stayed **6** throughout (winner 9) — opening *expansion never sped up*. The veto removes fragment
  launches but the agent responded by firing *fewer*, NOT by *concentrating* → a veto-mask removes the bad
  behavior but doesn't supply the good one (aggregation). No degeneracy (fire_frac WON 0.17, garrison low).
  Best ckpt is noise-level, none beat the p2rev5 4M base → **future runs still resume p2rev5 4M.** 14 ckpts +
  log harvested to `gpu_run_artifacts/p2rev6/`. **Narrow remaining hope for the IDEA (new run, not this one):**
  relaxed factor **0.6 / neutrals-only** (1.0 may over-constrain, blocking softening waves). Lower priority than
  p2rev7 (defense_coef) + the K-nearest-mask idea. **DECISION RULE confirmed:** a delta's target metric flat over
  many checkpoints = it isn't working; end + harvest (don't burn GPU to 10M hoping a pinned metric unsticks).

- **Jarvis A100-80GB ON-DEMAND, machine id `425850`, IP `217.18.55.147`, 28 cores. ⚠️ ON-DEMAND (not
  spot) — DESTROY when done (it bills ~₹2x/hr while Running, even after the run auto-stops at 10M).**
  ⚠️ **The local `gpu_run_artifacts/p2rev6/{instance.txt,.watch_env,launch.out}` initially referenced a
  DEAD spot box (425837 / 217.18.55.30) — a prior spot launch that died; the on-demand box was re-created.
  Always reconcile with `jl list` before trusting those files. instance.txt now fixed to 425850/.147.**
- **DESTROY:** `jl destroy 425850 --yes` (after final ckpts sync; needs venv + key — see JARVIS_RUNBOOK §Auth).
- **Run = the SUFFICIENT-COMMIT MASK delta** (`docs/phase2.md` §P-OPEN #2): resume **p2rev5 4M**
  (`seed_checkpoints/torch_step_4194304_p2rev5_20260612_083540.pt` — p2rev5's held-out Ajay WR peak, 5.9%)
  + **ONE delta `--sufficient-commit-factor 1.0`** (veto attack launches whose ships ≤ target defense →
  fragments impossible, forces concentration). Everything else IDENTICAL to p2rev5: deb-only pool @0.25,
  gate3 / **garrison-floor 0** / **no forward-only**, early_capture OFF, speed 0.3, expansion 0.03,
  fire-entropy 0.005, LR 5e-5, gae-lambda 0.99, 256 envs / rollout 128 / ppo-epochs 2 / 8 heuristic workers.
  Script: `gpu_run_artifacts/p2rev6/run_remote_p2rev6_jarvis.sh` (cwd `/home`).
- **TARGETS the held-out panel's dominant unmoved gap: opening conversion** — `open<50 cap/atk (WON)`
  stuck ~0.38 vs winner 0.51 across all of p2rev5's 12 ckpts. WATCH: `open<50 cap/atk (WON)` should lift
  toward 0.51. GUARD: if the opening goes forced-passive (no neutral takeable from the start garrison),
  relax factor 1.0→0.6 (or neutrals-only). Threat head = PARKED (held-out panel showed reinforce already
  winner-like vs Ajay — not the indicated lever).
- **LAUNCH NOTES (2026-06-12, took over a half-finished setup):** (1) `kaggle_environments` was missing on
  the box (a pause/resume wiped conda site-packages while `/home` persisted) → reinstalled via
  `pip install -q kaggle-environments && bash setup/install_orbit_wars.sh`. (2) tmux does NOT survive SSH
  disconnect on this image + a `pkill -f train_torch.py` over SSH self-kills the shell → left duplicate runs;
  cleaned to ONE tmux-managed run. Verify single run: `pgrep -c -f "[t]rain_torch"` == 1. iter 1 healthy
  (EV 0.69, clip 0.095, KL 0.008, SPS ~600–755 once de-duped). See memory `feedback_ssh_pkill_selfkill`.
- **Watchers via the CONTROLLER** repointed to .147: `bash gpu_run_artifacts/run_watchers.sh start p2rev6
  jarvis 217.18.55.147` (tore down the stale .30 watchers). Held-out **Ajay** full panel per ckpt, masks
  **gate3 / floor 0 / NO forward-only** (match training) → `gpu_run_artifacts/p2rev6/eval_ajay_1200.csv` +
  `eval_logs/`. `… status` / `… stop` as usual. First eval = step 500k.
- Monitor: `ssh -i ~/.ssh/jarvis-labs-key root@217.18.55.147 "grep '^iter' /home/train_gpu_phase1_p2rev6_*.log | tail"`.

---

## ✅ CONCLUDED — p2rev9 comet-faithful relaunch (KILLED @1.28M 2026-06-13, box DELETED, no billing)

- **⭐ This was the comet-fixed validation relaunch** (resume p2rev5 4M + POOL pivot on the FIXED torch_env).
  **Verdict: the fix works live** — iter 1 ran past the step-50 comet spawn cleanly, comets active in the full
  loop (SPS ~485), and **in-training pin WR dropped with comets** (rev38 ema 0.60→0.53, rev53b 0.70→0.60),
  difficulty recalibrating toward kaggle reality. Harvested 524k+1M ckpts → `gpu_run_artifacts/p2rev9/`; old
  comet-free artifacts → `p2rev9_cometfree_old/`. Killed after confirming the fix (it was a validation run, not a
  result run). **Next = the comet FEATURE phase (is_comet + expiry, parity) then from-scratch.** Original live
  block kept below for reference.
- **GCP L4 `g2-standard-8`, name `orbit-wars-p2rev9`, IP `8.231.115.108`, zone asia-south1-b. DELETED.**
- **DELETE (manual):** `gcloud compute instances delete orbit-wars-p2rev9 --zone=asia-south1-b`.
- **Run = the POOL pivot (NOT a reward delta).** Resume **p2rev5 4M** + ONE delta = the pool: `--pool-seed-rl
  rev38_5M.pt,rev53b_10M.pt` (winnable RL champions) + **deb demoted 0.25→0.10** (`--pool-external-fraction 0.15`
  × `--pool-fraction 0.65` ≈ 10%) + **35% self-play** (pool-fraction 0.65). Reward = p2rev5 CLEAN baseline:
  **`--early-capture-coef 0` + defense_coef OFF** (both p2rev7/p2rev8 levers DROPPED). gate3/floor0/no-forward,
  speed 0.3, expansion 0.03, LR 5e-5, 256 envs/128 rollout/ppo-2, 4 workers. Script:
  `gpu_run_artifacts/p2rev9/run_remote_p2rev9_gcp.sh`.
- **WHY:** win-starvation ([[feedback_win_starvation]]) — 4 reward/mask levers all left `planets@50` at 6 because
  we can't learn from a peeler (deb) we never beat. **Cross-eval 2026-06-13: p2rev5 4M beats rev38 27% / rev53b
  37.5%** (vs deb ~5%) = matched difficulty, a real transferable win-gradient; rev38 (aggressor) punishes
  under-expansion at a beatable difficulty. Also confirmed p2rev7 1M (27%/25%) — picked p2rev5 4M (cleaner reward
  lineage, no defense_coef baggage, slightly more winnable vs rev53b).
- **WATCH (held-out Ajay panel):** **`planets@50` climbs toward 9** (pins' target) + **WR climbs from the 27-37%
  base**. Secondary `peel-rate WON` (deb). GUARD: clip<0.25 (resume into new pool = warmup); ship0 ~0. KILL read:
  `planets@50` flat over many ckpts = pins didn't break it either → the ceiling is deeper than the pool.
- **Controller watcher** (sync + held-out Ajay per ckpt): `bash gpu_run_artifacts/run_watchers.sh start p2rev9 gcp
  orbit-wars-p2rev9.asia-south1-b.orbit-wars-rl` → `gpu_run_artifacts/p2rev9/`. iter 1 healthy (EV 0.70 warmup, KL
  0.007, clip 0.08, both pins + deb loaded, GPU 100%, SPS ~580 pre-warmup). GCP ckpt dir is a real dir (no symlink
  sync bug). ~6-7h to 10M (deb only 0.10 → less CPU than p2rev7).
- Monitor: `gcloud compute ssh orbit-wars-p2rev9 --zone=asia-south1-b -- "grep '^iter' ~/orbit_wars_rl/train_gpu_phase1_p2rev9_*.log | tail"`.

---

## ✅ CONCLUDED — p2rev8 on Jarvis A100-80GB SPOT (KILLED @5.77M 2026-06-13, box 425956 destroyed)

**VERDICT: FAILED — early_capture 0.2 did NOT move `planets@50` (dead flat at 6 over all 9 held-out ckpts;
winner 9), opening conversion flat ~0.37, WR flat ~4.5%.** Killed together with p2rev7 on the **win-starvation
finding** (`docs/training.md` Current State): we can't learn to beat a peeler we almost never beat — early_capture's
within-game expansion reward has no terminal deb-win that *requires* 9 planets to anchor it. `planets@50=6` is now
invariant across FOUR levers → opening ceiling is STRUCTURAL, not reward-tunable. 11 ckpts harvested → 5.77M.
**Next = pool-seed-RL pivot, not another reward delta** (see pool levers below). Original live block kept for
config reference:



- **Jarvis A100-80GB SPOT, machine id `425956`, IP `217.18.55.39`, 28 cores. ⚠️ SPOT — DESTROY, never pause
  (preemption may lose data; the sync watcher below is mandatory).**
- **DESTROY:** `jl destroy 425956 --yes` (from the launch box; needs venv + key — JARVIS_RUNBOOK §Auth).
- **Run = the QUANTITY/opening lever** (`docs/next-steps.md` P-OPEN #3): resume **p2rev7 1M**
  (`torch_step_1048576_p2rev7_20260612_144503.pt`) + **ONE delta `--early-capture-coef 0.2`** (0→0.2,
  always-on / anneal-frac 0). Everything else IDENTICAL to p2rev7 incl. **`--defense-coef 0.02` carried
  forward** (the 1M base was trained with it → keeping it is what makes early_capture the single delta).
  deb-only pool @0.25, gate3 / floor0 / no-forward, speed 0.3, expansion 0.03, LR 5e-5, 256 envs / 8 workers.
  Script: `gpu_run_artifacts/p2rev8/run_remote_p2rev8_jarvis.sh`.
- **CLEAN A/B:** p2rev7 continues on GCP from the SAME 1M point with defense_coef ONLY; p2rev8 branches here
  adding early_capture, so the divergence past 1M isolates early_capture's marginal effect.
- **WHY:** `planets@50` is stuck at **6** (winner 9) across p2rev5/6/7 in BOTH wins and losses — the QUANTITY
  gap. The sufficient-commit MASK (p2rev6) failed to lift it (veto removes fragments, supplies no
  concentration → [[feedback_veto_mask_removes_not_teaches]]). early_capture rewards the successful-capture
  OUTCOME (clamped count delta, NOT rev49's production-weighted delta → carpet-bomb) so failed fragments pay
  0 → positive gradient toward real commitment. It was the rev28 breakthrough (passive→expanding).
- **WATCH (held-out panel):** `planets@50` should climb toward 9 + `open<50 cap/atk WON` toward 0.51.
  **GUARD:** `ship0` must stay ~0 (1-ship spray canary — NOT using min-ship-bin; early_capture self-punishes
  probes since they capture nothing); `fire_frac`/`shipspp@50` must NOT balloon (carpet-bomb). KILL read:
  `planets@50` flat over 3–4 ckpts = failed like the mask.
- **Watchers via the CONTROLLER:** `bash gpu_run_artifacts/run_watchers.sh start p2rev8 jarvis 217.18.55.39`
  (sync + auto held-out **Ajay** full panel per ckpt, masks gate3/floor0/no-forward) → `gpu_run_artifacts/p2rev8/`.
  iter 1 healthy (EV 0.57 resume-warmup, KL 0.008, clip 0.08, single proc, `Early capture coeff: 0.2` confirmed).
- Monitor: `ssh -i ~/.ssh/jarvis-labs-key root@217.18.55.39 "grep '^iter' /home/train_gpu_phase1_p2rev8_*.log | tail"`.
  ⚠️ At 10M the run auto-stops but the SPOT box keeps billing — `jl destroy 425956 --yes` after final ckpts sync.

---

## ✅ CONCLUDED — p2rev7 on GCP L4 (KILLED @4M 2026-06-13, instance orbit-wars-p2rev7 deleted)

**VERDICT: hold metric moved but it's a self-play MIRROR artifact, not transferable.** `peel WON` declined
0.64→0.52 (toward winner 0.43, first lineage result to move it; flood guard GREEN) BUT **WR DECLINED 5.9→3.1**
and `planets@50` slipped 6→5 (`mid cap/atk` rose to 0.71) — defense_coef + perpetual-deb-loss = good at holding a
SMALL empire, expands less (conservatism). The peel gain is self-play copies out-holding *each other*, which is why
it does NOT lift WR vs deb. Killed on the win-starvation finding (`docs/training.md`). 7 ckpts harvested → 3.67M.
Original live block kept for config reference:



- **GCP L4 `g2-standard-8` (23 GB, 8 vCPU), zone asia-south1-b, name `orbit-wars-p2rev7`, IP 34.100.224.184,
  RUNNING + BILLING (~$1.13/hr, on-demand).**
- **DESTROY (manual, no auto):** `gcloud compute instances delete orbit-wars-p2rev7 --zone=asia-south1-b`
  (DELETE not stop — stopped instances still bill for disk; [[feedback_gcp_instance_cleanup]]).
- **Run = the defense_coef test** (`docs/phase2.md` flood history): resume **p2rev5 4M** (same base as p2rev6)
  + **ONE delta `--defense-coef 0.02`**, `--sufficient-commit-factor` OFF, else IDENTICAL to p2rev6
  (deb-only pool @0.25, gate3 / floor0 / no-forward, early_capture 0, speed 0.3, expansion 0.03, LR 5e-5,
  4 heuristic-workers for the L4's 8 cores). Script: `gpu_run_artifacts/p2rev7/run_remote_p2rev7_gcp.sh`.
- **WHY (the untested combo):** defense_coef is the one reward term that gives reinforcing/holding a
  near-immediate signal (avoid production lost) → fills the reinforce credit-assignment void. rev58 said it's
  the *flood pump*, BUT that was a **symmetric, pool-LESS mirror**; Tier-1 then dropped defense_coef AND added
  the aggressive pool in one move, so **defense_coef + peeler-pool was never isolated**. The deb pool should
  now punish hoarding. **A/B framing:** p2rev6 = sufficient-commit (opening), p2rev7 = defense_coef (hold),
  both single-delta off the SAME p2rev5 4M base → read which lever moves the gap.
- **WATCH:** **peel-rate WON should FALL** (winner 0.43 vs our ~0.6) — the payoff. **FLOOD GUARD:** flood =
  reinf rate up **WITH** volume exploding (`p90`/`shipspp@`); healthy = reinf up, volume flat, peel down.
  Magnitude 0.02 (rev58's 0.03 floods pool-less; started smaller). If it floods → it's the pump again (pool
  insufficient); if too conservative (game length rises, attacks drop) → lower it.
- **MONITORING = the CONTROLLER (since 2026-06-12, after p2rev6 concluded and freed it).** p2rev7 now owns the
  single-run controller: sync + auto held-out **Ajay** panel per checkpoint, masks gate3/floor0/no-forward (NO
  sufficient-commit — matches training). Procs were 51273/51274 → `gpu_run_artifacts/p2rev7/{logs,checkpoints,eval_ajay_1200.csv}`.
  **Restart if the laptop bounces:** `bash gpu_run_artifacts/run_watchers.sh start p2rev7 gcp orbit-wars-p2rev7.asia-south1-b.orbit-wars-rl`
  (`… status` / `… stop`). The launch script's original standalone sync loop (PID 41464) was RETIRED to avoid
  double-sync. ⚠️ If a SECOND concurrent run starts, it can't use the controller (single-run) — give it the
  standalone-sync + manual-eval treatment instead, and DON'T `start` the controller for it (that tears down
  p2rev7's). Cross-eval vs the 4M baseline is still MANUAL + needs a solo window (thrashes swap alongside the panels).
- iter 1→3 healthy (EV 0.69→0.84, KL low, clip 0.10→0.20 warmup, `Defense coeff: 0.02` confirmed).
  **SPS ~250** (deb/orbit_lite is CPU-bound on 8 cores; ~½ the A100's rate) → ~11 h for 10M, first 500k ckpt ~1 h.
- Monitor: `gcloud compute ssh orbit-wars-p2rev7 --zone=asia-south1-b -- "grep '^iter' ~/orbit_wars_rl/train_gpu_phase1_p2rev7_*.log | tail"`.

---

## ✅ CONCLUDED — p2rev5 on Jarvis A100-80GB SPOT (launched 2026-06-12 ~08:22 UTC; verdict → p2rev6 sufficient-commit mask)

- **Jarvis A100-80GB SPOT, machine id `425730`, IP `217.18.55.11`, 28 cores. ⚠️ SPOT — DESTROY, never pause
  (preemption may lose data; that's why the sync watcher below is mandatory).**
- **DESTROY:** `JL_API_KEY=$JARVIS_API_KEY jl destroy 425730 --yes` (from the launch box; needs `jl` + key).
- **Run = the validated peeler-in-pool delta** (`docs/phase2.md`): resume **p2rev3 0.5M**
  (`seed_checkpoints/torch_step_524288_p2rev3_20260611_153903.pt` — the documented **best-vs-deb (3.9%)**,
  before Nash reform erased holding; uploaded under its ORIGINAL name, no `*_resume.pt` rename) +
  **debatreya_1300 as the sole external pool opponent**
  (the peeler, `--pool-external-fraction 0.25`, `--heuristic-workers 8`). 256 envs / rollout 128.
  Masks: gate3, **garrison-floor 0**, **forward-only DROPPED** (decided 2026-06-12 — the run's point is
  learning to *hold* a peeled rear planet; forward-only blocks pulling ships back to defend). early_capture
  OFF, defense_coef 0, speed 0.3, expansion 0.03, fire-entropy 0.005. Script:
  `gpu_run_artifacts/p2rev5/run_remote_p2rev5_jarvis.sh` (cwd `/home`; box log `/home/train_gpu_phase1_p2rev5_*.log`).
- **⚠️ TORCH FIX APPLIED AT LAUNCH:** the box image shipped torch `+cu130` vs driver 570 (CUDA 12.8) →
  `cuda.is_available()=False` → it silently trained on **CPU** (A100 idle, no iter 1). Reinstalled
  `torch/vision/audio 2.11.0+cu128` (`cuda=True`) and relaunched. Setup now self-heals this
  (`setup/install_orbit_wars.sh` cu128 guard). Watch for it on any fresh Jarvis box.
- **Watchers via the CONTROLLER** `gpu_run_artifacts/run_watchers.sh` (NEW 2026-06-12 — fixes the recurring
  "stale prior-run watcher" bug): `run_watchers.sh start p2rev5 217.18.55.11` launches sync + held-out eval,
  tearing down ALL existing watchers first; each watcher self-terminates when `.active_run` changes. `… status`
  = active run + live procs; `… stop` = kill all. Sync pulls log + `torch_step_*/pool_step_*/torch_best_*`
  every 120s → `gpu_run_artifacts/p2rev5/{logs,checkpoints}`. Restart with `start` if the laptop bounces.
  Per-run ad-hoc `*_watch.sh`/`sync_watcher.sh` are DEPRECATED (removed) — see CLAUDE.md GPU rules.
- **⚠️ EVAL/EXPORT MASKS = gate3 / floor 0 / NO forward-only** (must match training). `cross_eval/run_cross_eval.sh`
  updated; `export_agent.py` defaults flipped to forward_only=False/floor=0. Old p2rev1-3 exports need the
  explicit `--reinforce-forward-only --reinforce-garrison-floor 10`.
- **EARLY READ (2026-06-12, 5-pt held-out Ajay trend @ ~4M train):** WR dropped off the 500k high then
  STABILIZED in the ~3% band — `500k 4.7 · 1M 2.7 · 1.5M 2.0 · 2M 3.1 · 2.5M 3.1` (not collapsing, but Ajay
  is a noisy non-LB-predictive guardrail). **peel_WON DEAD FLAT ~0.63 across all 5** (winner 0.43) → the hold
  gap is NOT closing. Training diag drifts the WRONG way: late reinforce `>100` stuck 0.73–0.80 (winner 0.31),
  `garrfrac@50` 0.59→0.71 + `shipspp@50` 38→42 (over-garrison INTENSIFYING), **`H_tgt` rising 1.12→1.71** (the
  "where-head going uniform" canary) ⇒ reinforcement is **mis-TARGETED, not under-rate** — reproduces the
  06-11 finding, points at the **THREAT HEAD** (self-supervised P(lost within K)) as the principled fix.
  Training mechanically healthy (EV ~0.85, KL low, clip ~0.19). **DECISION (user, 2026-06-12): let it run,
  re-evaluate at the 6M checkpoint** before committing to cut+threat-head.
- **6M RE-CHECK (2026-06-12) — the 4M pessimism was PREMATURE; run turned a corner ~3M.** 11-pt Ajay trend:
  WR troughed at 1.5M (2.0%) then RECOVERED/climbed — `3M 3.5 · 3.5M 4.3 · 4M 5.9 · 4.7M 4.3 · 5.2M 4.7 ·
  5.77M 5.1` → now a ~4.5–5% band (peak 5.9, above the 500k base, near best-ever Ajay; rev38 3.1%→994 LB).
  Health improved: `peel_WON` 0.63→~0.57 mid-run (wobbled back to 0.61–0.63 last 2 pts), `fire_frac`
  0.44→0.32 (less spray), `shipspp@50` 42→33 (over-garrison REVERSED), self-play `pl@100` 8→9, `owned`→10.8.
  **BUT unresolved:** `H_tgt` STILL rising → 2.22 (highest), late reinforce `>100` still 0.79 — so the WR
  gains come from better expansion/conversion/less-spray, NOT from fixed reinforce *targeting*. **DECISION
  (user): CONTINUE to 10M + harvest; threat-head = the NEXT run** for the unresolved H_tgt/targeting. Train
  ~6.7M, healthy (EV 0.87, KL low, clip crept 0.19→0.23 — minor, watch). ⚠️ At 10M the run auto-stops but the
  SPOT box keeps billing — DESTROY it (`jl destroy 425730 --yes`) after final ckpts sync.
- **⭐ FULL HELD-OUT PANEL TREND (2026-06-12, 12 ckpts 500k→6.3M) — CORRECTS the threat-head conclusion.**
  The 4M/6M notes above leaned on the SELF-PLAY training diag (H_tgt, garrfrac 0.71, reinf>100 0.79); those
  are mirror-game artifacts (self-play runs to 500 steps → heavy late reinforce). The **held-out Ajay panel
  logs** (`eval_logs/*.log`, parsed via `/tmp/parse_panels.py`) tell a different, trustworthy story. Trend
  (WON-game, decision-grade):
  ```
  metric      500k→6.3M trend          vs winner    verdict
  opnWON      ~0.38 FLAT (no trend)    0.51         ✗ opening conversion STUCK — the dominant gap
  peelW       0.64→0.56 (marginal)     0.43         ✗ mid-game hold barely moved
  WR%         osc ~4.5 (peak 5.9@4M)   —            ~ not a sustained climb
  rf<50       0.15→0.09 (WORSENING)    0.29         ✗ early reinforce DECLINING, away from winner
  fwd%        49→58                    57           ✓ reinforce DIRECTION fixed (cosmetic)
  garf50/spp  0.52→0.44 / 29→23        0.54 / 22    ✓ less hoarding (cosmetic)
  rf>100      ~0.24 (NOT 0.79!)        0.31         ✓ vs Ajay reinforce is moderate/under, NOT a flood
  ```
  **Conclusion:** deb-in-pool moved reinforce-SHAPE metrics (direction, hoarding) but NOT the two outcome
  levers — **opening conversion** and **hold**. Vs the real opponent reinforce is already winner-like (fwd
  58%, rate moderate) ⇒ **the THREAT HEAD is NOT the indicated next lever.** The flat-stuck signal is
  **opening conversion (P-OPEN)** → **the SUFFICIENT-COMMIT MASK is the next run** (veto fragment launches
  `ships ≤ target defense`; §"P-OPEN levers" #2). Early-reinforce decline (rf<50 0.15→0.09) is a secondary
  concern. **Lesson: read held-out PANEL logs, not self-play diag, for behaviour conclusions** (the diag is
  a mirror artifact — cf. [[project_train_eval_sim_gap]], [[feedback_selfplay_collapse_metrics]]).
- **EVAL via the controller (2026-06-12):** the `_eval` loop auto full-256-panels each `torch_step_*p2rev5*.pt`
  vs HELD-OUT **candidate_ajay_1200** (deb is now IN training so no longer held-out; Ajay = documented primary
  metric), masks gate3/floor0/NO-forward-only → `gpu_run_artifacts/p2rev5/eval_ajay_1200.csv` + `eval_logs/`.
  Runs LOCALLY. The stale p2rev4 eval watchers (deb/zach/cross, wrong dir+opp+masks) were KILLED by the
  controller's tear-down. First eval: step 524288 in progress. (Ajay only; add a zach/ladder opp later if wanted.)
- **TWO restarts on 2026-06-12 (current log `_083532`):** (1) LR HALVED 0.0001→0.00005 — the first run
  (from p2rev4 1.5M) drifted: `clip_frac` climbed monotonically 0.157→**0.265** by iter 10 under LR 1e-4
  (KL low 0.006–0.038, EV 0.74–0.90 = LR-too-hot, not KL blowup). (2) BASE SWITCHED p2rev4 1.5M → **p2rev3
  0.5M** — p2rev4 1.5M descended from p2rev3 *4M* (drifted) AND regressed during p2rev4 (churn 16→23,
  peel 0.61→0.64); p2rev3 0.5M is the best-vs-deb point, the coherent base for a deb-in-pool run meant to
  *preserve* the early holding skill Nash later erases. Restart iter 1: resumed from the p2rev3 0.5M ckpt,
  `peak_lr=5e-05`, clip 0.077→ plateau **~0.28** by iter 11. **DECISION 2026-06-12: clip ~0.28 is BENIGN —
  let it run, do NOT chase it with more restarts.** Evidence: (a) the 1e-4→5e-5 LR halve did NOT lower the
  plateau (0.265→0.284) ⇒ LR is not the driver; (b) KL stays low (0.004–0.027, << 0.05 target), EV healthy
  (0.72–0.90), estop=0 — clip-high + KL-low + EV-stable = a subset of actions moving (p2rev3 0.5M re-learning
  hard-commit under floor=0 vs the deb peeler), NOT divergence. If we ever DO want to dampen it the indicated
  low-KL levers are entropy_coef (0.05→0.01) or ppo_epochs 2→1 — NOT LR. **WATCH instead: held-out WR + EV/KL
  actual degradation** (act only if those move). Selection PURE: held-out WR vs a DIFFERENT strong opp
  (ajay/producer — deb is now IN training, no longer the clean held-out).

---

## ⚠️ ~~LIVE INSTANCE~~ TERMINATED — p2rev4 on GCP L4 (instance deleted 2026-06-12; checkpoints synced locally through 1.5M)

- **GCP L4 `g2-standard-8` (23GB), zone asia-south1-b, name `orbit-wars-p2rev4`, IP 34.100.181.138,
  RUNNING + BILLING (~$1.13/hr).** Running **p2rev4** in tmux `training`. The ONE training delta vs p2rev3:
  **`--reinforce-garrison-floor 10 → 0`** (unblock reinforcement — veto probe showed the floor blocked 62%
  of wanted reinforces). Resume of the p2rev3 4M checkpoint (`seed_checkpoints/p2rev4_resume.pt`).
  **256 envs / 4 workers** (fits L4 23GB with `expandable_segments` — no OOM). GCP-adapted start script:
  `gpu_run_artifacts/p2rev4/start_p2rev4_gcp.sh` (repo-root paths, no `--terminate-on-done` → manual delete).
  iter 1 healthy: EV 0.726, clip 0.154, SPS 622, no OOM. Launched ~18:46 UTC 06-11.
- **⚠️ Predecessor Jarvis A100 SPOT (IP 217.18.55.74) — user stopped it; VERIFY it is DESTROYED
  (`jl list`/`jl destroy`), else it keeps billing ₹179/hr. Could not check from this box (no `jl`/key).**
- **Baseline to beat (p2rev3 4M vs deb, 32g, masks floor10):** WR 6.25%, `cap/atk-launch 0.462`,
  `planets@ 2/4/6/3` (collapses), `lost-cap 0.91`, **`launch_rate 0.068` / `fire_frac 0.34` (≈2× Isaiah
  0.036/0.17 — the spray, now measured)**, reinf ramp suppressed mid-empire (.08–.12 vs ref .30).
- **WATCH:** EV recovery + clip<0.25 (resume warmup), then the PAYOFF metrics — **`lost-cap` should FALL /
  `median-hold` should RISE** (reinforcement now unblocked → can hold) and **`reinf@13+` should climb**.
  NEW guardrail: **`launch_rate`/`fire_frac` should NOT balloon** past ~0.07/0.34 (floor=0 → more reinforce
  launches → flood risk; spray may instead FALL if better holding cuts the frantic recapture churn).
  Caveat: floor=0 lets hard-commit reinforces strip a source to 0 → if `lost-cap` WORSENS, that's over-drain
  → add a small floor back (2-3). Selection PURE: held-out win-rate.
- **Eval stack LIVE (2026-06-12, re-pointed at GCP):** launcher sync watcher (PID 34615) pulls
  checkpoints+log from the box → `gpu_run_artifacts/p2rev4/` every 180s. Three local eval watchers
  poll that dir (PY = `orbit_wars_rl/.venv`): deb full-256-panel every ckpt (`eval_debatreya.csv`),
  zach full-panel @2M bucket (`eval_zach.csv`), cross-eval 48g @3M bucket. First ckpt (500k) lands
  ~13min after iter1. Evals run LOCALLY (never on the training box).
- **⚠️ EVAL MASKS MUST MATCH TRAINING = floor 0 for p2rev4** (gate>=3, forward-only, **garrison-floor
  0**). Watchers originally inherited floor=10 from p2rev3 → mismeasured the floor-0 policy (vetoes the
  reinforces it was trained to make); FIXED 2026-06-12 in all 3 watcher scripts + run_cross_eval.sh.
  **Same gotcha at EXPORT:** export_agent defaults floor=10 — any p2rev4 export MUST pass
  `--reinforce-garrison-floor 0` (see [[feedback_phase2_export_reinforce_parity]]).
- SSH: `gcloud compute ssh orbit-wars-p2rev4 --zone=asia-south1-b`.
  Tail: `... -- "grep '^iter' ~/orbit_wars_rl/train_gpu_phase1_p2rev4_*.log | tail"`.
- **TERMINATE when done (manual — no auto):** `gcloud compute instances delete orbit-wars-p2rev4 --zone=asia-south1-b`.

---

## ⭐ 2026-06-11 session — findings + next levers (prioritized)

**Biggest finding — the gap vs strong planners is the MID-GAME (steps 50→100) hold, NOT the opening.**
⚠️ **CORRECTED 2026-06-11 by the conversion metrics** (was "the gap is the opening"). p2rev2 @4.7M
conversion eval (masks-on) vs Zach 64g (77% WR) and Deb 32g (0% WR): **`planets@16/32/50/100` is the
clean win/loss discriminator** — Zach **2/4/7/10** (monotone, end 17) vs Deb **2/4/6/3** (peaks @50,
collapses to 3 by @100, end 0.1, dead ~step 122). **The opening is IDENTICAL win/loss (2/4 at steps
16/32) — we do NOT lose the early exchange; the divergence is the mid-game.** Metric verdicts:
`redundant-launch` is at the elite floor in BOTH (0.12/0.17; ref ~0.15) → **the over-fire /
friendly-coverage roi-deflation p2rev2 is built on is a non-bottleneck, no headroom** (not hurting,
but misdirected for the Deb gap). `cap/atk-launch` fine (0.59/0.44). `churn` degenerate on elimination
(end→0). Mechanism: we over-garrison mid-game (garr_frac@50 0.63–0.75, ships/pp 34–46 vs Isaiah 0.54/22);
Deb's planner peels the parked planets, Zach can't. **Lever the data points to = `--pool-seed-rl`** (a
strong RL self in the loop that punishes the over-garrison → forces holding), not deflation or First-Strike.
Tooling: `churn`+`redundant-launch` added to `eval.py game_conversion()`; `conversion_from_replays.py`
for the top-2 baseline; defs/confounds in `docs/metrics.md`. Won-game HTML: `/tmp/p2rev1_WIN_*_seed{1948,9013}.html`.
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
- 🟢 **RELAX `--reinforce-garrison-floor` (TOP lever — evidence-backed, cheapest).** Veto probe
  (`orbit_wars_rl/garrison_floor_probe.py`, 4M ckpt vs deb, 2026-06-11): the policy WANTS to reinforce 26%
  of fire-decisions but only **13% are allowed — the garrison-floor blocks 62%** (82% of forward-legal
  reinforces); at floor=0 allowed jumps to 55%. The floor=10 directly FIGHTS forward-staging (drain-rear vs
  keep-10) and the policy commits hard so nearly any reinforce trips it. **Next single-delta run: resume
  p2rev3 4M with garrison_floor dropped to 0 (or 2-3), else identical; watch `lost-cap`/`median-hold`.**
  Caveat: no floor → hard-commit can strip a source to 0 → policy must re-learn commitment. **Secondary
  blocker = `--reinforce-forward-only`** (blocks pulling ships BACK to defend a peeled rear planet; 23%→42%
  of intents once the floor is gone) — relax next if floor-relax alone isn't enough. **Sequencing: relax
  masks BEFORE/WITH deb-as-pool** — a peeler in the pool is useless if 87% of intended reinforces are masked out.
- 🟢 **THREAT HEAD (retention lever — strongest principled candidate).** Per owned planet, predict
  P(lost within K steps), self-supervised from the trajectory; feeds reinforce target selection
  (reinforce the planet you predict you'll lose). Full spec: `docs/phase2.md` "Deferred track". **Decision
  gate = is the target head undertrained?** Evidence as of p2rev3 ~2.5M: **`H_tgt` is RISING 1.07→1.70**
  (the documented "where-head going uniform" canary) AND aggregate reinf is high (0.5+) while empire-binned
  `reinf@13+` is LOW (0.25) with poor retention ⇒ **reinforce is mis-TARGETED, not under-rate** — exactly
  the threat head's domain. Build it ONLY as a self-supervised PREDICTION head (NOT target reward = rev49
  carpet-bomb graveyard; NOT KL-to-heuristic-targets = rev54 crater). Build landmine: per-planet labels hit
  the VDN planet-id-reorder corruption (scatter in planet-id space, not slot space). Highest-value/lowest-risk
  per phase2.md; do this if the p2rev3 retention metric (`lost-cap`) stays bad through 5M.
- 🟡 **Deb (or Ajay/producer) as the ONLY external pool opponent — "learn by losing to the exploit."**
  Fast hammers don't peel mid-game planets the way a planner does → never punish our retention weakness; a
  planner would. MECHANICALLY FEASIBLE: route via the `--external-opponents X --heuristic-workers N`
  SUBPROCESS path (NOT in-process `--planner-externals`, which stalls) — rev56 ran debatreya_1300 as sole
  external stably (256 envs / 4 workers / ext-frac 0.15, SPS ~400-660). Tradeoffs: **(1) deb is our HELD-OUT
  eval — training on it forfeits the clean generalization signal; swap in Ajay/producer as the new held-out.
  (2) sim-gap — deb-in-torch_env plays weaker than in Kaggle, so transfer to LB is not guaranteed (still a
  stronger/differently-exploiting opponent than hammers). (3) throughput retune** vs the 512/6 fast-hammer
  config. Queue as a single-delta experiment after p2rev3 concludes. (rev56 mechanism: memory
  `feedback_orbit_lite_is_slow`.)
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
- ✅ **Eval/export gate parity (DONE 2026-06-11, first Phase-2 submit sub 53574885).** Correction: `action_mask.py`
  `actions_from_target_policy` ALREADY had the full reinforce-discipline parity (gate/forward-only/garrison-floor, both
  logit-mask + post-argmax reject). The real gap was `export_agent.py`: it baked only `allow_reinforce` and let the three
  discipline params fall to their off-defaults (gate0/forward-off/floor0) → exported agent would reinforce <3 planets /
  backward / drain source = self-sabotage. The three params are NOT stored in the checkpoint (only `allow_reinforce` is),
  so export now takes them as CLI flags defaulting to the locked Tier-1 values (gate3/forward/floor10) and bakes them into
  the template. Also fixed a pre-existing export crash: the template didn't `import os`, surfaced by features.py's
  module-level `os.environ` (friendly-deflation) read. Verify after any reinforce export: grep `_REINFORCE_GATE_MIN`/etc.
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
