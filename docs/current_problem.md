# Current Problem — the Force-Concentration Wall, diagnosed

_Last updated: 2026-06-22. Supersedes the scattered "wall" notes; this is the standing statement of
what the wall is, what's ruled out, and where the fix has to come from._

## TL;DR

**The wall is the value/fire head not learning to STAGE — it hoards spare garrison idle instead of
feeding it, across turns, into a building wave toward a contested target.** It surfaces as
under-aggregation → losing the early expansion (planet-count) race; everything downstream
(out-massed ~96%, mid-game collapse, low Ajay WR) is shadow of that. Pinned precisely: when a contested
attack falls short and a spare source is available, the spare sits **idle 80%** of the time (not the
ship head — undersend 0–2%; not the target head — it isn't aiming elsewhere).

**The architecture is sufficient — the wall is NOT architectural.** Factored heads decide each source
independently (no within-turn coordination), but the coordination that *wins* is **cross-turn staging**
(accumulate inbound over turns), which factored heads express fine. The within-turn coordinator (AR head)
was built-as-plan and killed because winners barely use it (same-turn floor-cross 1.6%). So the fix is a
**value/reward** signal, not a decode/arch rebuild.

**It is an *escapable PPO fixed point*, not a true Nash.** `value_spare_diagnostic.py` shows the critic
is well-calibrated (V|won +1.57 / V|lost −0.59) but advantage A≈0 on the spare-fire action — a
self-reinforcing loop: nobody fires → spare-use never correlates with outcome in self-play data →
critic can't price it → A≈0 → nobody fires. It is **not** a true Nash (Ajay beats us *by* aggregating,
so aggregation is a profitable deviation we never explored) and **not** a broken critic. That's the
optimistic reading — it's an exploration/credit gap, the regime shaping is built to break.

**~~Chosen lever: PBRS staging reward~~ — TRIED, RULED OUT (2026-06-22, see CLOSED section below).**
PBRS (`r += α(γΦ(s')−Φ(s))`) + a fire-entropy spike un-trapped the idle fire head (mechanism worked) but
did **not** break the wall: fired-spare plateaued ~6.7%, WR collapsed to 4.3% @4M, and `delta_V≈0` shows
firing spare is **value-neutral** in the current Nash — the policy-invariance limit, exactly the deepest
risk we flagged. With this, the **reward/action-space family is exhausted with a mechanistic reason**; the
only remaining lever is the **GAME** (a win-gradient opponent we can beat). The pre-test "LEVER PULLS" was
necessary-only and is now known insufficient.

Baseline reference (ild 6M vs Ajay): WR ~23%, out-massed 96%, caps@50 LOST 6.1, planet-deficit ~−2.

---

## CLOSED RUNS (2026-06-22) — both destroyed; NEITHER dented the wall (avg_sources pinned ~2.2)

Two parallel Jarvis A100-80GB spot runs, both resumed the **3M peak** of r32_stage_h12 (shared start →
comparable), h12-only beatable pool, LR 5e-5. **Both destroyed; no live boxes.** Net: WR is movable
(retention/execution), pooling width (`avg_sources`) is NOT — by either lever.

**1. `r32_stage_hlr` (inst 431294 @217.18.55.19)** — *was the collapse LR-thrash?* One delta vs the dead
PBRS run: **LR 1e-4 → 5e-5**, resume the 3M peak (the 4M/5M ckpts are collapsed: WR 9.0→4.3→1.2%). Keeps
PBRS staging + entropy spike 0.05. Read: WR holds ≥9% past ~2M + ship0 stays ~24% = collapse was thrash;
slides under ~6% with ship0 climbing = LR wasn't the driver → entropy-decay 0.05→0.02 is the next delta.
**RESULT (confirmed thrash, not structural): WR 9.0(start)→10.5→16.0→13.7% @+1/2/3M (settled ~13-14%; the
16.0 was a +1σ blip, NOT a climb), ship0 25% (not 44), open cap/atk WON 0.54 (≥ winner ref 0.51) — vastly
better than the original collapse (4.3→1.2%). KILLED @4M, ckpt saved. The "PBRS NEGATIVE /
delta_V≈0 / kill" call was PREMATURE: delta_V≈0 was measured on the 4M ckpt that was ALREADY mid-collapse
(ship0 44%), so it reflected thrash, not a fixed property. At LR 5e-5 the same PBRS+spike config climbs. Wall
itself still intact (out-massed 94%) — half-LR salvaged+improved the run, doesn't break the wall. TODO: re-measure
delta_V on the healthy 16% ckpt (torch_step_2097152) to see if value-neutrality survives outside the collapse.**
**AGGREGATION-WIDTH metric (the real wall steering metric, USER-flagged): `avg_sources` from
`multi_source_events.py --our-name Saheb` (on our-loss replays vs Ajay) = 2.19 IDENTICAL at 3M-start (WR 9%) AND
half-LR 2M (WR 16%) — DEAD FLAT across the WR climb (winners pool 4-5; we pool the minimum 2 + leave ~2.8 idle).
The 9→16% climb came from RETENTION (held@+15 45→51%) + single-source execution, NOT wider pooling → WR is a POOR
wall proxy. Steer by avg_sources (pinned 2.19 by every lever). elim-planet's REAL success criterion = avg_sources > 2.19.
The per-NEUTRAL idle in value_spare_diagnostic is MISLEADING (denominator inflation); per-SOURCE idle (multi_source_why) 87% is the real one.**

**2. `r32_elimplanet` (inst 431296 @217.18.55.29)** — *is the TERMINAL reward the root of `delta_V≈0`?*
The reward-neutrality fix: **drop PBRS** (policy-invariant, confirmed inert) and **fix the terminal reward** —
`--eliminate-to-win --timeout-planet-coef 0.5`: elimination → ±1; timeout → 0.5·planet-share-margin (zero-sum,
graded). Kills the most-ships hoard attractor AND the draw-neutral starvation; rewards expansion+retention (the
root). Thesis: firing is neutral only because hoarding wins at timeout → make planets-held decide the timeout →
the fire→capture→hold→win chain becomes the only win path → **credit assignment makes firing reward-positive
with NO shaping.** Code: `torch_env._check_done` planet-margin branch + `timeout_planet_coef` (test_eliminate_to_win
6/6). Read: **WR↑ AND out-massed↓ with reinforce HELD high** = real wall-break (not the disengagement confound
[[feedback_outmassed_reinforce_confound]]). Tripwire: all-equal-planet draws (starvation even vs h12) or spray.
**RESULT (NO DENT, killed): WR 9.0(start)→4.7(+1M, reward-adaptation dip)→9.4(+2M, recovered) — NOT a
scenario-style collapse, but recovered only to baseline and trailed half-LR. Wall UNMOVED: out-massed 95-96%,
`avg_sources 2.20` (= 2.19 baseline), per-source idle 85% (= 87% baseline) on +2M loss replays. The terminal-
reward fix did not price the 3rd/4th pile-on source by +2M (early for slow credit-propagation, but ZERO signal).
Joins the failed reward-change family on the wall metric. Code (`timeout_planet_coef` + planet-margin _check_done,
test 6/6) is sound and retained for any future league run — the lever, not the code, is what didn't fire.**

**SESSION VERDICT (2026-06-22): the wall = pooling WIDTH (`avg_sources`), and it is robustly pinned at ~2.2
(winners 4-5) against half-LR, the terminal-reward fix, PBRS, and every prior reward/LR lever. WR moved
(9→16) via retention + single-source execution but that is NOT wall progress (WR is a poor proxy). The
honest open question: can anything short of a league/anti-cycling rebuild move pooling width — or is ~2.2 the
ceiling of this self-play setup. Both boxes destroyed; clean stop.**

---

## CLOSED — `r32_stage_h12` PBRS staging reward: NEGATIVE (killed @4M, 2026-06-22)

**Result up front:** PBRS + fire-entropy spike got the *mechanism* (fire head un-trapped) but **not the
outcome** (no wall break). Killed at 4M, Jarvis box destroyed. Trajectory vs held-out Ajay: WR
7.0 → 5.9 → 9.0 → **4.3%** @1/2/3/4M; out-massed flat 93–96%; planets@50 stuck at 7. **ship0 24% → 44%**
between 3M and 4M = the entropy spike's firing pressure discharging as 1-ship probes (the cheap-out),
actively harmful.

**The kill is mechanistic, not "flat WR" (`value_spare_diagnostic.py` @4M, 24 seeds):**
- **fired-spare plateaued** 3.9 (pre) → 6.2 (2M) → **6.7% (4M)** — the un-trap was a one-time step, not a
  trajectory. Idle 93.3%.
- **`delta_V ≈ 0`** (won +0.002 / lost +0.012, both noise) with a **well-calibrated critic** (V|won −0.166
  > V|lost −0.330). Firing spare carries ZERO value advantage *even though the agent self-selects favorable
  moments* (the confound runs in our favor). PBRS cannot relocate an optimum where idle-spare is
  value-neutral — the policy-invariance limit.
- **ship-size split** (newly added to `value_spare_diagnostic.py`): fired-spare is mostly **real mass**
  (mean 19 / median 14), not junk. Only **9% is "had-mass probe"** (could solo but sent ≤4 ships) = the
  *entire* min-ship-bin headroom; **65% is the `agg` bucket** (needs pooling, zero single-send headroom).
  ⇒ min-ship-bin is a symptom-stabilizer (closes the ship0 hatch), **not** a wall-breaker — Lesson 3 verbatim.
- **PBRS pre-test "LEVER PULLS" (+0.181) is necessary-only** (script's own caveat: "not a Nash-vs-fixedpoint
  test"). Green before launch; the live run is the sufficient test and it failed.

**Net:** under-aggregation is value-neutral in the current self-play Nash (a HABIT), and the majority
opportunity is multi-source **pooling** that no single-send/shaping knob touches → the lever is the GAME
(win-gradient opponent we can beat), per `project_h14_wingradient` / win-starvation.

<details><summary>Run design history (what shaped r32_stage_h12)</summary>

The PBRS staging reward was **built, tested, and run** (Jarvis A100-80GB spot, inst 431251). Env-side
`_staging_potential` (torch_env) + `r += coef·(γΦ(s')−Φ(s))`; flags `--staging-shaping-coef 0.2
--staging-topk 2`; unit test `tests/test_staging_shaping.py` proves neutral-only/cap/top-k, the per-step
formula, and the telescoping (spray-safe) identity.

**Config:** jake-BC seed (`bc_jake_unfiltered_pw6.2`) → PPO, **h12-only** external pool, **no anchor**,
critic-warmup ev0.8, **fire-entropy spike 0.05** (fire-head only). Throughput: **4096 envs / 24 workers /
num-minibatches 64** (~800 SPS; the model is tiny so the box was CPU-serial-bound — num-envs is the SPS
lever, not workers/GPU).

**Two findings that shaped the run:**
- **Jake BC does NOT carry the aggregating shape** (fire-spare 3.9% ≈ the trained policy's 3.4%; Jake is a
  reinforce-and-hold player, *not* an expander). So PBRS must **build** staging from a ~4% prior, not
  maintain it — the harder test. This elevated the **PPO-clip bottleneck** risk (raising p(fire) off 0.04
  blows past the ±0.2 clip; and you can't reward fires you never sample).
- **Fix = the fire-entropy spike** (0.05, fire-head only) to force fire-sampling so PBRS can reinforce the
  good ones. Constant, **decayed manually by observation** (not a timer — critic-warmup would burn a timer).

**First real signal (@2M, ~0.9M post-warmup) — cautiously positive:**
- Warmup released clean (EV 0.87); **fire-clip spiked 0.11→0.01 = the fire head moved through the clip, not
  trapped** (the clip-bottleneck risk did not bite).
- **fired-spare rate 3.9% → 6.2% vs held-out Ajay** (223/3605, ~5.7 SE — real, and it generalizes since
  it's measured vs Ajay not the h12 we train on). **First intervention all session to measurably lift the
  idle fire head.**
- Entropy spike adds broad spray noise (overall tgt-enemy up, dm-cross down), but PBRS lifts the *specific*
  neutral-spare subset it rewards; caps holding (pl@100=10) ⇒ **not** the kill-spray failure.

**The verdict is the 3–4M trajectory** (pending): fired-spare climbing toward 8–10%+ with caps@50 /
out-massed responding = PBRS breaking the wall; plateau ~6% with caps flat = marginal → weigh **Door 1**
(permanent aggregating opponent). Lower the entropy spike 0.05→0.02 once staging is clearly established.

**Open risks being watched** (tripwires): (a) fired-spare/fire_frac rises then *falls* by 3–4M = the
DAgger-style race lost; (b) **h12-WR↑ but held-out Ajay flat = overfit to h12** (Ajay panel is the
arbiter, not in-train h12-WR); (c) fire_frac/launch_rate balloon *without* caps↑ = spray. Deepest structural
risk: PBRS is policy-invariant, so a converged critic nets the shaping advantage back to ~0 — it's a *race*
to reach the staging basin before the critic neutralizes the boost. **[This is exactly what happened —
see CLOSED result above.]**

</details>

---

## The diagnostic chain (2026-06-21)

Each step killed a hypothesis. Tools: `orbit_wars_rl/{transition_autopsy,expansion_autopsy}.py`
(both vs Ajay, `ORBIT_GAME_PHASE_FEATURES=1`, logs under `gpu_run_artifacts/r32_ildecay03/eval_logs/`).

1. **3-arm batch (anchor-decay / eliminate-to-win / scenario, off the jake-5M seed): none broke the
   wall.** out-massed stayed 92–97% whenever WR was healthy. ild drifted 23.4→18.4%@8M; elimwin peaked
   27%@4M then collapsed to 6.6%@7M; scenario regressed 28→12% (disengagement). Reward/curriculum/anchor
   knobs do not move the wall.

2. **transition_autopsy (mid-game 50–100): the mid-game reinforce/triage lever is DEAD.** Features
   separate hopeless-from-defendable cleanly (+16.5/+27.3 vs −13.5/−15.1); heads behave identically at
   matched owned-count (the multi-launch gap is pure composition — winners own 12.4 planets vs 5.8); the
   "98% to-lost mass → hopeless" is a *symptom*, not a cause. By step 50 the divergence is already
   material (mass share 0.61 vs 0.34).

3. **expansion_autopsy (early 10–50): the loss is an EXPANSION-RACE deficit, not garrison/concentration.**
   Ships-per-planet is *matched* to the enemy (~47) — we don't under-garrison or over-extend. We lose the
   **planet-count race** (winners hold parity with enemy 8.9 vs 8.8; losers fall behind 6.2 vs 8.0).
   out-massed 96% is the *downstream shadow*. This **reversed** the "consolidate harder" hypothesis.

4. **static-vs-rotating split: the loss mechanism is UNIFORM.** Rotating-specific hypotheses (aiming,
   defensive bleed) both falsified — peel ≈ 0 everywhere. Rotating changes loss *frequency* (WR 56%→16%),
   not *shape*. Capture count is ~constant (~7–8) regardless of board or outcome. (Note: split must use
   panel archetype labels — the obs `angular_velocity` scalar is ~identical 0.03–0.05 for both classes;
   rotation is positional/orbit-radius.)

5. **Causal gate + 4-way neutral fork: resolves to AGGREGATION/COMMITMENT.** Board isn't tapped (13–19
   neutrals remain at step 50), isn't free-land hoarding (cheaply-takeable-but-skipped is small), isn't
   reach-limited (`far` resolves by step 40). The live bucket is `agg`: 3–5 neutrals per decision are
   takeable *if we pool 2+ sources' spare*, and we don't. **Converges with the original finding** —
   under-aggregation by choice, now shown for **expansion** as well as **attacks** (72% of attack
   floor-misses had a spare source in range; multi-source used on only ~7% of contested targets).

6. **DAgger warmstart→PPO (`r32_ajayclone`): the bet, and it FAILED.** The Ajay-clone seed verifiably
   carried a 2–5× more-active multi-launch shape, but **PPO erased it by 3M** — caps identical to ild
   (6.2 vs 6.1), deficit slightly worse, *more* per-planet hoarding (52 vs 47), out-massed unchanged. WR
   tracked at/below ild at every matched step (9.4 vs 11.7 @3M). Spot-preempted ~3–4M; verdict already in.
   The injected shape dissolved back into the under-aggregating Nash.

7. **Root-cause decomposition (`multi_source_why.py --our-name Saheb` on 292 us-vs-Ajay loss replays):
   the failure is the IDLE FIRE HEAD.** Of available-but-unused spare sources at a shortfall: **idle 80%**
   / attack-elsewhere 18% / reinforce-elsewhere 2%; 89% of policy-short events had ≥1 idle spare. Not the
   ship head (refuted), not the target head (not aiming elsewhere). The value fn rates *hold > spend*.

8. **Architecture question (model.py + torch_env + docs/autoregressive-head.md): the arch is SUFFICIENT.**
   Physics: fleets sum vs a garrison. Heads: factored, per-source, single pass — no within-turn coordination
   (source B can't condition on A's same-turn launch). BUT the winning form of aggregation is **cross-turn
   staging** (prior inbound + later adds; winner within-turn floor-cross only 1.6%), which factored heads
   express via fleet-token awareness. The AR within-turn coordinator was built-as-plan and killed for that
   reason. ⇒ the idle fire head = *doesn't stage*, not *can't coordinate*. No decode/arch rebuild needed.

9. **V(s) diagnostic (`value_spare_diagnostic.py`, phase4e 3.67M vs Ajay): the idle is an ESCAPABLE PPO
   fixed point.** Fired-spare rate 3.4% (96.6% idle). Critic well-calibrated (V|won +1.57 / V|lost −0.59 —
   *not* over-optimistic), but ΔV(fired−idle)≈0 within outcome (−0.13/−0.17 = pure selection). ⇒ A≈0 on the
   spare-fire action — the self-reinforcing loop. Not a true Nash (aggregation isn't dominated — Ajay wins by
   it), not a broken critic. *Entropy-knob rejected:* undirected fire samples bad launches → critic learns
   fire=lose → reinforces idle (explains the arms' 0.02→0.04 bump doing nothing).

10. **PBRS pre-test (`--shaping-coef`): the staging gradient exists and is well-scaled.** Counterfactual
   potential rise Φ′−Φ from firing pooled spare to floor = +0.93; α·shaping @0.2 = **+0.19/step** ≫ the A≈0
   floor. Σ-vs-MAX resolved by surface data (5.3 contestable targets/state): **use top-k (k≈2–3)** — pure Σ
   sprays, max kills serial breadth. (Confirms the gradient + calibrates α; not itself a wall-break proof.)

---

## What's ruled out

- Mid-game reinforce / hopeless-recycling / triage (symptom, not cause).
- Target selection; opening expansion count; 1-ship probes.
- Consolidate-harder / mass-per-planet (ships/planet already matched).
- Rotating-specific aiming or defensive-bleed mechanisms.
- Reward proxy for concentration — **decmass confirmed negative**.
- Scenario curriculum (overfits tiny boards, regresses transfer).
- Anchor as a *fix* — it's only a stabilizer; see below.
- Imitation **warmstart** (one-time shape injection) — dissolves under self-play PPO.
- **Ship head** (undersend) — `single` = 0–2%, send/garr ≈ 1.0; firing planets send ~their whole garrison.
- **Target head / coordination (architecture)** — the spare source is idle (80%), not aiming elsewhere, so
  it's not an inability to converge sources; no decode/arch fix needed.

## What's located

Under-aggregation by choice → lose the early **planet-count race** → out-massed downstream. Same root
for attacks and expansion. It is **systematic** (WON ≈ LOST), so it's a *ceiling-raiser*, not a
per-game flip — don't expect a clean WR jump from fixing it.

**Root cause refined (2026-06-21) — it's the IDLE FIRE HEAD, not the ship head or target head.**
On us-vs-Ajay loss replays, when a single-source attack fell short and a spare source was available
(`multi_source_why.py --our-name Saheb`), that spare source was **idle 80%** of the time (fired nothing),
attack-elsewhere only 18%, reinforce-elsewhere 2%; **89% of policy-short events had ≥1 idle spare source.**
So the 2nd source doesn't fire *at all* — the value/fire head doesn't value the supporting launch. It is
**not** the ship head (`single`-undersend = 0–2%, send/garr ≈ 1.0) and **not** the target head (spare isn't
aiming elsewhere). This is one behavior across attacks, expansion, and hoarding: the value function rates
**hold garrison > spend it** — the self-play mutual-turtle equilibrium.

## Why anchoring isn't the fix

The IL-anchor (frozen-teacher KL) is a λ-controlled tug-of-war between the teacher (aggregating) and the
self-play Nash (under-aggregating). Low λ → Nash reforms → wall. High λ → pinned near the teacher's
quality. So **anchoring caps you at the teacher**, and we have no teacher that is both aggregating *and*
strong (the clone is weak/spray; Ajay is a planner, not an il-ref-able policy). Self-anchor re-anchoring
reached best-ever 18% Ajay but still plateaued at the wall. **Decision: next run drops the anchor.**

---

## The plan — PBRS staging reward first, Door 1 as the evidenced fallback

**Step 1 — PBRS staging reward (chosen; build + run).** Potential-based shaping so the idle fire head
gets the directed gradient it's missing, spray-safe by the telescoping guarantee.
- **Φ(s)** (vectorized in `torch_env`, reusing `_decisive_mass_fields` `mass`/`floor` tensors): `top-k Σ of
  min(1, friendly_inbound_to_target / capture_floor)` over the k highest-progress **NEUTRAL** targets
  (owner < 0, reachable, below floor). **k = 2** to start.
  - **Gate on `owner < 0`, NOT the existing `is_enemy` gate** (torch_env.py:1112). Enemy-owned targets are
    deliberately excluded for v1: staging onto them = the reactively-defended contest we lose (out-massing) =
    `decmass` territory (failed). Neutral-only keeps PBRS a single clean lever aimed at the planet-count race.
  - "Neutral" still includes *contested* neutrals — the floor folds in enemy inbound + reactive mass, so the
    pooling case (2+ sources to take & hold a fought-over neutral) is in scope. Out-massing is addressed
    *indirectly* (more neutrals → more mass). Defensive staging (own threatened planets) = deferred v2.
  - Revisit enemy targets only if neutral-only plateaus *because* neutrals get scarce (they don't in the
    decisive 10–50 window: 13–19 remain).
- **Reward**: `r_t += α·(γ·Φ(s_{t+1}) − Φ(s_t))` (strict PBRS → telescopes → can't be farmed by spray).
- **α = 0.2** (pre-test-calibrated), sweep {0.1, 0.2, 0.3}. Flags `--staging-shaping-coef`, `--staging-topk`.
- **Run design (EXECUTED as `r32_stage_h12` — see LIVE RUN section above for actual config + results):**
  warmstart from the **jake-BC seed** (not a PPO checkpoint — the BC doesn't carry the shape, so PBRS builds
  it), **no anchor**, **h12-only** external (winnable → the win-gradient bootstraps), + **fire-entropy spike
  0.05** (added after the BC-shape finding to beat the clip/sampling bottleneck). PBRS the only reward delta.
- **Success**: caps@50↑ + planet-deficit closes + out-massed↓ with reinforce held + fired-spare rate ↑ from
  ~3%. **Tripwire**: `fire_frac`/`launch_rate` balloon *without* caps↑ = spray → kill.
- **Pre-build check (DONE)**: unit test proves the telescoping property (Σ shaping ≈ γᵀΦ_T − Φ₀). Status:
  live; first signal fired-spare 3.9%→6.2%@2M (positive, early — verdict at 3–4M).

**Step 2 — Door 1 (only if PBRS plateaus).** A *permanent* aggregating opponent in the pool so spare-use
predicts winning in the data (breaks the loop at the data layer, not just the signal). More expensive
(needs a strong aggressor); justified only if PBRS bootstraps then stalls — which would mean the symmetric
*data*, not the signal, was the bottleneck. SSDR/h14 were the right instinct but *transient* (reverted to
self-play); the fix is permanence.

**Why prior reward attempts don't apply:** `decmass` rewarded the *instantaneous floor-cross outcome*
(a lone launch never reaches it → no gradient → idle); PBRS rewards *progress* toward the floor.
`expansion-coef 0.03` was a level bonus, too weak and farmable; PBRS is a capped, telescoping difference.

## Pointers

- Memory: `project_undermass_by_choice`, `project_ajay_dagger_seed`, `project_aggregation_probe`,
  `project_force_concentration_wall`, `project_decisive_mass_lever`, `feedback_win_starvation`.
- Tools (all in tree): `transition_autopsy.py`, `expansion_autopsy.py`, `value_spare_diagnostic.py`
  (+`--shaping-coef` PBRS pre-test), `gpu_run_artifacts/multi_source_why.py` (+`--our-name`),
  `plot_train_log.py` (5×4 staging-first trend plot of a train log). Mostly untracked — commit if keeping.
- PBRS reward: `torch_env._staging_potential` + step() shaping; flags in `train_torch.py`; test
  `tests/test_staging_shaping.py`.
- **Current run: `r32_stage_h12`** (Jarvis spot 431251) — LIVE; PBRS staging from jake-BC, first signal
  positive (fired-spare 3.9→6.2%@2M). DESTROY 431251 when done.
- Prior run: `r32_ajayclone` (431164) — warmstart bet FAILED (shape dissolved by 3M); destroyed.
