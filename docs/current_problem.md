# Current Problem — the Force-Concentration Wall, diagnosed

_Last updated: 2026-06-21. Supersedes the scattered "wall" notes; this is the standing statement of
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

**Chosen lever: PBRS staging reward** (potential-based: `r += α(γΦ(s')−Φ(s))`, Φ = top-k staging
potential toward contestable floors). It injects *directed* exploration the idle head is missing —
validated by a cheap pre-test (α·shaping ≈ +0.19/step ≫ the A≈0 floor). Run PBRS alone first; if it
plateaus, that's the signal to add Door 1 (a permanent aggregating opponent). Prior reward attempts
failed for nameable reasons that PBRS avoids — see below.

Baseline reference (ild 6M vs Ajay): WR ~23%, out-massed 96%, caps@50 LOST 6.1, planet-deficit ~−2.

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
- **Run design**: warmstart off the jake/ild lineage (~5–6M), **no anchor**, base pool (prod-share,
  h12+h14 @0.60) — PBRS the only delta. Pure self-play first to isolate it.
- **Success**: caps@50↑ + planet-deficit closes + out-massed↓ with reinforce held + fired-spare rate ↑ from
  ~3%. **Tripwire**: `fire_frac`/`launch_rate` balloon *without* caps↑ = spray → kill.
- **Pre-build check**: a unit test proving the telescoping property (Σ shaping ≈ γᵀΦ_T − Φ₀) before any GPU.

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
  (+`--shaping-coef` PBRS pre-test), `gpu_run_artifacts/multi_source_why.py` (+`--our-name`). Untracked —
  commit if keeping.
- Last run: `r32_ajayclone` (Jarvis spot 431164) — warmstart bet FAILED (shape dissolved by 3M); **destroyed**.
