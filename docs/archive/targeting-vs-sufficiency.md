# Targeting vs sufficiency — session findings (2026-06-17)

Investigation that started from the head-audit "fire/target selection is the bottleneck" hypothesis and
ended concluding the opposite: **target SELECTION is systematic but not the lever; the binding flaw is
mass SUFFICIENCY / sizing** (how much force we commit relative to the target's defense), in both the
opening neutral race and enemy contests. Companion to `docs/head-audit.md`, `docs/outmass-limits.md`,
`docs/metrics.md`.

Checkpoint under the microscope: `gpu_run_artifacts/revedge1/checkpoints/torch_step_4718592_revedge1_20260616_095906.pt`
(revedge1 4.72M, reinforce + reverse-edge-cooldown lineage). Opponent: `opponents/candidate_ajay_1200.py`.

---

## TL;DR

0. **⭐ THE OVERRIDE SERIES (read first) — the policy out-decides every heuristic we tried, at fire, target,
   AND defense. Forcing it toward our "better" choices ties or destroys WR.** This is the dominant result:
   the micro-policy is *not* the wall; our heuristics are not oracles. See the override table below.
1. **Target selection is NOT a deficiency — the trained head out-targets our best heuristic.** Forcing every
   attack onto the top-holdable-ROI target → **0/256 (0.0%)** even with funneling ruled out (distinctness 0.96)
   and sizing fixed (`--retarget-resize`). So "we pick the holdable-ROI-best target only 29% of the time" is the
   policy **correctly overriding a greedy myopic metric**, not a flaw. holdable-ROI is a good *sufficiency*
   diagnostic but a **bad target oracle** — which retroactively undercuts the dom%/best% "selection is weak" read
   (it measured agreement with a non-oracle). The won/lost gap was small because the metric, not the policy, is weak.
2. **The fire head is fine — confirmed by direct manipulation, not the rate argument.** Force-firing the vetoed
   high-ROI sources → **3.1% (8/256)**: firing more sprays mass thin (`dm cross` 0.54→0.06) and collapses WR.
   The head's conservatism is correct and downstream of the sufficiency constraint. (We already fire *more* than
   Isaiah: 0.079 vs 0.036.) No training floor gates it (decisive-mass is a `coef=0` reward, not a mask).
3. **ROI *is* used by the net — but for COMMITMENT/conversion, not target ranking.** Scrambling roi nearly
   halved WR (23.8→10.2%) while every target-selection metric stayed flat; what collapsed was mass-assembly
   (`dm cross` 0.54→0.41) and conversion (`cap/atk` 0.59→0.51).
4. **Sizing is mis-calibrated in BOTH directions.** vs the winner of a real lost game (80267509): we
   **under-commit big rich neutrals** (16 ships vs g25–30) *and* **over-commit the small ones** (median
   send 2.7× garrison vs the winner's lean 1.1–1.8×). The winner is "just enough, almost always."
5. **The lever is sizing/sufficiency — and it's already instrumented** by the decisive-mass `dm` line
   (enemy contests), now extended with a **neutral arm** for the opening land-grab. But note (0): no per-action
   *override* installs it either → the residual is the **training equilibrium**, i.e. curriculum.

---

## The override series — the policy beats every heuristic (the dominant finding)

We repeatedly built a heuristic "better" action and forced the policy toward it at eval. Every one ties or destroys
WR (control = **23.8%**):

| override (eval-time) | what it forces | WR | reading |
|---|---|--:|--:|
| defensive selection (value overlay) | force value-gated saves | 23.0% | ≈ control |
| defensive overfill ×1.25–2.0 | bigger saves | 18–21% | ≤ control (saves still die) |
| **force-fire** (ROI≥0.3) | fire vetoed high-ROI sources | **3.1%** | spray → mass diluted → collapse |
| **retarget→top-ROI** | best-ROI target, keep ships | **0.0%** | ROI is a bad target oracle |
| **retarget→top-ROI + resize** | best-ROI target, sized to capture | **0.0%** | confirmed: not sizing/funnel — ROI targets are strategically wrong |

**Interpretation:** the policy's per-action choices — fire, target, ship — are each *better* than our best static
heuristic; overriding any of them ties or destroys WR. A self-play-trained head is **context-dependent and
multi-objective** (phase, role, sequencing, opponent); our heuristics are single-objective myopic snapshots. So
the wall is **not in the micro-policy** — it's in what the policy was *trained to value* (the self-play equilibrium
that never prices force concentration). **Methodological correction:** use these heuristics as *diagnostic lenses*,
never as *oracles to override toward*. The repeated "low agreement with planner/ROI" numbers (fire 13–30%, dom% 62%,
best% 29%) were the policy being **right** and the heuristic being a decoupled proxy — not head defects.

Mechanism note on retarget: `cap/atk` stayed ≈ control (0.50 vs 0.59) — we still capture at a normal rate, we just
capture the **wrong (ROI-greedy) planets** → out-massed 99% → eliminated fast. Not a sizing failure; a strategic one.

---

## What we built this session (durable tooling)

| Tool | Where | What it measures |
|---|---|---|
| Replay winner-vs-loser head audit | `orbit_wars_rl/audit_replay_head_labels.py` | our heads' agreement with rank-1 replay winner vs loser moves; projection-loss (move/mass); same-source target baseline + fire-source contrast |
| `near-vs-far` + `dom%` conversion line | `eval.py` `game_conversion` | target choice: nearest%, chosen/nearest dist ratio, `dom%` = passed up a closer-AND-richer target (by phase × won/lost) |
| `holdable-ROI rank` conversion line | `eval.py` `game_conversion` + `_holdable_roi` | reactive-aware target quality: `best%`/`top3%` using the decisive-mass capture floor (by phase × won/lost) |
| `--ablate-roi` / `--ablate-sun` | `eval.py` + `features.set_ablate_roi`/`set_ablate_channels` | permute a pairwise channel across targets per slot → test whether the net USES that feature |
| `decisive-mass NEUTRAL` line | `eval.py` `_decisive_gap_step(targets="neutral")` | neutral capture sufficiency: `mass / static_garrison` (neutrals don't grow/reinforce), by phase |

All eval lines are step-weighted like the existing `dm` line and ride every panel (watchers pick them up).
The per-game autopsy approach (send vs target-garrison; flight-time/tempo) was a throwaway script — reproducible
from a replay JSON with `eval._resolve_launch_target` / `_cap_cost_at_arrival` / `_holdable_roi`.

---

## Findings (with numbers)

### 1. Selection is systematic but not the lever
Full panel vs Ajay, by phase × outcome:
- `near-vs-far`: opening `dom%` 62% (WON 56 / LOST 64), chosen target ~3× farther than nearest, nearest% ~16%.
- `holdable-ROI best%` (chose the top reactive-aware target): early WON 32 / LOST 28; mid 23 / 19; late 36 / 28.

Winners pick the #1 holdable-ROI target only ~32% of the opening and **still win**; the won-vs-lost gap is only
4–8pp. → target selection is imperfect across the board but **does not separate outcomes** → not the binding constraint.

### 2. Fire head exonerated
- `fire_frac` WON 0.18 (≈ Isaiah ref 0.17), LOST 0.29 (inflates on losses).
- `--decisive-mass-coef` defaults to **0.0** (`train_torch.py`); the `dm`/floor is the `--decisive-diag` *measurement*,
  and even when on it is a **reward bonus** (`torch_env._decisive_mass_bonus`), never a fire gate. So a closed fire
  gate is *learned* behavior, not a hand-set floor. (Masks that DO gate in training: reinforce-gate, min-ship-bin,
  sufficient-commit — none blocks a neutral attack.)

### 3. ROI ablation — roi drives commitment, not targeting
Control vs `--ablate-roi` (roi_20/roi_50 permuted across targets):

| metric | control | roi-scrambled |
|---|--:|--:|
| WR | 23.8% | **10.2%** |
| decisive-mass cross | 0.54 | **0.41** |
| cap/atk-launch | 0.59 | 0.51 |
| reinf_share | 0.34 | 0.20 |
| dom% (open) / holdable-ROI best% (open) / nearest% | 62 / 29 / 16 | **61 / 28 / 16 (flat)** |

roi is heavily used (WR halved), but the change is concentrated in **mass-assembly + conversion**, not target
choice. `cap/atk` and `caps/game` are conversion *outcomes* (blend targeting + sufficiency); the *pure* selection
metrics stayed flat while the *pure* sufficiency metric (`dm cross`) dropped → the conversion loss flows through
the **sufficiency** channel.

**Placebo control (`--ablate-sun`, ch4 sun_safe, flips on ~16% of pairs): WR 21.1% (54/256) ≈ control**
(within ~1 SE of 23.8%), with `dm cross` 0.53 and `cap/atk` 0.58 also ≈ control. Scrambling a real channel
did NOT hurt; scrambling roi did → **the roi effect is roi-specific, not general brittleness.**
Caveat: sun_safe is binary while roi is continuous, so the placebo isn't a perfectly matched perturbation; and
permutation feeds out-of-distribution inputs, so −13.6pp is technically an **upper bound** on roi's functional
importance — but the placebo rules out the brittleness explanation, so most of it is real.

### 4. Sizing is mis-calibrated both ways (game 80267509, a real LB loss vs Mike Kim)
Neutral capture sufficiency (`mass / static_garrison`), by phase, both seats:

| phase | Saheb (LOST) cross / p50 | Mike Kim (WON) cross / p50 |
|---|--:|--:|
| <50 | 0.86 / 1.33 | 0.98 / 1.12 |
| 50–100 | 1.00 / 2.67 | 1.00 / 1.82 |
| ≥100 | 1.00 / 2.78 | 1.00 / 1.54 |

- **Under-commit:** opening cross 0.86 vs winner 0.98 — we throw ~16-ship fleets at g25–30 neutrals that can't
  capture them (16 < garrison). Confirmed in the launch autopsy (t23→g25, t39→g26 [Mike then took it, g26→46],
  t45→g30; all wasted).
- **Over-commit:** our median send is 1.3–2.8× the garrison vs the winner's lean 1.1–1.8× — surplus ships that
  should have expanded elsewhere.
- **Tempo:** far targets → long flights (eta 16–35) → first capture at t21 vs the winner's t16; the winner chained
  expansion off his earlier capture (a close prod-4 grab at holdable-ROI 0.86, rank #1).

The winner is **lean-and-sufficient**: high cross (~0.98–1.00) with low p50 (~1.1–1.8). Our flaw is *calibration*
(match the send to the target's defense), not "send more."

**Panel-scale confirmation:** aggregate `decisive-mass NEUTRAL cross` over 256 games = **0.87** (≈ the single
game's 0.86) → the opening neutral under-commit is **systematic**, not a one-game anecdote.

### 5. The overlay corroboration (from `docs/head-audit.md`)
Forcing "correct" defensive saves (value-gated k=3 overlay) only reached control (23.0% vs 23.8%), and the forced
saves *died* (reinf-mass-to-lost 38% vs control 27%) — selecting the right contest doesn't win if the mass is
insufficient. Same conclusion from the offensive side here: **selection isn't the lever; sufficiency is.**

---

## The lever, stated precisely

Match committed mass to the target's actual defense — in both directions (stop under-committing the big valuable
planets, stop over-committing the small ones), and aggregate across sources when one planet isn't enough. This is
the force-concentration / out-mass wall (`docs/outmass-limits.md`), now localized to **two measurable surfaces**:
- **enemy contests** → `decisive-mass` `dm` line (`cross`/`gap`/`p50`), long-standing.
- **opening neutral race** → `decisive-mass NEUTRAL` line (new this session).

Read both by phase, and against a winner reference (lean-and-sufficient: high cross, low p50).

---

## Caveats / honesty

- **Ablation magnitude is an upper bound** — permutation injects out-of-distribution inputs; the placebo (sun)
  firms up roi-specificity but isn't a perfectly matched perturbation.
- **dom% uses raw production; holdable-ROI uses the reactive floor** — prefer holdable-ROI (dom% over-flags
  unholdable closer-richer targets).
- **`dm`/neutral lines are step-weighted** — a sufficient fleet on a long flight contributes many `cross≥1`
  entries; read the `<50` phase split and don't read aggregate `cross` as "no problem."
- **Single-game autopsy is an anecdote** — game 80267509 is one opponent; the panel aggregates are the robust read.
  The winner reference there is one strong player.
- **A better feature ≠ the net using it.** The roi ablation shows the net *does* use roi (for commitment), so a
  reactive-aware hold-ROI *input* is plausible for sizing — but provision never guarantees usage; test via the
  ablation (does scrambling the new channel move `dm cross`?) and watch for the reward-proxy/imitation traps that
  killed decmass and rev54.

---

## Override series — verdicts (all done)

- [x] Sun placebo: **21.1% ≈ control** ≫ roi run 10.2% → roi effect is **roi-specific**, not brittleness.
- [x] Aggregate neutral `dm` cross **0.87** → opening under-commit is **systematic at panel scale**.
- [x] **Force-fire** (ROI≥0.3) → **3.1%** → fire veto is correct; firing more sprays/dilutes.
- [x] **Retarget→top-ROI** → **0.0%**; **+resize, funnel-ruled-out (distinctness 0.96)** → **0.0%** → the trained
      target head **out-targets argmax-holdable-ROI**; holdable-ROI is a sufficiency diagnostic, **not** a target oracle.
- [x] **Conclusion:** no per-action override (fire/target/save/sizing) beats the policy → the wall is the **training
      equilibrium**, not the micro-policy → **curriculum** is the indicated lever.

## New eval flags (this session)
`--ablate-roi` / `--ablate-sun` (feature-usage), `--force-fire-high-roi [--force-fire-roi-threshold]` (fire isolation),
`--retarget-top-roi [--retarget-resize]` (selection isolation), `decisive-mass NEUTRAL` line (opening sufficiency).

## Open / next

- [ ] **Curriculum** (the indicated bet): make lean-sufficient concentration *necessary to win* vs a beatable,
      **co-improving** opponent (fix the h-ladder's win-starvation + fixed-bot cheese). Not a reward proxy, not an override.
- [ ] If a *capability* gap is suspected (factored heads can't size-to-target / aggregate cross-source):
      **autoregressive heads** (fire→target→ship, and/or cross-source) — but it needs the signal too; cold-start/BC risk.
- [ ] Optional clincher: per-failed-attack decomposition (undercommit vs wrong-target) to split `cap/atk` losses directly.
- [ ] ⚠️ Demote dom%/holdable-ROI `best%` from "target-quality" reads — the retarget collapse shows holdable-ROI is
      not a valid target oracle. Keep it only as the **sufficiency** (`dm cross`) diagnostic.

[[outmass-limits]] [[head-audit]] [[metrics]]
