# Experiment Queue

One line per experiment, in rough priority order. One change per run; record hypothesis in
`docs/training.md` before launching, verdict after. Details live in `docs/writeup_lessons.md`.

> ## ⭐ Read this before proposing anything (2026-07-16)
> 1. **Verdict is Yijie, not Ajay.** Ajay saturates ~77–80%. Our 80.5%-Ajay champion is **0/256 vs
>    Ender** and 3.9% vs Yijie. Ajay moved 57→80% across the whole timeline+binary program and the
>    north star (`open<50 cap/atk` vs Ender) did not move at all (0.58 → 0.517).
> 2. **"Train longer" is not an explanation** — check the curve. tl100m_s2 added 35M for ~+1pp/10M.
> 3. **Probe before you propose.** Three beliefs and a ~30h experiment died to <1h of measurement
>    (`ender_sizing.py`, `peel_diagnosis.py`, gate-pressure probe). See CLAUDE.md Key Lessons 12–14.
> 4. **Suspect the hardcoded constants.** They deleted 80.2% of the action space and nobody had
>    ever measured it. The winners use soft priors; we shipped hard masks.

**Verdict metric is YIJIE.** Ajay saturates ~77–80% and is a regression guard only — the column
below exists to show that it moved 57→80% while nothing that matters moved at all.

| Frontier | Status | Ajay (guard) | **Yijie (verdict)** | Decision |
|---|---|---:|---:|---|
| **`binarygates100m_l4`** (Arm B) | ⭐ **RUNNING** | — | — | `--binary-commit-gates minimal`, from scratch 100M. **Bar: >6–7% Yijie** |
| `shipkl_probe` (absolute + soft ship-KL, ~136M cum.) | Plateaued | ~80% | **5.9–7.0%** | ⭐ **BEST YIJIE — the bar.** "dead flat 1M→8M" |
| Exact-marginal binary 40.108M | Best Ajay | **80.5%** | 3.9% | **0/256 vs Ender**, wiped 100%. Ajay peak bought nothing vs strong play |
| Target counterfactual 45.711M | Complete | 74.2% | 5.9% | No promotion; best Yijie of the binary lineage |
| Target+source counterfactual + L4 25.068M | Complete | 75.8% | 3.9% | Source channels active; no promotion |
| tl100m_s2 (+35M → ~135M) | **Plateaued** | ~77% ±4 | — | ~+1pp/10M. Killed the "budget will fix it" story |
| Forced projected hold | **Rejected** | 1/16 vs 13/16 (paired slice) | — | Underprices the opponent response |
| Learned middle commitment | **Rejected (measurement)** | — | — | Ender all-ins 97.3% vs Ajay / **97.7% vs itself** ⇒ worth ≤3% of launches |
| Submitted-agent cross-eval | Complete | 69.9% `presres1` · 64.1% `stgpr1` | — | Not a sweep; retain both as regression gates |
| Best-checkpoint anchor + gate | **Built, unrun** | — | — | Back-pocket for 200M+ (tl100m ran 100M unanchored, no collapse; noopkl2 was cold-Adam, fixed). ~15–20% throughput |
| Global economy series | **Built, unrun** | — | — | Opt-in `--global-econ`; ground-truthed vs engine, parity 0 error |

## ⛔ "Just train longer" — CHECKED AGAINST OUR OWN CURVES, AND IT DOES NOT HOLD (2026-07-16)

The tempting story is that every flat Yijie verdict is a budget artifact (25–55M vs Yijie's 13B).
**Our own data refutes it.** `tl100m_s2` continued tl100m from 99.5M for **+35M more steps**:

| cumulative steps | ~100M | ~110M | ~120M | ~130M | ~135M |
|---|---:|---:|---:|---:|---:|
| Ajay | 74.6 | 76.6 | 73.0 | 81.2 | 78.1 |

That is a **plateau at ~77% ±4**, i.e. ~+1pp/10M — not the "+5pp/10M at the tail" the stage-1 note
projected (that read the noise band of a saturating metric as a trend). And the Yijie curves are
flat over long stretches within a lineage: `binarymarg` 1.2 → 4.3 over 45M; `binary100m_scratch`
Ajay **plateaued at 48–49% from 30M through 50M** while Yijie sat at 1–2%.

**So plateaus are real and we do detect them.** Budget is not the free explanation. What the
curves also show, and this is the uncomfortable one:

| lineage | style | Yijie |
|---|---|---:|
| `shipkl_probe` (ship-KL on the tl100m timeline lineage, ~136M cumulative) | spray-ier, reinf 0.35 | **5.9–7.0%** |
| `binarymarg` / binary all-in (~70M cumulative) | disciplined, reinf 0.21–0.36 | **1.2–4.3%** |

**The binary all-in program bought Ajay (57→80%) and appears to have COST Yijie (~7%→~3%).**
Confounded by cumulative budget (136M vs 70M) and lineage, so it is not a clean A/B — but it is the
opposite of what a "we're on the right track, just under-trained" story predicts. The pre-binary
lineage was closer to the winners on reinforce share AND better against the strong opponent.

**A concrete mechanism for that regression now exists — see docs/training.md "THE REINFORCEMENT
LEGALITY WALL".** Measured on 758 real own-target cells: only **14.6% of reinforce options are
legal**, because binary mode resolves own targets to `maintain = enemy_mass_soon + 1` and requires
`>= 5` ships to be feasible — and `enemy_mass_soon` (enemy fleets arriving within **6** steps) is
**0 in 81.3% of cells**. Infeasible targets are stripped from the target softmax. **Binary mode
cannot pre-emptively reinforce; only react within a 6-step window.** `ship_bin_mode="absolute"`
(the better-vs-Yijie lineage) has no such gate — the walls are **binary-mode-only**
(`if ship_bin_mode == "binary"`), so ship-KL was never subject to them. The theory is therefore
NOT "ship-KL was hobbled and deserves a retry"; it is the reverse: **ship-KL ran without these
gates, with a learned size head and a soft prior, and produced our best Yijie — then the binary
design replaced it with hard masks AND disabled `ship_kl` (ppo.py:308 `and not binary_mode`).**
Arm B (#0) is the test.

## Measurement: we are flying with two broken instruments

- **Ajay saturates** (~75–80%) and is ~+240 Elo below the level we care about.
- **Yijie floors** (0–6%): at 1640 Elo he is ~+440 over an Ajay-class bot, so an agent that is
  *genuinely* Ajay+240 still scores ~10–15%. A 256-game panel at 4% has a ±2.5pp band — most
  real improvements are invisible on it.
- Nothing gives gradient in the 1300–1500 band. Consequence: use paired production/material
  deltas and loss-depth as the graded signal (docs/metrics.md), not the Yijie WR.
- The two panels also disagree about our strength (Ajay says +240, Yijie says −450). Both
  cannot be true of one number: something about strong *learned* play punishes us specifically,
  and the replays already named it — the production race / source drain.

## Completed validation

- **Submitted-agent cross-eval integrity audit** — the old wrappers errored before acting, so their
  256/256 results are void. Corrected 256-game panels against the hash-validated final payloads gave
  179/256 (69.9%; seats 66.4/73.4%) versus `presres1` and 164/256 (64.1%; seats 55.5/72.7%) versus
  `stgpr1`. The evaluator now requires `DONE/DONE` and hashes the tracked archives before play.

## Next in line

0. ⭐ **`--binary-commit-gates minimal`** — **RUNNING** (`binarygates100m_l4`, from scratch, 100M,
   GCP L4 asia-south1-b, launched 2026-07-16). Verdict metric: **Yijie** panel; Ajay = guard. Deletes the two
   hand-tuned walls (`capture_required`, `maintain`/`defend_ok`); COMMIT = all-in at any target,
   gated only on `S >= MIN_BINARY_COMMIT_SHIPS`. Measured: action space **19.8% → 83.7%** legal on
   the same states. Nothing new is added — it is pure deletion; ch10/ch20/ch22-25 remain as
   FEATURES so the model still sees cap-cost/threat/resolved-sizes, it just isn't overruled by
   them. This is SimJeg's shipped design and matches Ender's measured 97.7% all-in.
   **Bar: Yijie ~6–7%** (the ship-KL/absolute plateau — the repo's best). Beating binary's 3–4% is
   not success. Expect cap/atk and possibly Ajay to fall; that is the trade. Full contract,
   tripwires and the "what it bets against" in docs/training.md.

1. ~~**Best-ckpt anchor + promotion gate**~~ **BUILT 2026-07-16** (`--anchor-kl-coef`,
   `--anchor-value-coef`, `--anchor-promote-winrate/-min-games`, `--anchor-from`). KL(live ‖
   frozen best) over the exact NOOP/COMMIT distribution + value MSE; promotion adopts the live
   policy at ≥70% EMA h2h over ≥1024 games and demotes the old anchor into the league. Verified:
   6 unit tests (identity ⇒ KL 0; loss += coef·KL; gradient reduces KL) + an end-to-end CPU run
   where the gate fires and resets. **Unrun at scale.** Costs one no-grad forward per minibatch.
   ⚠ The anchor accrues gate games only when sampled — pass `--pool-pinned-fraction` (it is
   pinned) or as 1-of-20 members it sees ~5% of the pool slice and the gate crawls.
2. **Global economy series** — BUILT 2026-07-16 (global dim 15→63). Contract in docs/training.md.
3. ~~**Learned commitment (NOOP / HOLD / ALL-IN)**~~ **REJECTED 2026-07-16 by measurement.**
   Ender all-ins **97.3%** of launches vs Ajay and **97.7%** vs itself (opening attacks 100.0%) —
   `ender_sizing.py`. The strong-vs-strong control kills the "all-in only works vs weak play"
   confound. Our resolver already matches top-10 sizing ~97% of the time; a learned middle
   addresses ≤3% of launches. Sizing is settled — do not spend a run on it.
   ~~Still open: the single-source affordability mask~~ → **measured and folded into Arm B (#0)**:
   `capture_required` blocks **62.2%** of attack options, so pincers were inexpressible. It is one
   of the two walls `--binary-commit-gates minimal` deletes.
3b. **Delete the magic horizons** (follow-up to Arm B, cheap and high-information).
   `_THREAT_ETA_WINDOW=6`, `_REACH_HORIZON=18`, `_VALUE_HORIZON=40` are three arbitrary answers to
   "how far ahead should the model look", introduced together in one BC commit (`7bd0ffe`) with no
   justification and never ablated — why 6 and not 4 or 8? Meanwhile `TIMELINE_K=24` already hands
   the model the whole resolved future, so these scalars add **no information**, only an
   unjustified prior about which horizon matters. **Prediction: removing ch20/ch15's hand-picked
   summaries is a NO-OP.** If it hurts, the timeline isn't doing its job — also worth knowing.
   Pairs naturally with #5 (learned pooling picks the horizon instead). Do NOT bundle with Arm B.
4. **Combat-preview scalars** — endpoint owner/ships/flip-margin per planet (Jake); cheap add-on to the timeline, covers the one thing it doesn't hand over (margin).
5. **Conv1d timeline encoder** — SimJeg's 1D-CNN into the planet token; only if timeline signal looks bottlenecked by the linear projection. Yijie ran a 1D-CNN + attention pool over his series and notes most of his FLOPs landed there — and that Billy/Simon got away with flattening it into the MLP instead.
6. **Surrender / early-truncation** — cut compute on decided games (Jake: 60–70% of turns); sample-density multiplier, not raw SPS. ⚠ Yijie *tried and dropped* a resign rule (75% ships for 20 turns): models sometimes collapsed in games they had all but won.
7. **Capacity jump** — `--entity-dim`/`--num-layers` up (0.5M → 5–20M); needs H100/H200 (update-bound: bigger model = proportionally slower). Calibration: Yijie's final was **1.2M** (6-layer, 128-d) and his 4M attempt was *much worse*; we are at 0.5M.
8. **Exploiters** — train a fresh model purely to beat the main one, fold into the league (Ender/rank-55: +15–17pp first-place).
9. **Recipe deltas to fold into the long run** (not separate arms — confounded on purpose, adopt
   as a block with the winners' settings): **gamma 0.995 → 0.999** (Yijie; economic reversals play
   out over 100+ steps — his gamma=1 stalling note bounds it from above), **rollout 64 → 128**.
10. **Yijie-style model selection** — round-robin pool, score = mean WR over all pairs, instead of
   one saturating panel (writeup lesson 6; kiyotah's gate picked a 668M ckpt when 412M was better).

## ⭐ RETENTION — what the Ender panel actually locks (2026-07-16)

Champion (80.5% Ajay) vs Ender: **0/256**, wiped to 0 material in **100%** of games, **peel-rate
0.99** (5,277 of 5,309 captures lost), planets 8 @50 → **4** @100 vs Ender's 18.5 end. Production
delta **+0 at step 32 → −34 at 100**: level on the economy, then it compounds away — the same
shape as the Yijie/Ajay loss replays. `open<50 cap/atk` **0.517 vs Ender 0.75**, unmoved since
presres1's 0.58 despite Ajay 57.4%→80.5%.

**Forensics** (`peel_diagnosis.py`, 12 games, 250 captures — numbers below are the CORRECTED run;
see the retraction at the end of this section). Peel 0.99 is **tautological** under 100%
elimination (250 episodes, 249 lost, 1 held at end) and must stop being cited — it restates "we
lose". What the per-capture data actually says:

| read | value | reads as |
|---|---:|---|
| hold when landing **safe** & game even | **57st** | **we CAN hold what lands safely** |
| hold when landing **out-massed** | **14st** | 4× shorter |
| captures landing already out-massed | **61.6%** | |
| lost captures never reinforced | **63.9%** | |
| captures made while already behind | **65%** | thrash |
| churn (captures ÷ distinct planets) | **1.76×** | we re-take the same rock |
| "we take what we can't hold" | 21.4% of competitive captures | **not dominant** |
| "terminal collapse" | 13.7% of losses | **not dominant** |

**Neither classic hypothesis explains it.** Holding is not broken; selection into unholdable
positions plus churn is the profile. **The mechanism is now identified** — the reinforcement
legality wall above: we could not pre-emptively garrison a fresh capture (85.4% of reinforce
options illegal), so "capture, then abandon" was largely *structural*, not a policy choice.
Arm B (#0) is the test.

⚠ Do NOT respond with a reinforcement reward term (lesson 11). ~~Points at budget first~~ —
**that would contradict the plateau evidence at the top of this file.** The features that price a
capture's survival already exist (target-CF `held-through-horizon` / `mine-at-arrival`, 74.2%
Ajay / 5.9% Yijie at 45M, no promotion). If Arm B does not move it, the next suspect is
**decoding structure** (writeup_lessons §2, *not* item #2 here): after an all-in capture the
source is empty, so the follow-up must come from a *different* planet — coordination that fully
parallel per-source decoding cannot express.

⚠ **Two retractions, both mine, both worth remembering:**
1. *"92% of lost captures never reinforced"* — **wrong: 63.9%.** The probe built its agent without
   setting `allow_reinforce` on the model (`build_agent_fn` reads it off the object, eval.py:391/
   :1560), so reinforcement was **disabled in a diagnosis about reinforcement**. Every other number
   moved <2pp. Any probe must configure the model exactly as `evaluate_checkpoint` does.
2. *"reinforce share 0.21 vs Jake 0.56"* — **opponent-confounded.** Like-for-like vs Ajay:
   **us 0.49, Ender 0.434** — we reinforce slightly MORE. Never compare a rate across opponents.
   Surviving (weaker, state-confounded) observation: Ender *raises* reinforcement under pressure
   (0.434 vs Ajay → **0.708** vs Ender) while we *lower* it (0.49 → 0.42). Hypothesis, not a lever.

## Parked / conditional

- **Zero-pad warm-start** — presres1 + zero-padded timeline columns; only as a fast confounded side-signal, never the primary run.
- **4p variant work** (separate models, per-player value heads — Jake) — after 2p is competitive.
