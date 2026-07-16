# Experiment Queue

One line per experiment, in rough priority order. One change per run; record hypothesis in
`docs/training.md` before launching, verdict after. Details live in `docs/writeup_lessons.md`.

| Frontier | Status | Decisive evidence | Decision |
|---|---|---|---|
| Exact-marginal binary 40.108M | **Ajay baseline** | 80.5% Ajay · 3.9% Yijie | Retain; best Ajay checkpoint |
| Target counterfactual 45.711M | Complete | 74.2% Ajay · **5.9% Yijie** | No overall promotion; best Yijie read |
| Target+source counterfactual + L4 25.068M | Complete | 75.8% Ajay · 3.9% Yijie | Added source channels active; no promotion |
| Forced projected hold | **Rejected** | 1/16 vs 13/16 all-in on paired Ajay slice | Underprices the opponent response |
| Submitted-agent cross-eval | **Complete** | 69.9% vs `presres1` · 64.1% vs `stgpr1` | Current champion is stronger, not a sweep; retain both as regression gates |
| Best-checkpoint anchor + gate | **Built, unrun** | Unit + end-to-end verified (`tests/test_anchor.py`) | Back-pocket for 200M+ drift — NOT a prerequisite (tl100m ran 100M unanchored, no collapse; the noopkl2 collapse was cold-Adam, since fixed). Costs ~15–20% throughput |
| Global economy series | **Built, unrun** | Ground-truthed vs engine; parity 0 error | Feature arm — see contract in docs/training.md |
| Ender launch sizing | **Measurement** | Ender all-in 97.3% vs Ajay · **97.7% vs itself** | Kills learned commitment (#3) |
| Champion vs Ender panel | **Measurement** | **0/256**, peel 0.99, wiped 100%, prod +0@32 → −34@100 | Retention is the gap |

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
opposite of what a "we're on the right track, just under-trained" story predicts, and it is the
single most important open question in this table. The pre-binary lineage was closer to the winners
on reinforce share AND better against the strong opponent.

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
   Still open (small, unmeasured): the **single-source affordability mask** makes any planet no
   lone source can afford unattackable *by construction*, so combined-arms pincers are
   inexpressible. Cheap to probe before it is ever a run.
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

**Forensics (`peel_diagnosis.py`, 12 games, 235 captures) — the gap is FOLLOW-UP, not selection
or holding.** Peel 0.99 is tautological under 100% elimination (235/235 lost, 0 held at end) and
must stop being cited. What the per-capture data actually says:
- Captures that land **safe** while the game is even are held **55st** vs **16st** for out-massed
  ones — **we can hold what lands safely**. Hypothesis "we take what we can't hold" explains only
  19.5% of competitive captures; "terminal collapse" only 26.4% of losses. Neither dominates.
- **92% of lost captures never receive a single reinforcing launch**, and 60% land already
  out-massed. **We capture and abandon.** Our 0.42 reinforce share is going to the core, not the
  conquests. Ender spends **70.8%** of launches reinforcing and ends at 18.5 planets.
- 65% of our captures are made when already behind → thrash; churn 1.79× per distinct planet.

⚠ Do NOT respond with a reinforcement reward term (lesson 11). The features that price a
capture's survival **already exist** (target-CF `held-through-horizon` / `mine-at-arrival`) and
scored 74.2% Ajay / 5.9% Yijie at 45M with no promotion — we can already SEE it and haven't
trained long enough to ACT on it. Points at **budget** first, **decoding structure** (#2 —
follow-up must come from a *different* planet than the emptied source, which parallel per-source
decoding cannot coordinate) if budget stalls.

⚠ **Retracted:** the "reinforce share 0.21 vs Jake 0.56" deficit was **opponent-confounded**.
Like-for-like vs Ajay: **us 0.49, Ender 0.434** — we reinforce slightly more. Never cite a
reinforce-share gap across different opponents. The surviving (weaker, state-confounded)
observation: Ender raises reinforcement under pressure (0.434 vs Ajay → 0.708 vs Ender) while we
lower it (0.49 → 0.42). Hypothesis, not a lever.

## Parked / conditional

- **Zero-pad warm-start** — presres1 + zero-padded timeline columns; only as a fast confounded side-signal, never the primary run.
- **4p variant work** (separate models, per-player value heads — Jake) — after 2p is competitive.
