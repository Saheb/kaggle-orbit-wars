# Orbit Wars — Training & Eval Metrics Reference

What each number means, what's normal, and **which to trust**. This describes the *current*
metrics only; the history of how they evolved (target-resolution fix, dump prunes, superseded
corrections) is in git.

**Principle (2026-07 cull):** keep only metrics that are **outcome-grounded and provably track skill**;
rather have none than a conflated/misleading one. All model-based decisive-mass reads and the saturating
out-massed% were removed from eval — see the "dm culled" note in the Ender reference for the failure that
motivated it.

## Trust order

1. **Held-out win-rate** (Ajay / zach / Ender panels) — the only signal that sees *absolute*
   regression (our exploitability proxy, the real arbiter). Everything else is a leading indicator.
2. **Conversion** (eval, vs the top-player reference) — *why* we win/lose: capturing efficiently,
   expanding, deploying — or churning/hoarding? Printed on every eval.
3. **Behavioral degeneracy** (unconfounded tripwires): `fire_rate→0` (passive fire=0 Nash),
   `fire_frac→1` (carpet-bomb), `ship0→high` (1-ship-probe collapse).
4. **EV, entropy, KL/clip** — *training health* (is the optimiser sane), **not** policy quality.
5. **Vμ, rewμ** — weak corroborators. **Ignore the sign.** Never kill a run on these alone.

**Why held-out eval is mandatory:** internal metrics are blind to *mutual* self-play collapse — if
both policy copies degrade together, value / KL / win-rate-vs-self look fine while the agent gets
globally worse. Exploitability (how badly a best-response beats you) is the principled measure;
win-rate vs a fixed panel approximates it. (Same reason PPO logs value *explained variance*, never
the value mean.) Conversion metrics in the **training diag** are self-play-confounded too — read the
**eval** conversion numbers as the trusted ones; the diag ones as behavioral colour.

---

## Training lines

### `iter` (every iteration)

| Field | Meaning | Normal | Signal |
|---|---|---|---|
| `SPS` | env steps/sec | L4 ~600 (externals cap it; heuristic pool ~250–300) | ops only |
| `EV` | explained variance of value head | ~0.85–0.95 | **dropping = critic collapse** (trust) |
| `KL` | approx KL old/new policy | <0.02 healthy; target 0.05 | >0.05 sustained = unstable → halve LR |
| `clip` | clip_frac (`fire:` = per-slot) | **≤0.25 healthy** | **HARD 0.25 → halve LR** (>0.28 degrades). Higher entropy 0.02→0.05 pushes clip ~0.20→~0.30 — pair with lower LR. |
| `H_fire` | fire-head entropy | ~0.1–0.25 (entropy-coef 0.05) | <0.07 = deterministic fire collapse |
| `V_loss` | value loss | <1 settled; spikes on resume | explosion = critic divergence |
| `r_p0 / r_p1` | mean reward seat 0 / 1 | seat-asymmetric, oscillates | NOT a quality signal (zero-sum + shaping) |
| `LR` / `estop` | learning rate / KL early-stop | per schedule / 0 | frequent estop=1 = updates too big |

### `diag` (every 5 iters — `train_torch.py`)

| Field | Meaning | Normal | Signal |
|---|---|---|---|
| `fire[0]` | slot-0 fire prob (sigmoid) | ~0.06–0.19 (top-player-normal) | **slot-0-ONLY** (fire[0]≫rest_max) = degenerate; low fire[0] alone is fine |
| `rest_max` | max fire prob among non-0 slots | tracks fire[0] | fire[0]≈rest_max = healthy spread |
| `fire_frac` | on firing steps, fraction of owned planets firing | ~0.3–0.5 (top players ~0.17) | **→1.0 = carpet-bomb** |
| `owned` | mean planets owned | ~6–10 | low = under-expanding |
| `ship0` | fraction of fires choosing bin 0 (=1 ship) | ~0 | **high = 1-ship-probe collapse** (trust) |
| `meanshipbin` | mean ship-size bin when firing | ~15–20 | low = undersized launches |
| `pl@16/50/100` | planets owned at episode-step 16/50/100 (player 0) | ramp ~6→9→10 | flat/low = expansion plateau |
| `garrfrac@50` | garrison fraction at step 50 = parked / (parked+in-flight) | ~0.5 | **>0.65 = hoarding** (army parked) |
| `shipspp@50` | ships per owned planet at step 50 (parked) | ~22 (Isaiah) | ≫ref = piling instead of spending |
| `H_ship` | ship-head entropy | ~3.4 | low = ship collapse |
| `Vμ` | mean of V(s) | swings | **NOT a collapse signal — ignore sign** |
| `Rμ / Rσ / Aσ` | mean/std returns, adv std | Rμ~0 (zero-sum) | Rμ→0 = equilibrium; Aσ→0 = no learning signal |
| `featσ p/f/g/pw` | feature std | ~0.3–0.45 | →0 = representation collapse |
| `reinf step<50/50-100/>100` | reinforce own-target share by episode-window | peaks MID (50-100) `[ref:win 0.29/0.41/0.31]` | watch <50 and 50-100 climb; back-loaded = too little early |

`pl@`, `garrfrac@`, `shipspp@` are player-0 snapshots at the exact episode step (16/32/50/100),
accumulated over the rollout — controlled-time, so not skewed by game length or end-state. Full set
goes to W&B under `hoard/*`.

### `dm` decisive-mass gap (every 5 iters — `--decisive-diag`, default ON)

Whether the policy is moving toward force concentration (not just the outcome symptom). For each
enemy target with inflight mass converging on it: `ratio = own_inflight_mass / floor`, using the
**same capture floor as the decmass reward** (`floor = garrison + prod·eta + enemy_inbound +
β·ρ(eta)·reachable_enemy_mass + overhead`; shared code `torch_env._decisive_mass_fields` so diag and
reward can't drift). Runs even at `--decisive-mass-coef 0`. Phase-split `<50 / 50-100 / >100`.

- **`gap`** = mean `max(0, floor−mass)/floor` over attacked targets (headline) — want **DOWN**.
- **`cross`** = fraction of targets with `mass ≥ floor` — want **UP**.
- `overkill` = mean `mass/floor` on crossed targets (≫2 = wasted 3× overkill); `nearmiss` = fraction
  in `[0.75, 1.0)`; `tgt/step` = duration-weighted attacked-target observations.

Read: `gap↓ + cross↑ + overkill steady` = teaching concentration. **WR↑ but gap flat** = adjacent
competence, not concentration. β via `--decisive-mass-beta` (default 2.2). SPS: `_decisive_mass_fields()`
runs every step — `--no-decisive-diag` for max throughput.
⚠️ **This is the TRAINING diag only** — the eval mirror of dm was **culled 2026-07** (model-based, and
`take+hold` was contradicted by observed retention; see the Ender reference). Trust it only as a check
that the decisive-mass *reward* is responding when you train with it; it is NOT a skill metric on its own.
[[project_decisive_mass_lever]] [[project_force_concentration_wall]]

### `CKPT_METRICS` (each checkpoint — parsed by track.py)

`step EV KL clip fire_frac owned garrfrac@50 shipspp@50 fire_rate Hfire reinf`
(`fire_rate` = unconditional fraction of owned-planet-slots firing; `fire_frac` conditions on firing steps.)

---

## Eval conversion dump

Printed on **every eval** (`--panel` and baseline), computed identically from top-player replays via
`conversion_from_replays.py`, so eval is directly comparable to the human reference. Implementation:
`eval.py:game_conversion()`.

```
Conversion: caps/game X  atk-launch/game X  cap/atk-launch X (open<50 X  mid50-100 X)  ships/cap X  reinf_share X
  planets@16/32/50/100 a/b/c/d  end X
  game-len  median WON Xst (Ng)  ·  LOST Yst (Ng)
     [planets@N WON/LOST milestone split]
  retention  peel-rate X (lost/total caps)  median-hold Xst
     [retention WON/LOST split]
  loss-depth  median own-material in LOST games X (0 = total wipeout)  ·  wiped-to-0 X%
  fire-rate  launch_rate X  fire_frac X   [ref:Isaiah 0.036 / 0.17]
     WON(Ng) lr X ff X  |  LOST(Ng) lr X ff X   (read WON; ff inflates on losses)
  ship0 1-ship-probe by phase  early<50 X% mid X% late X%
```

**Definitions** (a "launch" = a legal fire, `sent ≤ source ships`):

- **capture** = a planet whose owner transitions **to us** (gross — counts re-captures).
- **attack-launch** = a launch whose target is **not** owned by us. Reinforce launches (own target)
  can't capture and are excluded from the cap/launch denominator.
- **cap/atk-launch** = captures ÷ attack-launches — per-attack conversion efficiency (the skill axis).
  ⚠️ **PHASE-CONFOUNDED — read `open<50`**, the opening-windowed value. The whole-game number averages
  a bad opening with easy late cleanup. Winners are FLAT (open ≈ whole ~0.51); losers COLLAPSE in the
  opening. **The real opening discriminator: Jake `open<50` 0.70 vs ours ~0.54** — we over-launch /
  fire fragments *under* the target's defense (6 ships at a 43-ship neutral → annihilated), not the
  same as legit multi-wave. This gap lives in the BC seed and survives PPO.
- **ships/cap** = attack-ships ÷ captures. ⚠️ Deflated by churn (cheap re-captures) — read with
  caps/game vs `end`: caps/game ≫ end = churn.
- **planets@N** = owned planets at episode-step N (expansion/retention trajectory).
- **game-len** = **median** length split by outcome. Short WON = decisive snowball; long WON =
  stall-and-win attrition. Symptom of the expansion/hold root, NOT an independent lever — don't bribe
  decisiveness with `speed_coef` (caused the Rev26 ship-bin-0 collapse).
- **retention** = `peel-rate` (of captured planets, the fraction we then lose = `lost_caps ÷ captures`)
  + `median-hold` (median steps capture→loss over LOST episodes). Denominator-free (normalized by
  captures ~14/game, not `end`), so it doesn't inflate as `end→0` on elimination. peel-rate→1 + short
  hold = capture-and-lose ("can't hold the lead"); low + hold≈game-length = sticky. Home/initial
  planets excluded.
- **loss-depth** = in LOST games, median final own-material (0 = total wipeout) + `wiped-to-0%`. The
  **graded loss signal** — grades *how badly* we lost, and unlike out-massed% it actually **moves**.
  Want ↑ material as the wall breaks. [[project_ender_opponent_calibration]]
- **fire-rate** = `launch_rate` (fraction of owned-planet-steps with a legal launch) + `fire_frac`.
  ⚠️ `fire_frac` is WIN/LOSS-confounded — **read the WON value** (losing corners you to few planets,
  inflating "many of few"; winners ~0.19–0.21, losers ~0.31–0.33). The lever is *winning more*
  (retention), not a fire tax.
- **ship0** = fraction of opening launches that are 1-ship probes, by phase (degeneracy tripwire).

The **tiered summary** at the end of an eval re-prints the highest-signal reads in priority order:
win-rate → loss-depth → retention (peel-rate/median-hold) → expansion (planets@50/end, open<50 cap/atk) →
degeneracy tripwires. All the model-based / saturating / conflated families that used to print
(decisive-mass gap/cross/overkill/med, take+hold/can't-hold/too-few, out-massed%, hold-floor, triage,
launch-waste, reinf deep-dives, hoard, near/far, holdable-ROI) were **culled** — definitions in git
history (pre-2026-07-06). See the Ender-reference "dm culled" note below for why.

---

## Top-player reference

Timing-corrected replays (action@t paired with obs@t-1), lead-collision target resolver.

| player | rank | n | cap/atk-launch | ships/cap | reinf_share | planets@16/32/50/100 | garr_frac@16/32/50/100 | ships/planet@16/32/50/100 |
|---|---|---|---|---|---|---|---|---|
| Jake Will | #2 | 313 | **0.71** (open<50 **0.70**) | 83 | **0.56** | 2/5/8/11 | 0.59/0.59/0.60/0.77 | 13/16/24/43 |
| Isaiah @ Tufa Labs | #1 | 194 | 0.59 ⚠stale | 168 | 0.30 ⚠ | 2/6/9/10 | 0.50/0.51/0.54/0.87 | 11/15/22/60 |
| 213tubo | mid | 46 | 0.07 (carpet-bomber) | 199 | 0.45 | 2/5/8/10 | 0.52/0.51/0.45/0.49 | 11/15/18/21 |

⚠️ Only Jake's target-derived columns are recomputed on the fixed resolver; Isaiah/TonyK
`cap/atk-launch` + `reinf_share` are STALE (old angle resolver, understated) — recompute with
`conversion_from_replays.py` before citing. `ships/cap`, `planets@`, `garr_frac@`, `ships/planet@`
are not target-derived → valid. Elite play = high cap/atk-launch, steady expansion to ~10 planets,
~half the army deployed mid-game (garr_frac ~0.5). Contested window = steps 16–50; @100 is endgame.

**Fire-rate:** top players fire RARELY — Isaiah `launch_rate` **0.036** / `fire_frac` 0.17; Jake
0.081 / 0.17; carpet-bomber 213tubo 0.41 / 0.47. Our `fire[0] ~0.09` is top-player-normal (even
slightly high) — **firing more is the wrong lever**; the gap is *which* launches + deploy/hold.

---

## ⭐ Ender reference — the progress-tracking setup

The trusted "are we improving" signal is **held-out win-rate + loss-depth + peel-rate + expansion vs
Ender** (`opponents/candidate_ender.py`, a top-10 open-source agent), with **Ender-vs-Ajay as the
target line** — all outcome-grounded, none saturating (unlike out-massed%, pinned 93–99% everywhere).
Reproduce the reference column: `CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/ender_ref.py --seeds 128`
(256 games; runs Ender vs Ajay as path-agents, prints Ender's conversion). [[project-ender-opponent-calibration]]

256-game panels, presres1 = the HEAD-loadable blessed checkpoint (2026-07-06):

| metric | presres1 → Ajay | presres1 → Ender | **Ender → Ajay (target)** |
|---|---|---|---|
| win-rate | 40.2% | **0.0%** | 100% |
| loss-depth · wiped-to-0 | 0 · 97% | 0 · **100%** | — |
| cap/atk-launch (open<50) | 0.65 (0.58) | 0.64 (0.59) | 1.03 (0.75) |
| planets@50 · end | 8 · 10.8 | 7 · **0.0** | 9 · 18.5 |
| peel-rate | 0.68 | **0.99** | 0.41 |
| game-len | 164 WON | 99 LOST | 102 WON |

**Diagnosis this locks:** our `cap/atk-launch` is opponent-INVARIANT (0.65 vs Ajay ≈ 0.64 vs Ender) — so
per-launch conversion is NOT where strong play breaks us. The real, OUTCOME-grounded gaps vs Ajay are
**peel-rate** (0.41 Ender vs 0.68 us — retention), **expansion** (end 18.5 vs 10.8), and **cap/atk**
(1.03 vs 0.65). Against Ender we're wiped to 0 material 100% of games — categorically below top-10.

**⚠ The dm (decisive-mass, model-based) family was CULLED from eval 2026-07** — gap/take-rate/overkill/med,
take+hold/can't-hold/too-few, waste, and out-massed%. They were model-based (β/ρ floor assumptions),
non-discriminating in matched play, and take+hold was **contradicted by observed retention** (Ender read
MORE model-predicted can't-hold, 11% vs our 7%, yet HELD better, peel 0.41 vs 0.68 — the floor model is
blind to the friendly follow-up reinforcement that actually does the holding). See git history for the
numbers if ever needed.

**Track (outcome-grounded, non-saturating): WR-vs-Ender · loss-depth/wiped-to-0% · peel-rate · planets@N/end ·
cap/atk-launch(open<50).** Ignore anything model-based (dm floors, out-massed%) and gross averages (ships/cap).

---

## Decode / masks (eval + export)

Eval and export use **`fire_threshold=0.5`** (default), NOT `--sample` — firing only when confident is
more selective (256-game zach: 45.3% vs 35.9% sampled). Reinforce-discipline masks **must match
training** at inference or the policy reinforces where it was masked and self-sabotages:
`--reinforce-gate-min-planets 3 --reinforce-forward-only --reinforce-garrison-floor 10` (see
`action_mask.py`). Always `--target-decode` for Phase-1 checkpoints.

---

## Deprecated / don't-trust

- **`avgfleet` / `p90`** — REMOVED. Episode-average level metrics are end-step-skewed (late large
  empires inflate them for winners as much as hoarders). Use milestone `garr_frac@` / `ships/planet@`.
  Don't reintroduce a raw fleet-size average.
- **`srcs_multi`** — REMOVED. Empire-size-confounded, outlier-dominated; never moved wins.
- **Zach panel WR** — saturated ~88–89% for the Phase-1 lineage → sanity check only. (For a lineage
  that traded general strength for reinforcement skill it's informative, not a ceiling.) The
  **conversion line vs zach** is useful either way (long games expose the hoard).
- **Ajay/1166 WR as the *objective*** — NOT LB-predictive (rev53b 10.9% → 933 LB < rev38 2.7% → 994).
  Guardrail/yardstick, not north star. Only honest LB signal is submitting (`docs/submissions.md`).
- **out-massed%** — saturates 95–99% vs strong play; a floor, not a gradient (see above).
- **In-training pool `wr` / `ema_wr`** — NOT a performance metric. Measured inside `torch_env` with
  sampled actions where aiming heuristics play far weaker than in the real env. PFSP sampling signal
  only; use cross-eval for the comparable number. Full analysis: `docs/train-eval.md`.

## The honest hierarchy of "is it working"

```
LB submission                 ← ground truth (slow, rate-limited)
  └ held-out panel WR          ← exploitability proxy (Ajay/zach/Ender) — trust for regression
      └ conversion (eval, vs top-player ref) — WHY: cap/atk-launch, planets@, loss-depth
          └ behavioral degeneracy (fire_rate→0, fire_frac→1, ship0) — concrete failure modes
              └ EV / entropy / KL — optimiser is sane
                  └ Vμ / rewμ — confounded; do NOT decide on these
```
