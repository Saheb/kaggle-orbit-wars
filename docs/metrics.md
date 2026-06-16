# Orbit Wars — Training & Eval Metrics Reference

Which numbers mean what, what's normal, and **which to trust**. Two overhauls baked in:
- **Rev54 false-alarm** — we mis-called a collapse on `Vμ` + `avgfleet` + `srcs_multi`, all non-signals.
- **2026-06-11 conversion/hoard overhaul** — replaced the end-skewed `avgfleet`/`p90` with
  controlled-time **milestone** metrics, and added **conversion** metrics (eval + training) with a
  top-player reference. See "Conversion & hoard" below.

---

## TL;DR — trust order

1. **Held-out win-rate** (Ajay / 1166 / 1300 / zach panels) — the only signal that sees *absolute*
   regression (our exploitability proxy, the real arbiter). Everything else is a leading indicator.
2. **Conversion & hoard** (eval, vs the top-player reference) — *why* we win/lose: are we capturing
   efficiently, expanding, and deploying the army, or churning/hoarding? Printed on every eval.
3. **Behavioral degeneracy** (unconfounded):
   - `fire_rate → 0` — passive fire=0 Nash. (But note: a *low* fire rate is normal — see fire note.)
   - `fire_frac → 1.0` — carpet-bomb (firing from every owned planet).
   - `ship0 → high` — 1-ship-probe collapse.
4. **EV, entropy, KL/clip** — *training health* (is the optimiser sane), **not** policy quality.
5. **Vμ, rewμ** — weak corroborators. **Ignore the sign.** Never kill a run on these alone.

**Why self-play needs held-out eval:** internal metrics are blind to *mutual* collapse — if both
policy copies degrade together, value / KL / win-rate-vs-self look fine while the agent gets globally
worse. The principled measure is **exploitability** (how badly a best-response beats you), approximated
by win-rate vs a fixed held-out panel. PPO logs *explained variance* of the value head, never the
*value mean*, for exactly this reason. Corollary: the conversion/hoard metrics in the **training diag**
are also self-play-confounded — read them as behavioral colour; the **eval** conversion numbers are the
trusted ones.

---

## The `iter` line (every iteration)

| Field | Meaning | Normal | Signal |
|---|---|---|---|
| `SPS` | env steps/sec | L4 ~600 (externals cap it; heuristic-worker pool ~250–300) | ops only |
| `EV` | explained variance of value head | ~0.85–0.95 | **EV dropping = critic collapse** (trust) |
| `KL` | approx KL old/new policy | <0.02 healthy; target 0.05 | >0.05 sustained = unstable → halve LR |
| `clip` | clip_frac; `fire:` = per-slot fire | **≤0.25 healthy** | **HARD 0.25 → halve LR** (>0.28 degrades; don't wait for 0.32). Higher entropy 0.02→0.05 pushes clip ~0.20→~0.30 — pair with lower LR. |
| `H_fire` | fire-head entropy | ~0.1–0.25 (entropy-coef 0.05); recovers after warmup | <0.07 = deterministic fire collapse |
| `V_loss` | value loss | <1 settled; spikes on resume | explosion = critic divergence |
| `r_p0 / r_p1` | mean reward seat 0 / 1 | seat-asymmetric, oscillates | NOT a quality signal (zero-sum + shaping) |
| `LR` / `estop` | learning rate / KL early-stop | per schedule / 0 | ops; frequent estop=1 = updates too big |

## The `diag` line (every 5 iters — `train_torch.py`)

| Field | Meaning | Normal | Signal |
|---|---|---|---|
| `fire[0]` | slot-0 fire *probability* (sigmoid) | ~0.06–0.19 (top-player-normal; see fire note) | **slot-0-ONLY** firing (fire[0]≫rest_max) = degenerate. A low fire[0] alone is fine. |
| `rest_max` | max fire prob among non-0 slots | tracks fire[0] | context for fire[0]; fire[0]≈rest_max = healthy spread |
| `fire_frac` | **on firing steps**, fraction of owned planets that fire | ~0.3–0.5 (top players ~0.17) | **→1.0 = carpet-bomb**; well above 0.5 = scattershot |
| `owned` | mean planets owned | ~6–10 | low = under-expanding |
| `ship0` | fraction of fires choosing bin 0 (=1 ship) | ~0 (with `--min-ship-bin`) | **high = 1-ship-probe collapse** (trust) |
| `meanshipbin` | mean ship-size bin when firing | ~15–20 | low = undersized launches (undercommitment) |
| `pl@16/50/100` | planets owned at episode-step 16/50/100 (player 0) | ramp toward ~6→9→10 | flat/low = expansion plateau (the deploy-and-hold gap) |
| `garrfrac@50` | garrison fraction at step 50 = parked / (parked+in-flight) | ~0.5 (top players) | **high (>0.65) = hoarding** (army parked, not deployed) |
| `shipspp@50` | ships per owned planet at step 50 (parked) | ~22 (Isaiah) | high (≫ref) = piling ships instead of spending |
| `H_ship` | ship-head entropy | ~3.4 | low = ship collapse |
| `Vμ` | mean of V(s) | swings; rev38 mean +0.32 | **NOT a collapse signal — ignore sign** |
| `Rμ / Rσ / Aσ` | mean/std returns, adv std | Rμ~0 (zero-sum) | Rμ→0 = equilibrium not collapse; Aσ→0 = no learning signal |
| `rewμ / rewNZ` | mean per-step reward / fraction nonzero | shaping-dependent | weak |
| `featσ p/f/g/pw` | feature std | ~0.3–0.45 | →0 = representation collapse |
| `reinf / tgt n/e` | reinforce rate / target-share neutral/enemy (if `--allow-reinforce`) | ramps with empire size | flood = high reinf **+ ballooning hoard** (read with garrfrac/shipspp) |
| `reinf step<50/50-100/>100` | reinforce own-target share by episode-window (added 06-12; `[ref:win 0.29/0.41/0.31]`) | peaks MID (50-100) | our pre-deb-run shape was back-loaded (0.05/0.19/0.42) — too little early/mid. Watch <50 and 50-100 **climb** with deb-in-pool. Same metric as eval `reinf by step`. |

`pl@`, `garrfrac@`, `shipspp@` are player-0 snapshots taken **at the exact episode step** (16/32/50/100),
accumulated across the rollout — controlled-time, so not skewed by episode length or end-state. Full set
(`garr_frac@`, `ships_per_planet@`, `planets@` for all four milestones) goes to W&B under `hoard/*`.

## The `dm` (decisive-mass GAP) line (every 5 iters — `--decisive-diag`, default ON)

Measures **whether the policy is moving toward the decmass target** (force concentration), not just the
outcome symptom (`out-massed%`). For each enemy target the current policy has inflight mass converging on,
it computes `ratio = own_inflight_mass / floor` using the **EXACT same capture floor as the Lever-A reward**
(`floor = garrison + prod·eta + enemy_inbound + beta·rho(eta)·reachable_enemy_mass + overhead`, eta = MAX
arrival ETA) — shared code (`torch_env._decisive_mass_fields`) so the diag and reward can never drift. Runs
**even when `--decisive-mass-coef 0`** (reads on any lineage, incl. hladder). Train-mask-weighted (current
policy only); phase-split `<50 / 50-100 / >100`.

```
dm | gap <50/50-100/>100 g/g/g | cross c/c/c | ratio r overkill o nearmiss n tgt/step t
```

| Field | Meaning | If decmass/curriculum is working |
|---|---|---|
| **`gap`** | mean `max(0, floor−mass)/floor` over attacked targets (the headline) | **DOWN** (attacks closing the floor) |
| **`cross`** | fraction of attacked targets where `mass ≥ floor` | **UP** |
| `ratio` | mean `mass/floor` | rises toward/past 1.0 |
| `overkill` | mean `mass/floor` on **crossed** targets | NOT exploding (≫2 = dumb 3× overkill, wasted) |
| `nearmiss` | fraction in `[0.75, 1.0)` | useful tell: approaching but not yet crossing |
| `tgt/step` | attacked enemy **target-observations** per controlled env-step (duration-weighted: a long-ETA attack counts each step it's inflight — NOT unique targets/launches) | assembly breadth |

**Read rules:** `gap↓ + cross↑ + overkill steady + out-massed% eventually↓` = teaching concentration. **WR↑
but gap flat** = adjacent competence, NOT concentration (the decmass1/h14only failure signature). **gap↓ but
out-massed% flat** = learning *attack* concentration but not *post-capture retention*. W&B: `dm/*`.

Eval has the same line (`decisive-mass …` in the conversion block) and adds **p50** per phase + a
`target-steps/game` (the same duration-weighted count as `tgt/step`) — train tells whether PPO responds to
the reward, eval whether it survives argmax + held-out.

**Caveats.** (1) **beta:** the floor's reactive margin uses `--decisive-mass-beta` (default 2.2). It is an
env param, **not stored in the ckpt** — so eval defaults to 2.2 and prints `(beta X.X)` on the line; to read a
run trained with a non-default beta, pass the same `--decisive-mass-beta` to eval (parity-tested at non-default
beta). (2) **SPS:** `_decisive_mass_fields()` runs EVERY step (a P×P enemy-pressure pass + fleet-target
resolution) — benchmark the delta on the GPU box; `--no-decisive-diag` for max-throughput production once the
measurement need is satisfied. [[project_decisive_mass_lever]] [[project_force_concentration_wall]]

## The `CKPT_METRICS` line (at each checkpoint — parsed by track.py)

`step EV KL clip fire_frac owned garrfrac@50 shipspp@50 fire_rate Hfire reinf`
(`fire_rate` = overall fraction of owned-planet-slots firing, unconditional; vs `fire_frac` which
conditions on firing steps.)

---

## Conversion & hoard metrics (the 2026-06-11 overhaul)

Printed on **every eval** (`eval.py`, both `--panel` and baseline) and computed identically from the
top-player replays, so eval is directly comparable to the human reference. The whole-game read that
win-rate alone can't give you: *are we capturing efficiently, expanding, and deploying — or churning
and hoarding?* Implementation: `eval.py:game_conversion()`; training side mirrors it for player 0.

```
Conversion: caps/game X  atk-launch/game X  cap/atk-launch X (open<50 X)  ships/cap X  reinf_share X
  planets@16/32/50/100 a/b/c/d  end X   churn X (X/100st, len X)
  retention  lost-cap X (lost/total caps)  median-hold Xst
  launch-waste<50  redundant X (WG X)  underkill X (WG X)
  fire-rate  launch_rate X  fire_frac X   [ref Isaiah 0.036 / 0.17]
     WON(Ng) lr X ff X  |  LOST(Ng) lr X ff X   (read WON; ff inflates on losses)
  hoard  garr_frac@ a/b/c/d  ships/planet@ a/b/c/d
  reinf by empire size  1:r(n)  2-3:r(n)  4-6:r(n)  7-9:r(n)  10-12:r(n)  13+:r(n)   [ref ramp @1:0.00 @2:0.10 @9-12:0.30 @13+:0.34-0.61]
```
Printed on **every** eval (`--panel` and baseline). `churn`, `launch-waste<50 (redundant/underkill)` are
defined below; all compute identically from top-player replays via `conversion_from_replays.py`.

**Definitions** (a "launch" = a legal fire, `sent ≤ source ships`):
- **capture** = a planet whose owner transitions **to us**. Counts re-captures, so it's gross, not net.
- **attack-launch** = a launch whose aimed target is **not owned by us**. *Reinforce launches (target
  owned by us) CANNOT capture and are excluded from the cap/launch denominator* — counting them deflates
  the ratio (was the original bug). Launches whose target can't be resolved by angle are skipped
  (matches `fetch_analyze_top_replays._resolve_target`, so eval == replay analysis).
- **cap/atk-launch** = captures ÷ attack-launches — per-attack conversion *efficiency* (the skill axis).
  ⚠️ **PHASE-CONFOUNDED — read `(open<50 X)`, the opening-windowed value printed beside it (added
  2026-06-12).** The whole-game number averages a catastrophic opening with easy late-game cleanup
  captures, so it lands reassuringly even when the opening (which decides the expansion race) is failing.
  **Winners are FLAT (opening ≈ whole-game): snowball winners 0.51/0.53. Losers COLLAPSE in the opening:
  our losing replay 0.26 open vs 0.46 whole-game**; p2rev4 500k read 0.442 whole-game while its
  `underkill 0.41` (opening waste) was the tell. The disease behind a low opening value =
  **under-commitment / fragments fired *under* the target's defense** (e.g. 6 ships at a 43-ship neutral
  → annihilated, ships wasted, never captured) → poor opening conversion → under-expansion → loss. NOT
  the same as legitimate combining multi-wave (4+6 vs an 8-ship neutral captures fine). Pairs with
  `underkill<50`. (Mild window-edge bias: a t~48 launch capturing at t~55 deflates it slightly.)
- **ships/cap** = attack-ships ÷ captures — *force per capture*. ⚠️ **Deflated by churn**: re-captures
  of flip-flopping planets are cheap, so a low value can mean efficiency **or** churn. Always read with
  caps/game vs `end`: caps/game ≫ end_planets = churn.
- **churn** = gross captures ÷ end_planets (capture-then-lose-then-recapture the same flip-flopping
  planets — the "can't hold the lead" signal). ⚠️ **LENGTH-CONFOUNDED** like the removed `avgfleet`: more
  steps → more gross re-captures, so the longest games read highest *regardless of holding skill*. Top-2
  raw churn is **Isaiah 7.1 > Jake 3.5 purely because Isaiah's games run 447 vs 284 steps.** Read the
  printed **`churn/100st`** (caps/end normalized per 100 steps) instead — it collapses to a tight elite
  band: **Isaiah 1.59 · TonyK 1.16 · Jake 1.23 · 213tubo 1.48.** Even normalized it's a *secondary* read;
  the clean hold signal is the **planets@N trajectory turning over** (peak then decline) + the
  proximity crossover step (`deb_proximity.py`).
- **game-len** (`game-len  median WON Xst (Ng)  ·  LOST Yst (Ng)`, added 2026-06-16) — **median** game length
  split by outcome (the `churn` line's `mean-len` is the mean over ALL games — confounded by the loss/win mix).
  Tells whether our WINS are **decisive snowballs** (short) or **stall-and-win attrition** (long). Prior ad-hoc
  read: vs Ajay WON ground to ~320st, LOST ~116st — i.e. even our wins were attrition, never a fast kill (a
  symptom of the expansion/holding root, NOT an independent lever — don't bribe decisiveness with speed_coef,
  which caused the Rev26 ship-bin-0 collapse). Read WON median dropping over training = we're learning to close.
- **retention** (`retention  lost-cap X (lost/total caps)  median-hold Xst`, added 2026-06-11) — the
  **denominator-free** hold signal, built to replace churn's degeneracy. `lost-cap` = of the planets we
  CAPTURE, the fraction we then lose (`lost_caps ÷ captures`); `median-hold` = median steps from capture to
  loss over LOST episodes (held-to-end is censored, not counted). **Why it beats churn:** normalized by
  *captures* (≈stable ~14/game), not `end_planets` → it does NOT inflate as `end→0` on elimination. churn
  rising 16→26 across p2rev3 500k→2.6M was almost entirely `end` falling 0.9→0.5 (caps flat) — a denominator
  artifact, NOT worsening turnover; lost-cap/median-hold measure the turnover directly. Read: lost-cap→1 +
  short hold = capture-and-lose ("can't hold the lead"); lost-cap low + hold≈game-length = sticky. Home/
  initial planets excluded by construction (only planets that entered `cap_step` count). Diagnostic of the
  Phase-2 retention gap; expect high lost-cap/short hold vs a strong planner (deb), low/long vs weak (Zach).
- **hold-loss autopsy** (`hold-loss  out-massed X% · abandoned X% · too-late X% · other X%   garr@cap A→@loss B
  vs enemy-inbound C`, added 2026-06-15) — WHY a captured planet falls, classified at the step of loss from the
  t-1 state (reuses `_friendly_inbound` geometry): **ABANDONED** = garrison ≤2 (we left it undefended); **OUT-MASSED**
  = garrison >2 but enemy inbound fleet > our garrison (under-massed vs the threat); **TOO-LATE** = we had reinforcement
  inbound but not enough/in time; **OTHER**. `garr@cap`/`@loss` = median garrison just after capture / just before loss;
  `enemy-inbound` = median enemy ships racing in at loss. **⭐ 2026-06-15 ROOT FINDING: losses are ~100% OUT-MASSED vs
  planners** (deb ~96 inbound vs our ~59 garrison), UNIVERSAL across our lineage (rev38 98% too). NOT a defense-laziness
  gap (we garrison + reinforce) — a **force-concentration** gap: planners forward-project defenders + regroup multiple
  planets into one decisive strike; we fire per-planet sized to current defense. The incoming fleet IS in our features
  (planet ch13 `enemy_pressure` / pairwise feat 14 `enemy_contest`) → the fix is a training SIGNAL, not a feature. Watch
  `out-massed%` DROP as a concentration lever works. Standalone tool: `orbit_wars_rl/hold_autopsy.py`. Memory:
  `project_force_concentration_wall`.
- **hold-floor** (`hold-floor (garr+friendly_in)/(enemy_in+β·reach+1) β=2.2 … by phase … by age-after-capture 0-5/6-15/16+`,
  added 2026-06-16) — the **DEFENSIVE mirror** of the decisive-mass attack gap, and the most direct read on the
  capture-then-lose wall. For each OWN planet under an actual inbound threat (an enemy fleet converging on it — same
  condition as the hold-loss `out-massed` autopsy), `hold = (our_garrison + friendly_inbound) / (enemy_inbound +
  β·reachable_enemy_mass + 1)`; ratio `<1` = under-defended (will likely be peeled). Reported as `under%(ratio<1)/p50`,
  split **by phase** AND **by age-after-capture** (0-5 / 6-15 / 16+ steps, from `cap_step`; home/initial planets have no
  capture step → excluded from age buckets). **The age axis is the headline:** high under% at **0-5** = we lose captures
  IMMEDIATELY (can't route defensive mass in time → a timing/action-grammar problem, supports multi-move/source); high at
  **16+** = later logistics/churn. β/overhead/reachable reuse the decisive-mass floor constants (β via `--decisive-mass-beta`,
  printed). Duration-weighted (a planet threatened N steps = N observations, like dm `target-steps`). Eval-only (like the
  hold-loss autopsy). `_hold_floor_step`; `project_force_concentration_wall`.
- **reinforce-triage / save-efficiency** (`reinforce-triage  ships→ safe X% cheap-save X% exp-save X% HOPELESS X% …`,
  added 2026-06-16) — tests the **"are we reinforcing the WRONG planets?"** hunch: maybe the wall isn't "reinforce more,"
  it's **bad triage** — pouring mass into planets we then lose instead of recycling it into the next attack. Mirrors
  producer_v2's actual defense logic (`orbit_lite/planner_core.py`): `safe_drain` (a source sheds only what it can spare
  while holding ITSELF; a **doomed** source drains fully) + the `roi_threshold` gate (a reinforce fires only if its
  competitive net-ship-delta clears 1.5). Each reinforce launch's target is classified at decision time via `_threat_class`
  into **already-safe / cheap-save / expensive-save / hopeless** (`defense_cost = enemy_in + β·reach_enemy + 1 − garrison`;
  `friendly_available` = inbound arriving before the threat ETA + **safe-drain-able spare** of owned sources reachable in
  time, doomed-drain-full; `value = prod·horizon`; hopeless = `friendly_available < defense_cost`). Reads: `HOPELESS%` of
  reinforce mass, reinforce mass **on planets we then LOST** (to-lost), of LOST planets the **cheap-save-MISSED** vs
  **hopeless(ok-to-drop)** split, of hopeless losses **wasted-ships vs abandoned**, and **WON/LOST** split. **Hunch
  confirmed if:** high `HOPELESS%`/`to-lost`, high `cheap-save-MISSED`, and **winners lower** — i.e. we waste ships on
  doomed planets while winners abandon them and recycle into attacks. ⚠ **Diagnostic only — do NOT turn into reward yet;**
  if confirmed, the lever is a *selective* reinforce/abandon signal or mask-prior, NOT another global "defend more" reward.
  Eval-only. `_threat_class` / `_reachable_friendly_mass` (FORK #2 = the faithful safe-drain version). `project_reinforce_triage`.
- **outmassed by planets@32** (`outmassed by planets@32  <=4: outmassed X% WR X% (nN)  5: …  >=6: …`, added 2026-06-16) —
  the SAME out-massed% (+ WR) **conditioned on the opening-expansion bucket** (the game's planets@32). Separates two
  compounding gaps: **upstream economic tempo** ("we entered the midgame at 4 planets vs winner's 6+ → less production /
  fewer source planets / fewer regroup options → less mass to aggregate") from the **tactical** aggregation/retention wall.
  The **verdict is COMPUTED from the panel** (not asserted): if `>=6` has materially lower out-massed (≥4pp) OR higher
  WR (≥5pp) than `<=4` it prints "opening expansion looks UPSTREAM"; else "NOT the lever; wall is downstream"; needs
  ≥15 games in both end buckets else "inconclusive". Bucketed per game (needs panel-style `won`; games ending before
  step 32 are unbucketable → skipped). **⭐ 2026-06-16 Ajay panel finding: NOT the lever — `<=4` 97%/13% (n165), `5`
  96%/12% (n50), `>=6` 97%/15% (n41)** → reaching 6+ by step 32 gave ~0pp out-massed reduction and only +2pp WR. So
  opening expansion is **deprioritized as the main wall fix** (kept as a competence lever); the wall is downstream
  (defensive routing / post-capture positioning / action grammar) → read the hold-floor age split next.
  `project_force_concentration_wall`.
- **launch-waste<50** (printed `launch-waste<50  redundant X (WG X)  underkill X (WG X)`) — the OPENING
  (step <50) launch-discipline pair, both keyed to `cap_cost_at_arrival` (the SAME quantity the roi-deflation
  uses, replicated in `eval.py _cap_cost_at_arrival`/`_eta` from `_ETA_PROBE_SPEED`). Windowed to the opening
  because a whole-game fraction is inflated by *benign end-game surplus re-fire* in long won games (a
  phase-composition confound, not a volume one — length-normalizing does NOT fix it); the `(WG x)`
  whole-game value is kept for context.
  - **redundant (OVERKILL)** = attack-launch at a target ALREADY covered to capture by own fleets inbound
    *before* the launch (`friendly_inbound ≥ cap_cost_at_arrival`) ÷ opening attack-launches → pure surplus,
    exactly what the deflation zeroes. ⚠️ **Top-player floor is ~0.02** (Isaiah 0.02 · TonyK 0.03 · Jake 0.02
    · 213tubo 0.02) — top players essentially never pile on. (An earlier current-ships threshold read ~0.12,
    but that was almost all *enemy multi-wave* the deflation rightly leaves alone — the aligned metric is the
    true one.) So our agent reading ≫0.02 = real over-fire the deflation should cut.
  - **underkill (INEFFECTIVE)** = FORWARD-looking: the target never becomes ours within ~eta+10 steps of the
    launch (id-keyed lookahead, no slot-reorder issue) ÷ opening attack-launches → ships that didn't lead to
    a capture (the seed1030 18-at-23 lone-undercommit case). A per-launch `sent+inbound<cost` threshold is
    WRONG here — it mis-flags legit multi-wave (each wave < cost) at ~0.86; forward-looking fixes that
    (a target a later wave captures reads effective for all waves). Top-player ref **~0.43** (Isaiah 0.40 ·
    Jake 0.43 · TonyK 0.50). ⚠️ **CORRECTION (2026-06-12): underkill does NOT discriminate winners from
    losers** — winners sit ~0.40–0.43 and our p2rev4 aggregate is 0.41 (*below* winners) yet loses; ~40%
    of opening attacks "not capturing within eta+10" is just normal probing/multi-wave/contested-neutral
    play. Do NOT alarm on underkill ~0.4, and don't read it as "the conversion gap." **The real opening
    discriminator is `open<50 cap/atk-launch`** (captures *per launch*): winners ~0.51, our losses ~0.26 —
    i.e. we need ~4 launches per captured planet, winners need ~2. The disease is **launches-per-capture
    (over-launching / fragments under the target's defense), not "launches that never capture."** Anchor to
    the WINNER ref, never the `(WG)` self-reference (which normalizes a bad opening against our own padded
    whole-game average).
  (Same-step double-fires at one fresh target aren't counted for redundant — neither fleet exists at decision
  time — matching the fix's reach.)
- **reinf_share** = reinforce-launches ÷ all launches. ⚠️ **Opponent/success-confounded**: it co-moves
  with empire size (own planets only become legal targets above the gate, and their fraction rises as the
  empire grows — phase2 §6), so the *same policy* reads ~0.08 vs debatreya (eliminated, stuck small/
  below-gate), ~0.23 vs a winnable opponent, ~0.5 in self-play training. Don't compare the aggregate
  across opponents — use the per-empire-size ramp below.
- **reinf by empire size** = own-target share among launches made *at that owned-planet count* (with the
  launch count in parens — low-count bins are noisy). This is the apples-to-apples comparison to the
  top-player ramp (phase2 §2): @1 ≈0.00, @2 ≈0.10, @9-12 ≈0.30, @13+ 0.34-0.61. Empire size is measured
  at decision time (obs@t-1). The aggregate `reinf_share` is empire-mix-weighted; the ramp decouples it.
- **planets@N** = owned planets at episode-step N — the expansion/retention trajectory.
- **garr_frac@N** = parked ÷ (parked + in-flight) at step N — scale-free deployment ratio; high = army
  parked. (Snapshot mid-game; at the terminal step everything lands → 1.0 trivially.)
- **ships/planet@N** = parked ships ÷ owned planets at step N — pile-up per planet.

**Why milestones, not `avgfleet`/`p90`:** an episode *average* (or p90) is dominated by the late-game
large-empire steps, where high ship counts are *normal* — a winner with 12 planets producing every turn
shows a huge average garrison that is *winning*, not hoarding. Fixed-step **snapshots** read the hoard at
controlled, comparable points (immune to game length / end-state), and the **ratios** (`garr_frac`,
`ships/planet`) decouple empire size. The `@100` column intentionally shows the won-game accumulation
(Isaiah garr_frac 0.87) so you can *see* it instead of letting it poison an average.

### Top-player reference (timing-corrected replays: action@t paired with obs@t-1)

Computed over **269 replays** (`/tmp/fresh_validate` 180 + `/tmp/snowball` 89) by running the eval's own
`game_conversion()` over the replays — driver `conversion_from_replays.py <dirs> [--player NAME]`, so the
top-player numbers and the eval numbers come from the *same function* (true eval == replay parity).
`n` = games each player appears in (all outcomes, not just wins); ratios are pooled over all those games.

| player | rank | n | cap/atk-launch | ships/cap | reinf_share | planets@16/32/50/100 | garr_frac@16/32/50/100 | ships/planet@16/32/50/100 |
|---|---|---|---|---|---|---|---|---|
| Isaiah @ Tufa Labs | #1 | 194 | **0.59** | 168 | 0.30 | 2/6/9/10 | 0.50/0.51/0.54/0.87 | 11/15/22/60 |
| Jake Will | #2 | 83 | 0.42 | 140 | 0.43 | 2/5/8/11 | 0.59/0.59/0.60/0.77 | 13/16/24/43 |
| TonyK | — | 83 | 0.53 | 181 | 0.40 | 2/6/9/9 | 0.46/0.46/0.50/0.80 | 10/14/20/63 |
| 213tubo | mid | 46 | 0.07 | 199 | 0.45 | 2/5/8/10 | 0.52/0.51/0.45/0.49 | 11/15/18/21 |

Read it: elite play = **high cap/atk-launch** (Isaiah 0.59 vs carpet-bomber 213tubo 0.07), steady
expansion to ~10 planets, and **~half the army deployed** mid-game (garr_frac ~0.5, ~11–22 ships/planet).
The contested phase (steps 16–50) is the clean window; @100 is endgame accumulation.

### Fire-rate reference (how often top players actually launch)

`launch_rate` = fraction of owned-planet-steps with a legal launch (the empirical analog of `fire[0]`).

| player | launch_rate | active_step_frac | fire_frac |
|---|---|---|---|
| Isaiah (#1) | **0.036** | 0.25 | 0.17 |
| Jake (#2) | 0.081 | 0.43 | 0.17 |
| 213tubo (mid) | 0.41 | 0.77 | 0.47 |

As of 2026-06-11 `launch_rate` + `fire_frac` are emitted on the **eval** conversion line too (the
`fire-rate` row), not just the training diag — so spray (high rate, low `cap/atk-launch`) is now
visible on the metric we judge submissions on. Both compute identically from top-player replays via
`conversion_from_replays.py`, so eval reads are apples-to-apples with the Isaiah 0.036 / 0.17 reference.

⚠️ **`fire_frac` is WIN/LOSS-confounded — read the WON-game value** (the `WON(Ng) … | LOST(Ng) …`
sub-line, added 2026-06-12). `fire_frac` = fraction of *owned* planets firing; when you lose you get
cornered to few planets, so firing from "many of few" inflates it. Measured on the SAME boards in the
snowball replays: **winners `fire_frac` ~0.19–0.21, losers ~0.31–0.33.** Consequences: (1) the BC
clone's apparent 0.39 spray was largely a losing-position artifact (it loses 97% vs zach) — *not* a
contaminated seed (snowball winners median 0.19); (2) our p2rev3-4M reads 0.29 vs zach (84% WR) but
0.34 vs deb (6% WR) — the deb number is inflated by losing. **The honest "are we sprayers?" signal is
the WON-game `fire_frac`** (~0.21–0.29 vs Isaiah 0.17 = a small residual gap), and the lever that pulls
it down is *winning more* (retention), not a fire tax (rev41–45 Nash graveyard) or a BC re-curation.
This is the same length/position-confound class as `churn` (end→0) and `atk-launch/game`.

**The top players fire RARELY** (Isaiah 3.6%). A *high* fire rate (213tubo 41%) is a *losing* trait.
So our `fire[0] ~0.09` is top-player-normal, even slightly high — **firing more is the wrong lever**;
raising fire-entropy pushes toward 213tubo's carpet-bombing. The gap is *which* launches and *deploy/hold*,
not *how often*.

### Decode note (eval / export)

Eval and export use **`fire_threshold=0.5`** (the default), NOT `--sample`. On the 256-game zach panel,
threshold-0.5 = **45.3%** vs sample **35.9%** — firing only when confident is more selective and plays
better; sampling fires low-confidence slots and adds noise. (A noisy 4-game run suggested the opposite —
don't trust small-N win-rate; use the full panel.) The reinforce-discipline masks must also match training
at inference: `--reinforce-gate-min-planets 3 --reinforce-forward-only --reinforce-garrison-floor 10`
(see `action_mask.py`), else the policy reinforces where it was masked and self-sabotages.

---

## Deprecated / removed / misleading

- **`avgfleet` / `p90` (planet ship inventories)** — REMOVED from the diag (2026-06-11). Episode-average
  level metrics are **end-step-skewed**: late-game large empires inflate them for *winners* as much as
  hoarders, conflating "winning" with "hoarding". Replaced by milestone `garr_frac@`/`ships/planet@`
  (controlled-time, scale-free). Don't reintroduce a raw fleet-size average.
- **`srcs_multi`** — REMOVED (2026-06-09). Empire-size-confounded, outlier-dominated. Optimising it never
  moved wins. (The `--srcs-multi-penalty` shaping knob is separately deprecated — floor=0 → fire=0 Nash.)
- **Zach panel** — saturated ~88–89% as a *win-rate* metric **for the Phase-1 lineage only** (rev32b 88.7%,
  rev31/38 ~85%). ⚠️ **NOT saturated for Phase 2:** the reinforcement lineage (p2rev1/p2rev2) started from
  snowball-BC and trades general Zach-beating strength for reinforcement-skill acquisition vs the aggressive
  pool, so it sits far below ~88% (p2rev2 @1M was ~37% on the first 16 games). For Phase 2, Zach WR is an
  **informative, non-ceiling signal** — the gap to Phase-1's ~88% measures how much general strength the
  lineage has (re)gained; track it climbing as a real progress signal, don't dismiss it as saturated. The
  **conversion line vs zach** is useful for either lineage (long games expose the hoard). Use zach for
  conversion/hoard diagnostics + (Phase-2) general-strength tracking, not as a Phase-1 WR headroom ceiling.
- **Ajay/1166 panel as the *objective*** — NOT LB-predictive (rev53b 10.9% Ajay → 933 LB < rev38 2.7% →
  994). Guardrail/yardstick, not north star. Only honest LB signal is submitting (`docs/submissions.md`).
- **In-training pool `wr` / `ema_wr`** — NOT a performance metric. Measured inside `torch_env` with
  sampled actions; aiming-heavy heuristics play far weaker there than in the real env (p2rev1 8.25M:
  pool ema 0.48 vs kaggle 0/6% vs lb1152). It's a PFSP sampling signal only. For the comparable number
  use cross-eval. **Full analysis: `docs/train-eval.md`.**

## The honest hierarchy of "is it working"

```
LB submission                 ← ground truth (slow, rate-limited)
  └ held-out panel WR          ← exploitability proxy (Ajay/1166/1300/zach) — trust for regression
      └ conversion & hoard (eval, vs top-player ref) — WHY: cap/atk-launch, planets@, garr_frac@
          └ behavioral degeneracy (fire_rate→0, fire_frac→1, ship0) — concrete failure modes
              └ EV / entropy / KL — optimiser is sane
                  └ Vμ / rewμ — confounded; do NOT decide on these
```
