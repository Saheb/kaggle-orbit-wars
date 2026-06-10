# Phase 2 — Reinforcement-Enabled From-Scratch Run

**Purpose:** teach the agent to *reinforce* (send ships to its own planets) as a native,
empire-size-gated, instrumental behaviour — the #1 structural skill-gap vs the leaderboard
top tier. This is a fresh run with a redesigned reward model + one new action mask, NOT a
resume of the rev55–57 reinforce lineage (all of which flooded).

Status: **Tier-1 reward model locked + keystone built (2026-06-10).** The rev58/58b resume
probes flooded; root cause re-diagnosed (`defense_coef` is the flood pump, not the cure); design
pivoted to outcome-tied attribution (forward-staging mask + drop `defense_coef` + small aggressive
pool). Forward-staging mask built + unit-tested. **p2rev1 run script READY** (`gpu_run_artifacts/p2rev1/`):
snowball-BC warmstart + forward mask + drop defense + pool (lb1152 hammer + debatreya_1300 @0.25);
target-head diagnostics added. Awaiting launch. See the Update below.

---

## Update (2026-06-10): the rev58/58b resume probes flooded → Tier-1 redesign

Per the "try on rev38 5M FIRST" decision, we probed the locked design by **resuming rev38 5M**
*before* the from-scratch run:
- **rev58** (gate + `defense_coef 0.03` + fire-entropy 0.005, **`reinforce_cost 0`**) — the locked
  design exactly. Started healthy (reinf 0.21 @32k, `Vμ +0.63`) then **drifted into a flood** by
  ~400k (reinf 0.75, `p90` 408, `Vμ`→0).
- **rev58b** (+ `reinforce_cost 0.001`, the §3 back-pocket lever) — flooded **earlier**, ~330k
  (reinf 0.69, `p90` 357, `avgfleet` 175, `fire_frac` 0.83, `Vμ −1.25`); `clip` crept to 0.27.
  Instance deleted.

**Conclusions (both §3 bets refuted):**
1. **The cost knob is dead** — `0` floods, `0.001` floods. Both endpoints of the back-pocket lever
   are spent. Stop the cost debate.
2. **The empire gate is orthogonal to the flood** — flooding sets in at owned≈8.7, well above the
   min-3 gate. The gate only suppresses *early* reinforce; the flood is a mid/large-empire effect.
3. **§4's "self-capping by construction" is false, and the cause is `defense_coef` itself.**
   `defense_coef` rewards *holding* (avoiding production loss). In a symmetric self-play mirror the
   enemy always threatens something → reinforcing-to-hold is always rewarded; losing a planet is
   double-penalized (expansion + defense); attacking an equally-strong defended enemy is −EV. So the
   reward-maximizing policy is **hold-everything-via-reinforce, never take a risky attack** — exactly
   rev58's signature (reinf↑, `Vμ`↓ = not *winning*, just not losing planets). **The term §4
   designated as "the reinforcement incentive" is the flood pump.**

This is a *drift from a healthy gated start*, not a t=0 shock — i.e. a property of the reward
**objective**, which is identical from scratch. The is_mine-untrained / "mature-equilibrium shock"
story (§1) only ever explained the rev55 *t=0 spray*; it never explained the drift, and the drift
recurs from scratch ⇒ a clean from-scratch run under the *unchanged* reward model would flood the
same way.

### Tier-1 design (locked) — outcome-tied attribution
Reinforcement earns reward *only* through the outcomes it enables, and the rear-hoard outlet is
removed structurally:
- **Forward-staging mask (NEW — built + tested):** an own reinforce target is legal only if it is
  closer to the nearest enemy planet than the source (`--reinforce-forward-only`). Reinforcement
  flows rear→front; a safe rear hoard is impossible by construction. Matches the 66–70%
  forward-staging in top-player replays. Implemented in `torch_env.py` (allow_reinforce mask
  branch); test `tests/test_reinforce_mask.py::test_forward_staging_gate_blocks_rear_reinforcement`.
- **Drop `defense_coef` (the pump):** reinforcement has no shaping reward at all — credited purely
  via terminal + early_capture through GAE.
- **Small aggressive pool (rev53b-proven):** held-out LB archetypes in the pool so hoarding actually
  *loses games*. The asymmetry is what makes reinforcement instrumentally valuable; a pure symmetric
  mirror yields either flood (with `defense_coef`) **or** passivity (without it), never useful
  reinforcement. (c)-attribution alone removes the bad incentive but supplies no attack pressure →
  the pool supplies it. Both halves are needed.
- **Keep:** empire gate(3), garrison-floor(10), `expansion 0.03` (anti-passive, telescopes),
  `speed 0.3`, early_capture anneal, fire-entropy 0.005. **Drop `reinforce_cost`** (dead).
- **From-scratch** via snowball-BC warmstart (`seed_checkpoints/bc_snowball_pairwise15.pt`, aggressive
  winners, 53% reinforce coverage); pool = lb1152 hammer + debatreya_1300 @ external-fraction 0.25.
  Run script: `gpu_run_artifacts/p2rev1/run_remote_p2rev1.sh`.

The heavier "full causal fleet attribution" (Tier 2: tag fleets, route reward from a later
capture/defense back to the launch) is held in reserve if Tier 1 underperforms.

---

## Deferred track: target-head auxiliary supervision (measure first → threat head)

**Hypothesis (2026-06-11):** the reward model trains `fire` ("should I act?") and `ship` ("how
much?") adequately — both get near-immediate signal — but the `target` head ("where?") has a
long-horizon credit-assignment problem: most targets only reveal their value many steps later.
Reinforcement makes this acute: "which threatened frontline do I reinforce" is a pure target-head
decision with no established competence. So the reward may value good *outcomes* without efficiently
training the *where* head.

**Principle — add auxiliary PREDICTION, never target reward.** Two hard constraints from the project's
own graveyard:
- **No target-specific reward** ("+reward for high-prod / nearest / weak-enemy targets"). rev49
  (production-weighted capture) → carpet-bomb; rev47/48 (activity) → token-fire; rev41–45
  (srcs-penalty) → fire=0 Nash. Reward-shaping the target head hard-codes degenerate heuristics.
- **No KL toward a heuristic target distribution** (the tempting "counterfactual `softmax(target_score)`"
  idea). This is the **rev54 crater**: BC-seeding the target head toward a strong heuristic set
  own-target top1 6.3%→53.8% and passed every label gate, but held-out win-rate **collapsed (Ajay
  4.17%→0%, eliminated every game)** — the shared `target_scorer` got dragged toward the heuristic's
  enemy/neutral targeting and destroyed our aggressive winning play. **We win by targeting differently
  from the heuristics; copying their targets craters us. Label-accuracy is a misleading proxy.**

The safe form is **self-supervised prediction heads** (threat / future-owner / opportunity): they
shape the shared transformer representation, they do NOT bias the policy toward anyone's target
choice.

**Why MEASURE before building.** It is not established that the target head is the bottleneck. The
win-mechanism analysis says wins = aggression→decisive elimination, losses = passivity (a *fire-rate*
lever), and the Ajay gap is **"conversion *timing*, not routing"** — i.e. attack targeting is roughly
fine. The head's weakness, if any, is in **reinforcement** targeting specifically. So we instrument
first and let p2rev1 tell us.

**Diagnostics added (p2rev1 emits these — done 2026-06-11):**
- `H_tgt` (target entropy) — the "is the where-head uniform/undertrained" canary.
- target-owner share among launches: `reinf` (own) + `tgt n/e` (neutral/enemy) on the diag line;
  `policy/target_share_*` + `entropy/target` in W&B.
- (already present) `reinforce_rate` read by empire size.
- **Bad signs to watch:** `H_tgt` stays high while fire looks fine; own-targeting goes uniform once
  the empire gate opens; reinforces are mostly rear/self-loop (the forward mask should prevent this —
  if not, a build bug); captures happen but overkill is huge.
- **Deferred deeper diagnostics** (need multi-step fleet-outcome tracking, so they come WITH the
  threat-head build, not before): frontline-reinforce-usefulness (fraction of reinforces arriving at
  planets later attacked/threatened), capture-success-per-fleet, overkill/underkill ratio.

**Decision gate → if undertrained, build the THREAT head first (p2rev2), nothing else.** Per owned
planet, predict P(lost within K steps), self-supervised from the trajectory. It is the
highest-value / lowest-risk head and directly serves reinforcement target selection (reinforce the
planet you predict you'll lose). future-owner is costlier; `opportunity` overlaps the existing
roi/enemy_contest features. Keep any aux weight modest and annealable — teach geometry/economics,
don't let it override RL.

**Build landmine to bank now:** per-planet prediction labels require looking ahead in the trajectory
*and* handling that owned-planet slots reorder by planet array index on every capture/loss step —
the exact corruption that bit the VDN per-planet GAE work (scatter in planet-id space, not slot
space). Budget for it.

---

## 1. Why Phase 2 (and why not a resume)

Reinforcement is structurally forbidden in the rev38 lineage: the target mask was
`alive & (owner != me)`, so own planets were never legal targets and the model's `is_mine`
target-input weight was **never trained**. Four attempts to bolt reinforcement onto a mature
base all failed for the same root cause — unmasking own-targets on a policy whose equilibrium
assumes "never reinforce" causes a shock/flood:

| Attempt | What | Outcome |
|---|---|---|
| rev55 | hard unmask at t=0 | over-fire collapse <400K steps |
| BC-seed | teach target head reinforcement | cratered enemy/neutral play (shared scorer) |
| rev56 | annealed logit-bias curriculum | reinforcement emerged (0.73–0.80) but **floods** |
| rev57/b | + garrison-floor + reinforce-cost | still floods |

**Flood mechanism:** reinforcement is *near-costless* (friendly arrival does
`planet.ships += survivors` — can't lose a battle, material conserved). With any fire incentive
(`entropy_coef_fire`), the policy satisfies it *for free* by reinforcing — especially at large
empires where capture opportunities are exhausted. Band-aids (curriculum, LR throttle,
reinforce-cost tax) don't hold; taxing a costless action is the Nash-trap lever.

**Decision:** start fresh so reinforcement is acquired *natively* (own-targets legal from step 0
under a gate → the `is_mine` weight trains organically, no mature equilibrium to shock), with an
incentive structure that makes flooding impossible by construction.

---

## 2. What the LB top tier actually does (replay analysis, 2026-06-10)

Direct profile of top-2p replays (06-07/08/09), **timing-corrected** (the action at `steps[t]`
was decided on the obs at `steps[t-1]` — verified 100% vs 9.9% launch validity; same-step pairing
corrupted all prior analysis, esp. ship-commitment). Schemas: planet `[id, owner, x, y, radius,
ships, production]` (owner −1 = neutral, ids global); fleet `[id, owner, x, y, angle,
from_planet_id, ships]`; action `[from_planet_id, angle, num_ships]`.

| Metric | Isaiah (#1, controlled) | Jake Will (#2) | Aggressive cohort |
|---|---|---|---|
| game length (median) | 500 | 167 | 166 |
| planets @ end | ~11 (plateau) | 16 | 20 |
| reinforce @1 planet | **0.00** | **0.00** | **0.00** |
| reinforce @2 | 0.12 | 0.09 | 0.08 |
| reinforce @9–12 | 0.30 | 0.43 | 0.44 |
| reinforce @13+ | 0.34 | 0.38 | **0.61** |
| forward-staging | 68% | 70% | 66% |
| full-garrison commit | 55% | 90% | 78% |
| 1-ship probes | 0.0% | 0.2% | 3.7% |
| prod selectivity (tgt/avail) | 1.17× | 1.15× | 1.17× |
| distance rank (0=nearest) | 0.29 | 0.33 | 0.36 |

**Universal signatures (every top player):**
1. **Empire-size gate** — reinforce ≈0 at 1 planet, ~0.1 at 2, ramps with empire size. *Never*
   reinforce a tiny empire; expand first. This is the most reliable finding.
2. **Forward-staging (66–70%)** — reinforce targets are closer to the enemy than the source.
   Reinforcement flows rear→front (staging), not random redistribution.
3. **Hard commitment** — sends ≥75% of the source garrison ~80% of launches; ~never 1-ship
   probes. Validates `--min-ship-bin`.
4. **Positional targeting** — prefers near targets (rank ~0.3), no production selectivity (1.15×).
   Target head is tempo/reachability, NOT value-ranking.

**Style axis (not universal):** Isaiah grinds (plateau ~11 planets, 500 steps, reinforce ~0.34);
the aggressive cohort snowballs (16–20 planets, ~166-step decisive wins, reinforce up to 0.61).
The earlier "0.74 at 14+" headline was the snowball regime + same-step-timing artifact — heavy
*late* reinforcement is a winning trait; the flood was heavy *early* reinforcement.

**Implications for Phase 2:** gate reinforcement on empire size; do NOT cap `reinforce_rate` at
0.34 (let it climb); reward decisive/fast wins (`speed_coef`); keep the 1-ship ban; do NOT reward
target production. (Note: production-weighted *capture reward* — rev49 — is NOT a validated dead
end; that run had a broken target head. The replay finding is only that top players don't *target*
production, which is separate from whether a prod-weighted reward helps our agent.)

---

## 3. The reward model

Per-step reward for a player (2-player game):

```
r_t =

  ┌─ at game end only ──────────────────────────────────────────────┐
  │  WIN  at step t_win :  +1 · ( 1 + speed_coef · (T_max − t_win)/T_max )
  │  LOSS               :  −1
  └─────────────────────────────────────────────────────────────────┘

  ┌─ every step — dense shaping ────────────────────────────────────┐
  │  + expansion_coef     · Δ( P_mine − P_enemy )           (economy lead)
  │  − defense_coef       · max( 0 , −ΔP_mine )             (production lost)
  │  + early_capture_coef · w(t) · clip( ΔN_mine , −1, +1 ) (planets taken)
  └─────────────────────────────────────────────────────────────────┘
```

- `P` = my total owned production · `N` = my planet count · `Δx` = change since last step
- `w(t) = exp(−2.5·t/T_max) + 0.10`, ×`first_strike_mult` while `t < first_strike_steps`
- `T_max` = 500

In plain terms:
```
r_t = terminal win/loss
    + production-control shaping   (expansion_coef)
    + planet-count opening shaping (early_capture_coef, first-strike + decay)
    − production-loss defense      (defense_coef)
    × speed terminal modifier      (speed_coef, winner only)
```

### Locked baseline config

```yaml
# terminal
terminal_win:        1.0
terminal_loss:      -1.0
speed_coef:          0.3      # reward winning EARLY (snowball/decisive; LB-aligned)
win_margin_coeff:    0.0      # DROPPED — speed replaces margin

# dense shaping
expansion_coef:      0.03     # grow production lead (telescopes → passive ≈ 0)
defense_coef:        0.03     # penalty for production lost → reward HOLDING (the reinforce incentive)
early_capture_coef:  0.3      # capture spike, front-loaded
first_strike_mult:   on
early_capture_decay: on       # exp(-2.5 t/T)+0.10

# reinforcement (masks, NOT reward terms)
allow_reinforce:               true
reinforce_gate_min_planets:    3     # own-targets legal only at >= 3 planets (NEW)
reinforce_garrison_floor:      10    # veto a reinforce that drains source < 10 ships
reinforce_cost:                0.0   # DROPPED — no tax on a costless action

# PPO entropy (in the loss, not r_t)
entropy_coef_target: 0.05
entropy_coef_ship:   0.05
entropy_coef_fire:   0.005    # NOT 0.05 — see §4
```

### The two bets (overturn the rev57 approach, locked)
> ⚠️ **Bet #1 REFUTED — see the 2026-06-10 Update at the top.** rev58 (cost 0) AND rev58b (cost
> 0.001) both flooded; `defense_coef` itself is the pump. Tier-1 drops `defense_coef`. Kept below
> for history.

1. **`reinforce_cost = 0`** — the flood is not a reward problem; rely on the gate + `defense_coef`
   + low fire-entropy. Keep the cost in the back pocket only.
2. **`speed_coef 0.3` replaces `win_margin 0.5`** — time-to-victory rewards efficient snowball wins;
   margin risks rewarding over-extension/carpet-bomb.

### Double-count (intentional)
Losing a 5-production planet hits both `expansion_coef` (Δlead = −10) and `defense_coef`
(−5·coef). Accepted: holding is the behaviour we specifically need to recover. **If the agent
turns too conservative, lower `defense_coef` first — do NOT add a reinforce tax.**

### Anneal policy

We only anneal behavioural *kickstarts*, not persistent strategic gradients.

- **`early_capture_coef` — annealed** (frac 0.67). It is an exploration/kickstart term that
  teaches early expansion. Once the policy has learned to launch and capture, a large
  planet-count spike would distort late-game behaviour. (Also has an intra-episode `exp` decay.)
- **`expansion_coef` — stays on.** It is a production-control gradient (telescoping → low
  distortion) and fixes self-play collapse from passive losing policies.
- **`defense_coef` — stays on for the full run.** It is the outcome-tied reinforcement
  incentive: reinforcement has NO direct reward and earns value only by preventing production
  loss. Annealing defense would remove the late-training reason to reinforce and risks collapse
  back to the no-reinforce attractor. **If reinforcement is too conservative or too frequent,
  reduce `defense_coef` magnitude — do NOT anneal it.**
- **`speed_coef` — stays on.** It shapes decisive wins, not action choice directly.

No anneal mechanism exists on `expansion`/`defense`/`speed` (constant by construction); only
`early_capture` has the anneal flag — so the default config already implements this. The thing to
guard is never adding an anneal to `defense`.

---

## 4. The key design principle: reinforcement has NO reward term

> ⚠️ **The "self-capping by construction" claim below was REFUTED by rev58/58b (see top Update).**
> `defense_coef` is the flood *pump*, not the cure: in a symmetric mirror, hold-everything-via-
> reinforce dominates risky attacking. Tier-1 drops `defense_coef` and adds a forward-staging mask +
> aggressive pool. The "reinforcement has no *direct* reward" principle survives; the "masks +
> `defense_coef` make it self-cap" reasoning does not. Kept below for context.

Reinforcement is shaped entirely by **two hard masks on the action space**, never by a reward:

```
own planets are legal targets   ⟺   N_mine ≥ reinforce_gate_min_planets   (empire gate)
a reinforce launch is vetoed     ⟺   it would drain its source < garrison_floor
```

So reinforcing pays off **only instrumentally**: it raises `r_t` by *avoiding* the
`−defense_coef·(production lost)` term (holding threatened planets) and through the terminal win it
enables. It cannot be farmed for free. This is **self-capping** at the data's 0.3–0.6 ramp:
- reinforcing a *threatened frontline* planet → prevents loss → rewarded (and naturally
  forward-staging, matching the 68% measured);
- reinforcing beyond hold-need, or stacking an unthreatened rear planet → zero marginal reward;
- every reinforcing ship forgoes `expansion_coef` capture reward (opportunity cost).

**`entropy_coef_fire` is the one place the old config is actively dangerous.** `fire` is a binary
Bernoulli (max entropy ≈0.69); 0.05 is a hard push toward p≈0.5 firing. Once reinforcement is a
low-risk outlet, that entropy stops being neutral exploration and becomes pressure to emit harmless
same-owner fleets → flood. Start at **0.005** (the masks + dense shaping provide the activity floor).

---

## 5. Implementation

The reward terms (`speed_coef`, `expansion_coef`, `defense_coef`, `early_capture_coef`,
`garrison_floor`, `reinforce_cost`) **already exist** as flags in `train_torch.py` / `torch_env.py`.
The only new build is the **empire-size gate**:

- `torch_env.py` `__init__`: param `reinforce_gate_min_planets: int = 0` (0 = off, backward-compatible).
- `torch_env.py` `get_features` target-mask (`allow_reinforce` branch, ~line 593): own targets
  (`owner == player`) legal only where the env's owned-planet count ≥ threshold; enemy/neutral
  always legal; source always excluded.
- `train_torch.py`: `--reinforce-gate-min-planets` flag, passed to the env config + printed.
- `tests/test_reinforce_mask.py`: a gate test (own-target legality flips at the threshold).

Training-only, like `garrison_floor` — the real Kaggle env has no mask, the policy internalises the
gate. (Eval/export parity in `action_mask.py` is a follow-up to add at export time if we want to
*guarantee* the inference behaviour rather than rely on internalisation.)

---

## 6. Monitoring / dials

### Training reward ≠ selection metric (keep selection pure)

The shaped reward exists for the *gradient*. Champion/checkpoint selection runs on *outcomes
only* — never on shaped reward (confounded by expansion/defense/speed), `Vμ` (not a collapse
signal), or the Ajay panel alone (guardrail, not objective). This is the rev15 lesson:
`reinforce_rate` climbed while win-rate *declined*, and reading the behaviour profiler as
progress picked the wrong checkpoint.

- **Decider:** head-to-head **win-rate / Elo** (the checkpoint ladder vs prior milestones +
  external anchors) → ultimately **LB**. This and only this picks the champion.
- **Diagnostics (explain & health-check, never override):** `reinforce_rate` (0.3–0.6 by empire
  size), frontline-staging share, game length (decisive = shorter wins), capture rate,
  `clip_frac`. A checkpoint with great reinforce metrics but lower win-rate **loses**.

Primary dial = the `reinforce_rate` training metric (added rev57b; bias=0 offline read via
`behavior_analysis.py` is decision-grade).

**Caveat — `reinforce_rate` is partly mechanical.** Target choice at model init is ~uniform over
legal targets (measured 1.01× vs uniform — no architectural own-bias), so the aggregate
`reinforce_rate` co-moves with the board's own-fraction of legal targets, which rises as
neutrals/enemies are consumed (late-game, large empires). A high aggregate rate is therefore NOT
by itself a flood signal — read it **by empire size** and alongside **volume** (`avgfleet`, `p90`
ship counts) and win-rate. The flood is high rate **+** exploding volume (rev57: `p90` 368), not a
high rate per se. (Smoke @10k steps, untrained: reinf 0.67 at owned~6.8 with `p90`~190 = mechanical,
not flood.)

| Watch | Healthy | Action if not |
|---|---|---|
| `reinforce_rate` after N≥3 | 0.3–0.5 (may climb higher late) | **floods** → lower `entropy_coef_fire` first; **disappears** → raise `defense_coef` (never add reinforce-cost) |
| capture / expansion rate | does not collapse | if collapses with reinforce → fire-entropy too low (passivity) |
| avg game length | falls (decisive) | if rises → over-conservative; lower `defense_coef` |
| frontline reinforce share | high (rear→front) | low → gate/defense not inducing staging |
| `clip_frac` | < 0.25 | > 0.25 → halve LR (standing authority) |

---

## 7. Tooling & data

- **`orbit_wars_rl/fetch_analyze_top_replays.py`** — fetch top-score daily replays + behavioural
  analysis (timing-corrected). `--player NAME` / `--exclude NAME` / winner-mode. Metrics: reinforce
  ramp by empire size, forward-staging, ship commitment, prod/distance selectivity, expansion tempo.
- **Style selection without player names:** the daily `manifest.csv` has no TeamNames. Use
  `high avg_score + size_bytes < 3.5MB` as a *snowball-style selector* (short decisive games);
  it excludes Isaiah's 6.6MB/500-step grinds and yields the aggressive cohort (Jake 38%, TonyK,
  213tubo, Boey). Isaiah's grinds dominate plain top-by-avg_score (he is #1).
- Local replay sets (gitignored / `/tmp`): `/tmp/fresh_validate` (180 top-2p games),
  `/tmp/snowball` (89 aggressive-cohort games).

---

## 8. Open decisions / next steps

1. **Build the empire-size gate** (in progress) → unit-test → smoke-test a short run.
2. **Starting point:** fresh PPO from the standard BC warmstart tests "can it learn to reinforce on
   its own" under the new reward model (the original goal). A reinforcement-aware BC warmstart
   (Isaiah controlled vs Jake/cohort snowball — full reinforce ramp, better `is_mine` coverage) is
   the fallback/next iteration if emergence is too slow. **Sequence: reward model → gate → train →
   (BC seed only if needed).**
3. Launch on GCP L4 (one delta = the reward model + gate); first real `reinforce_rate` read early.
