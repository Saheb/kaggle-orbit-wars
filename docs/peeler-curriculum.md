# Peeler curriculum plan

Date: 2026-06-17

Status: next curriculum candidate. Build only after the de-risk gate passes.

## Starting Point

The autoregressive action-list route is parked. Stage 0 showed that top-player
native action lists do not contain much cloneable same-turn or co-arrival
floor-covering aggregation:

```text
winner native list p50/p90/max: 1/3/29
same-turn multi-source floor-cross: 1.6%
co-arrival floor-cross, +10/+25/+50 windows: ~3.5%
```

Replay inspection (`leader-analysis/80315440.json`) showed that visible winner
coordination is mostly factored-expressible:

```text
shared board salience: multiple source slots independently fire at the same salient target
cross-turn staging: prior inbound is already en route, later launches add to it
economy / positioning: winners avoid many contests where only within-turn residual-fill helps
```

The next bet is therefore not a new action grammar. It is a curriculum that
prices the outcome we need: captures must stick.

## Core Hypothesis

Use a graded reactive peeler opponent:

```text
weak / beatable expansion
strong reactive peel against thin captures
reactive targeting, not a fixed opening script
```

The peeler replaces the dead BC seed. We do not need a self that already knows
concentration. We need an opponent that makes thin, unheld captures lose while
remaining beatable enough to give PPO a win gradient.

This is outcome pressure, not a reward proxy. Keep the existing win/material
reward. Do not add a concentration reward, capture reward, or floor-cross reward.

## Existing Hooks

The repo already has the main pieces:

```text
opponents/candidate_producer_v2.py
  orbit_lite planner, horizon, ROI threshold, reactive reinforcement floor,
  regroup, max_waves_per_turn, safe_drain sizing

opponents/candidate_producer_h4.py / h10.py / h12.py / h14.py
  horizon-tier precedent

opponents/candidate_landgrab.py
  weak-expansion baseline shape

orbit_wars_rl/opponent_pool.py
  external heuristics, EMA PFSP, --pfsp-externals

orbit_wars_rl/train_torch.py
  --pool-mode mixed
  --external-opponents
  --pool-external-fraction
  --pfsp-externals
  --pool-hard-ramp-steps
  --pool-pfsp-min-games
  teacher-KL / IL anchor flags
```

## Peeler Design

Implement a small family of `candidate_peeler_t*.py` files derived from
`candidate_producer_v2.py`, not a rewrite.

Separate two behaviors:

```text
Expansion strength:
  how aggressively the opponent grabs neutral/economy targets.

Peel strength:
  how strongly it retakes or reinforces planets we captured thinly.
```

For the first gate, weaken expansion while preserving peel.

Expansion-weakening knobs:

```text
raise roi_threshold
lower max_offensive_targets
lower max_waves_per_turn
raise min_ships_to_launch slightly
optionally reduce horizon for offensive planning
```

Do not weaken these initially:

```text
reinforce_size_beta
max_defensive_targets
enable_regroup
```

Those are the peel / retention pressure path. Weakening them risks producing a
beatable opponent that does not force the skill.

## Initial Tiers

Start with three files so `--pfsp-externals` can find the matched rung:

```text
candidate_peeler_t0.py  # weakest / gate target
  horizon=10
  max_offensive_targets=4
  max_defensive_targets=4
  max_sources_per_lane=8
  max_waves_per_turn=2
  roi_threshold=3.0
  min_ships_to_launch=8.0
  reinforce_size_beta=2.2
  enable_regroup=True
  max_regroup_time=7.0

candidate_peeler_t1.py
  horizon=14
  max_offensive_targets=6
  max_defensive_targets=4
  max_sources_per_lane=10
  max_waves_per_turn=3
  roi_threshold=2.2
  min_ships_to_launch=6.0
  reinforce_size_beta=2.2
  enable_regroup=True
  max_regroup_time=7.0

candidate_peeler_t2.py
  horizon=18
  max_offensive_targets=8
  max_defensive_targets=4
  max_sources_per_lane=12
  max_waves_per_turn=4
  roi_threshold=1.8
  min_ships_to_launch=5.0
  reinforce_size_beta=2.2
  enable_regroup=True
  max_regroup_time=8.0
```

These numbers are hypotheses. The gate owns the tuning. Do not train until the
weakest tier is matched.

## De-Risk Gate

Run no training until all are true:

```text
1. peeler beats random / trivial agents
2. revedge1 beats weakest tier about 40-60% on a full-enough eval
3. peeler specifically punishes thin captures
4. peeler does not win only by superior expansion/economy
```

The h-ladder failed by being win-starved. If revedge1 cannot get near 50%
against `t0`, soften `t0` before any GPU run.

Thin-capture punishment readout:

```text
our captures peeled within K steps
median hold duration after capture
out-massed% on lost captures
mass-to-lost / garrison-vs-inbound at loss
agg2 / failed-attack buckets
planets@16/32/50 so expansion strength is visible
```

Required pattern:

```text
thin captures get peeled
concentrated/staged captures stick more often
revedge1 still wins roughly half the weakest-tier games
```

If the peeler simply crushes economy, it is too strong or mis-aimed. If it is
beatable but does not peel thin captures, it does not force the target skill.

## De-Risk Gate Results

Date: 2026-06-17

Local peeler files:

```text
opponents/candidate_peeler_t0.py
opponents/candidate_peeler_t1.py
opponents/candidate_peeler_t1_sticky.py
opponents/candidate_peeler_t2.py
```

Eval checkpoint:

```text
gpu_run_artifacts/revedge1/checkpoints/torch_step_4718592_revedge1_20260616_095906.pt
```

Logs:

```text
gpu_run_artifacts/peeler/eval_logs/revedge1_vs_peeler_t0_32.log
gpu_run_artifacts/peeler/eval_logs/revedge1_vs_peeler_t1_32.log
gpu_run_artifacts/peeler/eval_logs/revedge1_vs_peeler_t2_32.log
gpu_run_artifacts/peeler/eval_logs/revedge1_vs_peeler_t1_64.log
gpu_run_artifacts/peeler/eval_logs/revedge1_vs_peeler_t1_sticky_32.log
gpu_run_artifacts/peeler/eval_logs/revedge1_vs_peeler_t1_sticky_64.log
```

Headline:

```text
t0 32g: revedge1 32/32, 100.0% WR  -> too easy as matched rung
t1 32g: revedge1 17/32, 53.1% WR   -> matched
t2 32g: revedge1  8/32, 25.0% WR   -> hard rung
t1 64g: revedge1 35/64, 54.7% WR   -> confirmed matched
t1_sticky 32g: revedge1 17/32, 53.1% WR -> matched
t1_sticky 64g: revedge1 35/64, 54.7% WR -> confirmed matched
```

The de-risk gate passes with `t1` as the initial matched rung. `t0` is an easy
warm-up tier, not the main pressure source. `t2` is useful as a harder PFSP rung
once the policy starts beating `t1`.

`t1_sticky` is a small T1 variant that adds a score bonus for enemy-owned,
low-garrison, productive targets. It is intended to make thin frontier retakes
more salient without strengthening expansion. The 64-game gate showed it is
not materially harder than T1; treat it as an alternate matched rung, not a new
hard tier.

The `t1` 64-game confirmation shows the right kind of pressure:

```text
peel-rate: 0.57 overall, 0.39 WON, 0.98 LOST
hold-loss: 97% out-massed
garr@cap -> @loss vs enemy-inbound: 42 -> 28 vs 90
hold-floor under%: <50 91%, 50-100 88%, >=100 87%
capture-born class: safe 57%, cheap 15%, exp 9%, hopeless 19%
nonterminal lost-rate by birth: safe 40%, cheap 64%, exp 65%, hopeless 89%
failed-attack agg2 LOST: early 17%, mid 17%, late 19%
failed-attack agg2 WON:  early  8%, mid 11%, late  5%
```

The `t1_sticky` 64-game confirmation is nearly identical:

```text
peel-rate: 0.57 overall, 0.39 WON, 0.98 LOST
hold-loss: 97% out-massed
garr@cap -> @loss vs enemy-inbound: 42 -> 28 vs 90
failed-attack agg2 LOST: early 17%, mid 17%, late 19%
failed-attack agg2 WON:  early  8%, mid 11%, late  6%
```

Read: `t1` is beatable but punishes thin / unheld captures. Losses are still
dominated by enemy reactive concentration, not by a simple expansion crush.
That is the pressure profile C1 wanted.

## C1 Run

Only after the de-risk gate passes:

```text
resume: revedge1 4.72M
pool: peeler tiers as external opponents, with t1 as the matched starting rung
external sampling: PFSP over externals enabled
teacher-KL: anchor to revedge1
reward: unchanged; no concentration proxy
SSDR: off unless pool assignment / mask is explicitly aligned
```

Suggested command shape:

```text
--pool-mode mixed
--pool-fraction 1.0
--pool-external-fraction 0.70-0.85
--pfsp-externals
--pool-pfsp-min-games 30
--external-opponents opponents/candidate_peeler_t0.py,opponents/candidate_peeler_t1.py,opponents/candidate_peeler_t2.py
--il-lambda 0.025-0.05
--il-decay-frac 0.8
```

Note: the current `--pool-hard-ramp-steps` code ramps external fraction only when
`--pool-pinned-fraction > 0`. Either add a tiny pinned fraction intentionally or
do not rely on hard-ramp for C1.

## C1 Read

Primary reads:

```text
training WR on matched peeler rung climbs from near 50%, not flat
held-out Ajay out-massed% bends below the ~95-96% wall
retention / peel-rate improves
agg2 failure share falls or shifts away from reachable-drainable failures
dm cross/gap improves without late-only overkill farming
```

Guardrails:

```text
Ajay/deb/Zach held-out panels do not collapse
opening planets@16/32/50 do not degrade
ship0 / tiny-probe collapse does not return
policy does not overfit by avoiding all captures
```

Kill / retune:

```text
flat WR on t0/t1 -> peeler still too hard or too noisy; retune tiers
WR climbs but Ajay out-massed flat -> peeler is not forcing transferable skill
out-massed improves but WR collapses elsewhere -> overfit / economy regression
```

## C2 Run

Only after C1 shows a gradient.

Add:

```text
past-self league / organic snapshots
ratcheted teacher anchor rather than forever-fixed anchor
possibly keep Ajay/deb as held-out, not necessarily training opponents
```

Purpose:

```text
prevent fixed-peeler cheese
keep the skill under co-improvement
avoid soft-capping forever to revedge1
```

## Coverage Features

Do not start with feature-only training.

Coverage/floor-gap features are a possible nudge, but they are not independently
testable in plain self-play: if the curriculum does not price retention, PPO can
ignore the feature and the null is uninterpretable.

Correct sequence:

```text
run read-only diagnostics in parallel with peeler work
if coverage/staging/economy metrics discriminate, add the smallest feature delta
test that feature only inside a curriculum run
require explicit uptake diagnostics before promoting
```

Pairwise feature width is shape-coupled to model layers:

```text
pair_kv
target_scorer first layer
```

Adding channels is a partial restart of the pairwise/target path, not a free
continuation. This is another reason the feature must be signal-gated.

## Why This Is Different

```text
SSDR / handicap:
  made games harder but did not specifically punish thin captures.

h-ladder:
  difficulty floor too high, uniform external sampling, fixed-bot cheese risk,
  and beating the rung did not necessarily require the target skill.

peeler:
  weak expansion keeps it beatable;
  reactive peel specifically prices retention;
  PFSP externals search for the matched tier;
  no reward proxy is introduced.
```

## Implementation Order

```text
1. create candidate_peeler_t0/t1/t1_sticky/t2 from candidate_producer_v2.py  [done]
2. run sanity eval: random/trivial vs peeler
3. run revedge1 full-ish panel vs t0/t1/t2  [done]
4. run thin-capture punishment audit on those evals  [done via eval metrics]
5. tune tiers until one rung is 40-60% for revedge1 and thin captures are peeled  [t1 and t1_sticky pass]
6. only then launch C1
```

A failed de-risk gate is a useful result. Do not force a GPU run if no tier is
both beatable and skill-punishing.
