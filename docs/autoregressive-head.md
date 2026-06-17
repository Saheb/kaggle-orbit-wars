# Autoregressive action-list head plan

Date: 2026-06-17

Status: **downgraded / parked after Stage 0**. Do not build next.

Stage 0 is the last cheap falsification of the pivot. A KILL there is a good
outcome if rank-1 data does not contain the missing floor-covering skill.

Companions:

- `docs/peeler-curriculum.md`
- `docs/targeting-vs-sufficiency.md`
- `docs/head-audit.md`
- `docs/outmass-limits.md`
- `docs/metrics.md`

Reference checkpoint and panel:

```text
gpu_run_artifacts/revedge1/checkpoints/torch_step_4718592_revedge1_20260616_095906.pt
opponent: opponents/candidate_ajay_1200.py
```

Key logs:

```text
gpu_run_artifacts/head_audit/eval_logs/faildecomp_revedge1_4718592_ajay_panel.log
gpu_run_artifacts/head_audit/eval_logs/agg2_revedge1_4718592_ajay_panel.log
```

## Verdict

Stage 0 killed the near-term AR action-list pivot.

The original AR thesis had two separable claims:

```text
BC premise:
  winners use deliberate same-turn floor-covering aggregation;
  clone that native action-list grammar.

Capability premise:
  AR can represent within-turn residual-fill coordination that factored heads cannot.
```

The BC premise failed:

```text
native winner list length p50/p90/max: 1/3/29
same-turn multi-source floor-cross: 1.6%
co-arrival floor-cross, +10/+25/+50 windows: ~3.5%
```

The capability premise remains technically true but is no longer the next bet.
AR's unique capability is **within-turn sequential residual accounting**:

```text
move 1 commits to target T
move 2 sees T is still short and adds more
move 3 sees T is covered and stops
```

Stage 0 and replay inspection say top winners rarely rely on that mechanism.
The observed winning coordination is mostly factored-expressible:

```text
shared board salience:
  multiple source slots independently see the same important target and fire

cross-turn staging:
  prior inbound is already en route, and later source slots add to it

economy / positioning:
  winners often avoid states where only within-turn residual-fill would solve the contest
```

Concrete replay check, `leader-analysis/80315440.json`:

```text
step 245:
  target 12
  source 7 sends 90
  source 21 sends 102
  timed prior inbound 194
  current + prior = 386 vs floor-ish 363

step 399:
  target 7
  source 6 sends 60
  source 23 sends 57
  current = 117 vs floor-ish 102
```

These look coordinated, but they do not require AR. A factored policy can express
them because every source slot sees the same global board, including in-flight
fleets. The gap is therefore more likely a **training-signal / shared-state
accounting** problem than an action-grammar problem.

Next bet:

```text
existing factored policy
+ curriculum that makes concentration/coverage necessary to win
+ optional cheap coverage/inbound/floor-gap features
```

Concrete curriculum handoff: `docs/peeler-curriculum.md`.

Revisit AR only if the factored+curriculum path plateaus and a later audit shows
within-turn residual-fill matters at the optimal-play margin.

## Original AR Thesis

The proposed architecture bet was not "a better single action head." It was an
autoregressive decoder over the native action list:

```text
repeat:
  stop_or_continue
  target pointer
  source pointer conditioned on target + prefix state
  ship bin conditioned on target + source + prefix state
```

The reason is specific:

- `single ~= 0%` in failed attacks: when the chosen source has enough ships, per-source ship sizing is not the observed wall.
- raw `aggregate` failures are large and outcome-skewed: lost games need multi-source mass more often than won games.
- tightened `agg2 reachable-drainable` remains large: most raw aggregate failures survive a reachability/drainability filter.
- Probe A says current policy already aggregates at roughly winner frequency. Therefore the gap is not aggregation frequency; it is deliberate floor-covering coordination in the situations that need it.

Stage 0 had to prove that rank-1 replay labels contain this missing skill before
any expensive model work. It did not.

## Evidence for the pivot

Full panel vs Ajay, revedge1 4.72M:

```text
Overall: 61/256 (23.8%)
```

Failed-attack decomposition:

```text
LOST early<50    fail 52%  single 0%  agg 25%  unaff 24%  suff 29%  red 20%
LOST mid50-100   fail 58%  single 0%  agg 35%  unaff 15%  suff 35%  red 10%
LOST late>=100   fail 47%  single 1%  agg 33%  unaff 18%  suff 32%  red 10%
```

Tightened aggregate estimate:

```text
agg2 reachable-drainable:
overall early<50    75% raw-agg (334/448), 17% failures
overall mid50-100   74% raw-agg (579/779), 23% failures
overall late>=100   80% raw-agg (102/128), 10% failures

LOST early<50       75% raw-agg, 19% failures
LOST mid50-100      72% raw-agg, 25% failures
LOST late>=100      75% raw-agg, 24% failures
```

Read:

- Per-source sizing aux is not the main lever.
- Cross-source coordination is a real capability gap candidate.
- After tightening, the realistic AR-actionable ceiling is still roughly 19-25% of lost-game failed attacks.
- The wall is still not only capability: `unafford` and `suff` remain large, so curriculum/outcome pressure is also required.

## Probe A reconciliation

Probe A found that the current factored policy already multi-source aggregates at winner-like frequency:

```text
aggTurn Ajay-WON ~= 0.094
rank1 winners ~= 0.082
sameSrc = 0
```

That does not refute the AR pivot. It refines it.

The current policy can produce incidental aggregation at the right frequency. The question is whether it aggregates deliberately to cover a target floor when the board requires it.

AR is justified only if rank-1 action lists show:

```text
when a target needs multiple reachable/drainable sources to cross floor,
winners continue adding sources until the contest is covered,
and our policy often stops short, scatters, or chooses unaffordable contests.
```

This is the first Stage-0 gate. If rank-1 labels do not contain more deliberate floor-covering aggregation than our rollouts, AR-BC has no special skill to learn and the pivot should be killed or downgraded.

## Design

### Decoder order

Use target-first list decoding:

```text
stop_or_continue -> target -> source -> ship_bin
```

Rationale:

- The wall is contest-centric: pick the objective, then muster enough mass onto it.
- Source-first recreates the factored policy's asset-routing bias: route each source and hope they converge.
- Target-first lets the decoder keep selecting sources for the same target until the running state says it is covered.

### Fire dissolves into stop

There is no per-source fire head in the action-list decoder. "Fire" becomes the sequence termination decision:

```text
continue = emit another fleet
stop = end this turn
```

This removes the old failure mode where fire, target, and ship heads each make locally plausible but globally incoherent choices.

### Running prefix state

The decoder must receive or update explicit prefix state. Without this, it is AR in form but not in substance.

Required state:

```text
remaining_ships_by_source
committed_ships_by_target
estimated_floor_by_target
coverage_ratio_by_target = committed / floor
sources_used_this_turn
moves_emitted
```

Useful derived features:

```text
target_already_covered
target_under_floor_gap
source_can_reach_target_window
source_drainable_spare
source_already_used
```

### Native action-list BC

The biggest advantage of AR is that it unlocks rank-1 replay data without lossy projection.

The factored policy had to project a native action list down to one move per source/top-16, throwing away exactly the multi-source coordination signal we now need. AR can consume:

```text
[target_1, source_1, ship_1]
[target_2, source_2, ship_2]
...
[stop]
```

The BC phase has no RL credit-assignment problem: the list itself is the label.

## Credit assignment contract

Principle:

```text
Improve attribution of the real curriculum/outcome signal.
Never create a new reward proxy.
```

Do not add capture-event reward. Capture quantity rewards are the rev49/caputil trap: they encourage spray, churn, and thin expansion.

### BC phase

BC sidesteps RL credit assignment. It should teach:

- action-list legality
- target-first contest selection
- source aggregation onto one target
- stop-when-covered behavior
- native rank-1 sequencing patterns

### PPO phase

Do not start with complex credit machinery. First test whether standard turn-level PPO can preserve the AR grammar under curriculum pressure.

If turn-level PPO smears good and bad moves inside the same turn, then add prefix-level value:

```text
V_prefix(s, prefix_k)
delta_k = V_prefix(prefix_k) - V_prefix(prefix_{k-1})
```

This treats intra-turn decoding as its own MDP:

```text
state = (board, partial action list)
action = next emitted move or stop
reward = 0 until the turn executes
terminal reward = normal environment/curriculum return
```

`delta_k` is then the intra-turn TD/GAE credit for adding move `k`, not arbitrary shaping. It depends on critic quality, so validate with offline counterfactual probes before trusting it.

### Launch registry

Use a launch registry for attribution and auxiliary labels, not reward.

Safe outcome-grounded labels:

```text
target_held_K_steps
captured_and_held
move_part_of_won_contest
stop_before_redundant_overcommit
```

Risky heuristic-grounded labels:

```text
floor_crossed
agg2_contributed
covered_by_our_floor_estimate
```

Use heuristic labels only as low-weight representation hints, if needed. They are diagnostics, not oracles. The retarget-to-holdable-ROI collapse is the warning case.

### Counterfactuals

Use deterministic re-simulation offline, not in the main PPO loop:

```text
take sampled trajectory
remove or alter one emitted move
re-sim short horizon
measure delta in ownership / hold_K / dm cross / outcome
```

Purpose:

- validate prefix critic quality
- inspect whether turn-level advantage is smearing credit
- train or calibrate optional value/aux heads

Do not use exact counterfactuals as a new dense reward term initially.

## Staged plan

One stage, one falsifiable gate. Do not stack AR architecture, BC, PPO, prefix critic, aux losses, teacher KL, and curriculum in one run.

## Parallel workstream: curriculum

AR supplies the action-list capability. It does not create the outcome signal.
Curriculum is a co-equal workstream, not a minor Stage-2 detail.

The curriculum must be designed and de-risked in parallel with Stage 0/1. Stage
2 blocks on it. The previous h-ladder failed at the relevant problem class:
win-starvation and fixed-bot cheese. Do not assume "matched difficulty +
co-improving + concentration-required" exists until it has its own concrete
design and smoke criteria.

Minimum curriculum design contract before Stage 2:

```text
matched-difficulty rung with nonzero but nontrivial train WR
co-improving or ratcheted opponent, not one fixed exploitable bot
boards where single-source probes fail and floor-covering aggregation wins
metrics that show the rung rewards concentration, not generic competence
clear tripwires for fixed-bot cheese
```

If AR-BC passes Stage 1 but no credible curriculum exists, do not run plain PPO.
Park the AR checkpoint and solve the curriculum workstream first.

### Stage 0: data and premise audit

Goal: decide whether rank-1 native action lists contain a floor-covering coordination skill that our policy lacks.

Data corpus (durable — copied off `/tmp`, see `gpu_run_artifacts/ar_stage0/MANIFEST.md`):

```text
gpu_run_artifacts/ar_stage0/replays/top2/   100 1v1  (top-player 1v1)
gpu_run_artifacts/ar_stage0/replays/jake/   272 1v1  (Jake-wins, filtered)
leader-replays/rank1/                        117 1v1  (repo fallback)
total 1v1 = 489   filter: len(rewards)==2 at load; winner = argmax(rewards)
```

(4-player Isaiah corpus excluded for the 1v1 gate; raw JSON and the durable
MANIFEST are gitignored under `gpu_run_artifacts/`, and this doc is the tracked
provenance pointer.)

Tasks:

1. Parse native rank-1 action lists.
2. Verify legality and representability under the installed env.
3. Measure list length distribution and repeated-source frequency.
4. Define max decode length and stop-token construction.
5. Count BC data volume:
   - replay count
   - usable turns
   - emitted moves
   - floor-needed targets
   - multi-source floor-needed examples
6. Decide whether more top-player replays are needed before model work.
7. Measure projection loss only as context; do not train on projected labels.
8. Compute winner floor-covering behavior:
   - target requires multiple sources to cross floor
   - winner emits multiple moves to the same target
   - committed mass crosses floor
   - committed mass is lean rather than gross overkill
9. Compute the same floor-needed cases in our rollouts, especially raw `agg` and `agg2`.
10. Compare winners vs our policy:
   - not aggregation frequency
   - deliberate floor coverage in the cases that need aggregation

Floor-definition requirement:

Use the same floor for winner replays and our rollouts:

```text
agg2 floor
enemy target: reactive floor
neutral target: static garrison + overhead
timing: steps[t-1] observation -> steps[t] action
seat perspective: compute floor from the acting seat's view
```

Reuse the existing agg2/eval/replay-audit helpers where possible so definitions do not drift.

Gate:

```text
PASS only if all are true:
  1. winners' multi-source floor-cross rate on floor-needed targets exceeds ours by >= 15pp
  2. winners' lean-overkill p50 is meaningfully below ours (target: >= 20% lower p50 ratio)
  3. there are enough examples to train BC, or Stage 0 identifies a concrete replay expansion path

KILL/DOWNGRADE if:
  winners aggregate at similar frequency and similar floor-coverage quality,
  or the floor-cross gap is < 15pp,
  or data volume is too thin and cannot be expanded.
```

Stage-0 output should include:

```text
available replay count
usable turn count
usable move count
winner floor-needed target count
winner multi-source floor-cross rate
winner lean-overkill p50
our comparable floor-cross rate
our stop-short/scatter/unaffordable split
native list length p50/p90/max
illegal/unrepresentable action rate
```

Stage-0 winner-side audit, 2026-06-17:

```text
script: orbit_wars_rl/audit_ar_action_lists.py
output:
  gpu_run_artifacts/ar_stage0/action_list_audit_winners_1v1.json
  gpu_run_artifacts/ar_stage0/action_list_audit_winners_1v1.md

paths scanned: 588
1v1 replays used: 489
non-1v1 excluded: 99
usable turns: 82,262
action turns: 36,507
native moves: 58,970
list length p50/p90/max: 1/3/29
turns with >16 native moves: 0.24%

winner floor-needed targets: 5,775
winner multi-source floor-cross rate: 1.6%
winner stop-short rate: 96.3%
winner lean-overkill p50/p90 on crossed floor-needed targets: 1.26/1.67
```

Temporal co-arrival follow-up:

```text
script: orbit_wars_rl/audit_ar_action_lists.py --coarrival-window {10,25,50}
outputs:
  gpu_run_artifacts/ar_stage0/action_list_audit_winners_1v1_coarrival10.*
  gpu_run_artifacts/ar_stage0/action_list_audit_winners_1v1_coarrival25.*
  gpu_run_artifacts/ar_stage0/action_list_audit_winners_1v1_coarrival50.*

co-arrival cross on floor-needed targets:
  window +10: all 3.5%, open 4.8%, mid 3.6%, late 3.0%
  window +25: all 3.5%, open 4.8%, mid 3.6%, late 3.1%
  window +50: all 3.5%, open 4.8%, mid 3.6%, late 3.1%
```

Corpus split check:

```text
top2: 100 1v1, floor-needed 1,475, multi-source cross 1.8%
jake: 272 1v1, floor-needed 3,000, multi-source cross 1.9%
rank1: 117 1v1, floor-needed 1,300, multi-source cross 0.5%
```

Read: under the current `agg2` floor, top-player winners do not usually emit
same-turn multi-source action lists that cross floor-needed targets. This is
consistent across the durable corpora, so the strongest version of the AR-BC
premise is not supported by the winner labels as currently measured.

The temporal check also does not revive the premise: counting existing friendly
inbound that co-arrives with the current launches only raises floor-needed
coverage from 1.6% to about 3.5%, and the result is stable even with a wide
50-step window. Combined with native list p50=1 / p90=3, winner labels do not
appear to contain much within-turn or staged floor-covering aggregation to clone.

Do not treat this as a final PASS/KILL until the comparable policy-side audit is
run, but the winner side is already too weak to justify an expensive AR build on
"learn deliberate same-turn floor covering from rank-1 labels" alone. The next
Stage-0 question is whether the floor definition is too conservative for replay
labels, or whether the missing skill is not present in native top-player action
lists either.

### Stage 1: AR-BC only

Goal: prove the decoder can learn the native action-list grammar before RL.

Train:

```text
encoder warmstart from revedge1 or strongest current checkpoint
new AR action-list decoder
BC on native rank-1 action lists
no PPO
no aux losses initially
```

Gate:

```text
legal action rate high
random sanity passes
Ajay panel does not collapse
faildecomp agg2 decreases
dm cross improves or at least does not collapse
stop behavior sane (no endless list / no immediate stop collapse)
```

PPO-ability smoke:

```text
first PPO checkpoint from AR-BC has nonzero clip_frac
entropy is not floored
policy is not frozen
```

If AR-BC does not lower `agg2`, stop. The architecture failed to capture the grammar.

Warmstart caveat:

The encoder is still worth reusing, but its embeddings were shaped under factored
source-local decoding. Expect adaptation shock under AR heads; do not over-interpret
the first few PPO updates unless entropy/clip indicate a true freeze or collapse.

### Stage 2: AR PPO + curriculum + KL

Goal: test whether the AR grammar survives outcome-pressure training and improves the wall.

Precondition: curriculum rung is designed before launch.

Curriculum requirements:

- matched difficulty
- co-improving opponent, not fixed-bot cheese
- concentration is necessary to win
- opening/contest setup where single-source probes fail and floor-covering aggregation wins

Train:

```text
AR-BC init
standard turn-level PPO advantage
teacher KL to BC policy
curriculum opponent/rung
no V_prefix yet
no aux losses yet
```

Readout:

```text
training WR on matched rung climbs = curriculum gives gradient
agg2 stays low = AR grammar survives PPO
dm cross up / out-massed down = concentration improves
Ajay WR improves or at least does not regress
```

Failure interpretation:

```text
rung WR flat -> curriculum/signal failure
agg2 drifts up -> PPO erased AR grammar
agg2 low but WR flat -> aggregation capability exists but is not sufficient scoring lever
dm cross up but out-massed flat -> attack concentration without retention
```

Teacher-KL note:

Fixed BC KL is a soft cap. It is acceptable for the first "does grammar survive?" run, but should eventually ratchet to improving selves if Stage 2 works.

### Stage 3: prefix critic

Only add if Stage 2 shows turn-level advantage is too coarse.

Signals that justify Stage 3:

- good and bad moves appear in the same turn
- PPO pushes the whole list together
- agg2 degrades despite rung WR improvement
- offline counterfactuals show strong within-turn marginal differences

Add:

```text
V_prefix(s, prefix)
intra-turn TD / GAE credit
optional offline counterfactual calibration
```

### Stage 4: outcome-grounded auxiliary losses

Only add if representation/attribution remains weak after Stage 3.

Prefer:

```text
target_held_K_steps
captured_and_held
redundant_overcommit
stop_after_actual_coverage
```

Avoid or low-weight:

```text
match holdable-ROI target
reward capture events
hard force floor-cross labels
hard force agg2 labels
```

## Promotion metrics

Primary:

```text
Ajay WR (guardrail/proxy, not final proof)
failed-attack agg2 down
decisive-mass cross up
hold-loss out-massed down
```

Secondary:

```text
planets@16/32/50/100
unafford down
cap/atk open<50 and mid50-100
retention peel-rate
decisive-mass NEUTRAL cross and p50
lean overkill p50
```

Do not promote on:

```text
holdable-ROI best/top3
dom%
capture count alone
training reward alone
```

`holdable-ROI` is a sufficiency diagnostic, not a target oracle. Retargeting to top holdable-ROI scored 0/256.

Panel-to-LB humility:

Ajay panel gains are not LB proof. The project has prior counterexamples where Ajay
panel strength did not translate to leaderboard improvement. Use Ajay WR as a
guardrail and local pressure test; an actual LB submission remains the promotion test
for leaderboard claims.

## Non-goals

- Do not implement another eval-time override.
- Do not add capture-event reward.
- Do not train a per-source ship-sizing aux as the main lever.
- Do not judge AR by aggregation frequency alone.
- Do not run plain PPO after AR-BC.
- Do not stack AR + PPO + prefix critic + aux + curriculum before the BC gate.

## Implementation notes

Likely files:

```text
orbit_wars_rl/model.py
orbit_wars_rl/action_mask.py
orbit_wars_rl/train_torch.py
orbit_wars_rl/ppo.py
orbit_wars_rl/bc.py or new ar_bc.py
orbit_wars_rl/eval.py
```

Stage 0 likely belongs in a new audit script, not the training path:

```text
orbit_wars_rl/audit_ar_action_lists.py
```

The first deliverable should be a read-only report, not model code.

Throughput risk:

Sequential AR decode will reduce SPS relative to one-pass factored decoding. Measure rollout SPS early on the actual training box. A 3x slowdown changes iteration economics.

Decision threshold:

```text
If AR rollout SPS is < 150 on the intended GPU box, do not launch long curriculum PPO
until decode throughput is optimized or the run budget is explicitly accepted.
If slowdown is > 3x versus the comparable factored checkpoint, treat throughput as a
blocking issue for iteration economics even if the model is behaviorally promising.
```

## Current next action

Build Stage 0:

```text
native rank-1 action-list parser
floor-needed target detector
winner floor-covering aggregation report
our-policy comparable agg2/floor-covering report
```

Decision after Stage 0:

```text
if winner floor-covering signal exists -> build AR-BC
if not -> pivot back to curriculum/economy; AR is not justified by replay data
```
