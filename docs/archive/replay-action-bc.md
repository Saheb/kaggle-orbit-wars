# Replay Action BC: Gate2 + Cooldown + Save-Quality Filter

This document records the current replay-action BC data path after the
`bc_replay_value5k_fp64_20260619` validation showed that unfiltered save labels
poisoned own-target behavior.

The goal is not to maximize offline BC accuracy. The goal is a PPO seed whose
rollout behavior has sane attack conversion, sparse but real reinforcement, no
early reinforce flood, and no obvious reinforce ping-pong.

## Problem Found

The first replay-action BC used winner action lists and included save/reinforce
labels too broadly. Validation found:

- the checkpoint had a metadata bug: `allow_reinforce=True` was not persisted,
  so eval masked own targets unless the checkpoint was manually patched;
- when own targets were enabled, reinforce share became too high;
- reinforce ping-pong was severe;
- many save labels were likely safe, late, hopeless, or strategically low-value.

The fix is to make the replay dataset match the intended target-decode
discipline before BC:

- gate own-target reinforcement at `owned_count >= 2`;
- block reverse-edge reinforce ping-pong for 3 steps;
- keep only threatened, reachable, holdable, value-positive save labels;
- persist all discipline flags in the BC checkpoint.

## Code Paths

- Builder: `orbit_wars_rl/build_replay_action_bc.py`
- Trainer: `orbit_wars_rl/bc.py`
- Focused tests: `orbit_wars_rl/tests/test_build_replay_action_bc.py`

`bc.py` now saves these checkpoint config fields:

- `allow_reinforce`
- `reinforce_gate_min_planets`
- `reinforce_forward_only`
- `reinforce_garrison_floor`
- `reverse_edge_cooldown`
- `sufficient_commit_factor`
- `pairwise_feature_dim`

This avoids the previous bug where training/eval/export silently disagreed about
own-target legality.

## Label Timing

Replay actions are interpreted as:

```text
steps[t][seat].action was selected from steps[t-1][seat].observation
```

The builder copies the observation at `t-1`, decodes each replay move's
`[source_id, angle, ships]` into a target planet with the same intercept recovery
used by `bc.py`, filters the move labels, dedupes to one move per source, then
emits standard `bc.py` samples.

Only winner actions are used by default:

```text
--mode winner
```

## Save-Quality Filter

The builder keeps an own-target save only if all of these are true:

1. `owned_count >= reinforce_gate_min_planets`  
   Current default: `2`.

2. It is not reverse-edge blocked.  
   If `B -> A` was a kept save within the last `K` steps, `A -> B` is dropped.
   Current default: `K = 3`.

3. The target has enemy inbound.  
   Saves to already quiet planets are dropped.

4. The source can arrive before the threat.  
   Late saves are dropped.

5. The target has a positive defensive deficit:

```text
floor   = enemy_inbound + beta * reachable_enemy_mass + overhead
cover   = target_garrison + friendly_inbound_before_enemy
deficit = floor - cover
```

Current defaults:

```text
beta = 2.2
horizon = 18
overhead = 1
```

6. The save is not hopeless:

```text
reachable_friendly_mass >= deficit
```

7. The save is value-positive enough:

```text
deficit / max(target_production * horizon, 1) <= max_save_cost_ratio
```

Current default:

```text
max_save_cost_ratio = 1.0
```

Attack labels are kept by default. Optional attack filters exist:

- `--min-attack-value`
- `--min-reactive-roi`
- `--min-keepability`

Do not relax save filters just to inflate label count. That recreates the
save-spam failure.

## Current Dataset Snapshot

Command run locally:

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/build_replay_action_bc.py \
  gpu_run_artifacts/ar_stage0/replays \
  --mode winner \
  --max-samples 5000 \
  --step-limit 500 \
  --samples-out gpu_run_artifacts/bc_rp5k/replay_value_filtered_bc_5k_gate2_cd3_quality.pkl \
  --summary-out gpu_run_artifacts/bc_rp5k/replay_value_filtered_bc_5k_gate2_cd3_quality.json
```

Result:

```text
sample_count: 5000
replays_used: 95 / 424
raw_moves_seen: 10236
attack_moves_seen: 5885
kept_attack_moves: 5885
save_moves_seen: 4196
kept_quality_save_moves: 82
```

Dropped save labels:

```text
no threat:      2754
late:           1030
already safe:    185
hopeless:        112
expensive:        32
gate:              1
```

Read this as: the filter is doing its job, but quality save labels are now sparse.
Before launching a long PPO run, rebuild with all local replays and check whether
the quality-save count is high enough.

## Full Existing-Replay Build

Use this before collecting more data:

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/build_replay_action_bc.py \
  gpu_run_artifacts/ar_stage0/replays \
  --mode winner \
  --step-limit 500 \
  --samples-out gpu_run_artifacts/bc_rp5k/replay_action_bc_gate2_cd3_quality_all.pkl \
  --summary-out gpu_run_artifacts/bc_rp5k/replay_action_bc_gate2_cd3_quality_all.json
```

Then inspect:

```bash
python3 - <<'PY'
import json
p = "gpu_run_artifacts/bc_rp5k/replay_action_bc_gate2_cd3_quality_all.json"
s = json.load(open(p))["stats"]
for k in [
    "samples", "replays_used", "attack_moves_seen", "kept_attack_moves",
    "save_moves_seen", "kept_quality_save_moves",
    "filtered_save_no_threat", "filtered_save_late",
    "filtered_save_already_safe", "filtered_save_hopeless",
    "filtered_save_expensive", "filtered_save_reverse_edge",
]:
    print(k, s.get(k, 0))
PY
```

If quality-save count is still only a few hundred, get more winner replays or add
a deliberate save-balanced training phase. Do not weaken the filter first.

## BC Training Command

For the 5k quality-filtered dataset:

```bash
python3 orbit_wars_rl/bc.py \
  --samples gpu_run_artifacts/bc_rp5k/replay_value_filtered_bc_5k_gate2_cd3_quality.pkl \
  --steps 5000 \
  --batch-size 512 \
  --save checkpoints/bc_replay_value5k_gate2_cd3_quality.pt \
  --allow-reinforce \
  --reinforce-gate-min-planets 2 \
  --reverse-edge-cooldown 3 \
  --sufficient-commit-factor 1.0
```

The smoke-tested metadata for this command should load as:

```text
action_decode=target
allow_reinforce=True
reinforce_gate_min_planets=2
reverse_edge_cooldown=3
reinforce_forward_only=False
reinforce_garrison_floor=0.0
sufficient_commit_factor=1.0
```

## Validation Before PPO

Do not accept offline BC loss alone. Run behavior evals first.

Minimum local panel:

```bash
CUDA_VISIBLE_DEVICES="" python3 orbit_wars_rl/eval.py \
  --checkpoint checkpoints/bc_replay_value5k_gate2_cd3_quality.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 8 \
  --seed-start 0 \
  --target-decode \
  --fire-threshold 0.8 \
  --natural-head-audit
```

Read these before PPO:

- `cap/atk-launch`, especially open and mid;
- neutral and contested `cross`;
- `reinf_share` by empire size and by step;
- `reinf ping-pong recip<=1/2/3st`;
- `reinforce-triage`;
- hold-loss `out-massed` and `garr@loss vs enemy-inbound`;
- natural-head audit attack/save target top-k.

Go/no-go shape:

- no ship0 collapse;
- early/small-empire reinforce share not flooded;
- ping-pong near zero with cooldown active;
- attack conversion does not regress versus the previous best decode;
- holds are not still almost entirely out-massed.

If attack conversion improves but hold losses remain out-massed, do not assume PPO
will fix it. Treat the BC as an attack seed and add more targeted hold/save data
or a separate defense curriculum.
