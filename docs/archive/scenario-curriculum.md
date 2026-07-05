# Scenario curriculum

Status: implemented as an opt-in reset-state curriculum in `VecTorchEnv`.

This is the narrow successor to the failed broad board curriculum. The broad
`neutral_garrison_scale=3.0` experiment made the whole board expensive and the
policy could respond by becoming passive. These scenarios instead create short
episodes where the advantaged player wins only by applying one concentration
skill before a deadline.

## Design contract

Do not use these as a new proxy reward on normal games. They are tiny terminal
states mixed into training resets:

* `agg_attack`: two advantaged sources can capture the focal neutral target, but
  no single source can. Missing the deadline loses.
* `stage_attack`: a friendly inbound fleet is already committed but stops short;
  one more launch to the same target wins. Missing the deadline loses.
* `hold_under_peel`: the focal planet starts ours but thin; an enemy peel is
  inbound. Reinforcing enough to hold through the deadline wins; losing the
  planet loses.
* `mixed`: samples the three families with attack-heavy weighting.

The advantaged player is randomized per reset and mirrored geometrically, so the
same policy sees both roles from both seats.

## Implementation

Training flags:

```bash
--scenario-curriculum {off,mixed,agg_attack,stage_attack,hold_under_peel}
--scenario-fraction 0.10
--scenario-deadline 20
```

The implementation is in `orbit_wars_rl/torch_env.py`:

* reset-time state replacement happens in both initial `reset()` and `_auto_reset()`;
* scenario metadata records id, advantaged player, focal target, and deadline;
* scenario terminal outcomes override the normal terminal winner only for
  scenario boards;
* `stage_attack` and `hold_under_peel` seed one inbound fleet before material
  baselines are captured;
* seeded fleets use the same intercept aimer and launch-from-surface geometry
  as normal launches, so orbiting targets are genuinely threatened;
* the train log reports `scen <success_rate>/<count>` on the diagnostic line.

`scenario_fraction=0` or `scenario_curriculum=off` is a no-op.

## First run shape

Start as a mixed-in curriculum, not a replacement distribution:

```bash
python3 orbit_wars_rl/train_torch.py \
  --resume seed_checkpoints/revedge1_4718592.pt \
  --total-steps 2000000 \
  --checkpoint-interval 500000 \
  --pool-mode self \
  --pool-checkpoint-interval 500000 \
  --pool-max-size 20 \
  --pool-pfsp-min-games 30 \
  --action-decode target \
  --allow-reinforce \
  --reinforce-gate-min-planets 2 \
  --reverse-edge-cooldown 3 \
  --scenario-curriculum mixed \
  --scenario-fraction 0.10 \
  --scenario-deadline 20 \
  --il-lambda 0.05 \
  --il-ref seed_checkpoints/revedge1_4718592.pt \
  --run-name scenec1
```

Gate the run in two stages:

1. Scenario uptake: `scen` success rate must climb materially above random while
   PPO health stays sane (`EV`, `clip`, entropy).
2. Transfer: normal-board Ajay panel must move the real wall metrics
   (`outmassed%`, `peel`, `dm cross`, `planets@16/32`) before treating this as
   useful.

If scenario success does not climb, the scenario is either too hard, too sparse,
or not representable by the current policy. If scenario success climbs but Ajay
wall metrics stay flat, the tiny skill is not transferring and should not be
scaled blindly.

## Verification

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/tests/test_scenario_curriculum.py
/Users/saheb/home/.venv/bin/python orbit_wars_rl/tests/test_neutral_garrison_scale.py
```

Tiny CPU integration smoke:

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/train_torch.py \
  --num-envs 2 --rollout-steps 4 --num-minibatches 1 --total-steps 8 \
  --checkpoint-interval 1000000 --pool-mode none --action-decode target \
  --allow-reinforce --scenario-curriculum mixed --scenario-fraction 1.0 \
  --scenario-deadline 2 --device cpu --run-name scenario_smoke
```

## Diagnostic Commands

Run these before a long curriculum job:

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/audit_scenario_agents.py \
  --mode oracle --games 32 --deadline 20

/Users/saheb/home/.venv/bin/python orbit_wars_rl/audit_scenario_agents.py \
  --mode noop --games 32 --deadline 20

/Users/saheb/home/.venv/bin/python orbit_wars_rl/audit_scenario_agents.py \
  --mode head \
  --checkpoint gpu_run_artifacts/revedge1/checkpoints/torch_step_4718592_revedge1_20260616_095906.pt \
  --games 1 --deadline 20 --device cpu

/Users/saheb/home/.venv/bin/python orbit_wars_rl/audit_scenario_agents.py \
  --mode sample \
  --checkpoint gpu_run_artifacts/revedge1/checkpoints/torch_step_4718592_revedge1_20260616_095906.pt \
  --games 64 --deadline 20 --device cpu
```

Current `revedge1` reading:

* Oracle passes all three scenarios at 32/32.
* Deterministic checkpoint is 0/16 on all three scenarios.
* Head audit: target ranking is mostly correct on the main sources and ship bins
  are usable, but fire probabilities are near zero (`~0.001-0.005` for attack
  and stage, `~0.013-0.022` for the two main hold reinforcers).
* Stochastic sampling is nonzero but sparse: `agg_attack` 1/64,
  `stage_attack` 1/64, `hold_under_peel` 5/64.

Scenario validity gate for the next version:

* oracle must pass all three scenarios;
* noop must fail all three scenarios.

If `noop` succeeds, the scenario is not load-bearing and should not be used in
training.

## Local Uptake Smoke

On 2026-06-18, a short local CPU smoke was run from `revedge1` with
`scenario_fraction=0.50`.

First run:

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/train_torch.py \
  --resume gpu_run_artifacts/revedge1/checkpoints/torch_step_4718592_revedge1_20260616_095906.pt \
  --num-envs 16 --rollout-steps 64 --num-minibatches 4 \
  --total-steps 20000 --checkpoint-interval 10000 \
  --pool-mode none --action-decode target --allow-reinforce \
  --scenario-curriculum mixed --scenario-fraction 0.50 \
  --scenario-deadline 20 --device cpu \
  --run-name scenario_local_smoke
```

Stopped after the 10k checkpoint. This run is not clean because it omitted the
`revedge1` discipline flags (`--reinforce-gate-min-planets 2` and
`--reverse-edge-cooldown 3`), so the saved checkpoint auto-loads as gate/cooldown
off. Do not use it as a promotion candidate.

Even with that caveat, it showed the scenarios are learnable:

* before training: deterministic checkpoint was 0/16 on all three scenarios;
* after 10k local steps:
  * `agg_attack`: 15/32;
  * `stage_attack`: 20/32;
  * `hold_under_peel`: 32/32.

The same checkpoint was weaker on the same 16-seed Ajay sanity check than the
source checkpoint:

* scenario-smoke checkpoint: 2/16;
* original `revedge1` checkpoint: 4/16.

Because the smoke omitted discipline flags and used an aggressive local LR, this
is a transfer warning, not a final curriculum verdict.

A corrected local run with `--reinforce-gate-min-planets 2`,
`--reverse-edge-cooldown 3`, and `--learning-rate 0.00005` had sane first-update
PPO health (`KL 0.0082`, `clip 0.075`) but was too slow on CPU (~7 SPS, ~143s for
the first update). A meaningful disciplined scenario run should be on GPU.
