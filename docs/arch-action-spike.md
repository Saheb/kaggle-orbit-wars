# Action-List Architecture Spike

Status: spike branch `codex/arch-action-spike`.

## Question

The official game accepts an action list: multiple moves per turn, and the same
source planet can appear in more than one move as long as ships remain after each
sequential debit. Our policy currently emits one action per owned source slot:

```text
(N, MAX_OWNED, 4) = [fire, angle_bin, ship_bin, target_idx]
```

This spike separates two possible architecture changes:

1. Raise `MAX_OWNED`.
2. Add multiple launch lanes per source.

## Finding

Raising `MAX_OWNED` is mostly a width change. It threads through:

- `torch_env.MAX_OWNED`
- `features.MAX_OWNED_PLANETS`
- `action_mask.MAX_OWNED_PLANETS`
- `ModelConfig.max_owned_planets`
- rollout storage in `train_torch.py`
- BC sample tensors in `bc.py`
- export constants in `export_agent.py`

The model heads are shared per slot, so there are no per-slot parameters to
resize. This is likely checkpoint-loadable if the checkpoint config is patched,
but it changes rollout/action tensor shapes, entropy/logprob scale, PPO joint
logprob magnitude, memory, SPS, and the global feature `owned/MAX_OWNED`.

Multi-move/source is a true action-grammar change. It touches rollout sampling,
PPO logprobs, env debit/order, BC labels, eval/export decode, and action-list
length. This is the scratch-run candidate.

## Recommended First Architecture

Use fixed launch lanes per selected source:

```text
MAX_OWNED = 16 or 24
MAX_LANES = 2
actions: (N, MAX_OWNED, MAX_LANES, 4)
```

Each lane has `[fire, target, ship]`. The source slot is shared; lane 0 and lane
1 can choose different targets. The environment flattens `(source_slot, lane)`
into an action-list order and debits sequentially, matching the official engine.

Avoid a lane-agnostic duplicate head. If lane 0 and lane 1 see identical slot
embeddings, they will emit identical moves. Add a small lane embedding before
the fire/ship/target heads:

```text
lane_entities = owned_enriched[:, :, None, :] + lane_embed[None, None, :, :]
```

Then heads run on `(B, MAX_OWNED, MAX_LANES, D)`.

## Warm Start Plan

For `MAX_LANES=2`:

- initialize shared backbone, pairwise heads, target scorer, ship head, fire head
  from current checkpoint;
- initialize `lane_embed[0] = 0`;
- initialize `lane_embed[1]` small random or negative-fire-biased;
- set lane-1 fire bias negative at launch, anneal to 0 over early training.

This makes lane 0 reproduce the old policy at step 0 and introduces lane 1
gradually. It is still a new learning problem, but not a blind cold start.

## PPO Changes

Current PPO sums per-slot fire/ship/target logprobs:

```text
new_log_prob = sum_MO(fire + fired * ship + fired * target)
```

Lane PPO becomes:

```text
new_log_prob = sum_MO,sum_L(fire + fired * ship + fired * target)
```

This mechanically increases joint-logprob variance and clip pressure. The
per-slot `clip_frac_fire` should be extended to per-lane; the joint clip should
not be the only health read.

Entropy coefficients may need downscaling or lane-1-only gating, because doubling
Bernoulli decisions doubles the free-fire pressure.

## Env Changes

`VecTorchEnv._apply_actions()` currently assumes one action per selected source
slot. The lane implementation should:

1. accept either `(N, MO, 4)` legacy or `(N, MO, L, 4)`;
2. expand `owned_idx`, `slot_valid`, source state, ship count, target decode, and
   masks across lane;
3. flatten to `(N, MO*L)` in deterministic source-major/lane-major order;
4. apply sequential debit semantics.

The current vectorized scatter debit is valid only when each source appears once:

```python
new_ships = ships_col.scatter_add(1, owned_idx, -debit)
```

With lanes, multiple debits can target the same source. `scatter_add` handles the
sum, but launch validity must be sequential, not based on the pre-debit source
ships for every lane. Otherwise two lanes can each spend the full garrison.

Minimal faithful semantics:

- lane 0 sees full source ships;
- lane 1 sees source ships minus lane-0 realized debit;
- more lanes repeat this scan.

This loop is over `MAX_LANES` only, so it can remain vectorized over env/source.

## BC Changes

`trajectory_to_training_sample()` currently overwrites labels when a source emits
multiple moves in the same turn. It builds:

```python
fire_target[slot] = 1
ship_target[slot] = ...
target_target[slot] = ...
```

For lanes, labels become:

```text
fire_target:   (MAX_OWNED, MAX_LANES)
ship_target:   (MAX_OWNED, MAX_LANES)
target_target: (MAX_OWNED, MAX_LANES)
```

For each source, sort that turn's moves in replay action-list order and assign
the first `MAX_LANES` moves to lanes. Extra moves are counted as truncation.

This is a concrete reason a from-scratch or BC-warmstarted arch run should not
start until replay coverage is measured.

## Eval/Export Changes

Eval/export should emit an unbounded action list up to the model's lane budget:

```text
max emitted moves = MAX_OWNED * MAX_LANES
```

Do not reintroduce the old fixed 8-move cap. The only cap should be the model
grammar and source-garrison availability.

## Acceptance Tests

Before any full run:

1. Legacy shape `(N, MO, 4)` still produces byte-equivalent env transitions.
2. Lane shape `(N, MO, 2, 4)` can fire two moves from one source when enough ships
   exist.
3. Lane 1 is dropped, not overdrawn, when lane 0 spends the source garrison.
4. Eval/action-mask/export agree on emitted move order and ship clamps.
5. BC lane-label builder preserves same-source replay moves instead of
   overwriting them.
6. PPO loss accepts lane tensors and reports per-lane fire clip/entropy.
7. SPS/memory benchmark at `MAX_LANES=2`, `MAX_OWNED=16` and optionally
   `MAX_OWNED=24`.

## Decision Rule

Run the replay action-list probe first. If strong replays show same-source
multi-move or same-turn multi-source aggregation is common in decisive phases,
the lane architecture is worth a scratch/BC-warmstarted run. If same-source
multi-move is rare and the dominant signal is multi-source same-target
aggregation, prioritize a training signal or target-centric objective before
lane architecture.

## Replay Probe Result

Command:

```bash
python3 orbit_wars_rl/analyze_action_list_replays.py leader-replays/rank1 --mode winners
python3 orbit_wars_rl/analyze_action_list_replays.py archive/replays/top_agent_replays --mode winners
```

Summary:

| corpus | turns | moves | same-source turn | same-source moves | aggregate turn | aggregate moves | >16 moves/turn |
|---|---:|---:|---:|---:|---:|---:|---:|
| rank1 winners | 9,738 | 17,843 | 4.1% | 5.9% | 8.2% | 10.4% | 0.0% |
| top-agent winners | 21,380 | 79,431 | 1.3% | 0.7% | 19.7% | 36.1% | 3.7% |

Phase split:

| corpus | phase | same-source turn | aggregate turn | aggregate moves | >16 moves/turn |
|---|---|---:|---:|---:|---:|
| rank1 winners | <50 | 0.6% | 1.5% | 2.8% | 0.0% |
| rank1 winners | 50-100 | 2.8% | 8.2% | 9.6% | 0.0% |
| rank1 winners | >100 | 7.1% | 11.4% | 13.2% | 0.0% |
| top-agent winners | <50 | 0.3% | 3.1% | 4.6% | 0.0% |
| top-agent winners | 50-100 | 2.1% | 10.3% | 14.2% | 0.5% |
| top-agent winners | >100 | 1.0% | 33.8% | 46.0% | 7.8% |

Interpretation:

- Same-source multi-move is not the dominant top-player behavior in these
  corpora. It is real for some agents (for example Shun_PI/Vadasz), but it is
  sparse overall and mostly mid/late.
- Same-turn multi-source aggregation is much more common. The current policy can
  represent some of it already by choosing the same target from multiple source
  slots, so this points more toward training signal/objective and source
  availability than toward same-source lanes as the first scratch run.
- `>16` moves/turn is rare in rank1 and concentrated in a few top-agent styles
  (notably 213tubo-like carpet-bomb/fan-out). Raising `MAX_OWNED` or adding more
  lanes may help late/won-game throughput, but it is not the cleanest first
  answer to the Deb/Ajay out-massed wall.

Spike conclusion:

1. Do not start with lane-based multi-move/source as the first expensive arch
   training run.
2. If we do an architecture experiment, the lower-risk first step is
   `MAX_OWNED=24/32` plus garrison ranking, because it targets source
   availability and `>16` action-list turns without changing per-source grammar.
3. For the force-concentration wall, prioritize a target-centric aggregation
   signal/probe: can the model learn to put multiple selected sources onto the
   same target at the same arrival floor? This aligns with the larger replay
   signal and with Deb's `_plan_regroup` behavior.
