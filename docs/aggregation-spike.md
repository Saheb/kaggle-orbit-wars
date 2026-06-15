# Target Aggregation Spike

Status: spike branch `codex/arch-action-spike`.

## Why This First

The action-list spike showed same-source multi-move is sparse in top replays,
while same-turn multi-source aggregation is much larger:

- rank1 winners: aggregated targets on 8.2% of firing turns, 10.4% of moves
- top-agent winners: aggregated targets on 19.7% of firing turns, 36.1% of moves

This matters because our current policy can already express basic aggregation:
multiple source slots can independently choose the same target in the same turn.
So the bottleneck may be training signal and coordination, not action grammar.

## Current Expressivity

Current model grammar:

```text
for each selected source slot:
  fire Bernoulli
  target categorical over planets
  ship categorical
```

This can produce:

```text
A -> T
B -> T
C -> T
```

in one turn, as long as A/B/C are in the selected source slots. Garrison-ranked
source selection improved this by keeping the largest available sources in those
slots.

It cannot produce:

```text
A -> T1 and A -> T2 in the same turn
```

but replay mining says that is the smaller phenomenon.

## Failure Mode

The wall is not per-source undercommit after the clamp fix: the ship head often
tries to send the whole source. The failure is target-level: several sources need
to coordinate on one target whose forward-projected defender count is larger than
any single source.

This is exactly the Deb/planner pattern:

1. choose a valuable target;
2. compute a capture floor at arrival;
3. collect enough sources to exceed that floor;
4. launch them as one wave.

Our PPO objective gives credit only after the resulting capture/hold dynamics.
Self-play rarely prices this, and Deb/Ajay wins are too sparse to teach it
directly.

## First Probe

Use `orbit_wars_rl/analyze_aggregation_replays.py` to measure whether top replay
aggregation is essential:

```bash
python3 orbit_wars_rl/analyze_aggregation_replays.py leader-replays/rank1 --mode winners
python3 orbit_wars_rl/analyze_aggregation_replays.py archive/replays/top_agent_replays --mode winners
```

For each turn, the script groups attacks by resolved target and reports:

- attack turns and attack moves;
- aggregated attack turns/moves;
- aggregated groups by phase;
- projected capture floor at the slowest contributor ETA;
- `essential`: combined ships meet the floor but no single source does;
- `solo_capable`: at least one source alone meets the floor;
- `under_floor`: combined ships still do not meet the floor.

The floor is approximate, not a bit-exact planner clone:

```text
neutral floor = ships_at_arrival + 1
enemy floor   = ships_at_arrival + production * 3 + 1
ships_at_arrival = current_ships + production * max_eta
```

That is sufficient for ranking whether aggregation is real versus cosmetic.

## Probe Result

Commands:

```bash
python3 orbit_wars_rl/analyze_aggregation_replays.py leader-replays/rank1 --mode winners
python3 orbit_wars_rl/analyze_aggregation_replays.py archive/replays/top_agent_replays --mode winners
```

Attack-only aggregation, corrected for replay timing:

| corpus | attack turns | attack moves | agg turns | agg moves | agg groups | essential | solo-capable | under-floor | coverage | src/group | max sources/target |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank1 winners | 7,603 | 10,902 | 6.1% | 9.2% | 481 | 15.8% | 29.7% | 54.5% | 1.63 | 2.07 | 5 |
| top-agent winners | 14,921 | 24,716 | 9.5% | 16.2% | 1,751 | 8.9% | 19.2% | 71.9% | 1.30 | 2.29 | 8 |

Phase split:

| corpus | phase | agg turns | agg moves | essential | solo-capable | under-floor | coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| rank1 winners | <50 | 1.3% | 2.5% | 8.0% | 24.0% | 68.0% | 0.95 |
| rank1 winners | 50-100 | 6.0% | 8.4% | 16.8% | 21.4% | 61.7% | 1.06 |
| rank1 winners | >100 | 9.4% | 13.5% | 15.8% | 36.5% | 47.7% | 2.12 |
| top-agent winners | <50 | 2.4% | 4.0% | 24.1% | 16.5% | 59.5% | 1.06 |
| top-agent winners | 50-100 | 6.7% | 10.1% | 10.1% | 21.3% | 68.6% | 0.99 |
| top-agent winners | >100 | 16.0% | 25.6% | 7.5% | 18.6% | 73.9% | 1.43 |

Player slices are important:

- Isaiah @ Tufa Labs: `aggTurn=4.2%`, `essential=31.5%`, `underFloor=29.6%`.
- kovi/rank1 slice: `essential=22.8%`, `underFloor=37.0%`.
- 213tubo: `aggTurn=40.0%`, but `essential=2.8%`, `underFloor=92.2%`
  — mostly fan-out/carpet behavior, not the clean floor-clearing wave we want.

Interpretation:

- Raw aggregation is common enough to monitor and train against.
- Essential aggregation exists, especially in stronger/non-carpet styles, but it
  is a smaller, higher-quality subset than raw same-target aggregation.
- A naive "reward aggregation" scalar would likely reward bad 213tubo-style
  under-floor fan-out. Any training signal must be **floor-aware**.
- The best first implementation is an eval/training diagnostic for target-level
  floor coverage, then a floor-aware auxiliary or group-level veto. Do not simply
  encourage more same-target launches.

## Candidate Implementation Path

### 1. Diagnostics In Eval

Add target-level launch metrics to `game_conversion`:

- `atk_agg_turn`: fraction of attack turns with >=2 sources on same target
- `atk_agg_moves`: fraction of attack moves in aggregated target groups
- `atk_agg_essential`: aggregated groups where sum clears floor and no source
  alone clears it
- `atk_floor_cov`: total ships sent / projected target floor

This tells whether our policy ever coordinates multiple sources and whether lost
captures are preceded by under-floor launches.

### 2. Training Signal, Not Mask

Avoid a hard sufficient-commit mask as the main solution. The current
`sufficient_commit_factor` is per-launch:

```text
block source if source_ships <= target_current_ships * factor
```

That can accidentally suppress exactly the multi-source wave we want, because
each individual contributor may be under the floor while the group is correct.

Better first signal:

- per target, after sampling all source actions, group attack launches by target;
- compute projected floor for that target;
- if total sent to target is below floor, optionally penalize/drop only the
  **whole under-floor group**;
- if total sent exceeds floor, leave contributors alone.

This prices target-level coordination without requiring a new action grammar.

### 3. Imitation Option

For a stronger structural signal, build a target-centric BC/DAgger dataset from
Deb/Producer/top replays:

- state features as usual;
- label selected target groups, not only independent source actions;
- for each target, teach which sources joined and total/floor coverage.

A small auxiliary loss can be added without changing execution:

```text
group_score[target] = logsumexp over source target logits weighted by fire prob
group_mass[target]  = sum expected ship_count(source) for sources choosing target
loss = target-group CE + floor-coverage regression/ranking
```

This is more invasive than eval diagnostics but less invasive than lane
architecture.

## Decision

Proceed one by one:

1. measure aggregation in top replays and our panels;
2. add target-level diagnostics;
3. test a target-floor group penalty/auxiliary on current grammar;
4. only then revisit lanes or `MAX_OWNED`.

If our policy cannot aggregate even when source slots are available, the first
fix is training signal. If it aggregates well but lacks enough source slots, the
next fix is `MAX_OWNED=24/32`. If top opponents rely heavily on same-source
splitting after those are solved, then lane architecture becomes worth the full
action-grammar rewrite.
