# Orbit Wars — RL Training Project

A self-play RL agent for the [Orbit Wars Kaggle competition](https://www.kaggle.com/competitions/orbit-wars).
Players send fleets to capture orbiting planets; most ships at turn 500 wins (full rules in
[Game Specification](#game-specification) below).

The agent is an entity-transformer policy trained with PPO self-play on a vectorised GPU env, warm-started
from behaviour cloning of top-leaderboard replays. The hard part has never been *capturing* planets — it's
**holding** them against forward-projecting planner opponents. Most of the work below is the search for a
training signal that produces retention instead of spray-and-churn.

## Phases

Each phase is one structural attack on that holding problem. Reward/mask deltas live in `archive/docs/training-till-submission.md`;
these docs cover the architecture- and dynamics-level changes.

| Phase | Idea | Design doc |
|-------|------|------------|
| **Phase 2** | Teach **reinforcement** (sending ships to your *own* planets) as a native, empire-size-gated behaviour — the #1 skill gap vs the top tier. Fresh run, redesigned reward + one new action mask. | [`docs/phase2.md`](docs/phase2.md) |
| **Phase 3** | Fix the **self-play drift / cycling** that no reward knob touched, via a ratcheted teacher-KL anchor + opponent league (Toad Brigade / Isaiah Pressman recipe). Changes the *training dynamics*, not the reward. | [`docs/phase3.md`](docs/phase3.md) |
| **Phase 4** | **Per-target conditioning** for the fire/ship heads (architecture change). Makes all three action heads read the same `[q_slot, k_target, pairwise]` inputs so ship can size to `garrison+1` and fire can see `enemy_contest`. | [`docs/phase4.md`](docs/phase4.md) |

The standing diagnosis that ties them together — **what the wall actually is** — lives in
[`docs/current_problem.md`](docs/current_problem.md): the long-assumed "out-massing wall" turned out to be a
measurement ghost; the real axis is **peel-rate / retention** (holding a higher fraction of *your* captures
than the opponent holds of theirs), and the open lever is **capital efficiency**.

## Results

Final pre-submission eval of the best exported agents (2026-06-24, 256-game panels on the GCP eval box,
0 draws across all panels). Full tables + raw logs in [`results/eval_results.md`](results/eval_results.md).

**2p head-to-head win rate** (256 games = 128 seeds × 2 seats):

| Agent | vs Ajay (~1200 LB) | vs Debatreya (~1300 LB) |
|---|---|---|
| `stgpr1` 0.5M (spray) | **57.4%** | **59.4%** |
| `presres1` 1.5M (~1000 LB ref) | 54.3% | 53.1% |
| `presres1` 0.5M (decisive) | 53.9% | 51.6% |

> `stgpr1`'s higher WR is consistent with its spray/churn style, which inflates head-to-head WR without
> reflecting cleaner positional play — a caveat for interpretation, not a cert problem.

**4p FFA mixed-field** (seat-rotated, win-rate = 1st-place share = the LB metric):

| Agent | Win-rate | Mean place |
|---|---|---|
| `ajay` | **39.1%** | **1.64** |
| `producer_v2` (prior 4p incumbent) | 28.1% | 1.73 |
| `stgpr1` (our 2p neural agent) | 18.8% | 1.81 |
| `deb` | 14.1% | 1.88 |

Verdict held against both producer variants → **4p slot = `ajay`** (a heuristic), since 2p strength does
not predict 4p and our 2p-trained neural agent only places 3rd–4th there.

**Cross-eval vs the held-out opponent set** — both submitted agents against public heuristics *and*
our own past-best checkpoints (full 256-game both-seats panels, same as the certs):

| Opponent | `presres1` (decisive) | `stgpr1` (spray) |
|---|---|---|
| zach | 99.6% | 99.6% |
| hellburner | 97.3% | 98.0% |
| h1043 (lb1043) | 98.4% | 98.4% |
| h1166 (lb1166 peak) | 89.5% | 93.8% |
| pool_lb1084 / lb1138 / lb1152 | 95.7 / 85.5 / 88.3% | 97.7 / 90.6 / 91.0% |
| past selves rev38 / rev53b / rev31 / rev32b | 94 / 91 / 93 / 93% | 97 / 91 / 95 / 97% |

Both sweep the held-out set — every public heuristic (85–99%) and every prior self (91–97%) — clear
absolute progress with no forgetting, seats near-symmetric (|asym| ≤ 5.5pp). `stgpr1` edges ahead on
most opponents, but that's the same spray/churn WR-inflation, not cleaner play. Full per-seat table +
the historical `corrpack3e` panel + the N=6 diversity matrix: [`results/eval_results.md`](results/eval_results.md).

## Current state

Snapshot in [`docs/current-state.md`](docs/current-state.md); standing diagnosis in
[`docs/current_problem.md`](docs/current_problem.md).

- **Two agents submitted** (2026-06-24, both daily slots): `presres1` 0.5M "decisive" and `stgpr1` 0.5M
  "spray", each routing 2p → neural agent, 4p → `ajay`. See [`docs/submissions.md`](docs/submissions.md).
- **Two bug fixes banked in code but not in a competition agent**: phantom-neutral-production and
  path-obstruction. Every retrain (`presfix1`/`pathobs1`/`stgpr2`) drifted back into the spray Nash within
  ~2M steps, and applying the fixes at inference-only on a frozen agent measured net-negative (~−1.5 to
  −2pp) — both genuinely need a retrain to cash in.
- **Key open finding**: phantom-fix / `--min-ship-bin` / path-obstruction-veto are all the *wrong lever
  class* — they clean the opening, then self-play retrains the agent back into flooding + non-holding. The
  next move is **structural** (an opponent/curriculum that punishes spray and forces holding), steered by
  `launch_rate → ~0.04` / `peel↓` / `hold↑` rather than by spray-inflated Ajay WR.

## Post-submission progress (2026-07)

The competition ended; the project continued as a study of the top-100 writeups
([`docs/writeup_lessons.md`](docs/writeup_lessons.md)), applying one lesson at a time. The first
structural lesson — **projected-future timeline features** (per-planet ownership/garrison rolled
24 steps forward, the most universal ingredient across winner writeups) — plus a 10–20× step
budget answered the open finding above:

- **`tl100m`** (2026-07-12): 100M steps from scratch, *pure self-play, sparse ±1 reward, no
  shaping* — **74.6% vs Ajay** (best panel 77.7%), past the shaped lineage's 57.4%, and clean:
  launch discipline (`launch_rate` 0.09) was *learned*, not masked in. The spray Nash that every
  competition-era retrain fell back into simply doesn't form when bad launches are visible to the
  critic the step they happen. Run record: [`docs/training.md`](docs/training.md).

The previously reported **256/256 against each submitted endpoint was invalid**. The local wrapper
used `__file__`, which `kaggle_environments` does not define when it executes a path agent; the old
evaluator then counted the errored opponent as a win. The identical aggregate statistics against two
different neural payloads were the tell. The evaluator now fails closed on non-`DONE` agent status,
and submitted-opponent checks use the archived standalone `neural_agent.py` payloads whose hashes
match the final tarballs.

A corrected full-panel audit of the best current checkpoint (exact-marginal binary 40.108M, 80.5%
vs Ajay) is in progress. The first independent canonical samples already disprove the perfect-sweep
claim: 12/16 against `presres1` and 14/16 against `stgpr1`. Final 256-game rates will replace this
interim note when both panels complete. Experiment history and the audit trail live in
[`docs/training.md`](docs/training.md).

## Repo navigation

| Where | What |
|-------|------|
| `orbit_wars_rl/` | All active RL code (training, env, model, eval, export) |
| `opponents/` | Eval + training opponents (Ajay, Zach, Debatreya, producers; `orbit_lite/` dep) |
| `results/` | Final eval panels + raw cert/FFA logs |
| `seed_checkpoints/` | Resume points uploaded to training instances |
| `docs/commands.md` | Copy-paste command reference (start here for ops) |
| `archive/docs/training-till-submission.md` | Full run history + key config (through first submission) |
| `docs/submissions.md` | Submission log with Kaggle IDs and checkpoint paths |
| `docs/GCP_RUNBOOK.md` · `docs/JARVIS_RUNBOOK.md` | GPU instance launch / monitor / teardown |
| `CLAUDE.md` | Agent operating rules (hard constraints for Claude Code) |
| `gpu_run_artifacts/` | Training scripts, watchers, synced checkpoints (gitignored) |

---

# Game Specification

Conquer planets rotating around a sun in continuous 2D space. A real-time strategy game for 2 or 4 players.

## Overview

Players start with a single home planet and compete to control the map by sending fleets to capture neutral and enemy planets. The board is a 100x100 continuous space with a sun at the center. Planets orbit the sun, comets fly through on elliptical trajectories, and fleets travel in straight lines. The game lasts 500 turns. The player with the most total ships (on planets + in fleets) at the end wins.

## Board Layout

- **Board**: 100x100 continuous space, origin at top-left.
- **Sun**: Centered at (50, 50) with radius 10. Fleets that cross the sun are destroyed.
- **Symmetry**: All planets and comets are placed with 4-fold mirror symmetry around the center: (x, y), (100-x, y), (x, 100-y), (100-x, 100-y). This ensures fairness regardless of starting position.

## Planets

Each planet is represented as `[id, owner, x, y, radius, ships, production]`.

- **owner**: Player ID (0-3), or -1 for neutral.
- **radius**: Determined by production: `1 + ln(production)`. Higher production planets are physically larger.
- **production**: Integer from 1 to 5. Each turn, an owned planet generates this many ships.
- **ships**: Current garrison. Starts between 5 and 99 (skewed toward lower values).

### Planet Types

- **Orbiting planets**: Planets whose `orbital_radius + planet_radius < 50` rotate around the sun at a constant angular velocity (0.025-0.05 radians/turn, randomized per game). Use `initial_planets` and `angular_velocity` from the observation to predict their positions.
- **Static planets**: Planets further from the center do not rotate.

The map contains 20-40 planets (5-10 symmetric groups of 4). At least 3 groups are guaranteed to be static, and at least one group is guaranteed to be orbiting.

### Home Planets

One symmetric group is randomly chosen as the starting planets. In a 2-player game, players start on diagonally opposite planets (Q1 and Q4). In a 4-player game, each player gets one planet from the group. Home planets start with 10 ships.

## Fleets

Each fleet is represented as `[id, owner, x, y, angle, from_planet_id, ships]`.

- **angle**: Direction of travel in radians.
- **ships**: Number of ships in the fleet (does not change during travel).

### Fleet Speed

Fleet speed scales with size on a logarithmic curve:

```
speed = 1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ^ 1.5
```

- 1 ship moves at 1.0 units/turn.
- Larger fleets move faster, approaching the maximum speed (default 6.0).
- A fleet of ~500 ships moves at ~5, and ~1000 ships reaches the max.

### Fleet Movement

Fleets travel in a straight line at their computed speed each turn. A fleet is removed if it:

- Goes out of bounds (leaves the 100x100 playing field).
- Crosses the sun (path segment comes within the sun's radius).
- Collides with any planet (path segment comes within the planet's radius). This triggers combat.

Collision detection is continuous -- the entire path segment from old to new position is checked, not just the endpoint.

### Fleet Launch

Each turn, your agent returns a list of moves: `[from_planet_id, direction_angle, num_ships]`.

- You can only launch from planets you own.
- You cannot launch more ships than the planet currently has.
- The fleet spawns just outside the planet's radius in the given direction.
- You can issue multiple launches from the same or different planets in a single turn.

## Comets

Comets are temporary extra-solar objects that fly through the board on highly elliptical orbits around the sun. They spawn in groups of 4 (one per quadrant) at steps 50, 150, 250, 350, and 450.

- **Radius**: 1.0 (fixed).
- **Production**: 1 ship/turn when owned.
- **Starting ships**: Random, skewed low (minimum of 4 rolls from 1-99). All 4 comets in a group share the same starting ship count.
- **Speed**: Configurable via `cometSpeed` (default 4.0 units/turn).
- **Identification**: Check `comet_planet_ids` in the observation to see which planet IDs are comets. Comets also appear in the `planets` array and follow all normal planet rules (capture, production, fleet launch, combat).

When a comet leaves the board, it is removed along with any ships garrisoned on it. Comets are removed before fleet launches each turn, so you cannot launch from a departing comet.

The `comets` observation field contains comet group data including `paths` (the full trajectory for each comet) and `path_index` (current position along the path), which can be used to predict future comet positions.

## Turn Order

Each turn executes in this order:

1. **Comet expiration**: Remove comets that have left the board.
2. **Comet spawning**: Spawn new comet groups at designated steps.
3. **Fleet launch**: Process all player actions, creating new fleets.
4. **Production**: All owned planets (including comets) generate ships.
5. **Fleet movement**: Move all fleets along their headings. Check for out-of-bounds, sun collision, and planet collision. Fleets that hit planets are queued for combat.
6. **Planet rotation & comet movement**: Orbiting planets rotate, comets advance along their paths. Any fleet caught by a moving planet/comet is swept into combat with it.
7. **Combat resolution**: Resolve all queued planet combats.

## Combat

When one or more fleets collide with a planet (either by flying into it or being swept by a moving planet), combat is resolved:

1. All arriving fleets are grouped by owner. Ships from the same owner are summed.
2. The largest attacking force fights the second largest. The difference in ships survives.
3. If there is a surviving attacker:
   - If the attacker is the same owner as the planet, the surviving ships are added to the garrison.
   - If the attacker is a different owner, the surviving ships fight the garrison. If the attackers exceed the garrison, the planet changes ownership and the garrison becomes the surplus.
4. If two attackers tie, all attacking ships are destroyed (no survivors).

## Scoring and Termination

The game ends when:

- **Step limit reached**: 500 turns.
- **Elimination**: Only one player (or zero) remains with any planets or fleets.

Final score = total ships on owned planets + total ships in owned fleets. Highest score wins.

## Observation Reference

| Field | Type | Description |
|-------|------|-------------|
| `planets` | `[[id, owner, x, y, radius, ships, production], ...]` | All planets including comets |
| `fleets` | `[[id, owner, x, y, angle, from_planet_id, ships], ...]` | All active fleets |
| `player` | `int` | Your player ID (0-3) |
| `angular_velocity` | `float` | Planet rotation speed (radians/turn) |
| `initial_planets` | `[[id, owner, x, y, radius, ships, production], ...]` | Planet positions at game start |
| `comets` | `[{planet_ids, paths, path_index}, ...]` | Active comet group data |
| `comet_planet_ids` | `[int, ...]` | Planet IDs that are comets |
| `remainingOverageTime` | `float` | Remaining overage time budget (seconds) |

## Action Format

Return a list of moves:

```python
[[from_planet_id, direction_angle, num_ships], ...]
```

- `from_planet_id`: ID of a planet you own.
- `direction_angle`: Angle in radians (0 = right, pi/2 = down).
- `num_ships`: Integer number of ships to send.

Return an empty list `[]` to take no action.

## Agent Convenience

The module exports named tuples for easier field access:

```python
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT

def agent(obs):
    planets = [Planet(*p) for p in obs.get("planets", [])]
    fleets = [Fleet(*f) for f in obs.get("fleets", [])]
    player = obs.get("player", 0)

    for p in planets:
        print(p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)

    return []  # list of [from_planet_id, angle, num_ships]
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `episodeSteps` | 500 | Maximum number of turns |
| `actTimeout` | 1 | Seconds per turn |
| `shipSpeed` | 6.0 | Maximum fleet speed |
| `sunRadius` | 10.0 | Radius of the sun |
| `boardSize` | 100.0 | Board dimensions |
| `cometSpeed` | 4.0 | Comet speed (units/turn) |
