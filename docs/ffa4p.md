# 4p FFA training — plan & status

_Started 2026-06-23. Supersedes the stale `codex/4p-ffa-handoff` branch (forked May 31,
pre-Phase-1 arch, results negative — see "What we took / left" below)._

## Hypothesis

Warm-start the strongest 2p policy (**phase4fs 7.34M, 49.6% Ajay** — best opening tempo,
from First Strike) into **4p self-play with rank-based terminal rewards** and let it adapt
to FFA. The May replay diagnosis found FFA winners = **fast expansion + sustained launch
pressure**, survival-oriented loses — exactly what First Strike + expansion shaping already
bias toward, so the 2p tempo policy is the right seed.

## What made this small (modern arch is already P-generic)

- `torch_env._check_done`, `get_features` (player-relative: `is_enemy = owner!=player` —
  all 3 opponents collapse to "enemy", the correct FFA encoding), GAE (folds P into the env
  axis), and the PPO update are **already `num_players`-generic**. `eval.py` already has
  `--num-players 4`.
- The only code changes (this branch):
  1. `torch_env`: `rank_reward_coef` param + rank interpolation in `_check_done`
     (`rank 0→+1 … P-1→-1`, blended; inert for P≤2). Raw winner mask unchanged → PFSP safe.
  2. `train_torch`: `--num-players` / `--rank-reward-coef` flags; `P = args.num_players`;
     `dones` written to all seats; a guard that **4p + pool>0 is not yet wired**.

## What we took / left from `codex/4p-ffa-handoff`

- **Took (as design, re-implemented on current arch):** rank-based terminal reward; the
  lessons (fast-expansion>survival; external pool 10–20% not 50%; final-ckpt mixed-field
  eval as the milestone, not `torch_best`).
- **Left:** all checkpoints (0.25M steps, stale arch, won't load), the AWS infra, the
  material-delta shaping (didn't move the objective), and the 50%-external curriculum
  (explicitly the thing that failed). The trained agents there were *worse* than the
  pre-existing champion (mixed-field 12.5% vs 16.7%).

## Validation (CPU smoke, this session)

- Fresh 4p self-play + rank reward: runs, terminal rewards graded per seat (not flat ±1).
- Warm-start phase4fs_stg → 4p: loads (32-bin, game-phase, reinforce gate2), V_loss spikes
  then EV recovers to ~0.92 as the value head re-fits to rank rewards. No key mismatches.
- Full production flag set parses + runs end-to-end.
- Test suite: 184 pass, **0 new failures** (4 pre-existing failures unrelated to this change).

## Launch

`gpu_run_artifacts/ffa4p/run_remote.sh` (mirrors the proven phase4fs_stg launcher). Run it
on a provisioned GPU box via the existing `launch.sh`-style wrapper. Key flags:
`--num-players 4 --pool-fraction 0 --rank-reward-coef 0.5`, resume the 7.34M peak, `num-envs
2048` (4 seats = 2× forward/step + 2× buffer; raise if the box has headroom).

## FFA mixed-field baseline (2026-06-23, `scripts/run_ffa_panel.py`, 128 games)

Field = neural (phase4fs_stg 2p) + producer_v2 + roman_v4 + hellburner, 32 seeds × 4 seat
rotations. Win-rate = 1st-place share (the FFA LB metric):

| agent | WR | mean place |
|---|---|---|
| **producer_v2** | **75.0%** | 1.25 |
| roman_v4 | 25.0% | 1.75 |
| neural (2p) | 0.0% | 2.00 |
| hellburner | 0.0% | 2.00 |

- **producer_v2 beats roman_v4 3:1** → the `ffa_leader_attack_bonus` is net-negative in this
  field (corroborates the 2p H2H where producer_v2 won 62.5%). ⇒ `53958174` (producer_v2) is
  the better 4p engine; `53958510` (roman_v4) is likely a 4p regression. Caveat: the field
  CONTAINS producer_v2; the real LB field is other competitors, so this is a head-to-head
  result, not an LB prediction.
- **neural 0% = the FFA floor** the warm-started 4p run must beat (it survives — mean place
  ~2 by material — but never converts a win). Target = producer-competitive.
- Log: `gpu_run_artifacts/ffa4p/eval_logs/mixed_neural_prod_roman_hb.log`.

## Monitoring while it trains (all synced locally every 120s)

Watcher launched with `FFA_EVAL=1 ... run_watchers.sh start ffa4p jarvis <IP>` → three loops:
`_sync` (logs+checkpoints), `_eval` (2p Ajay = retention tripwire), `_ffaeval` (the real metric).

1. **FFA win-rate (the objective)** — `gpu_run_artifacts/ffa4p/eval_ffa.csv`, one row per checkpoint
   from `eval_ffa_checkpoint.py` (loads the ckpt as a **4p** agent — `num_players=4` so it sees
   `mode_4p=1`, matching training — and plays it seat-rotated vs producer_v2 + roman_v4 +
   hellburner). Columns: step, wr_pct, mean_place, mean_game_len. **Baseline floor = the 2p seed
   at ~25% (chance).** WANT: WR climbing above 25% toward producer's level.
2. **Passive-mirror collapse tripwire** — read the synced training log diag line
   (`gpu_run_artifacts/ffa4p/logs/train_gpu_phase1_ffa4p_*.log`): `fire_frac` / `reinf` / `owned`
   / `pl@16/32/50`. Collapse = these heading toward 0 / games all hitting the 500 cap. (At 0.5M:
   fire_frac 0.37, reinf 0.50, owned 3.5 — healthy.)
3. **2p retention** — `gpu_run_artifacts/ffa4p/eval_zach*.csv` / `eval_ajay*.csv`: does 4p training
   tank the policy's 2p strength.

Critic warmup (value-only, policy frozen) recovered EV 0.64→0.94 by iter 4 then unfroze — working.

**Live config (2026-06-23, 15M run):** LR **1e-4** (2× the quick-probe 5e-5), `--total-steps
15000000`, and a **homogeneous self-snapshot league** (`--pool-mode self --pool-fraction 0.5
--pool-checkpoint-interval 500000`): half the games put the learner vs 3 copies of a sampled
PAST self, half are full current-mirror. This is the anti-cycling fix for the passive-mirror
collapse a pure 4-way mirror is prone to. The seat machinery was generalized (homogeneous fill
of all P-1 non-learner seats + placement-based PFSP credit); external heuristics in 4p remain
guard-blocked (not validated). CPU-smoke-tested (preseed 12 snapshots, pool-fraction 0.8 forces
the override) + 184 unit tests pass.

Foundational fix made for any 4p neural eval: `build_agent_fn` now takes `num_players` (was
hardcoded 2) — `eval.py:279`. Without it a 4p-trained model is fed `mode_2p=1` (a train/eval
mismatch that was a latent bug in eval.py's existing `--num-players 4` path).

## Open items (each a SEPARATE delta)

1. **4p eval for progress tracking.** `eval.py --num-players 4` exists but the watcher isn't
   wired for it, and the **mixed-field eval** (our agent + 3 varied opponents, the real LB
   objective) from the stale branch is NOT ported. This is the next thing to build — also
   lets us finally measure producer_v2 vs roman-v4 vs neural in FFA (the submission choice
   is currently unmeasured).
2. **External 4p pool.** The per-episode seat assignment + PFSP attribution assume one
   opponent seat. Generalizing to 3 opponent seats (mix of self-snapshots + heuristic
   producers at 10–20%) is the follow-up once self-play 4p shows a healthy baseline.
3. **Dead-player masking.** Eliminated-player transitions are currently trained on; they're
   near-harmless (no owned planets → no valid actions → ~0 policy loss) but masking them
   tightens the signal. Low priority.
