# Final Evaluation Results

Final eval of the best exported agents ahead of Kaggle submission.
Raw log: [`cert_results.log`](cert_results.log). Generated 2026-06-24.

## Agents

| Label | Checkpoint | Role |
|---|---|---|
| `presres_05` | presres1 0.5M | decisive — primary candidate |
| `stgpr1_05` | stgpr1 0.5M | spray — hedge candidate |
| `presres_15` | presres1 1.5M | ~1000 LB reference (current submission's `neural_agent`) |

Exported via `export_agent.py` (`--target-decode`); byte-identical format to the
submission `neural_agent.py`.

## 2p panel (256 games = 128 seeds × 2 seats, win/loss/draw)

All panels run on the GCP eval box, sharded with `OMP_NUM_THREADS=1` (single-threaded
shards — oversubscription causes per-step timeouts → no-op defaults → invalid draws).
**0 draws across all panels** confirms clean timing.

### vs Ajay (`candidate_ajay_1200.py`, ~1200 LB — primary metric)

| Agent | W | L | D | Win rate |
|---|---|---|---|---|
| `stgpr1_05` | 147 | 109 | 0 | **57.4%** |
| `presres_15` | 139 | 117 | 0 | 54.3% |
| `presres_05` | 138 | 118 | 0 | 53.9% |

### vs Debatreya (`candidate_debatreya_1300.py`, ~1300 LB)

| Agent | W | L | D | Win rate |
|---|---|---|---|---|
| `stgpr1_05` | 152 | 104 | 0 | **59.4%** |
| `presres_15` | 136 | 120 | 0 | 53.1% |
| `presres_05` | 132 | 124 | 0 | 51.6% |

### Summary

| Agent | vs Ajay | vs Deb |
|---|---|---|
| `presres_05` (decisive) | 53.9% | 51.6% |
| `stgpr1_05` (spray) | 57.4% | 59.4% |
| `presres_15` (~1000 ref) | 54.3% | 53.1% |

> Note: `stgpr1_05`'s higher WR is consistent with its spray/churn play style, which
> inflates head-to-head WR without necessarily reflecting cleaner positional play — a
> caveat for interpretation, not a cert problem.

## 4p FFA mixed-field panel

Field: `producer_v2` + `ajay` + `deb` + `stgpr1_05`. Seat-rotated (each agent occupies
every seat equally per seed), 256 games (64 seeds × 4 rotations). Metric:
**win-rate = 1st-place share** (the FFA LB metric — rating is gained only by winning) and
**mean placement** (1=best, 4=worst).

| Agent | Win-rate (1st-place share) | Mean place |
|---|---|---|
| `ajay` | **39.1%** (100/256) | **1.64** |
| `producer_v2` (current 4p incumbent) | 28.1% (72/256) | 1.73 |
| `stgpr1_05` (our 2p agent) | 18.8% (48/256) | 1.81 |
| `deb` | 14.1% (36/256) | 1.88 |

**Verdict: `ajay` for the 4p slot** — highest 1st-place share (the LB metric) and best
mean placement, clearly ahead of the incumbent `producer_v2`.

> 2p strength does **not** predict 4p: `deb` is the strongest 2p bot (59.4% vs us) but
> the *worst* here, while `ajay` is the reverse. Our 2p-trained neural agent runs in 4p
> without erroring but places 3rd — confirming a heuristic is the right 4p choice.

### Robustness check — producer_v4 instead of producer_v2

Same field with `producer_v4` (the prior 4p incumbent, "Producer Hybrid v4") swapped in
for `producer_v2`, 256 games:

| Agent | Win-rate (1st-place share) | Mean place |
|---|---|---|
| `ajay` | **39.5%** (101/256) | 1.68 |
| `deb` | 24.2% (62/256) | 1.82 |
| `producer_v4` | 23.0% (59/256) | 1.77 |
| `stgpr1_05` | 13.3% (34/256) | 2.02 |

`ajay` wins decisively here too (~39%, stable across both panels), and **`producer_v4`
(23.0%) is no better than `producer_v2` was** — slightly worse. The 4p verdict (ajay)
holds against both producer variants. Raw: [`ffa_v4_results.log`](ffa_v4_results.log).

---

# Cross-eval panel — submitted agents vs the held-out opponent set

Broad cross-eval of **both submitted 2p agents** against the full held-out set: public heuristic
opponents **and** our own best past checkpoints (a forgetting/cycling detector — a drop vs a past
self we used to beat = regression self-play WR is blind to). Generated 2026-06-25.

**Full 256-game both-seats panels per opponent** (same methodology as the certs above), via
[`run_submitted_cross_eval.sh`](run_submitted_cross_eval.sh) +
[`run_stgpr1_remaining_parallel.sh`](run_stgpr1_remaining_parallel.sh). Raw:
[`submitted_cross_eval.log`](submitted_cross_eval.log) (presres1 + stgpr1 zach/hellburner) and
[`submitted_cross_eval_stgpr1_parallel.log`](submitted_cross_eval_stgpr1_parallel.log) (stgpr1 rest).

> **Eval lineage note:** these checkpoints were trained **before** the phantom-neutral-production
> feature fix, so the panels were run from the `orbit-prephantom` worktree (commit `8f78555`) — the
> code the agents were trained on. Running them on current `main` (phantom + path-obstruction fixes
> applied) would make their trained features OOD; see `docs/current-state.md` for the same reasoning
> behind the inference-only A/B. Checkpoints live in the `orbit-audit` worktree; opponents + venv
> from main. 0 errored panels.

### Win-rate (overall, both seats)

| Opponent | `presres1` 0.5M (decisive) | `stgpr1` 0.5M (spray) |
|---|---|---|
| **Public heuristics** | | |
| zach | 99.6% | 99.6% |
| hellburner | 97.3% | 98.0% |
| h1043 (lb1043 simple) | 98.4% | 98.4% |
| h1166 (lb1166 peak heuristic) | 89.5% | 93.8% |
| pool_lb1084 | 95.7% | 97.7% |
| pool_lb1138 | 85.5% | 90.6% |
| pool_lb1152 | 88.3% | 91.0% |
| **Our own past-best checkpoints** | | |
| self_rev38 (5M) | 94.1% | 96.5% |
| self_rev53b (10M) | 91.0% | 91.0% |
| self_rev31 | 93.0% | 94.5% |
| self_rev32b | 93.0% | 96.9% |

Both submitted agents sweep the held-out set: every public heuristic (85–99%) and every prior self
(91–97%) — no forgetting, clear absolute progress over the rev31/32b/38/53b lineage. Seats are
near-symmetric (|asymmetry| ≤ 5.5pp on every panel; per-seat splits in the raw logs). The `pool_lb*`
rows are the same heuristic hammers used in the training pool, eval'd here in the real Kaggle env (a
train/eval sim-fidelity check); `pool_lb1138`/`pool_lb1152` are the hardest of the heuristic set.

`stgpr1` edges `presres1` on most opponents (e.g. h1166 93.8 vs 89.5, lb1138 90.6 vs 85.5) — the
same spray/churn WR-inflation flagged for the head-to-head panels, not a cleaner-play signal. Note
these heavy-heuristic panels run far slower for `stgpr1` than `presres1` precisely *because* spray
drags games to the 500-step limit instead of ending them decisively.

### Historical comparison — `corrpack3e` 4.7M (earlier best, smaller panel)

For reference, our best-ever Ajay+Zach checkpoint `corrpack3e` 4.7M on the same opponent set, but at
the older **32 games/opp, seat-0** setting (2026-06-16), via
[`gpu_run_artifacts/cross_eval/run_cross_eval.sh`](../gpu_run_artifacts/cross_eval/run_cross_eval.sh).
Raw: `gpu_run_artifacts/cross_eval/xeval_corrpack3e_4718592.out`. Lower-N/single-seat, so read as a
coarse panel, not a cert:

| Public heuristics | WR | Our past selves | WR |
|---|---|---|---|
| hellburner | 93.8% | self_rev38 (5M) | 81.2% |
| zach | 90.6% | self_rev31 | 78.1% |
| h1043 | 90.6% | self_rev32b | 78.1% |
| h1166 | 75.0% | self_rev53b (10M) | 71.9% |
| pool_lb1084 / lb1138 / lb1152 | 78.1 / 75.0 / 71.9% | | |

The submitted agents score higher across the board than `corrpack3e` here, but the methodologies
differ (256-game both-seats vs 32-game seat-0), so treat it as directional, not a clean delta.

There is also a small **N=6 round-robin diversity matrix** (hellburner / zach / h1166 / h1043 /
rev53b / rev38, used to pick pool-worthy = behaviourally-distinct opponents) in
[`gpu_run_artifacts/cross_eval/study_result.log`](../gpu_run_artifacts/cross_eval/study_result.log).
