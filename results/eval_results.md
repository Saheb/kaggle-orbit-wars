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
