# Current State — 2026-06-23

Snapshot of live work. Companion to [`current_problem.md`](current_problem.md) (the standing
diagnosis) and [`eval_box.md`](eval_box.md) (the GCP eval box). Newest at top.

---

## ⭐ ACTIVE: roi-deflation + pressure-resolver A/B (two parallel Jarvis runs)

**Lineage:** root cause in replay `submission_analysis/81454632.json` (we LOST as player 1) —
the **target head chases cheap-but-contested neutrals** into second-arrival traps (our 32 ships
fired at a far 8-ship neutral the enemy captured first, crashed into 41 defenders, took nothing;
a near 31-ship clean neutral our 32 would have taken sat untouched). Diagnosis: the pairwise
`roi_20/roi_50` channels (ch12/13) are deflated by *friendly* inbound only, **never enemy
contest** — so a cheap neutral an enemy fleet captures first reads as identically attractive
(+1.00) to a clean win. The contest-aware channels (ch17 reactive_roi, ch19 keepability, ch14/20/21)
all flag it correctly; the head leans on the raw-cheapness roi instead. Two fixes built behind
flags (default off, persisted + eval/export-synced like game-phase):
- `--roi-enemy-deflate` — deflate roi_20/50 by enemy contest (mirror of the friendly term).
- `--zero-roi-channels` — zero ch12/13 entirely (test redundancy given reactive_roi_40 ch17).

**First A/B (`roidefl1` deflate / `roizero1` zero) — PREEMPTED at 1.57M** (spot). roidefl1 tracked
Ajay 41.8/49.2/46.1% over 0.5/1.0/1.57M; roizero1 ~41%. ⚠️ **those roidefl1 numbers are suspect** —
the checkpoint config didn't persist `roi_enemy_deflate` (launched just before the persist line
landed), so eval ran with deflation OFF on a deflation-ON policy (train/eval mismatch).
**Backfilled `roi_enemy_deflate=True` into the 3 ckpts + re-evaling correctly locally** (latest→back):
`gpu_run_artifacts/roidefl1/local_eval_correct_summary.txt`. roizero1 dropped.

**Current A/B — both resume `roidefl1@1.57M`, differ by ONE flag:**

| run | instance | IP | delta | destroy |
|---|---|---|---|---|
| `roidefl2` | 432364 | 217.18.55.81 | `--roi-enemy-deflate` (resolver OFF) | `jl destroy 432364 --yes` |
| `roideflpr1` | 432365 | 217.18.55.120 | deflate **+ `--pressure-precise-resolver`** | `jl destroy 432365 --yes` |

Both: --roi-enemy-deflate, externals producer_v2+deb, LR 1e-4, gate2, sufficient-commit 1.0,
game-phase, 10M, A100-80GB spot. **Read = roideflpr1 − roidefl2** = the resolver fix's marginal
effect on the deflate arm. Tripwire: Ajay must not drop. Scripts in
`gpu_run_artifacts/{roidefl2,roideflpr1}/`.

---

## Feature-accuracy audit (2026-06-23) — train/eval consistency

Swept the feature surface (features.py numpy ↔ torch_env.py torch). **All 22 pairwise channels
verified byte-identical across both paths** (no numpy↔torch divergence); the bug class is
"feature models the world wrong, but *consistently*" (so the model trained on it and the parity
test stayed green).

| # | finding | status |
|---|---|---|
| 1 | **Pressure channels (ch14 enemy_contest / ch19 keepability / ch20 enemy_mass_soon / ch21 threat_imminence) used the deprecated ~85% loose corridor** (perp<r+1.5, launch heading, no orbital lead, double-counts a fleet onto every planet in its path) vs the 98.4% `_fleet_target_idx` resolver the veto/reward already use | **FIXED** behind `--pressure-precise-resolver` (live in roideflpr1). Removes ~42% phantom over-count in enemy_contest; parity 0.0000 (numpy `_resolve_fleet_targets` == torch). |
| 1b | **Planet-level pressure features ch12/13** (`friendly_pressure` / `enemy_pressure`) STILL use the corridor — left out of #1 to keep the blast radius tight (they're train/eval-consistent, both paths corridor, so no parity break). For full consistency, fold them under the SAME `--pressure-precise-resolver` flag. | **FIXED** — ch12/13 now route through `incoming_pw` (torch) / `_resolve_fleet_targets` (numpy) under the existing flag; no new flag/wiring/dim change. Measured: corridor→resolver drops ch12/13 abs-sum ~42%/25% (same over-count as #1); planet parity 0.0000 (now hard-asserted in `run_pressure_resolver_parity`). |
| 2 | **Phantom neutral production** — roi/cap-cost adds `prod·eta` to neutrals, which don't regrow (confirmed: 4 neutrals flat 31 ships for 18 steps in the replay) | **queued** (feature change → retrain; interacts with #1 + roi; raises far-neutral ROI) |
| 3 | **Threat-ETA uses planet center not surface** (omits `−pr` that the resolver uses) → threat reads ~½ step under-urgent | **queued** (minor; consistent train/eval) |
| 4 | **`SHIP_COUNTS`/`NUM_SHIP_BINS`/`FRACTION_BIN_VALUES` hand-triplicated** across action_mask/model/torch_env (identical now, silent landmine) | **FIXED** — single source in action_mask, derived elsewhere |
| 5 | **Pairwise parity test only asserted *planet* feats** — the 22 pairwise channels (where #1/#2 live) were untested | **FIXED** — all 22 channels + roi flags + resolver now hard-asserted in `tests/test_torch_env_features.py` (pytest-collected) |

#1/#4/#5 committed (`72b20ba`, `80244cb`). #2/#3 to stack on whichever roi arm wins as a bundled
"feature-accuracy pass". All flags are checkpoint-compatible (no dim change), default-off,
persisted + eval/export-synced.

**#1b — DONE (gated under the existing `pressure_precise_resolver` flag, no new flag):**
- **torch** `torch_env.py` get_features: moved the `incoming_pw` block above `friend`/`enemy` and
  derived `friendly_pressure`/`enemy_pressure` (planet ch12/13) from it; `friendly_pressure_pw`
  now just aliases `friendly_pressure` (same attribution). Corridor preserved when flag off.
- **numpy** `features.py`: precompute `_resolve_fleet_targets(...)` ONCE before the planet loop →
  per-fleet target array-index; in the loop `incoming = (pp_resolved_tgt == i)` when the flag is
  on, else the corridor. (Planet array index `i` == resolver target index since both index
  `planets[:max_planets]`.)
- **parity** `tests/test_torch_env_features.py`: `run_pressure_resolver_parity` now also diffs the
  full **planet** feature vector under the flag (`max_planet_diff`, hard-asserted) — ch12/13 locked
  alongside the pairwise channels. `3 passed`; planet diff 0.0000.
- **wiring**: none — reuses the persisted `pressure_precise_resolver` config + eval/export sync.
- Verified: corridor→resolver drops ch12/13 abs-sum ~42%/25% (the same phantom over-count as #1);
  checkpoint-compatible, default-off. Lands consistency: no more two-ways-to-measure-the-same-thing
  in one feature vector. (Not yet committed — local edits.)

### Prior thread (concluded): redundant-target veto A/B

`phase4fs2` (432127, no veto) / `phase4fs_rv` (432208, `--redundant-target-factor 1.0`) — both
dead (preempted). Decode-only test was −3.5pp vs Ajay but +3pp h2h (train/eval mismatch → folded
into training). `--redundant-target-factor` is in the codebase (config/ppo/torch_env/train/eval +
`tests/test_redundant_target.py`). Superseded by the roi-deflation thread above.

---

## Eval infrastructure

- **GCP box** `orbit-wars-eval` (n2-standard-32, asia-south1-a) — sharded Ajay panels ~4 min.
  Bills ~$1.55/hr: `gcloud compute instances delete orbit-wars-eval --zone=asia-south1-a` when done.
- **`auto_eval_loop` daemon** (local, pid varies) now tracks `roidefl2:roidefl2` +
  `roideflpr1:roideflpr1`, evals each new checkpoint on the box, appends
  `gpu_run_artifacts/<run>/eval_ajay_1200.csv`. **Box code re-synced 2026-06-23 for BOTH the
  roi-flag and the pressure-resolver eval auto-load** (`set_roi_enemy_deflate` /
  `set_pressure_precise_resolver` / `_resolve_fleet_targets`) — older box snapshots would eval
  roideflpr1 without the resolver transform. Start/stop: [`eval_box.md`](eval_box.md) §7.
  **Local `_sync` watchers** kept (land checkpoints); local `_eval` watchers **killed** (slow).
- **roidefl1 correct re-eval (one-off, local):** `/tmp/roidefl1_local_eval.sh` runs full panels
  latest→back on the backfilled ckpts → `gpu_run_artifacts/roidefl1/local_eval_correct_summary.txt`.
- **Gotcha:** box needs `gpu_run_artifacts/<run>/checkpoints/` pre-created or the daemon's rsync
  fails (mkdir won't make nested parents). **Persist gotcha:** a run launched before its flag's
  persist line lands won't round-trip the flag → eval runs the wrong feature regime (this bit
  roidefl1; backfill the ckpt config or pass the flag explicitly).

---

## Session findings (2026-06-23) feeding the above

1. **`cu` overfired = 1-ship spray (confirmed).** phase4fs_cu 51.6% Ajay but `ship0=49%`,
   launch_rate 0.36 (vs base 0.095), worse concentration+retention → spray artifact, won't
   transfer. **Keep base 4fs 7.34M** as the clean 2p agent. stg = same pathology, milder (39%).
2. **2p LB loss analysis** (44 real LB games, `/tmp/sub_replays/53958174`): the loss is
   **mid-game (step 40–75), NOT the opening** (launch rate 0–25 identical won/lost). At step 40
   games are tied; the winner pulls away 50→75. Mechanism = **retention** (mid-game hold% 47%
   lost vs 58% won) at *equal launch effort*. Steer by mid-game hold% / planets@50, not opening
   aggression or firing volume. Confirms `current_problem.md` v3 on real LB games.
3. **Capture floor** (`eval._attack_capture_floor` = `torch_env._decisive_mass_fields`):
   `garrison + production·eta + enemy_inbound + β·ρ(eta)·reachable_enemy_mass + overhead`
   (β=2.2, ρ=clamp((eta−3)/12,0,1), horizon 18, overhead=1 = the strict-capture +1 margin).
   The **`sufficient_commit` decode/train mask** uses a simpler static floor (garrison +
   enemy_inbound). Only `sufficient_commit` (and now `redundant_target`) are in **training**;
   the reactive decisive-mass floor is computed for **diagnostics only** (decmass reward off).

---

## Submission state (Kaggle)

`submission_2p4p/` dispatches by player count: **2p → neural** (phase4fs 7.34M), **4p →
`producer_v4.py`** (Producer Hybrid v4, FFA leader-guard 0.035 + target-prod 0.08). Last two
subs: 53958174 (2p 61% / 4p 29%), 53958510 (2p 57% / 4p 31%). 2p above par (50), 4p barely
above par (25). The 4p RL self-play (`ffa4p`, 15M, done) plateaued ~28% and **collapsed 2p
Ajay to ~2%** — abandoned; producer dominates 4p (won 100% vs the RL agents in a panel).

---

## Cleanup checklist (when done)

```bash
jl destroy 432364 --yes     # roidefl2 (deflate)
jl destroy 432365 --yes     # roideflpr1 (deflate + pressure-resolver)
pkill -f auto_eval_loop.sh                                  # stop daemon
pkill -f roidefl1_local_eval.sh                             # stop local roidefl1 re-eval loop
bash gpu_run_artifacts/run_watchers.sh stop                 # stop _sync watchers
gcloud compute instances delete orbit-wars-eval --zone=asia-south1-a   # eval box
```
