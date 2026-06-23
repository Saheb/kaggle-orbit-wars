# Current State — 2026-06-23

Snapshot of live work. Companion to [`current_problem.md`](current_problem.md) (the standing
diagnosis) and [`eval_box.md`](eval_box.md) (the GCP eval box). Newest at top.

---

## ⭐ ACTIVE: redundant-target veto A/B (two parallel Jarvis runs)

**Hypothesis:** the 2p agent wastes fleets *redundantly reinforcing already-won captures*
(a planet whose in-flight friendly mass already clears it) instead of taking a new neutral —
the "capital efficiency" leak. A **redundant-target mask** vetoes such launches so the policy
retargets the spare.

**Two runs, identical config except the one flag** (clean A/B):

| run | instance | IP | delta vs the other | destroy |
|---|---|---|---|---|
| `phase4fs2` | 432127 | 217.18.55.81 | **no veto** (baseline) | `jl destroy 432127 --yes` |
| `phase4fs_rv` | 432208 | 217.18.55.102 | **`--redundant-target-factor 1.0`** | `jl destroy 432208 --yes` |

Both: resume base 4fs **7.34M** (Ajay peak 49.6%), externals **producer_v2 + deb** (fresh pool),
**LR 1e-4 (2×)**, 10M steps, A100-80GB spot. Launchers + run scripts in
`gpu_run_artifacts/{phase4fs2,phase4fs_rv}/`.

### Ajay WR so far (256-game box panels, matched steps)

| step | `phase4fs2` (no veto) | `phase4fs_rv` (veto) |
|---|---|---|
| 0.5M | 51.6 | — |
| 1.05M | — | 44.5 |
| **1.57M** | **41.4** | **45.3** ← veto +3.9pp |
| 2.1M | 43.4 | (running) |
| 4.2M | 39.8 | — |
| 4.7M | 45.3 | — |
| 5.2M | 43.0 | — |

**Early read (noisy, ~1 SE):** `phase4fs_rv` tracks *at or above* `phase4fs2` at matched steps —
directionally consistent with the head-to-head edge. **Caveat:** *both* runs dipped below the
49.6% baseline into the low-40s — the 2× LR + producer_v2 curriculum causes an early dip
(adaptation or **win-starvation** — producer_v2 may be too hard, [[feedback_win_starvation]]).
The A/B signal is the *relative* fs_rv−fs2 gap; if both keep sliding, the curriculum/LR is the
culprit, not the veto. Training health: clip ~0.37 (high but EV ~0.88 / KL ~0.02 / estop 0 =
benign adaptation under 2× LR).

### The veto, decode-only (what motivated the training fold)

Tested as an **inference-time** target-logit mask first (redirects, doesn't drop). Results
(256-game panels in `gpu_run_artifacts/redundant_veto_ab/`):

| variant | vs Ajay | h2h vs no-veto |
|---|---|---|
| baseline | **49.6%** | — |
| static-floor veto | 46.1% | **53.5%** |
| reactive-floor veto | 45.3% | 53.1% |

Decode-only: **−3.5pp vs Ajay but +3pp head-to-head** (both floors). The Ajay loss is the
**train/eval mismatch** (policy never trained under the mask → its committed fleets get
overridden). Reactive floor (adds `β·ρ·reachable_enemy_mass`) masks a strict *subset* → less
effect, no better → **static floor chosen** for the training fold. Hence: fold into **training**
so the policy adapts (this A/B).

### Implementation (folded into training behind a flag)

`--redundant-target-factor` (default 0 = off, everything else unchanged). 5-file change:
`config.py` (ModelConfig field) · `ppo.py` (persist to ckpt config) · `torch_env.py`
(`_apply_actions` `can_fire` veto, mirrors `sufficient_commit`; static floor =
`friendly_inbound > (garrison + enemy_inbound)·factor`) · `train_torch.py` (CLI + auto-load +
env) · `eval.py` (auto-load from ckpt for parity). Verified: 22 existing mask tests pass +
new `tests/test_redundant_target.py` (3 pass). Decode path also still env-gated via
`OW_REDUNDANT_TGT_FACTOR` / `OW_REDUNDANT_REACTIVE` (the A/B harness).

---

## Eval infrastructure

- **GCP box** `orbit-wars-eval` (n2-standard-32, asia-south1-a) — sharded Ajay panels ~4 min.
  Bills ~$1.55/hr: `gcloud compute instances delete orbit-wars-eval --zone=asia-south1-a` when done.
- **`auto_eval_loop` daemon** (local, pid varies) tracks `phase4fs2:fs2` + `phase4fs_rv:rv` +
  `roidefl1:roidefl1` + `roizero1:roizero1` (the two roi-channel arms, 2026-06-23),
  evals each new checkpoint on the box, appends `gpu_run_artifacts/<run>/eval_ajay_1200.csv`.
  NB: box code re-synced 2026-06-23 for the roi-flag eval auto-load (`set_roi_enemy_deflate`/
  `set_zero_roi_channels`) — older box snapshots would eval the roi runs without the transform.
  Start/stop: see [`eval_box.md`](eval_box.md) §7. **Local `_sync` watchers** kept (land
  checkpoints); local `_eval` watchers **killed** (the slow 68-min local panels).
- **Gotcha fixed:** box needs `gpu_run_artifacts/<run>/checkpoints/` pre-created or the daemon's
  rsync fails (mkdir won't make nested parents).

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
jl destroy 432127 --yes     # phase4fs2
jl destroy 432208 --yes     # phase4fs_rv
pkill -f auto_eval_loop.sh                                  # stop daemon
bash gpu_run_artifacts/run_watchers.sh stop                 # stop _sync watchers
gcloud compute instances delete orbit-wars-eval --zone=asia-south1-a   # eval box
```
