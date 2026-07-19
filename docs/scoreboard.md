# Scoreboard — where our agent stands

One table, kept current, of our best checkpoints against the opponent ladder. Update after each run
completes or hits a milestone. Numbers are **our agent's win-rate** (as the agent-under-test).

## The ladder

| Milestone (checkpoint) | steps | Ajay | **Yijie** ⭐ | presres1 | stgpr1 | yusa | Ender |
|---|--:|--:|--:|--:|--:|--:|--:|
| stgpr1 — our June submission | 0.5M | 57.4% | — | — | — | — | — |
| tl100m — timeline, from scratch | 100M | 74.6% | — | — | — | — | — |
| binarymarg — binary + gates | 40M | 80.5% | 3.9% | — | — | — | 0/256 |
| shipkl — ship-KL plateau | ~136M | ~80% | ~7% | — | — | — | — |
| **binarygates100m_l4 — ⭐ champion** | 100M | **98.0%** | **14.1%** | 96.9% | 90.6% | 78.1% | 0/32 |
| econblock — econ feats + γ0.999 | *running* | … | … | — | — | — | — |
| cap128x6 — 1.44M capacity | ❌ stopped @40M | ~82 | ~5 | — | — | — | — |

⭐ **Yijie** (rank 13, 1640 Elo) is the verdict metric — Ajay saturates ~77–80% and is blind to what
beats us. `0/N` = wiped, zero wins.

**The wall (reference / target line):** a top-10 agent (Ender) beats Ajay **100%**, and reproduces
`cap/atk open<50 0.75`, `planets@50 8`, `end 14.8` — the conversion we're trying to close toward.
We are **0/32 vs Ender**, wiped 100%. Ajay 98% ≠ strong play; that's why Yijie is the verdict.

## What each milestone changed

Each row is one deliberate change from the row above it (the project moves one lever at a time).

- **stgpr1** — our actual June competition submission. 0.5M-param model, "spray" style (many small
  launches). The pre-timeline baseline everything else is measured against.
- **tl100m** — *added timeline features*: the model now sees each planet's **projected future**
  (who will own it, garrison size) rolled 24 steps ahead, not just the current board. Trained 100M
  from scratch, pure self-play, no reward shaping. 57%→75% Ajay — the biggest single structural jump.
- **binarymarg** — *switched to a binary action space*: instead of choosing **how many** ships to
  send, the model decides only NOOP vs COMMIT (send all-in), and **hardcoded "gates" decided which
  commits were even legal**. Best Ajay at the time (80.5%) but Yijie fell to 3.9% — the red flag.
- **shipkl** — a different take on the same idea: keep a graded ship-count head but add a soft nudge
  (KL prior) toward all-in instead of a hard binary. ~7% Yijie — the standing bar before the champion.
- **binarygates100m_l4 ⭐** — *same binary action space as binarymarg, but we removed the hardcoded
  limiters* (`--binary-commit-gates minimal`). Those gates had been deleting **~80% of legal moves**
  — no multi-source pincer attacks, no pre-emptive reinforcement — by imposing a fixed verdict
  computed from features the model already saw. Letting the model decide for itself took Yijie
  3.9%→**14.1%** *and* Ajay to **98%**. Lesson: the **limiters**, not the binary action space, were
  the problem.
- **econblock** *(running)* — from the champion recipe, *add long-horizon economy awareness*: feed
  the model its projected production/material trajectory, and raise the discount (`gamma`
  0.995→0.999) so it actually values economic swings that play out over 100+ steps (how we lose to
  strong agents).
- **cap128x6** *(❌ stopped @40M — tracked below the 0.53M baseline; RL capacity ceiling looks real)* — from the champion recipe, *make the model bigger*: 128-wide × 6 layers
  = **1.44M** params (up from 96×3 = 0.53M), matching the ~1.2M scale the winners used. Everyone
  competitive was 2–3× our size.

## Precision & methodology (read before trusting a cell)

- **Ajay, Yijie** — 256-game watcher panels, ±~2–4pp. Auto-written to
  `gpu_run_artifacts/<run>/eval_{ajay_1200,yijie}.csv` by `run_watchers.sh`; take the latest row.
- **presres1, stgpr1, yusa, Ender** — 32-game one-off quick panels, ±~7pp (seat-0 only). The
  champion column was measured at step 95.256M. These are NOT watcher-tracked — re-run manually.
- Different panel sizes ⇒ **don't compare a 32g cell to a 256g cell as if equal precision.** Single-
  digit Yijie moves are inside the band; weight the graded metrics (loss-depth, peel, planets@50 —
  see [`metrics.md`](metrics.md)) when the WR is ambiguous.

## Opponents

| Column | Opponent file | Tier |
|---|---|---|
| Ajay | `opponents/candidate_ajay_1200.py` (needs `orbit_lite/`) | 1200 Elo · saturated guard |
| Yijie | `opponents/candidate_yijie.py` | rank 13 · **verdict** |
| presres1 | `final_submissions/submission_presres05.tar.gz` (our "decisive" sub) | our own baseline |
| stgpr1 | `final_submissions/submission_stgpr1.tar.gz` (our "spray" sub) | our own baseline |
| yusa | `opponents/candidate_yusa_rank166.py` (rank ~151, 7.2M BC) | mid-tier · graded loss-depth |
| Ender | `opponents/candidate_ender.py` (top-10, with search) | the wall · 0/N |

## How to refresh a cell

- **Ajay / Yijie** (auto): `tail -1 gpu_run_artifacts/<run>/eval_{ajay_1200,yijie}.csv` → col 3 is WR.
- **presres1 / stgpr1 / yusa / Ender** (manual): quick 32g read —
  `CUDA_VISIBLE_DEVICES="" orbit_wars_rl/.venv/bin/python3 orbit_wars_rl/eval.py --checkpoint <ckpt>
  --opponent <opp> --games 32 --target-decode --reinforce-gate-min-planets 2
  --reinforce-garrison-floor 0` (add `--panel` for the full 256g). Grep `^Win rate vs` / `T2 THE WALL`.
- ⚠ Ender runs WITH search (~1h/256g panel; ~8min/32g). Gate it first: Ender must beat Ajay ~100%
  (`ender_ref.py`) or its config is wrong and the cell is garbage.
