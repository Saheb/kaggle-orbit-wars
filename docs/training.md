# Orbit Wars — Training State & History

---

## Current State (2026-05-30)

**Active run:** None — Rev7 killed at 4.6M steps (2026-05-30 ~17:40)

**Next:** Rev8 — decide single delta based on Rev7 1M results below

**Rev7 1M panel results** (`torch_step_1015808_20260530_104829`, 256 games, 2026-05-30):

| Opponent | Score | vs Phase 1 6M peak |
|----------|-------|--------------------|
| Hellburner | 40.2% | +1.5pp vs 38.7% |
| Zach | 50.8% | -3.5pp vs 54.3% |
| Suneet | 59.8% | -1.9pp vs 61.7% |

HB up slightly vs 6M peak; Zach/Suneet slightly below — all expected at 1M steps.
Trajectory looks healthy (Suneet ~60% at 1M is strong). Rev7 6M checkpoint will be the real comparison.

**Rev7 post-mortem:**
- Instance `i-0a5129fba17cf3de6` terminated; all checkpoints + log pulled locally
- Ran to 4.6M steps — blew past 3M hard cap due to lack of automated kill
- `fire[0]` declined across all 4 checkpoints: 0.34 → 0.32 → 0.29 → 0.25 (same passivity pattern as rev5)
- `avgfleet` rose steadily: 72 → 76 → 88 → 92 (passivity signal)
- `srcs_multi` stayed clean: 1.0–1.7 throughout (penalty working, not the problem)
- IL removal alone was not enough to prevent passivity drift — just slowed it slightly
- **Best checkpoint expected: 1M** (`torch_step_1015808_20260530_104829`) — consistent with 141208 pattern

**Rev7 fire[0] progression:**

| Checkpoint | Steps | fire[0] | avgfleet |
|---|---|---|---|
| 1M | 1,015,808 | ~0.32 | ~74 |
| 2M | 2,031,616 | ~0.29 | ~82 |
| 3M | 3,047,424 | ~0.27 | ~86 |
| 4M | 4,063,232 | ~0.25 | ~90 |

**Ops fix:** panel eval watcher had 3 bugs fixed (2026-05-30):
1. Opponent paths broken after moving to `opponents/` — fixed
2. `EVAL_EVERY_N_CHECKPOINTS=3` skipped 1M checkpoints — changed to 1
3. N/A — python path was valid

**Kill signals (standard, for all runs):**
1. `fire[0]` declining 3 consecutive 1M checkpoints → kill immediately
2. `fire[0] < 0.25` at any checkpoint → kill immediately
3. `srcs_multi > 4.0` or `fire[0] > 0.55` → kill (collapse)
4. Hard cap: 3M steps regardless — **enforce this strictly**

---

## Architecture

### Phase 1 (current)
- **Ship bin mode:** absolute, 32 bins
  - `SHIP_COUNTS = [1,2,3,4,5,6,7,8,9,10,12,14,16,19,22,26,30,35,42,50,60,72,86,102,122,145,173,206,245,290,350,420]`
  - Bin 0 = 1 ship — 0-masking built in by design
- **Features:** planet=20, fleet=13, global=11, pairwise=12, max_owned=16
- **Warmstart:** `bc_phase1_warmstart.pt` (BC on top-200 Kaggle replays)
- **Action decode:** `target` mode

### Old architecture (141208 chain — best ever)
- **Ship bin mode:** fraction, 10 bins (0.1–1.0 × source fleet)
- **Features:** planet=18, fleet=9, global=10, pairwise=10
- `min_ship_bin=1` masked the 0-ship bin (a no-op fire)
- No IL anchor in any blitz run
- ⚠️ Not interchangeable with Phase 1 checkpoints

---

## Best Checkpoints Ever

| Checkpoint | Arch | Steps | HB | Zach | Suneet |
|---|---|---|---|---|---|
| `torch_step_1015808_20260526_141208` | Old | ~63M effective | **55.5%** | 74.2% | 80.1% |
| `torch_step_1015808_20260526_123203` | Old | ~62M effective | 44.5% | 75.4% | 75.8% |
| `torch_step_1015808_20260526_174758` | Old | ~63M + 1M blitz | 42.6% | 76.6% | 75.0% |
| `torch_step_6094848_20260529_160908` | Phase 1 | 21.7M | **38.7%** | 54.3% | 61.7% |

**Target:** >75% on all three simultaneously.

---

## Run History

### Old Architecture

#### Foundation (62M self-play, no IL, no penalty)
- Pure self-play, no external opponents, no IL
- Result: `fire[0]` dropped from 0.25 → 0.09 over 62M steps
- **All checkpoints scored 0% vs HB** — too passive to attack
- But built deep game-sense that the blitz chain exploited

#### 123203 chain → 141208 blitz (best ever)
- `torch_best_123203` (44.5% HB) was produced by an unknown blitz on the foundation
- **141208 run:** resumed from 123203, +1M steps, no IL, no penalty
  - 1M = **55.5% HB** ← peak, best ever
  - 2M = 47.3%, 3M = 46.1%, 4M = 45.3%, 5M = 44.1%
  - **Peaked exactly at 1M, declined every checkpoint after**
- Key: foundation had ~63M of self-play game-sense; blitz injected HB aggression before passivity reset

#### Post-141208 blitz (174758)
- +1M more blitz on 141208 best
- HB=42.6%, Zach=76.6%, Suneet=75.0% — HB regressed, Zach/Suneet held

---

### Phase 1

#### Rev1–Rev3 (BC warmstart, il-decay-frac=0.5)
- Resume: `bc_phase1_warmstart.pt`
- `--il-decay-frac=0.5` — IL decayed to 0 halfway through, too short
- Peak: Zach=12.9%, HB=0.8%
- IL anchor kept policy too conservative; not enough self-play to build game-sense

#### Rev4 (resume rev3 3M peak, +6M)
- Same config as rev3
- `srcs_multi` collapse at iter 200+ — carpet-bomb failure mode
- Peak same as rev3, no improvement

#### Rev5 (resume rev4 3M, +6M, srcs-multi-penalty added)
- Added `--srcs-multi-penalty 0.001 --srcs-multi-threshold 2.0` to guard collapse
- **6M peak: HB=38.7%, Zach=54.3%, Suneet=61.7%** ← Phase 1 best
- Regression after 6M: fire[0] 0.35→0.25 by 12M, avgfleet rising
- `srcs_multi` stayed 1.2–1.8 throughout (penalty worked, not the cause of passivity)
- IL anchor (`--il-lambda=0.01`) + self-play equilibrium together drove passivity

#### Rev6 blitz (resume rev5 6M, 3M, cosine penalty decay)
- Tried cosine decay on the srcs-multi penalty to 0
- HB=0%, Zach=19.9% — penalty decayed too aggressively, carpet-bomb returned

#### HB blitz attempt (2026-05-30, pool-fraction=0.9, pool-external-fraction=1.0)
- Tried to replicate 141208 blitz on Phase 1: 90% HB exposure, lr=3e-5, no IL
- **Destroyed everything in 500K steps: HB=0%, Zach=4.3%**
- Root cause: Phase 1 at 21.7M effective steps lacks the deep foundation (~63M) the 141208 blitz relied on
- Heavy HB overwrites fragile generalisation — blitz only works as final polish on fully-developed model

#### Rev7 (killed 2026-05-30, resume rev5 6M, no IL)
- **Single delta from rev5: IL anchor removed entirely**
- Rationale: 141208 chain had no IL; IL pulls toward conservative BC → amplifies self-play passivity drift
- Config: pool-fraction=0.75, pool-external-fraction=0.05 (5% HB, 75% self-play), lr=3e-4 cosine over 60M schedule
- `fire[0]=0.57` at iter 1 (optimizer transient), settled ~0.32 at 1.5M, declined to 0.25 by 4M
- srcs_multi stayed 1.0–1.7 throughout — penalty working, not the problem
- **Verdict:** IL removal alone insufficient; passivity still set in, just slightly slower than rev5
- **Best checkpoint: 1M** — panel scores pending
- Ran to 4.6M before kill (3M hard cap was not enforced automatically)

---

## Why Passivity Sets In

Self-play equilibrium rewards holding over firing — `fire[0]` declines monotonically in long runs. Observed in every unconstrained run:
- 62M foundation (no IL, no penalty): `fire[0]` 0.25 → 0.09
- Rev5 (with IL): `fire[0]` 0.35 → 0.25 by 12M

**IL makes it worse:** `--il-lambda` penalises KL divergence from the BC warmstart each update. BC is methodical/conservative (it imitates top human players who build large fleets before attacking). IL + self-play passivity compound each other.

**srcs-multi-penalty is NOT the cause** — in rev5, `srcs_multi` stayed 1.2–1.8 the entire run, well below the 2.0 threshold. Passivity set in anyway.

---

## Short-Run Chain Strategy (current)

Replicate the 123203→141208 structure on Phase 1:

1. Run 2–3M steps, kill before passivity takes hold
2. Resume from best checkpoint → next run
3. Each run is a single delta from the previous
4. Build foundation depth through chaining, not one long run
5. HB blitz only as final polish once foundation is solid (>60M equivalent)

**Chain so far:**
```
bc_phase1_warmstart → rev1/2/3 → rev4 → rev5 (6M peak) → rev7 (active)
```

---

## Key Config (Rev7 / current standard)

```bash
python orbit_wars_rl/train_torch.py \
  --resume seed_checkpoints/phase1_resume.pt \
  --total-steps 30000000 \
  --lr-schedule-steps 60000000 \   # cosine over 60M so LR doesn't hit 0 at 3M
  --learning-rate 0.0003 \
  --num-envs 512 --rollout-steps 64 --num-minibatches 32 \
  --ppo-epochs 2 \
  --checkpoint-interval 1000000 \
  --pool-checkpoint-interval 500000 --pool-max-size 20 \
  --pool-mode mixed --pool-fraction 0.75 \
  --external-opponents opponents/candidate_hellburner.py \
  --pool-external-fraction 0.05 \
  --win-margin-coeff 0.5 \
  --action-decode target \
  --pool-pfsp-min-games 30 \
  --pool-mastered-threshold 0.99 \
  --pool-mastered-min-games 500000 \
  --srcs-multi-penalty 0.001 \
  --srcs-multi-threshold 2.0 \
  --terminate-on-done
```

**Flags intentionally absent vs rev5:**
- No `--il-lambda` (removed in rev7 — key delta)
- No `--il-ref` / `--il-decay-frac`

---

## Metrics to Watch

| Metric | Healthy range | Warning | Kill |
|--------|--------------|---------|------|
| `fire[0]` | 0.30–0.55 | Declining 2 consecutive | Declining 3 consecutive or <0.25 |
| `srcs_multi` | 1.0–2.5 | 2.5–4.0 | >4.0 (carpet-bomb collapse) |
| `avgfleet` | 50–90 | >100 and rising | — |
| `clip_frac` | 0.10–0.25 | >0.30 sustained | — |
| `H_fire` (entropy) | 0.05–0.15 | <0.03 (deterministic) | — |
