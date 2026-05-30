# Orbit Wars: Top 10 Strategy Plan

## Context

We have a working PPO + Entity Transformer training pipeline in `orbit_wars_rl/`. The current heuristic agent (`main.py`) is estimated at ~900–1000 LB. The top-10 cutoff appears to be ~1100+, with Roman at 1224 and Suneet at 1200 holding top spots. No published top-10 agent uses deep RL — the field is entirely heuristic-based. This is the opportunity: a well-trained RL agent that generalises across map seeds can discover coordination patterns that hand-coded heuristics miss.

This plan is two-track: **close the gap now with heuristic improvements** (days), then **blow past the ceiling with RL** (weeks on GPU). The tracks are independent — improvements to `main.py` also produce better BC training data.

---

## What Is the Entity Transformer and Why It Helps Here

### The Problem with Heuristics

Every heuristic agent (Roman, Suneet, Pascal) works by **computing pairwise scores**: for each owned planet × each target planet, calculate a hand-crafted utility (travel time, production value, contest risk, enemy strength). This has three hard limits:

1. **Features are manually designed.** Roman hardcodes 20+ multipliers (ELIMINATION_BONUS = 55, GANG_UP_VALUE_MULT = 1.4, WEAKEST_ENEMY_VALUE_MULT_4P = 1.5, etc.). Tuning them by hand takes weeks and overfits to the current opponent pool.
2. **No joint reasoning.** Each planet scores its targets independently. There's no way to learn "if I send from planet A *and* planet B simultaneously, they arrive together and overwhelm the garrison."
3. **Fixed architecture.** When the opponent pool changes (new top agents emerge), the hardcoded weights can't adapt.

### What the Entity Transformer Does Differently

The **Entity Transformer** treats every planet and fleet as a *token* with a feature vector and runs Transformer attention across the entire set simultaneously:

```
Input tokens = [global_state | planet_0 | planet_1 | ... | fleet_0 | fleet_1 | ...]
```

**Self-attention** lets every planet token look at every other planet and fleet in one forward pass. Practically this means:

- **Planet A "sees" that fleet X is converging on planet B** → A can learn to time a coordinated strike.
- **The global token aggregates army strength, production ratio, step number** → emergent phase detection without hard-coded thresholds.
- **Enemy fleet vectors (angle, speed, ships) are in context** → implicit opponent move prediction.

The three output heads (fire/angle/ship) act on each owned-planet token independently, so decisions are per-planet but informed by the global context. This is exactly the right inductive bias for Orbit Wars.

### Why 307K Params Is the Right Scale

- Large enough (3 layers × 4 heads × 96 dims) to learn complex multi-entity strategies.
- Small enough (~10 ms per inference on CPU) to fit within Kaggle's per-turn time budget.
- Fits in 3–4 GB VRAM, trainable on a single A10G/V100.

### The RL Advantage Over Pure Heuristics

| Capability | Roman (heuristic) | Entity Transformer (RL) |
|---|---|---|
| Multi-planet coordination | ✗ (greedy per-planet) | ✓ (joint attention) |
| Opponent adaptation | ✗ (fixed weights) | ✓ (learned from self-play) |
| Phase detection | ✓ (8 hand-coded phases) | ✓ (emergent from global token) |
| Elimination timing | ✓ (ELIMINATION_BONUS=55) | ✓ (learned from reward signal) |
| Gang-up timing | ✓ (2 turn post-battle delay) | ✓ (learns arrival timing) |
| Map generalisation | ✓ (geometry is explicit) | ✓ (geometry baked into features) |
| New opponent adaptation | ✗ | ✓ (opponent pool self-play) |

---

## Competition Landscape

| Agent | LB Score | Type | Key edge |
|---|---|---|---|
| Roman | 1224 | Heuristic | Elimination bonus, gang-up, weakest targeting |
| Suneet | 1200 | Heuristic (mislabelled "PPO") | Same WorldModel as Roman, slightly tuned |
| Pascal | ~1100 | Heuristic | Fleet sweep, extended horizon |
| Marco | 1060 | Beam search (K=3) | Chain precompute, 2-hop lookahead |
| Rahul | ~1000? | Hybrid (MCTS + tiny MLP) | 420ms MCTS budget |
| **Our heuristic** | **~900–1000** | Heuristic | Production-EV scoring, phase system |

**Top-10 gap:** We need ~100–300 LB points. Roman's three key innovations over a base heuristic are worth an estimated ~150–200 pts combined (elimination, gang-up, weakest targeting). The Entity Transformer, once trained, has the ceiling to exceed Roman.

---

## Two-Track Strategy

```
Week 0–1  │  Track A: Port Roman's 3 innovations into main.py  →  +100–200 LB
           │  Track B: BC warm-start on improved heuristic data
           │
Week 1–2  │  Track A: Evaluate + submit improved heuristic
           │  Track B: 5–10M local PPO steps (validates learning)
           │
Week 2–4  │  Track B: Cloud GPU, 50–200M PPO steps  →  top 10 target
```

---

## Track A: Heuristic Improvements (Local, Days)

Port exactly three innovations from Roman's kernel into `main.py`. These are the highest-LB-per-line-of-code changes:

### A1 — Weakest Enemy Targeting
**File:** `main.py`  
**Roman's code:** Lines 68–69, 723–730, 1476–1480 of `kernels/roman-lb-1224/submission.py`  
**What it does:** Multiplies attack value by 1.25× (2P) or 1.5× (4P) for planets belonging to the weakest enemy (by total ship count). Focuses fire to eliminate one opponent quickly.  
**Implementation:** In the target scoring function, compute `weakest_enemy = min(enemies, key=lambda e: total_ships(e))`, then apply multiplier when `target.owner == weakest_enemy`.  
**Estimated gain:** +30–60 LB (esp. in 4P matches).

### A2 — Elimination Bonus System
**File:** `main.py`  
**Roman's code:** Lines 160, 1470–1474 of `kernels/roman-lb-1224/submission.py`  
**What it does:** Adds a flat +55 bonus to any planet belonging to an enemy whose total strength is below a threshold (~90% of our total). Forces decisive kill when ahead.  
**Implementation:** Add constant `ELIMINATION_BONUS = 55.0`. In `score_for_send()`, check if `enemy_total_ships < 0.9 * our_total_ships` and add the bonus.  
**Estimated gain:** +40–80 LB (win-rate in advantaged positions increases sharply).

### A3 — Gang-Up Missions (Inter-Enemy Exploit)
**File:** `main.py`  
**Roman's code:** Lines 2217–2304 of `kernels/roman-lb-1224/submission.py`  
**What it does:** Detects when two enemies are fighting each other. Schedules our fleet to arrive `battle_turn + 2` turns after the battle resolves, capturing the depleted winner.  
**Implementation:** Scan all `(enemy_planet_A, enemy_fleet_targeting_B)` pairs. If we own a planet within range, schedule a fleet to arrive 2 turns post-battle. Apply 1.4× value bonus.  
**Estimated gain:** +30–50 LB (especially in 4P where inter-enemy battles are frequent).

**Evaluation after A1–A3:** Run `python evaluate_head_to_head.py --candidate improved --baseline main --seeds 20` and `python evaluate_baseline.py --seeds 20`. Target: +20% win rate vs current main.py.

---

## Track B: Entity Transformer RL

### B0 — Benchmark Local Throughput First

Before committing to a training run, measure real pipeline SPS:

```bash
cd orbit_wars_rl
time .venv/bin/python -c "
from env import OrbitWarsEnv
from features import extract_features
from action_mask import compute_action_masks
from model import EntityTransformer
from config import Config
import torch, time

cfg = Config()
model = EntityTransformer(cfg.model)
model.eval()
env = OrbitWarsEnv(num_players=2)
obs = env.reset()
t0 = time.time()
for i in range(500):
    f = extract_features(obs, 0, num_players=2)
    m = compute_action_masks(obs, 0)
    with torch.no_grad():
        model(f['planet_features'].unsqueeze(0), f['fleet_features'].unsqueeze(0),
              f['global_features'].unsqueeze(0), f['planet_mask'].unsqueeze(0),
              f['fleet_mask'].unsqueeze(0))
    obs, r, done, _ = env.step([])
    if done: obs = env.reset()
print(f'{500/(time.time()-t0):.1f} SPS full pipeline')
"
```

**Expected:** 30–80 SPS on M-series MPS. Use this to calibrate local training budget.

### B1 — Behavioral Cloning Warm-Start

Before any PPO, run BC on the **improved heuristic** (Track A output):

```bash
cd orbit_wars_rl
.venv/bin/python bc.py --agent ../main.py --num-games 500 --steps 10000
```

**Why:** BC initialises the ET to be roughly as good as the heuristic in ~10K gradient steps. Without it, PPO wastes thousands of episodes learning basic game mechanics from scratch (what "fire" does, orbital mechanics). BC warm-start cuts this wasted phase entirely.

**Files:** `orbit_wars_rl/bc.py` (fully functional after our rewrite)  
**Expected output:** Val loss < 0.5 on fire/angle/ship targets; the agent should play recognisably heuristic-like games after BC.

### B2 — Local PPO Validation (1–5M Steps)

```bash
cd orbit_wars_rl
.venv/bin/python train.py --total-steps 5_000_000 --shaping-coef 0.001 --wandb
```

**Goal:** Confirm the learning curve is positive. Specifically:
- Reward history should trend from ~+1/−1 random (50% win rate) toward +0.5+ (60%+ win rate) against "random" opponent
- `clip_frac` should stay below 0.3 (it was 0.64 in the initial random-policy run because there was no warm-start; BC will fix this)
- Value loss should decrease over first 200K steps

**Estimated time at 40 SPS:** 5M steps ÷ 40 SPS = 35 hours (~1.5 days). Run overnight.

**What to watch:** If win rate vs. random doesn't improve past 65% by 1M steps, the reward signal is too sparse. In that case, increase `shaping_coef` to 0.005.

### B3 — Evaluation Loop

After each checkpoint, evaluate using:

```bash
.venv/bin/python eval.py \
  --checkpoint checkpoints/step_5000000/checkpoint.pt \
  --games 32 \
  --opponent ../main.py
```

Target thresholds before moving to cloud:
- > 55% win rate vs `random` within 1M steps
- > 40% win rate vs `main.py` (heuristic) within 5M steps

### B4 — Cloud GPU (AWS/GCP)

**When to move:** After confirming positive learning signal locally (B2 passes thresholds), or when local budget is exhausted (5M steps).

**Recommended instance:**
| Option | GPU | VRAM | Est. SPS | Cost/hr | 50M steps cost |
|---|---|---|---|---|---|
| AWS g5.2xlarge | A10G 24GB | 24 GB | 200–300 | ~$1.21 | ~$60 |
| AWS p3.2xlarge | V100 16GB | 16 GB | 150–250 | ~$3.06 | ~$100 |
| GCP a2-highgpu-1g | A100 40GB | 40 GB | 400–600 | ~$3.67 | ~$85 |

**Recommended:** `g5.2xlarge` — best cost/performance for this model size (307K params). 50M steps in ~6–10 hours at ~200 SPS. 200M steps in ~24–40 hours at ~$30–$50.

**Setup:**
```bash
# On cloud instance:
pip install torch kaggle-environments wandb numpy
git clone <repo>
cd orbit_wars_rl
# Copy BC-pretrained checkpoint as starting point
python train.py \
  --total-steps 200_000_000 \
  --shaping-coef 0.001 \
  --wandb \
  --device cuda
```

**Self-play progression:** The opponent pool (size=8, 30% sampling) automatically gets populated every 50 episodes. By 50M steps the pool will contain snapshots from multiple stages of training — this is the core mechanism that prevents the agent from overfitting to "random" and forces it to learn robust strategies.

### B5 — Export and Submit

```bash
cd orbit_wars_rl
.venv/bin/python export_agent.py \
  --checkpoint checkpoints/final.pt \
  --output ../main_submitted.py
```

The export embeds model weights as base64, inlines `features.py` and `action_mask.py`, and produces a single self-contained file ready for Kaggle submission.

---

## Milestone Schedule

| Week | Deliverable | Metric |
|---|---|---|
| 1 | A1+A2+A3 in main.py | +20% win rate vs current main.py |
| 1 | Submit improved heuristic | Target: ~1100+ LB |
| 1 | BC warm-start runs clean | Val loss < 0.5 |
| 2 | 5M local PPO steps | >40% win rate vs heuristic |
| 2 | Evaluate checkpoint vs main.py | Confirm learning |
| 3 | Launch cloud GPU run (50M steps) | Policy clearly beats heuristic |
| 4 | Cloud run (200M steps) | Target: competitive with Roman (~1200 LB) |
| 4 | Submit ET agent | **Top 10** |

---

## Critical Files

| Purpose | File |
|---|---|
| Heuristic agent to improve | `main.py` |
| Roman's innovations to port from | `kernels/roman-lb-1224/submission.py` |
| Entity Transformer model | `orbit_wars_rl/model.py` |
| Feature extraction | `orbit_wars_rl/features.py` |
| Action masks | `orbit_wars_rl/action_mask.py` |
| BC trainer | `orbit_wars_rl/bc.py` |
| Training loop | `orbit_wars_rl/train.py` |
| PPO learner | `orbit_wars_rl/ppo.py` |
| Evaluation | `orbit_wars_rl/eval.py` |
| Export to submission | `orbit_wars_rl/export_agent.py` |
| Head-to-head evaluation | `evaluate_head_to_head.py` |
| Baseline evaluation | `evaluate_baseline.py` |
| Training config | `orbit_wars_rl/config.py` |

---

## Verification Plan

1. **After Track A (heuristic improvements):**
   ```bash
   python evaluate_head_to_head.py --candidate improved --baseline main --seeds 32
   # Pass: ≥55% win rate
   python evaluate_baseline.py --seeds 20 --opponent random
   # Pass: ≥80% win rate vs random
   ```

2. **After BC warm-start (B1):**
   ```bash
   cd orbit_wars_rl
   .venv/bin/python eval.py --checkpoint checkpoints/bc_warmstart.pt --games 16 --opponent random
   # Pass: >50% win rate vs random (better than coin flip from the start)
   ```

3. **After local PPO (B2, 5M steps):**
   ```bash
   .venv/bin/python eval.py --checkpoint checkpoints/step_5000000/checkpoint.pt \
     --games 32 --opponent ../main.py
   # Pass: >40% win rate vs heuristic
   ```

4. **After cloud GPU (B4, 200M steps):**
   ```bash
   .venv/bin/python eval.py --checkpoint checkpoints/final.pt \
     --games 64 --opponent ../kernels/roman-lb-1224/submission.py
   # Target: >45% win rate vs Roman (means ET ≈ Roman on LB)
   ```

5. **After submission:** Monitor Kaggle LB score. If stalled, increase opponent pool diversity (add Roman/Suneet as fixed opponents to the pool during training).
