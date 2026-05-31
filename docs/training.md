# Orbit Wars — Training State & History

---

## Phase 1 Run History — Quick Reference

| Run | Delta from prev | Expected | What happened | Killed because | Best ckpt LB |
|---|---|---|---|---|---|
| **Rev5** | Added srcs-multi-penalty | Guard carpet-bomb collapse | 6M peak: HB=38.7%, Zach=54.3%, Suneet=61.7%. Passive drift after 6M. | fire[0] 0.35→0.25 by 12M | — |
| **Rev6** | Cosine decay on srcs penalty | Reduce over-penalisation | Carpet-bomb returned immediately | HB=0%, collapsed | — |
| **HB blitz** | 90% HB, lr=3e-5, no IL | Replicate 141208 blitz | Destroyed everything in 500K steps | HB=0%, Zach=4.3% | — |
| **Rev7** | Removed IL anchor | IL amplifies passivity; remove it | 1M peak HB=40.2%, then passive drift | fire[0] 0.32→0.25 by 4M | 750 LB |
| **Rev8** | `--shaping-coef 0.05` | Reward material gain per step → more firing | HB dropped to 30% at 1M. fire[0] hit 0.22 by 1.5M | Passive + panel regression | — |
| **Rev9** | `--entropy-coef-fire 0.05` | Higher H_fire suppresses passive locking | H_fire rose (✓) but HB=25%, Zach=42% — entropy = random firing, not strategic | Panel regression on all metrics | — |
| **Rev10d** | `rollout-steps 512, envs=64` | Better credit assignment for sparse reward | HB=28.1%, Zach=46.9%, Suneet=53.8%. avgfleet oscillated 85-92 post-1M | Panel below rev7 on all metrics | 772 LB (1M) |
| **Rev11** | Pure self-play, 10M target, pool=40, kill floor 0.20 | Passive phases may recover at scale; HB was wrong opponent | clip_frac=0.199↓ (lowest ever). 2M best (fire=0.34, fleet=77). 4M/5M/6M declined. vs 141208: 2M=17%, 4M=8%, 5M=6%. vs Zach: 2M=41%, 5M=36%. Killed at 6M. | **Best: 2M** (`torch_step_2031616_20260531_065423.pt`) — submit tomorrow | — |

**What we know:**
- Every run peaks at ~1M steps then passive drift sets in
- HB panel is the **wrong metric** — we beat HB by being passive (low fire rate wins vs HB)
- Use Zach and Suneet as proxies; LB score is ground truth
- Export requires `--target-decode` for Phase 1; all pre-fix submissions scored ~87

**LB scores:** 141208 (old arch) = **894** | rev10d 1M = **772** | rev7 1M = **750**
**Target:** > 894 to beat current best. Top 100 needs ~1153.

---

## Current State (2026-05-31)

**Active run:** None — Rev11 killed at 6M (2026-05-31)
- Pure self-play (no external opponents), pool-max-size=40, 10M step target
- Kill threshold lowered: fire[0] < 0.20 (allow passive phases)
- 1M checkpoint: fire[0]=0.32, avgfleet=78.1 — comparable to rev7 1M
- Behaviour post-1M: avgfleet oscillating 78-85, NOT monotonically climbing like previous runs
- clip_frac=0.213 still drifting down (healthy)
- **Tomorrow (2026-06-01):** Submit rev11 2M as slot 1. Then consider rev12 direction.

**Rev12 candidates:**
1. Resume from rev11 2M, longer pure self-play (get to 10M+ total)
2. Resume from rev11 2M, try pool-only (no current-self games, pure historical self-play)
3. Try `--run-name rev12` — all checkpoints now embed revision name (no more file confusion)

**LB scores for correctly-exported Phase 1 agents (2026-05-31):**

| Submission | Steps | LB Score | Notes |
|---|---|---|---|
| 141208 (old arch) | ~63M eff | **894.0** | Best ever, target to beat |
| rev10d 1M (Phase 1) | 1M | 772.4 | Better than rev7 1M on LB |
| rev7 1M (Phase 1) | 1M | 750.3 | Panel best, but LB < rev10d |
| rev10d 2M (Phase 1) | 2M | 600.0 | Passive drift hurt 2M ckpt |

Key insight: **rev10d 1M (772) > rev7 1M (750) on LB despite worse panel** — panel and LB don't correlate perfectly. More firing = worse vs HB (panel) but better on LB.

**Export bugs fixed (2026-05-31) — all previous Phase 1 submissions were broken:**
1. `def agent(obs)` → `def agent(obs, cfg=None)` — was crashing every step silently (score 87)
2. Missing `--target-decode` — was using angle decode (wrong for Phase 1)
Always use: `python3 orbit_wars_rl/export_agent.py --checkpoint <ckpt> --output <out> --target-decode`

---

---

## LB Episode Analysis (2026-05-31)

### How to fetch our episodes

```python
# Step 1: list our submission's episodes
import requests, json

with open('/Users/saheb/.kaggle/access_token') as f:
    token = f.read().strip()
with open('/Users/saheb/.kaggle/kaggle.json') as f:
    creds = json.load(f)

# EpisodeService API (not the standard v1 API)
url = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
resp = requests.post(url,
    auth=(creds['username'], creds['key']),
    json={"submissionId": 53076736},  # our best submission
    headers={"Content-Type": "application/json"})
episodes = resp.json()['episodes']
# Save: json.dump({'episodes': episodes}, open('/tmp/our_episodes.json','w'))

# Step 2: download individual replay JSON
# IMPORTANT: standard v1 API requires Bearer token (not basic auth)
ep_id = 78025831
r = requests.get(
    f"https://www.kaggle.com/api/v1/competitions/episodes/{ep_id}/replay",
    headers={"Authorization": f"Bearer {token}"}   # ~/.kaggle/access_token
)
open(f"/tmp/{ep_id}.json", "wb").write(r.content)
```

> ⚠️ The replay endpoint **requires Bearer token** (`~/.kaggle/access_token`), not basic auth.
> Basic auth returns 401. The `access_token` file is separate from `kaggle.json`.

### Key findings from 159 LB episodes (submission 53076736, score 894.4)

**Win/loss summary:**
- 63 wins (40%) / 96 losses (60%)
- We are **not** matched against top-10 agents — highest opponent score that beat us was ~1036
- **38 out of 95 losses (40%) are to weaker opponents** (their score < ours at game start)

**Loss breakdown by opponent score:**
| Bracket | Losses | % of losses |
|---------|--------|-------------|
| < 800 | 6 | 6% |
| 800–900 | 38 | 40% |
| 900–1000 | 46 | 48% |
| 1000–1100 | 5 | 5% |
| > 1100 | 0 | 0% |

**Root cause: bimodal fire behavior (same checkpoint, two modes)**

From replay analysis of 10 losses + 5 wins:

| Metric | Our WINS | Our LOSSES | Winner in losses |
|--------|----------|------------|-----------------|
| fire_rate | **0.468** | **0.223** | 0.528 |
| multi_rate | 0.502 | 0.316 | 0.633 |
| avg_ship_bin | 31.0 | 34.3 | 53.3 |
| n_steps | 232 | 281 | 281 |

**Same agent, same checkpoint — we fire 2× more often in games we win.**
In losses we play passively (22% fire rate); the winner fires on 53% of steps.
Longer game length in losses (281 vs 232 steps) confirms we're being snowballed.

**Implication:** The policy has a passive mode it locks into for specific game states.
The training self-play equilibrium creates states where holding is locally optimal.
Training `fire[0]~0.32` is an average that masks this bimodal behavior.

### Rev9 fix: entropy-coef-fire

`--entropy-coef-fire` (default 0.01) controls the entropy bonus on the fire head specifically.
Increasing it prevents the policy from becoming deterministically non-firing.

Rev9 uses `--entropy-coef-fire 0.05`. H_fire rose from ~0.09 (rev7/rev8) to 0.143 by iter 3.

---

## Leaderboard Reality Check (2026-05-31)

| Rank | Agent | Score | Notes |
|---|---|---|---|
| 1 | Isaiah @ Tufa Labs | 1751.4 | Target for top 10 |
| 2 | typeIIIfairy | 1716.7 | |
| 3 | Vadasz | 1614.9 | |
| 7 | **Zachary Ruhe** | 1596.0 | ← Our `candidate_zach_public.py` (old version) |
| 17 | kovi | 1412.1 | Beats even rank1; large ship commitment |
| 23 | Shun_PI | 1354.1 | Was rank1 when replays were collected |
| 100 | ttmn | 1153.0 | Score needed for top 100 |
| **669** | **Saheb (us)** | **894.4** | Current position |
| 962 | **Suneet Saini** | 827.4 | ← Our `candidate_suneet_lb1200.py` — **below us** |
| N/A | Hellburner | — | Not on LB — local test bot only |

**LB game type split (rev7 1M submission, 159 episodes):**
- 1v1: 76 games — **50% WR** (we're competitive in head-to-head)
- FFA: 83 games — **30% WR** (badly losing in 4-player games)

**The FFA problem:** Our agent is trained in 2-player mode only. 52% of LB games are FFA, where it plays a game it was never trained for. This is the primary LB score drag. Top-10 team trained *separate* 2p and 4p policies.

**Critical implication**: Our panel opponents don't represent actual LB threats.
- Suneet (rank 962) is below us — optimising vs Suneet adds no signal.
- HB not on LB at all.
- Zach (rank 7) is real but `candidate_zach_public.py` is an old version.
- Top threats (Isaiah, typeIIIfairy, Vadasz, 213tubo, bowwowforeach, 3Comets) untested.

**What kovi does differently** (beats rank1 Shun_PI — 8 wins from 116 games):
- avg_ship_bin=40.6 vs Shun_PI 27.5 → commits ~50% more ships per attack
- multi_src_rate=0.46 vs 0.43 → fires from more sources simultaneously
- fire_rate=0.47 vs 0.43 → slightly more aggressive

**Panel opponent priority for next update:** find/build proxies for Isaiah, typeIIIfairy, Vadasz.

---

## Current State (original — 2026-05-30)

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

## Training Flag Glossary

### PPO / Optimiser

| Flag | Default | What it does |
|---|---|---|
| `--learning-rate` | 3e-4 | PPO learning rate (Adam). |
| `--lr-schedule-steps` | — | Total steps over which LR cosine-decays to 0. Set to 60M so LR barely moves in short runs. |
| `--num-envs` | 512 | Parallel game environments. More = faster SPS but more GPU memory. |
| `--rollout-steps` | 64 | Steps collected per environment before each PPO update. Longer = better credit assignment for sparse terminal reward, but uses more GPU memory. |
| `--num-minibatches` | 32 | How many minibatches to split each rollout into for PPO gradient updates. Fewer = larger batches = more memory. Keep `total_buffer / num_minibatches ≈ 2000` to avoid OOM on L4. |
| `--ppo-epochs` | 2 | How many times to iterate over each rollout during the PPO update. |
| `--total-steps` | 30M | Training runs until this many env steps. Watcher hard-kills at 3M (10M for rev11). |

### Self-play pool

| Flag | Default | What it does |
|---|---|---|
| `--pool-mode` | mixed | `mixed` = sample both recent snapshots and older ones. `latest` = only recent. |
| `--pool-fraction` | 0.75 | Fraction of games played against the pool (vs current self). |
| `--pool-max-size` | 20 | Max checkpoints kept in the opponent pool. Bigger = more diverse opponents. |
| `--pool-checkpoint-interval` | 500K | How often to snapshot current model into the pool (env steps). |
| `--pool-external-fraction` | 0 | Fraction of pool games guaranteed to go to external opponents (bypasses PFSP). |
| `--pool-pfsp-min-games` | 30 | Use uniform sampling until an opponent has played this many games (prevents death-spiral from noisy early win rates). |
| `--pool-mastered-threshold` | 0.99 | Win rate above which an opponent is considered "mastered" and removed from pool. |
| `--external-opponents` | — | Path(s) to external rule-based agent files. We used `opponents/candidate_hellburner.py`. |

### Reward shaping

| Flag | Default | What it does |
|---|---|---|
| `--win-margin-coeff` | 0.5 | Scales the reward by win margin (more ships remaining = bigger reward). Encourages decisive wins. |
| `--shaping-coef` | 0 | Per-step reward = `coef × (my_material_delta)`. Tried in rev8 — hurt performance because passive resource collection also gives material gain. |

### Entropy / exploration

| Flag | Default | What it does |
|---|---|---|
| `--entropy-coef-fire` | 0.01 | Entropy bonus specifically on the fire head. Higher = more random firing decisions. Tried in rev9 — raised H_fire metric but made play less strategic, hurting panel/LB. |

### Collapse guards

| Flag | Default | What it does |
|---|---|---|
| `--srcs-multi-penalty` | 0 | Per-step penalty when the agent fires from too many sources simultaneously (carpet-bomb). Set to 0.001. |
| `--srcs-multi-threshold` | 2.0 | Sources above this number get penalised. |

### IL (behaviour cloning anchor) — NOT used in rev7+

| Flag | What it does |
|---|---|
| `--il-lambda` | KL penalty toward the BC warmstart on every PPO update. Removed in rev7 — it amplifies passivity drift. |
| `--il-ref` | Reference checkpoint for IL. |
| `--il-decay-frac` | Fraction of training over which IL weight cosine-decays to 0. |

### Action decode

| Flag | What it does |
|---|---|
| `--action-decode target` | Phase 1 mode: model outputs a target planet logit; argmax at inference. Required for Phase 1 checkpoints. **Always pass `--target-decode` when exporting.** |

---

### Key training metrics (in log lines)

> ⚠️ **Thresholds are empirical** — derived from watching our own runs, not from studying top-ranked agents' training internals. They are proxy signals, not ground truth. LB score is ground truth.

### Data-driven targets from rank1 replays (Shun_PI, 116 1v1 games)

| Metric | Rank1 WINS | Rank1 LOSSES | kovi (beats rank1) | Our rev7 wins | Our rev7 losses |
|---|---|---|---|---|---|
| fire_rate | **0.320** | 0.366 | 0.310 | 0.468 | 0.223 |
| multi_rate | 0.325 | 0.364 | 0.326 | 0.502 | 0.316 |
| avg_ship/attack | 31.7 | 17.6 | **45.7** | ~31 | ~34 |
| n_steps | **145** (decisive) | 167 | 167 | 232 | 281 |

**Key insights:**
1. **Lower fire_rate = winning** — rank1 fires *less* in wins (0.32) than losses (0.37). Selective, decisive attacks beat reactive spray.
2. **Ship commitment is the differentiator** — kovi beats rank1 not by firing more often but by sending **45.7 avg ships per attack** vs rank1's 31.7. Big forces at the right time.
3. **Our fire_rate analysis may be misleading** — our LB losses show fire_rate=0.223, but this may be *reactive* (responding to being attacked) rather than passive. Rank1 also has lower fire_rate in wins.
4. **Our avg_ship (~31) matches rank1** — the problem may not be ship commitment size. It may be tactical timing.
5. **Shorter games = winning** — rank1 wins in 145 steps, loses in 167. We win in 232, lose in 281. We should be closing games faster.

**Revised focus:** Rather than maximising fire_rate, target *decisive* attacks — fewer but larger (meanshipbin ≥ 18), and learn to close games before step 200.

| Metric | Healthy range | Warning | Why it matters |
|---|---|---|---|
| `fire[0]` | 0.28–0.45 | < 0.25, or declining 3 consecutive checkpoints | Fraction of steps the agent fires from planet-slot 0 (its first owned planet). Proxy for aggression — if it stops firing from there, it's hoarding ships. Kill floor 0.25 was set empirically: below this, panel scores reliably crash. |
| `avgfleet` | 60–85 | > 95 and trending up | Average ships on planets. **Rising = passive.** Ships accumulate when you *hold* rather than *attack*. An aggressive agent keeps fleets lean by sending them constantly. If avgfleet climbs, the agent has learned to build up fleets rather than spend them. |
| `srcs_multi` | 1.0–2.5 | > 4.0 | Average number of source planets firing simultaneously. A value > 4 signals the **carpet-bomb collapse**: agent fires from all planets at once, empties defences, rarely wins. This is a degenerate training artifact. Normal multi-source attacks use 2-3 sources. Threshold 4 was empirically tuned. |
| `clip_frac` | 0.10–0.22 | > 0.30, creeping upward | PPO's measure of how often the policy update was clipped (policy changed too much). Creeping upward = gradients are too large = value function estimates go stale = training destabilises. From standard PPO/IMPALA/AlphaStar literature: sustained > 0.30 is a red flag. Our rev11 shows clip_frac drifting *down* to 0.205 which is unusually good. |
| `H_fire` | 0.08–0.15 | < 0.05 (deterministic) | Entropy of the fire-head distribution. Low = policy is near-deterministically choosing fire or no-fire in each state. Too low = rigid passive policy that can't adapt. **Note:** higher is NOT automatically better — rev9 showed that boosting H_fire with `--entropy-coef-fire` made firing random rather than strategic, hurting LB. |
| `meanshipbin` | 15–18 | > 19 trending up | Mean ship-count bin when firing (Phase 1 has 32 bins; bin 16 ≈ mid-range fleet fraction). Rising alongside avgfleet = passive (building bigger fleets before attacking). Rising with stable avgfleet = sending larger forces per attack (may be OK). |
| `SPS` | 650–730 | < 500 | Environment steps per second. Drops if: rollouts are longer (rev10d hit 499 with envs=64), pool workers are slow, or GPU is doing other work. |

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
- **All checkpoints scored poorly on full panel eval** — too passive to attack vs real opponents
- ⚠️ **The foundation checkpoints themselves were bad agents — panel score, not fire[0], is truth**
- But built deep game-sense that the blitz chain exploited

#### 123203 chain → 141208 blitz (best ever)
- `torch_best_123203` (44.5% HB) was produced by an unknown blitz on the foundation
- **141208 run:** resumed from 123203, +1M steps, no IL, no penalty
  - 1M = **55.5% HB** ← peak, best ever
  - 2M = 47.3%, 3M = 46.1%, 4M = 45.3%, 5M = 44.1%
  - **Peaked exactly at 1M, declined every checkpoint after**
- Key: foundation had ~63M of self-play game-sense; blitz injected HB aggression before passivity reset

#### ⚠️ What we don't fully understand about the 1M blitz mechanism
The 45M and 62M foundation checkpoints all scored poorly on the panel. The 1M blitz on 123203
produced a large jump (+11pp HB to 55.5%). But the *same* blitz on 141208 immediately regressed
(-13pp HB to 42.6%). Open questions:
1. **Why did the 1M blitz work on 123203 but not 141208?**
   - 123203 was itself the product of a prior blitz on the foundation — was it at an optimal
     "tipping point" in weight-space that made the HB gradient land well?
   - 141208 had already absorbed the HB signal; a second blitz overfit?
2. **Is foundation depth actually the key variable?**
   - The story is "foundation gives game-sense, blitz injects aggression" — but we haven't
     proven the foundation checkpoints had useful game-sense. They scored 0% vs HB.
   - Alternative: 123203 was good *despite* being on a bad foundation, not *because* of it.
3. **Fire[0] is a proxy; panel eval is ground truth.**
   - We've been killing runs based on fire[0] decline, but the 62M foundation had terrible
     fire[0] AND terrible panel scores. A run could have healthy fire[0] but still be a bad agent.
   - Implication: **run panel eval on every 1M checkpoint**, don't just trust fire[0].

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
- **Best checkpoint: 1M** (`torch_step_1015808_20260530_104829.pt`) — HB=40.2%, Zach=50.8%, Suneet=59.8%
- Ran to 4.6M before kill (3M hard cap was not enforced automatically)

#### Rev8 (killed 2026-05-31, resume rev7 1M, shaping-coef)
- **Single delta from rev7: `--shaping-coef 0.05`**
- Rationale: per-step material-delta reward to incentivise attacking over passive fleet-building
- Panel at 80 games: **HB=30%** (vs rev7 1M baseline 40.2%) — clear regression
- fire[0] hit 0.22 by iter 45 (1.5M steps), killed early
- H_fire stayed ~0.09 throughout — entropy unchanged, passive mode unchecked
- **Verdict: failed.** Shaping rewards material gain, which can be achieved passively (collect more planets); did not incentivise firing specifically
- **Best checkpoint: 1M** (`torch_step_1015808_20260530_184448.pt`) — HB~30%, worse than rev7

#### Rev9 (killed 2026-05-31, resume rev7 1M, entropy-coef-fire)
- **Single delta from rev7: `--entropy-coef-fire 0.05` (5× default 0.01)**
- Rationale: LB episode analysis showed bimodal behavior — same checkpoint fires at 0.47 rate in wins vs 0.22 in losses
- H_fire rose from ~0.09 to 0.121–0.153 at 1M (entropy working mechanically)
- Panel at 1M (112/192 games): **HB=25%, Zach=42%** — worse than rev7 AND rev8
- **Verdict: failed.** Entropy coef raises H_fire but does not improve strategic play.
  - Higher entropy = more random firing, not more strategic firing
  - HB and Zach are rule-based opponents that punish inefficient/random attacks
  - The bimodal fire behavior is game-state-driven (not entropy-driven); entropy can't fix it
  - The passive mode in losses reflects strategic holding in specific positions, not entropy collapse
- Pattern: rev7→rev8→rev9 on HB = 40.2%→30%→25% — each reward/entropy tweak regresses
- **Best checkpoint: 1M** (`torch_step_1015808_20260530_193824.pt`) — HB~25%, worst so far
- Lesson: stop tuning reward signals and entropy. Try training setup changes instead.

#### Rev10 (killed 2026-05-31, resume rev7 1M, longer rollouts)
- **Attempted deltas:** rollout_steps=512 (OOM × 3 attempts), final working config: `num_envs=64, rollout_steps=512, num_minibatches=32`
- Note: 512×512 and 256×512 both OOM'd on L4 23GB; 64×512 = same 65K buffer size as rev7, worked
- SPS=499 (vs rev7's 680 — fewer envs = slower)
- **Panel results at 1M:** HB=28.1%, Zach=46.9%, Suneet=53.8% — all below rev7 1M
- **Panel results at 2M:** fire[0]=0.32, avgfleet=88, passive drift 1.5-2M
- **LB scores:** rev10d 1M = **772.4**, rev10d 2M = **600.0**
- **Key finding from local replay analysis:**
  - Rev10d fires MORE aggressively than rev7 in losses (fire_rate 0.479 vs 0.387)
  - vs HB: we WIN by playing PASSIVELY (fire_rate=0.238 in wins, 0.479 in losses) — HB is a rush aggressor
  - More aggression = worse vs HB panel, better on LB
  - **HB is the WRONG panel metric** — it rewards passivity which is anti-LB strategy
- **Verdict:** Longer rollouts did not clearly help. Panel misleading due to HB. LB: rev10d 1M (772) > rev7 1M (750) — slightly better.
- Export bugs fixed during this run — all prior Phase 1 LB submissions were broken (scored ~87)

#### Rev11 (active 2026-05-31, resume rev7 1M, pure self-play 10M)
- **Key deltas from rev7:**
  1. **No external opponents** — pure self-play only (HB was wrong opponent, rewarding passivity)
  2. **10M step target** — never ran past 3M; top-10 team ran 600M
  3. **pool-max-size=40** (doubled) — more diverse self-play partners
  4. **fire[0] kill threshold 0.20** (vs 0.25) — allow passive phases rather than killing them
- Resume: `torch_step_1015808_20260530_104829.pt` (rev7 1M, 32-bin absolute)
- SPS=674, clip_frac=0.213 (still drifting down — healthier than all previous runs)
- **1M checkpoint:** fire[0]=0.32, avgfleet=78.1 — comparable to rev7 1M
- **Post-1M behaviour:** avgfleet oscillating 78-85 (NOT monotonically climbing like rev7-rev10)
- clip_frac=0.213 is lowest steady-state of any run — policy settling more stably
- **Submission plan:** not submitting today (4/5 slots used). Submit 3M or 5M checkpoint tomorrow if fire[0] holds.
- **Success criterion:** LB score > 894 (beat 141208 old arch)

---

## External Reference: Top-10 Team Approach (discussion/697725)

Team "Light" + Claude Opus (top-10, was #1 briefly). Posted 2026-05-31. Read the full post.

### Their setup vs ours

| Item | Theirs | Ours (rev9) |
|---|---|---|
| Environment | **JAX rewrite** | Vectorised PyTorch GPU |
| SPS | **~10,000** (basic) → ~2,000 (complex) | ~685 |
| Total steps | **600M** (3 days, RTX 5090) | ~22M effective |
| Model size | ~600K params | 404K params |
| Architecture | Entity transformer | Entity transformer |
| Self-play | **Pure** — no external opponents | Mixed + HB 5% |
| Reward | **`+1/-1` only** (2p mode) | +1/-1 + win-margin coeff |
| Rollout steps | **512** | 64 |
| Num minibatches | **1** | 32 |
| Grad clip | **99** | 0.5 (standard) |
| GPU cost | ~$150 (5090, 3 days) | ~$50 so far |

### Key quotes

> *"About 100M samples with pure self-play should beat all public agents by 90%"*

> *"clip_frac starts creeping up monotonically (0.10 → 0.30+) before entropy_fire collapses or KL spikes. When you see that creep, cut lr or revert capacity. Don't wait for the blow-up."*

> *"+1/-1 is enough for 2p mode."*

> *"Forget sample efficiency. You are doing RL. [Fast environment] is non-negotiable."*

> *"We don't have scale. So, put as many inductive biases as possible."*

> *"Add one architecture delta at a time. Always."* (they violated this, broke training, had to recover)

> *"entropy is like 50% of max"* — far higher than our H_fire ~0.11

### What this means for us

1. **SPS is the primary bottleneck.** 685 vs 10,000 = 14× slower. 100M steps takes us ~41 hours; takes them ~3 hours. A JAX rewrite is the path to their training scale, but it's a large project.

2. **`rollout_steps=512` worth trying** (rev10 candidate). Longer rollouts = better credit assignment for sparse terminal reward. Directly comparable experiment, cheap delta.

3. **`num_minibatches=1`** — full-rollout batch updates. More conservative gradient steps. Worth pairing with longer rollouts.

4. **Pure self-play** — they reached top-10 without any external opponent. Our HB external may be adding overfitting risk not signal.

5. **Entropy 50% of max** — they run with far more entropy than us. Consistent with our rev9 hypothesis (entropy suppresses passive collapse). Their target is much higher than our H_fire~0.12.

6. **Current training chain is structurally sound** — same entity transformer, same PPO + self-play, same "one delta at a time" rule. The gap is scale (600M vs 22M) and environment speed.

### What NOT to copy

- JAX rewrite: too large a change to validate mid-competition
- Their exact features: not shared publicly
- `num_minibatches=1` alone without testing: changes gradient dynamics significantly

---

## ⚠️ HB Is The Wrong Benchmark (discovered 2026-05-31)

Replay analysis of local games (10 games each, rev7 and rev10d vs HB) revealed:

| | Rev7 WINS vs HB | Rev7 LOSSES vs HB | Rev10d WINS vs HB | Rev10d LOSSES vs HB |
|---|---|---|---|---|
| fire_rate | **0.185** | 0.387 | **0.238** | 0.479 |
| n_steps | **500 (timeout)** | 243 | **500 (timeout)** | 186 |

**We beat HB by playing PASSIVELY.** HB is a rush aggressor. We win by holding ships and outlasting it on timeout. We lose when we attack too much (HB captures our empty planets).

This is the **opposite** of LB strategy. LB analysis showed:
- LB wins: fire_rate=0.468, shorter games (232 steps)
- LB losses: fire_rate=0.223, longer games (281 steps)

**Consequence:** Every experiment that increased fire rate (shaping-coef, entropy-coef) improved LB strategy but hurt HB panel score. This explains why rev8/rev9 showed lower HB scores despite training metrics looking healthier. We were optimizing against the wrong opponent.

**Rules going forward:**
1. **Do NOT use HB panel score for kill/keep decisions.** HB rewards passivity which is the opposite of LB strategy.
2. **Use Zach and Suneet panel scores** — they are better proxies (both are on LB, play more strategically).
3. **LB submission score is ground truth.** Panel is for directional guidance only.
4. **Rev10d interpretation:** HB 25.9% (we fire more = worse vs HB), Zach 44.9%, Suneet 65.6%. Net vs rev7: Zach -6pp, Suneet +6pp. Roughly a wash. LB score will decide.

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
bc_phase1_warmstart → rev1/2/3 → rev4 → rev5 (6M peak) → rev7 (1M peak=40.2% HB)
  → rev8 (failed, shaping-coef) → rev9 (active, entropy-coef-fire=0.05)
```

**Rev10 candidate (if rev9 1M looks healthy):** `--rollout-steps 512 --num-minibatches 4`
- Rationale: top-10 team uses rollout_steps=512 vs our 64; longer rollouts improve credit assignment for sparse terminal reward
- Single delta — keep everything else from rev9
- num_minibatches=4 (not 1 — their extreme is untested; 4 keeps batch size sane)

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
