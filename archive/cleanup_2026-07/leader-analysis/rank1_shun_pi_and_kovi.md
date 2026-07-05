# Leader Analysis — Shun_PI (rank1) & kovi

**Source:** `leader-replays/rank1/` — 116 1v1 games  
**Players:** Shun_PI (1354 score, was rank1 when collected), kovi (1412 score, rank17, beats rank1)  
**LB context (2026-05-31):** rank1 Isaiah @ Tufa Labs = 1751; top-100 threshold = 1153

---

## Win rates

| Player | Games | W | L | WR |
|--------|-------|---|---|----|
| Shun_PI | 48 | 40 | 8 | **83%** |
| kovi | 77 | 76 | 1 | **99%** |

---

## Opening pattern (steps 1–64)

Both players open almost identically:

| Metric | Shun_PI wins | Shun_PI losses | kovi wins |
|--------|-------------|----------------|-----------|
| First fire step (avg) | 3.3 | 4.1 | 4.2 |
| First fire step (median) | 2 | 2 | 3 |
| Fire within step 5 | 37/40 (93%) | 6/8 (75%) | 64/76 (84%) |
| First fire ships (avg) | 13.1 | 13.2 | 13.9 |

**First fire ship size is nearly identical across wins and losses (~13-14 ships). The difference is purely timing.**

---

## Planet expansion snowball (avg planets owned)

| Step | Shun_PI wins | Shun_PI losses | kovi wins |
|------|-------------|----------------|-----------|
| @10 | 1.8 | 1.5 | 1.6 |
| @20 | 3.2 | 2.8 | 3.1 |
| @30 | 5.2 | 4.6 | 4.9 |
| @40 | 7.5 | 6.1 | 7.5 |
| @64 | **13.9** | 11.0 | **14.4** |

Both players reach ~14 planets by step 64 in wins, starting from 1. That is roughly one new planet captured every 4–5 steps — driven by firing from each newly captured planet as soon as it produces enough ships.

---

## What "decisive decision making" actually looks like

It is not big strikes or high fire rate. It is **small, fast, continuous probing that compounds**:

1. **Step 2–3:** Fire ~13 ships at the nearest neutral. This is just enough to capture it.
2. **Step 5–7:** Probe lands. Now 2 planets producing simultaneously.
3. **Step 7–9:** Fire again from both planets toward the next neutrals.
4. **Step ~12:** 3 planets. Fire from all three.
5. **Step 20:** 3+ planets. Opponent with 1–2 planets cannot catch up in production.

Each captured planet shortens the time to the next capture. The advantage compounds geometrically. One probe landing at step 5 instead of step 7 is worth 2 extra production cycles before step 20 — that difference propagates through the entire game.

---

## Why kovi beats Shun_PI (99% vs 83%)

The opening is the same. The difference documented in training.md:
- kovi avg ships/attack = **45.7** vs Shun_PI **31.7**
- kovi closes games faster once ahead — commits larger forces at decision points to prevent opponent recovery

kovi does not fire more often; it sends larger forces when it commits, which denies the opponent a chance to stabilize.

---

## Contrast with our agent (141208, 894 LB score)

From `submission_analysis/53076736_894.md`:

- 141208 first fires at **step 20** (28 ships) in upset losses vs opponent firing at step 3
- By the time 141208 fires once, rank1 already has 3 planets
- 141208 learned "save and strike" — accumulate a big fleet, then fire — instead of small continuous probes
- The terminal ±1 reward cannot distinguish a planet captured at step 3 vs step 20. The compounding production value is invisible to the agent.

---

## Key training implication

The entire game is decided in the first 20 steps by whether the early probe compounds correctly. A shaped reward for **planet count above starting count, weighted by how early it is captured**, would give gradient signal exactly where the terminal reward is blind.

Target behavior: fire ≤ step 5 with ~13 ships, then fire continuously from each new planet as soon as ships are available.

---

## Multi-source firing (srcs_multi) targets

How many planets fire simultaneously per step:

| Metric | kovi wins | Shun_PI wins | Rev24@5M (carpet-bomb) |
|--------|-----------|--------------|------------------------|
| mean srcs/fire step | **1.51** | **1.65** | ~6.84 |
| p90 srcs | **2.7** | **3.0** | carpet-bomb |
| max srcs ever | **4.7** | **4.0** | 6.84+ |

Rank1 fires from 1-2 sources on average. They occasionally coordinate 3-4 sources simultaneously at key moments, but never sustain it. The srcs_multi=4.0 warning threshold in training is well-calibrated to this data — above 4 sustained means degenerate multi-source spray, not tactical coordination.

---

## Shaped reward design — coefficient sanity check

Proposed reward: `+coeff * max(0, 1 - step/100)` for each planet owned above starting count, applied per step.

**Cumulative return for a single planet captured at step 3 and held to step 100:**

$$G_{\text{shaped}} = \sum_{t=4}^{100} \text{coeff} \cdot \left(1 - \frac{t}{100}\right) \approx 97 \times 0.48 \times \text{coeff}$$

| coeff | Cumulative bonus (1 planet) | As % of terminal win |
|-------|----------------------------|----------------------|
| 0.02  | ~0.93 | **93% — swamps terminal reward; agent becomes expansion bot** |
| 0.01  | ~0.46 | **46% — still too large** |
| 0.003 | ~0.14 | 14% — upper edge of acceptable |
| **0.0025** | **~0.12** | **12% — recommended start** |
| 0.002 | ~0.09 | 9% — lower edge |

**Rule:** total cumulative bonus for a perfect opening should be ≤ 10–15% of a terminal win. At 0.0025, a step-3 capture yields ≈ 0.11 — loud enough for GAE to propagate back to step 3, small enough that a terminal loss still dominates the gradient.

**Recommended coeff: 0.0025** (range 0.002–0.003).
