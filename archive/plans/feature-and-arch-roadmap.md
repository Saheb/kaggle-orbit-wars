# Feature & Architecture Roadmap

**Written:** 2026-05-27  
**Context:** delta1_arrival_eta running (55.5% → ? vs Hellburner). Target: >75% all three opponents simultaneously.

---

## Where We Are

| What | Value |
|------|-------|
| Model params | ~350k |
| entity_dim | 96 |
| num_layers | 3 |
| num_heads | 4 |
| planet_feature_dim | 18 |
| fleet_feature_dim | 9 |
| global_feature_dim | 10 |
| pairwise_feature_dim | 10 |
| max_owned_planets | 10 |
| Action decode | target (48-planet selection) |

The delta1 run added arrival-corrected pairwise features (sin/cos to predicted arrival pos, ETA, sun-safe, ownership flags, production). That closes the biggest geometry gap.

---

## Feature Gaps

### Gap 1 — Fleet destination unknown *(High impact, Medium effort)*

Fleet features are `[x, y, angle_cos, angle_sin, ships, speed, owner, dist_sun, active]`. There's no indication of **which planet a fleet is heading for** or **when it arrives**.

`friendly_pressure` / `enemy_pressure` on planet features uses a rough geometric heuristic — is the fleet aimed vaguely at this planet? Not actual destination.

**Consequence:** The model can't properly evaluate defense urgency. It has to back out destination from angle+position geometry, which the transformer would struggle to learn reliably.

**Fix:** Add to fleet features (fleet_feature_dim 9 → 13):
```python
target_planet_idx_norm  # target_pid / max_planets (normalized)
eta_normalized          # eta / 500.0
dist_remaining_norm     # remaining dist / BOARD_SIZE
is_threatening_owned    # bool: fleet heading toward one of your planets
```
Requires resolving fleet destination in `extract_fleet_features`. The game obs has `angle` which points at the target; combined with known planet positions, exact destination can be decoded geometrically (find closest planet along trajectory).

### Gap 2 — Owned planet cap at 10 *(High impact, Easy)*

```python
max_owned_planets: int = 10  # hard cap in config + storage
```

Late-game you can own 15-20 planets. Planets beyond slot 9 are **completely invisible to the action head** — no fire decision is possible for them. This is a silent correctness bug, not just missing signal.

**Fix:** Bump `max_owned_planets` to 16 (or 20). Requires:
- `config.py`: `max_owned_planets = 16`
- `train_torch.py` storage: angle_mask buffer `(T, N, P, 16, 144)` — slightly larger
- `action_mask.py`: already dynamic, just pass new max
- Must retrain from scratch (changes model architecture / action head size)

### Gap 3 — Planet position prediction uses fixed 5-turn horizon *(Medium impact, Easy)*

```python
future_angle = init_angle + angular_velocity * (step + 5)
```

Hardcoded 5-turn lookahead for the `pred_x, pred_y` planet features (indices 10-11). But real fleet ETAs range from 2 to 50+ turns depending on distance and ship count. A planet 40 units away won't be where the 5-turn prediction puts it.

The pairwise features already compute ETA-matched arrival position correctly. The single-entity planet features still use the stale approximation.

**Fix:** Pass per-planet ETA into `extract_planet_features`. For owned-planet self-prediction, use ETA from nearest source. For general planets, use median ETA or remove the fixed-horizon prediction entirely and rely on the pairwise path (which already covers it).

### Gap 4 — Production-over-time valuation missing *(Medium impact, Easy)*

The model sees `production` (per turn) and `ships` (current count) separately. It has no feature encoding:
```python
ships_at_eta = ships + production * eta_to_here
```

An enemy planet with 10 ships + production=5 at ETA=8 will have ~50 ships when you arrive, not 10. The model has to learn to multiply `production × (1/close_fast_preference)` implicitly — hard.

**Fix:** Add to pairwise features (dim 10 → 12):
```python
out[slot, :n_p, 10] = (tgt_ships + tgt_prod * eta) / 200.0  # ships at arrival
out[slot, :n_p, 11] = tgt_ships_at_arrival / my_sendable     # capture ratio at arrival
```
Pure compute, no env changes. High signal for capture timing decisions.

### Gap 5 — Connectivity / supply line invisible *(Medium impact, Easy)*

`min_owned_dist` (planet feature index 15) captures only the nearest friendly planet. Loses:
- Is this planet isolated or well-supported?
- How many friendly planets can reinforce within N turns?

**Fix:** Add to planet features (dim 18 → 20):
```python
owned_within_15 / 5.0   # count of your planets within range 15
owned_within_30 / 10.0  # count within range 30
```
Fast to compute (vectorize distances). Strong signal for "this is a safe investment vs exposed frontier."

### Gap 6 — Enemy fleet vs planet ship split in global features *(Low impact, Easy)*

```python
total_enemy_ships / 2000.0  # single lump sum
```

Ships on planets (static, defensive) are very different from ships in fleets (committed, can't be recalled). Conflating them loses strategic info.

**Fix:** Split into two features in `extract_global_features` (dim 10 → 11):
```python
enemy_ships_on_planets / 2000.0
enemy_ships_in_fleets  / 2000.0
```

### Gap 7 — No trend / momentum features *(Medium impact, Hard)*

All turns are fully independent. No sense of "you just captured a planet" or "enemy is collapsing." Step-based features tell the model *when* it is but not the *trajectory*.

**Skip for now** — requires carrying history across turns, which means env changes. Not worth complexity until other gaps are closed.

### Gap 8 — No fleet convergence signal *(Medium impact, Hard)*

Two enemy fleets converging on the same planet are far more dangerous than one (near-impossible to intercept). Model sees them as two separate entities.

**Skip for now** — requires O(N²) fleet-fleet comparison. Complex, deferred.

---

## Architecture Gaps

### A1 — Value head too weak for PPO *(High impact, Easy)*

The value head mean-pools all entities, losing spatial structure:
```python
pooled = (x * valid_float.unsqueeze(-1)).sum(dim=1) / valid_float.sum(...)
value = self.value_out(F.gelu(self.value_fc2(F.gelu(self.value_fc1(pooled)))))
```

In PPO, `explained_variance` of the value function directly controls advantage quality. Better value → better advantages → better policy gradient.

**Fix:** Concat global token (attends to everything) + owned planet pool (your position specifically):
```python
global_token = x[:, 0, :]                    # (B, D) — already attends to all
owned_pool   = owned_entities.mean(dim=1)     # (B, D) — your planets
value_input  = torch.cat([global_token, owned_pool], dim=-1)  # (B, 2D)
# value head: Linear(2D → D) → Linear(D → D//2) → Linear(D//2 → 1)
```
Breaks checkpoint compat (adds new weight), but very cheap and high ROI.

### A2 — Model too small for problem complexity *(Medium impact, Medium effort)*

~350k parameters, 3 layers of 96-dim attention, is thin for a game with up to 48 planets, 128 fleets, and 10 owned planets each needing coordinated decisions.

**Conservative scale-up (wait for evidence of plateau first):**
```
entity_dim:  96 → 128   (+33%)
num_layers:   3 → 4     (+1 reasoning step)
num_heads:    4 → 4     (head_dim 24 → 32, better)
```
New total: ~750k params (2×). Safe for PPO. Going beyond 256-dim or 6 layers without strong evidence of plateau is premature.

---

## Prioritized Plan

### Phase 0 — Current run (delta1_arrival_eta)
*No action needed — let it run. Baseline for all comparisons.*

Hypothesis: arrival-corrected pairwise features improve Hellburner score because the model can now correctly predict where orbiting planets will be and learn intercept timing. Watch for improvement beyond 55.5% on the 2M checkpoint panel.

---

### Phase 1 — High-ROI features, no architecture change
*Can start preparing now, launch after delta1 panel results arrive.*

| Change | Files | Breaks compat? |
|--------|-------|----------------|
| Fix owned planet cap (10 → 16) | `config.py`, storage buffer | Yes — retrain |
| Better value head (concat global + owned) | `model.py` | Yes — retrain |
| Fleet destination features (9 → 13) | `features.py`, `config.py` | Yes — retrain |
| Ships-at-arrival in pairwise (10 → 12) | `features.py`, `config.py` | Yes — retrain |
| Connectivity features on planets (18 → 20) | `features.py`, `config.py` | Yes — retrain |
| Enemy fleet/planet split in global (10 → 11) | `features.py`, `config.py` | Yes — retrain |

All of these break checkpoint compat (feature dims change), so bundle them all into **one new training run** from scratch (with BC warm-start since the model is fresh). Don't mix and match across runs.

BC first → then PPO with pool. The BC target will need to be updated for `fleet_feature_dim=13` input.

---

### Phase 2 — Architecture scale-up
*Only launch if Phase 1 plateaus before reaching target.*

- `entity_dim: 96 → 128`, `num_layers: 3 → 4`
- BC warm-start again (new architecture, no compat with Phase 1 checkpoint)
- Hypothesis to validate: larger model learns better multi-planet coordination

---

## Decision Rule

**Don't move to Phase 1 until delta1_arrival_eta panels confirm or deny the arrival-correction hypothesis.** One delta at a time — if delta1 hits 65%+ vs Hellburner, the geometry fix is working and Phase 1 is additive. If delta1 stalls at ~55%, there's a different bottleneck and we diagnose before adding more features.

---

## What NOT to do

- Don't scale up architecture and add features in the same run (can't attribute)
- Don't increase `external_fraction` to 1.0 again — it caused regression (CLAUDE.md lesson 1)
- Don't run BC without confirming `target_top1 ≥ 0.30` gate (from `validate_bc`)
- Don't trust quick_eval for go/no-go decisions — full 256-game panel only
