# Phase 2 — Reinforcement-Enabled From-Scratch Run

**Purpose:** teach the agent to *reinforce* (send ships to its own planets) as a native,
empire-size-gated, instrumental behaviour — the #1 structural skill-gap vs the leaderboard
top tier. This is a fresh run with a redesigned reward model + one new action mask, NOT a
resume of the rev55–57 reinforce lineage (all of which flooded).

Status: **design locked, building the empire-size gate.** (2026-06-10)

---

## 1. Why Phase 2 (and why not a resume)

Reinforcement is structurally forbidden in the rev38 lineage: the target mask was
`alive & (owner != me)`, so own planets were never legal targets and the model's `is_mine`
target-input weight was **never trained**. Four attempts to bolt reinforcement onto a mature
base all failed for the same root cause — unmasking own-targets on a policy whose equilibrium
assumes "never reinforce" causes a shock/flood:

| Attempt | What | Outcome |
|---|---|---|
| rev55 | hard unmask at t=0 | over-fire collapse <400K steps |
| BC-seed | teach target head reinforcement | cratered enemy/neutral play (shared scorer) |
| rev56 | annealed logit-bias curriculum | reinforcement emerged (0.73–0.80) but **floods** |
| rev57/b | + garrison-floor + reinforce-cost | still floods |

**Flood mechanism:** reinforcement is *near-costless* (friendly arrival does
`planet.ships += survivors` — can't lose a battle, material conserved). With any fire incentive
(`entropy_coef_fire`), the policy satisfies it *for free* by reinforcing — especially at large
empires where capture opportunities are exhausted. Band-aids (curriculum, LR throttle,
reinforce-cost tax) don't hold; taxing a costless action is the Nash-trap lever.

**Decision:** start fresh so reinforcement is acquired *natively* (own-targets legal from step 0
under a gate → the `is_mine` weight trains organically, no mature equilibrium to shock), with an
incentive structure that makes flooding impossible by construction.

---

## 2. What the LB top tier actually does (replay analysis, 2026-06-10)

Direct profile of top-2p replays (06-07/08/09), **timing-corrected** (the action at `steps[t]`
was decided on the obs at `steps[t-1]` — verified 100% vs 9.9% launch validity; same-step pairing
corrupted all prior analysis, esp. ship-commitment). Schemas: planet `[id, owner, x, y, radius,
ships, production]` (owner −1 = neutral, ids global); fleet `[id, owner, x, y, angle,
from_planet_id, ships]`; action `[from_planet_id, angle, num_ships]`.

| Metric | Isaiah (#1, controlled) | Jake Will (#2) | Aggressive cohort |
|---|---|---|---|
| game length (median) | 500 | 167 | 166 |
| planets @ end | ~11 (plateau) | 16 | 20 |
| reinforce @1 planet | **0.00** | **0.00** | **0.00** |
| reinforce @2 | 0.12 | 0.09 | 0.08 |
| reinforce @9–12 | 0.30 | 0.43 | 0.44 |
| reinforce @13+ | 0.34 | 0.38 | **0.61** |
| forward-staging | 68% | 70% | 66% |
| full-garrison commit | 55% | 90% | 78% |
| 1-ship probes | 0.0% | 0.2% | 3.7% |
| prod selectivity (tgt/avail) | 1.17× | 1.15× | 1.17× |
| distance rank (0=nearest) | 0.29 | 0.33 | 0.36 |

**Universal signatures (every top player):**
1. **Empire-size gate** — reinforce ≈0 at 1 planet, ~0.1 at 2, ramps with empire size. *Never*
   reinforce a tiny empire; expand first. This is the most reliable finding.
2. **Forward-staging (66–70%)** — reinforce targets are closer to the enemy than the source.
   Reinforcement flows rear→front (staging), not random redistribution.
3. **Hard commitment** — sends ≥75% of the source garrison ~80% of launches; ~never 1-ship
   probes. Validates `--min-ship-bin`.
4. **Positional targeting** — prefers near targets (rank ~0.3), no production selectivity (1.15×).
   Target head is tempo/reachability, NOT value-ranking.

**Style axis (not universal):** Isaiah grinds (plateau ~11 planets, 500 steps, reinforce ~0.34);
the aggressive cohort snowballs (16–20 planets, ~166-step decisive wins, reinforce up to 0.61).
The earlier "0.74 at 14+" headline was the snowball regime + same-step-timing artifact — heavy
*late* reinforcement is a winning trait; the flood was heavy *early* reinforcement.

**Implications for Phase 2:** gate reinforcement on empire size; do NOT cap `reinforce_rate` at
0.34 (let it climb); reward decisive/fast wins (`speed_coef`); keep the 1-ship ban; do NOT reward
target production. (Note: production-weighted *capture reward* — rev49 — is NOT a validated dead
end; that run had a broken target head. The replay finding is only that top players don't *target*
production, which is separate from whether a prod-weighted reward helps our agent.)

---

## 3. The reward model

Per-step reward for a player (2-player game):

```
r_t =

  ┌─ at game end only ──────────────────────────────────────────────┐
  │  WIN  at step t_win :  +1 · ( 1 + speed_coef · (T_max − t_win)/T_max )
  │  LOSS               :  −1
  └─────────────────────────────────────────────────────────────────┘

  ┌─ every step — dense shaping ────────────────────────────────────┐
  │  + expansion_coef     · Δ( P_mine − P_enemy )           (economy lead)
  │  − defense_coef       · max( 0 , −ΔP_mine )             (production lost)
  │  + early_capture_coef · w(t) · clip( ΔN_mine , −1, +1 ) (planets taken)
  └─────────────────────────────────────────────────────────────────┘
```

- `P` = my total owned production · `N` = my planet count · `Δx` = change since last step
- `w(t) = exp(−2.5·t/T_max) + 0.10`, ×`first_strike_mult` while `t < first_strike_steps`
- `T_max` = 500

In plain terms:
```
r_t = terminal win/loss
    + production-control shaping   (expansion_coef)
    + planet-count opening shaping (early_capture_coef, first-strike + decay)
    − production-loss defense      (defense_coef)
    × speed terminal modifier      (speed_coef, winner only)
```

### Locked baseline config

```yaml
# terminal
terminal_win:        1.0
terminal_loss:      -1.0
speed_coef:          0.3      # reward winning EARLY (snowball/decisive; LB-aligned)
win_margin_coeff:    0.0      # DROPPED — speed replaces margin

# dense shaping
expansion_coef:      0.03     # grow production lead (telescopes → passive ≈ 0)
defense_coef:        0.03     # penalty for production lost → reward HOLDING (the reinforce incentive)
early_capture_coef:  0.3      # capture spike, front-loaded
first_strike_mult:   on
early_capture_decay: on       # exp(-2.5 t/T)+0.10

# reinforcement (masks, NOT reward terms)
allow_reinforce:               true
reinforce_gate_min_planets:    3     # own-targets legal only at >= 3 planets (NEW)
reinforce_garrison_floor:      10    # veto a reinforce that drains source < 10 ships
reinforce_cost:                0.0   # DROPPED — no tax on a costless action

# PPO entropy (in the loss, not r_t)
entropy_coef_target: 0.05
entropy_coef_ship:   0.05
entropy_coef_fire:   0.005    # NOT 0.05 — see §4
```

### The two bets (overturn the rev57 approach, locked)
1. **`reinforce_cost = 0`** — the flood is not a reward problem; rely on the gate + `defense_coef`
   + low fire-entropy. Keep the cost in the back pocket only.
2. **`speed_coef 0.3` replaces `win_margin 0.5`** — time-to-victory rewards efficient snowball wins;
   margin risks rewarding over-extension/carpet-bomb.

### Double-count (intentional)
Losing a 5-production planet hits both `expansion_coef` (Δlead = −10) and `defense_coef`
(−5·coef). Accepted: holding is the behaviour we specifically need to recover. **If the agent
turns too conservative, lower `defense_coef` first — do NOT add a reinforce tax.**

---

## 4. The key design principle: reinforcement has NO reward term

Reinforcement is shaped entirely by **two hard masks on the action space**, never by a reward:

```
own planets are legal targets   ⟺   N_mine ≥ reinforce_gate_min_planets   (empire gate)
a reinforce launch is vetoed     ⟺   it would drain its source < garrison_floor
```

So reinforcing pays off **only instrumentally**: it raises `r_t` by *avoiding* the
`−defense_coef·(production lost)` term (holding threatened planets) and through the terminal win it
enables. It cannot be farmed for free. This is **self-capping** at the data's 0.3–0.6 ramp:
- reinforcing a *threatened frontline* planet → prevents loss → rewarded (and naturally
  forward-staging, matching the 68% measured);
- reinforcing beyond hold-need, or stacking an unthreatened rear planet → zero marginal reward;
- every reinforcing ship forgoes `expansion_coef` capture reward (opportunity cost).

**`entropy_coef_fire` is the one place the old config is actively dangerous.** `fire` is a binary
Bernoulli (max entropy ≈0.69); 0.05 is a hard push toward p≈0.5 firing. Once reinforcement is a
low-risk outlet, that entropy stops being neutral exploration and becomes pressure to emit harmless
same-owner fleets → flood. Start at **0.005** (the masks + dense shaping provide the activity floor).

---

## 5. Implementation

The reward terms (`speed_coef`, `expansion_coef`, `defense_coef`, `early_capture_coef`,
`garrison_floor`, `reinforce_cost`) **already exist** as flags in `train_torch.py` / `torch_env.py`.
The only new build is the **empire-size gate**:

- `torch_env.py` `__init__`: param `reinforce_gate_min_planets: int = 0` (0 = off, backward-compatible).
- `torch_env.py` `get_features` target-mask (`allow_reinforce` branch, ~line 593): own targets
  (`owner == player`) legal only where the env's owned-planet count ≥ threshold; enemy/neutral
  always legal; source always excluded.
- `train_torch.py`: `--reinforce-gate-min-planets` flag, passed to the env config + printed.
- `tests/test_reinforce_mask.py`: a gate test (own-target legality flips at the threshold).

Training-only, like `garrison_floor` — the real Kaggle env has no mask, the policy internalises the
gate. (Eval/export parity in `action_mask.py` is a follow-up to add at export time if we want to
*guarantee* the inference behaviour rather than rely on internalisation.)

---

## 6. Monitoring / dials

Primary dial = the `reinforce_rate` training metric (added rev57b; bias=0 offline read via
`behavior_analysis.py` is decision-grade).

| Watch | Healthy | Action if not |
|---|---|---|
| `reinforce_rate` after N≥3 | 0.3–0.5 (may climb higher late) | **floods** → lower `entropy_coef_fire` first; **disappears** → raise `defense_coef` (never add reinforce-cost) |
| capture / expansion rate | does not collapse | if collapses with reinforce → fire-entropy too low (passivity) |
| avg game length | falls (decisive) | if rises → over-conservative; lower `defense_coef` |
| frontline reinforce share | high (rear→front) | low → gate/defense not inducing staging |
| `clip_frac` | < 0.25 | > 0.25 → halve LR (standing authority) |

---

## 7. Tooling & data

- **`orbit_wars_rl/fetch_analyze_top_replays.py`** — fetch top-score daily replays + behavioural
  analysis (timing-corrected). `--player NAME` / `--exclude NAME` / winner-mode. Metrics: reinforce
  ramp by empire size, forward-staging, ship commitment, prod/distance selectivity, expansion tempo.
- **Style selection without player names:** the daily `manifest.csv` has no TeamNames. Use
  `high avg_score + size_bytes < 3.5MB` as a *snowball-style selector* (short decisive games);
  it excludes Isaiah's 6.6MB/500-step grinds and yields the aggressive cohort (Jake 38%, TonyK,
  213tubo, Boey). Isaiah's grinds dominate plain top-by-avg_score (he is #1).
- Local replay sets (gitignored / `/tmp`): `/tmp/fresh_validate` (180 top-2p games),
  `/tmp/snowball` (89 aggressive-cohort games).

---

## 8. Open decisions / next steps

1. **Build the empire-size gate** (in progress) → unit-test → smoke-test a short run.
2. **Starting point:** fresh PPO from the standard BC warmstart tests "can it learn to reinforce on
   its own" under the new reward model (the original goal). A reinforcement-aware BC warmstart
   (Isaiah controlled vs Jake/cohort snowball — full reinforce ramp, better `is_mine` coverage) is
   the fallback/next iteration if emergence is too slow. **Sequence: reward model → gate → train →
   (BC seed only if needed).**
3. Launch on GCP L4 (one delta = the reward model + gate); first real `reinforce_rate` read early.
