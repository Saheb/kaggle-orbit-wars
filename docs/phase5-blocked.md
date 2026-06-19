# Phase 5 — BLOCKED by loss-mode audit (2026-06-19)

**Status:** Phase 5 build is **paused at Step 4**. A loss-mode audit (run after a Phase 5 design
review) shows the phase's core thesis — *synchronized-wave concentration / arrival timing* — is
**not** the loss mode it was built to fix. Steps 0–4 are committed (`ebde763`); Steps 5–7 (wave
planner, gates, from-scratch BC→PPO) are **not** justified by the evidence and should not be built
as specced.

This doc records the audit, the evidence, what survives, and the candidate directions. No
direction has been chosen yet.

---

## TL;DR

- The Phase 5 spec assumes we get out-massed because our mass arrives **staggered** (far sources
  don't launch early, near sources don't wait), and builds heavy machinery (`tol` windows,
  `ready_now`, ETA/count coupling, from-scratch heads) to make fleets **arrive together**.
- **Measured: the "arrived too late / staggered" loss mode (`TOO-LATE`) is ~0–1% of lost captures.**
  We almost never have reinforcement inbound that misses the window. We simply **don't send it** —
  and usually have nothing nearby to send.
- The real wall is the **mid-game `planets@50` stall + economy snowball**: even on mass at step 25,
  we stall at ~6 planets through steps 25–75 while winners climb to 8→10+, then get snowballed
  (18/19 game losses are **eliminations**, material→0).
- The only Phase-5 component on-target is **"size a strike to cross the reactive floor"** — but
  decisive-mass *as a reward* was already **confirmed negative** ([[project_decisive_mass_lever]]),
  and this wall is a **training-signal** problem, not a feature/architecture gap
  ([[feedback_win_starvation]], [[project_force_concentration_wall]]).
- Therefore the **from-scratch rebuild (Step 4) and arrival-timing planner (Steps 5–7) are hard to
  justify.** Arrival synchronization addresses ~1% of our losses.

---

## How the audit was run

The audit subject is our **existing trained policy**, not the (untrained) Phase 5 model — a fresh
model's losses are noise. The Phase 5 branch can't load Phase-4 checkpoints (Step 4 changed the
architecture), so the audit ran in an **isolated git worktree off the last pre-Phase-5 code commit
`8c8f6ce`** (`F_pair=16`, `MAX_OWNED=16`, prior+residual heads — exactly how these checkpoints were
trained/evaluated). The Phase 5 branch was untouched.

Checkpoints audited (both load faithfully at `8c8f6ce`; both verified ~100% vs random first):
- **phase4e 3.67M** — the actual Phase 4 lineage arm whose inert-fire-residual result *motivated*
  the from-scratch decision (`F_pair=16`, `sufficient_commit=1.0`, matches `8c8f6ce` exactly).
- **corrpack3e 4.7M** — CLAUDE.md's documented best-Ajay baseline (`F_pair=15`, zero-pad loaded).

Tools (in the audit worktree, NOT on the Phase 5 branch):
- `orbit_wars_rl/hold_autopsy.py` (existing) — ABANDONED / OUT-MASSED / TOO-LATE / OTHER.
- `orbit_wars_rl/audit_lossmode.py` (new) — decomposes OUT-MASSED into economy / position /
  concentration, plus a mass-ratio + planet-count trajectory split by eventual outcome.

Reproduce:
```bash
git worktree add /tmp/orbit-audit 8c8f6ce
cd /tmp/orbit-audit
CUDA_VISIBLE_DEVICES="" python3 orbit_wars_rl/audit_lossmode.py \
  --checkpoint <abs path to phase4e or corrpack3e .pt> \
  --opponent opponents/candidate_ajay_1200.py --games 24 --gate 2
```

---

## Findings

### Loss-mode breakdown (lost captures vs Ajay)

| metric | corrpack3e 4.7M (24g) | phase4e 3.67M (24g) | phase4e OFFICIAL 256g panel |
|---|---|---|---|
| captures / lost / peel-rate | 616 / 495 / 0.80 | 648 / 509 / 0.79 | 9900 caps, peel 0.69 |
| OUT-MASSED | 95.6% | 94.3% | **94%** |
| **TOO-LATE (staggered arrival)** | **0.2%** | **0.8%** | **1%** |
| ABANDONED | 2.4% | 1.4% | 2% |
| garrison@loss vs enemy-inbound (median) | 27 vs 64 | — | 24 vs 54 |

The audit reproduces the official panel (out-massed 94%, too-late 1%) → the integration is faithful.
**`TOO-LATE` — the exact failure Phase 5's wave-synchronization exists to fix — is ~1%.**

### OUT-MASSED decomposition (loss-moment; confounded by late-game collapse)

| bucket | corrpack3e | phase4e | definition |
|---|---|---|---|
| ECONOMY | 44.8% | 44.2% | global mass ratio < 0.8 (median 0.43–0.47 — half the enemy's mass) |
| POSITION | 44.2% | 45.2% | global ok but no mass reachable to the target in time |
| CONCENTRATION | 11.0% | 10.6% | reachable uncommitted mass ≥ deficit (Phase 5's target) |

Median OUT-MASSED loss had **reachable = 0, 0 source planets** → mostly *not* "had the mass, failed
to bundle it." Concentration-at-loss-moment is a weak, late-confounded signal.

### Mass-ratio trajectory (our total / enemy total, mean by eventual outcome) — the decisive cut

phase4e:

| step | WIN ratio (planets) | LOSS ratio (planets) |
|---|---|---|
| 25 | 1.42 (3.6) | 0.90 (3.4) — roughly even |
| 50 | 1.04 (7.6) | 0.74 (6.6) — gap opens |
| 75 | 1.36 (10.8) | 0.60 (6.2) — **we stall at ~6 planets** |
| 100 | 1.99 (10.0) | 0.39 (4.2) — collapse |
| 150 | 142 (18.7) | 0.25 (2.7) — elimination |

corrpack3e is identical in shape (loss: 1.03 → 0.78 → 0.57 → 0.36 → 0.21; planets stall ~6 then
collapse). **The gap opens in the mid-game (steps 25–75), driven by failure to keep expanding /
holding planets — not the opening (even at 25), not arrival timing.** This is the documented
`planets@50 ≈ 6–7` wall ([[project_phase3_compass_wall]]).

### Corroborating signals already in phase4e's panel log

- `outmassed by planets@32: <=4 → WR 45% · >=6 → WR 58%`, annotated *"opening expansion looks
  UPSTREAM"* — **+13% WR at the same out-massed rate** when we have more early planets.
- `decisive-mass cross 0.61, overkill 6.04` — we **overkill** some contested targets 6× while only
  crossing the floor on 61% (mis-allocation, not arrival timing).
- `reinf by step 0.18 / 0.36 / 0.58` vs winner `0.29 / 0.41 / 0.31` — we **under-reinforce early,
  over-pour late** into hopeless planets (HOPELESS 9%; 95% of to-lost mass wasted).

---

## What this means for the Phase 5 components

| Component | Verdict |
|---|---|
| Arrival synchronization (`tol`, `ready_now`, ETA/count coupling, wave anchor) | **Dead** — solves ~1% of losses. |
| From-scratch rebuild (Step 4: direct heads, NO_OP, no resume) | **Unjustified** — discards working PPO to fix a non-bottleneck. |
| Wave planner + gates (Steps 5–7) | **Do not build as specced.** |
| Wave *features* (Steps 2–3, pairwise 21:40) | **Reusable** as inputs, but only the size/floor channels are on-target. |
| "Size a strike to cross the reactive floor" (decisive-mass floor) | **On-target idea, but** decmass-as-reward already **negative** ([[project_decisive_mass_lever]]); the wall is a *signal* problem. |

The mid-game expansion/retention wall + economy snowball is the dominant, unaddressed failure.
Consistent with the whole prior probe history: [[project_force_concentration_wall]],
[[project_aggregation_probe]] (we aggregate at winner rates; wall = sufficiency + retention),
[[project_overextension_probe]], [[feedback_win_starvation]] (planets@50 invariant across 4 levers;
fix the signal, not reward knobs).

---

## Candidate directions (not yet decided)

1. **Warm-start + size-to-floor only.** Resume phase4e/corrpack3e, add *only* the
   size-to-reactive-floor wave features (drop all arrival-timing machinery and the from-scratch
   rebuild), continue PPO. Cheapest test of the one on-target component; keeps existing skill. Also
   doubles as the experiment that would justify/refute from-scratch (if features go inert under
   continued PPO, that's the positive evidence the spec lacked).
2. **Reframe to the `planets@50` / expansion wall.** Attack mid-game expansion+retention via a
   training **signal**/curriculum or opponent-side approach (h14-style: beat a forward-projector so
   out-massing is learnable). The history-favored "fix the signal not features" path. Shelves most
   of Phase 5.
3. **Target the mis-allocation directly.** decisive-mass cross 0.61 / overkill 6.04 + mis-phased
   reinforce (under early / over late into hopeless): spread overkill mass to missed targets, fix
   reinforce phasing. Narrower, evidence-direct.
4. **Proceed full Phase 5 anyway.** Lowest expected value given arrival timing is ~1% of losses.

---

## Artifacts

- Audit worktree: `git worktree add <path> 8c8f6ce` (delete with `git worktree remove` when done).
- New script: `orbit_wars_rl/audit_lossmode.py` (lives in the worktree).
- Memory: [[project_lossmode_audit_phase5]].
- Phase 5 code (Steps 0–4) remains committed on `main` (`ebde763`); nothing here reverts it.
