# Review: Phase 5 — Synchronized-Wave Concentration

## Verdict

The thesis is correct and the engineering discipline (I1–I3, single source of truth, §9 poisoning checks) is the strongest part of your pipeline. But the spec makes **three bets that can sink you at rank 900–1000**:

1. **From-scratch** discards 6M+ PPO steps on the *unverified* hypothesis that the existing architecture can't use wave features.
2. **Concentration is diagnosed as the bottleneck** without loss-mode attribution to confirm it.
3. **Several deterministic tiebreaks** (source-id order, single-pass hold_class) bake in suboptimalities that PPO can't fix because they're in the labels.

If any one of these is wrong, you'll be worse off than Phase 4. Below is the breakdown.

---

## What's strong

- **Single source of truth (I1–I3)** — one `floor` / `choose_anchor` / `hold_class` across planner/features/eval. The §9 poisoning checks enforce it. This is the right invariant.
- **ETA/count coupling (§3.0.1)** — the observation that ETA depends on ship count, *and that this is a control knob* (more ships ⇒ faster ⇒ earlier arrival) is genuinely insightful. The contract is dense but correct.
- **Sticky-then-fresh anchor (§3.3)** — avoids memoryless churn. Good.
- **Deferred `wave_head` / `commit_head`** — restraint in not building learned heads until diagnostics demand is the right call.
- **NO_OP target column** — clean factored-policy handling of no-fire slots; keeps PPO chain rule well-posed.
- **§10 parity contract** — choosing ready-wave quota (masked sums + division) *because* it's parity-trivial is good engineering judgment.

---

## Strategic concerns

### 1. From-scratch is the wrong default

The lineage para says Phase 4's `fire` residual is inert (`flip_f ≈ 0.04`). You read this as "the residual formulation is implicated in inert-fire." The simpler explanation: **the base policy already encodes fire decisions, so the residual has nothing to add**. That's expected behavior, not a bug — inert residuals are what you want when the base is doing its job.

From-scratch throws away 6M+ steps of working PPO on the bet that "heads + features are load-bearing from step 0." But features can become load-bearing during PPO if they're informative — that's the whole point of PPO. If they go inert, that's *diagnostic*: it tells you concentration isn't actually the bottleneck.

**Lower-risk path**: add the wave features (§5) as new inputs to the Phase 4 policy, continue PPO. If features go inert → re-diagnose. If they help → you keep existing skill plus gain concentration. From-scratch is only justified if you have positive evidence the existing architecture *can't* use the features, and you don't.

### 2. The diagnosis isn't grounded

You have "out-massed ~95% in both" from the Phase 4 A/B. You jump to "synchronized concentration is the missing skill." But out-massed has three causes:

- **Economy deficit** — total mass is genuinely lower → no synchronization helps.
- **Positioning** — mass is fine but in the wrong place → waves help, but only somewhat.
- **Timing** — mass is fine and roughly positioned but arrives staggered → waves are exactly the fix.

Only the third case is what Phase 5 solves. The §9 poisoning checks validate the *planner*, not the *loss modes*. Before building, take 50 losses from your current policy and categorize them. If concentration failures aren't dominant, you're building the wrong thing.

### 3. Rank 900–1000 is rarely won by mid-game tactics alone

At that rank, the typical killers are:
- **Opening blunders** (steps 1–15): bad expansion → economy deficit forever.
- **Defense leaks**: losing planets to small raids because reinforce logic is wrong.
- **Late-game doomstack**: when opponent has 10× ships, no tactics matter.

Phase 5 directly addresses none of these. The `MAX_OWNED=24` change helps opening source *visibility* but not opening *logic*. There's no `capture_value` feature for target prioritization — the model has to infer "should I attack here vs. expand elsewhere" from `floor_for_wave` and `wave_feasible_by_D`, which is a weak signal for strategic target choice.

---

## Technical issues in the spec

### A. §6 residual rule uses ascending source-id — arbitrary and harmful

> "shortfall is assigned to ready sources in ascending source-id order, each ceiling to its next legal bin"

Planet IDs are arbitrary. This means planet 0 systematically gets over-asked across all waves. Use **ETA-ascending** (closest sources first — they have the tightest slack to absorb rounding) or **spare-capacity-descending**. Source-id is the worst choice.

### B. §3.4 single-pass hold_class over-counts help

Pass 2 computes `optimistic_reachable_help(P)` from "all other planets' maximum currently legal send capacity." This includes garrisons from other CANDIDATEs that will reserve for their own defense. A CANDIDATE classified "saved" in pass 2 can flip to DOOMED in pass 3 — but pass 2's pool composition was already locked in.

Pass 3 claims greedily and releases failed claims, but a *late-processed* (later-`D_def`) candidate can fail because *early-processed* candidates already ate the pool — even if the late candidate was more defensible. Ascending-`D_def` order mitigates urgency but not pool composition.

Fix: in pass 2, compute `optimistic_reachable_help` from **SAFE planets only**; expand the pool in pass 3 as planets flip to DOOMED. Still no fixed-point, but no over-counting.

### C. Bin schedule is unspecified (§3.0.1)

The entire ETA/count contract depends on bin granularity. Log-spaced bins (1, 2, 4, 8, 16, 32, 64) give wide ETA range but coarse sizing — `n_ontime` could be 2× `ship_target`, causing overshoot. Linear bins give fine sizing but narrow ETA range — many sources won't have a viable bin. Specify the schedule explicitly and add bin-resolution to the §9 checks.

### D. STICKY anchor's `min_material_ships=5` doesn't scale

Late-game on small planets, 5 ships might be the whole garrison — meaning any trickle becomes a sticky anchor and prevents fresh-wave selection. Scale it: `max(5, 0.05 * floor(T))`.

### E. §5 dropped `source_rank` — be careful

`source_rank` told the model "am I the best source for this target?". Without it, every source sees the same target features + its own pair features. The model must infer "am I the best" from `my_safe_sendable`, `eta_fast`, `marginal_needed_from_me` — doable but harder, especially in tie-break cases. Watch the §11 gate for evidence the model can arbitrate; if it can't, restore `source_rank`.

### F. §3.5 quota is proportional, not ETA-aware

`my_quota = remaining * my_safe_sendable / max(ready_safe, eps)` gives every ready source a share proportional to its `safe_sendable`. But a far source (ETA=8) and a near source (ETA=2) have different flexibility — the near source can wait, slow down, or send fewer. Should the near source get a bigger share (it can absorb more of the gap with less risk) or smaller (the far source is more constrained)? This is a design choice that's not explicitly justified. Consider ETA-weighted allocation.

### G. §11 promotion gate is metric-soft

"held-out Ajay up" — by how much? Specify a target (e.g., +5pp win rate vs Phase 4 baseline at 1000 games). Without a number, "up" is unfalsifiable.

---

## What's missing

1. **Head-to-head gate vs Phase 4 baseline.** The §11 promotion gate checks behavioral metrics but not direct win rate. Add: "Phase 5 must beat Phase 4 in a 1000-game head-to-head at ≥55% before promotion." Without this, you can pass all behavioral gates and still be worse.
2. **Loss-mode attribution** (should be §0, before §1).
3. **`capture_value` feature** — `hold_value` exists for owned planets but there's no analog for targets. A simple `production(T) * min(remaining_steps, value_horizon)` gives the model target priority signal. Without it, target choice is driven by `floor_for_wave` and `wave_feasible_by_D` — necessary but not sufficient for "is this target worth attacking at all?"
4. **Opening / early-game logic.** Wave machinery doesn't help in steps 1–15. Either acknowledge this gap or add opening-specific features (nearest-neutral distance, expansion-rate).
5. **Opponent modeling.** At rank 900–1000, opponents have predictable patterns. A simple enemy-aggression-rate or enemy-expansion-rate feature would help defense planning.
6. **Computational budget and deadline.** From-scratch BC+PPO is expensive. If you have a competition deadline, from-scratch might not finish in time. This needs explicit treatment.

---

## Recommendations (ordered by leverage)

1. **Before any build**: do loss-mode attribution on 50 current losses. If concentration isn't dominant, stop and re-plan.
2. **Don't go from-scratch.** Add wave features to Phase 4 policy, continue PPO. Treat inert features as diagnostic, not as a reason to rewrite.
3. **Add the head-to-head gate.** ≥55% vs Phase 4 in 1000 games, no exceptions. This is the only gate that actually answers "should I promote?"
4. **Fix the source-id tiebreak** in §6 → use ETA-ascending.
5. **Fix the §3.4 over-counting** — pass-2 help should come from SAFE planets only.
6. **Specify the bin schedule** in §3.0.1 and add bin-resolution to §9 checks.
7. **Add `capture_value`** to §5.
8. **Keep the reinforce overlay as a fallback** until the policy defends in real games, not just synthetic states. Retiring it on synthetic-state evidence alone is risky.
9. **Add opening-specific features** if opening is a loss mode.

---

The spec is the best-engineered thing in your pipeline and the thesis is sound. But the from-scratch decision and the unverified diagnosis are the two bets that can sink you. Resolve those first — a 50-game loss-mode audit takes an afternoon and could save you two weeks of building the wrong thing.

Want me to draft the loss-mode attribution protocol (categories, replay-sampling scheme, decision rules), or sketch the lower-risk "add features to Phase 4, continue PPO" alternative in more detail?