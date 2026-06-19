# Phase 5 — Synchronized-Wave Concentration (from-scratch architecture)

**Status:** locked spec / build started. Steps 0-4 in §14 are implemented: the decisive-mass
floor uses the Phase 5 deadline/window object, source width is `MAX_OWNED=24`, the shared
scalar wave primitives are in place, pairwise wave channels `21:41` are wired with train/eval
parity, and the model now uses **direct per-(source,target) fire/ship heads (no slot prior +
residual)** with a synthetic always-legal **NO_OP target column** at index `max_planets`
(target width `max_planets+1`). `bc.py` evaluates fire/ship at the teacher's chosen target
column and labels NO_OP for valid no-launch slots. The model is checkpoint-incompatible with
the revedge1 lineage by design (from-scratch).

**Lineage:** Phase 4 added per-target fire/ship heads as zero-init residuals on the
revedge1 lineage. The phase4e (deb) vs h14feat (h14) controlled A/B (both 6M, same base,
opposite opponent regimes) converged to **near-identical weights**: `fire` residual inert
(`flip_f ≈ 0.04`, declining, despite `σ≈0.9` and `ρ≈0.42`), `ratio < 1` (undercommit),
out-massed ~95% in both. Conclusion: the opponent regime is not the constraint and the
residual formulation can carry weight without changing behavior. The missing skill —
**concentrating enough mass, arriving together, to cross a contested floor** — is unlearnable
by a factored policy because *no source can observe the bundle the others form*. Phase 5
makes that bundle a **deterministic, state-derived feature** so each source can decide its
role locally, and trains it **from scratch** so the heads + features are load-bearing from
step 0. The ETA-binned `reachable_enemy_mass` feature (pairwise ch15-20, already shipped) is
an input to the floor below.

---

## 1. Thesis

Concentration = **the right mass arriving inside a tight window** to cross a contested floor
before the defense reacts. Two facts make it hard for a parallel/factored policy:

1. **Aggregation is unobservable.** When logits are computed, no source has committed, so a
   source cannot see whether the others will cross the floor. `already_committed_mass` is
   action-dependent and unavailable at decode — the wrong abstraction.
2. **Arrival timing, not launch timing.** A "sufficient" bundle that arrives staggered still
   gets out-massed. Far sources must launch *now* while near sources *wait*, so they land
   together.

The fix is to expose a **state-anchored wave plan** as deterministic features: an absolute
arrival deadline `D_abs` anchored in observable state, the reactive `floor(D_abs)`, how much friendly
mass is already committed to that wave, and each source's **quota** of the remaining gap. Then
"am I needed, and is now my launch moment, and how much do I send" becomes a *local* decision.

We do **not** go autoregressive (keeps PPO/eval/export factored) and we **defer** the learned
`wave_head` and `commit_head`; v1 uses a deterministic anchor and deterministic quota.

Phase 5 v1 also raises source width to **`MAX_OWNED=24` with `MAX_LANES=1`**. This increases
visible source availability for wave feasibility/hold classification without introducing
same-source multi-move lanes. Same-source lanes remain deferred because replay analysis showed
same-source multi-move is sparse relative to same-turn multi-source aggregation.

---

## 2. Core objects

- **Wave** = (target `T`, absolute arrival deadline `D`, `tol`). All contributing fleets aim
  to arrive in `[D − tol, D + tol]`.
- **Attack wave**: take a target `T` (neutral/enemy) by crossing `capture_floor`.
- **Defense wave**: hold an owned planet `P` against the earliest enemy wave that can flip it,
  by crossing `hold_floor`.
- Every planet is classified each step and routed: attack target, defense target (`HOLDABLE`),
  or attack source (incl. `DOOMED` planets draining into attack).

---

## 3. Shared functions — the single source of truth

These have **one implementation each**, imported by (a) the feature path, (b) the BC planner,
(c) eval/diagnostics. Drift between them poisons BC (features say one thing, labels another).
Hard invariant in §9.

### 3.0 Constants
```
wave_tol_steps      = 2        # base tolerance "tol"
max_owned_sources   = 24       # one move/source; no same-source lanes in v1
max_lanes_per_source = 1       # multi-move/source deferred
value_horizon       = 40       # cap for hold_value lookahead (steps)
min_reserve         = 2        # garrison a SAFE source keeps (never leave a planet empty)
safety_pad          = 1        # margin = capture_overhead(+1) + safety_pad
margin              = 2        # = capture_overhead(+1) + safety_pad(1)
material_frac       = 0.10     # min friendly/enemy group mass (vs floor) to count as a wave anchor
min_material_ships  = 5        # absolute floor so tiny floors don't let 1-ship trickle anchor
NO_OP_IDX           = max_planets   # synthetic target column, always legal for valid source slots
tau                 = D_abs − current_step    # TIME-TO-ARRIVAL (steps). Use tau, never D_abs, in all
                                              # time-dependent floor terms (prod·tau, rho(tau), inbound windows).
                                              # D_abs is only the wave's stable identity across steps.
rho(tau)            = clamp((tau − DM_ETA_FREE) / DM_ETA_SCALE, 0, 1)   # reactive ramp (existing dm constants)
```
`tol` interval semantics (do not collapse these to one inequality). `D_abs` is
an absolute arrival step; `tau = D_abs - current_step` is the relative duration
used for production, reach, and `rho`.
```
ready_now(src, D_abs)       : §3.0.1 actual-count readiness, not probe-ETA slack
inbound_counts_for_floor(f) : arrival_step <= D_abs + tol
same_wave_arrival_spread    : <= 2 * tol            # because readiness is +/- tol
```
### 3.0.1 ETA / ship-count contract  [P0 — load-bearing]

Fleet speed rises with ship count (`_ship_speed`), so **ETA depends on how many ships are
launched** — and the current pairwise `eta_to_target` uses a fixed 20-ship probe, which is wrong
for garrison-scale fleets (a 100-ship fleet can arrive ~3 steps before a 20-ship one — larger
than `tol`). If a source is "ready" under probe-ETA, then quota rounds it to fewer ships, it
moves slower and misses `D`. So timing and sizing are coupled and need one basis.

The coupling is also a **control knob**: more ships ⇒ faster ⇒ *earlier* arrival. So a source
can hit a deadline `D` by choosing the right count. The contract:

```
ETA(n)        = dist / _ship_speed(n)           # ALWAYS the actual planned count n, never the 20-probe
eta_fast      = ETA(safe_sendable)              # most ships  → earliest possible arrival
eta_slow      = ETA(min_legal_bin)              # fewest ships → latest possible arrival
arrival_range = [current_step + eta_fast, current_step + eta_slow]

ready_now(s, D_abs) = exists legal bin n <= safe_sendable such that
                      current_step + ETA(n) in [D_abs-tol, D_abs+tol]

# size = max(share, on-time) capped at capacity — resolves the quota>safe case:
n_ontime      = smallest legal ship count/bin with current_step + ETA(n) <= D_abs + tol
ship_target   = min(quota, safe_sendable)                                        # share capped at source capacity
n_want        = max( ship_target , n_ontime )                                    # share AND on-time
n_cap         = largest legal ship count/bin <= safe_sendable                    # capacity
chosen_count  = min( n_want , n_cap )
chosen_bin    = legal bin that emits chosen_count
viable(s)     = chosen_count satisfies current_step + ETA(chosen_count) in [D_abs-tol, D_abs+tol]
               # if n_ontime > n_cap (even full safe arrives too late) → NOT viable this step (defer / it's too far)
               # if quota > safe_sendable (full safe < share) → send full safe; wave under-fills THIS step, the
               #   remaining gap is covered by other ready sources now or by the same set in later steps
               #   (as long as wave_feasible_by_D held when the wave was chosen). This is the partial-wave case.
post_round    : assert viable(s); Σ_ready chosen_count recorded vs (floor − cover)   # parity-critical; log misses
```

`choose_anchor`, the floor's inbound/cover windows, and `ready_now` all use this same
actual-count ETA. Features expose `eta_fast`/`eta_slow` so the achievable arrival range is
visible. This subsumes the rounding rule (§6) — rounding is "smallest bin ≥ ship_target that
lands on time," not a free nearest-bin choice.

### 3.1 floor / cover / remaining (attack and defense are structurally identical)
Both sides are `remaining = relu(floor − cover)`; only the contents of `floor`/`cover` differ.

All time-dependent terms use `tau = D_abs − current_step`, never the absolute step `D_abs`.
```
ATTACK  (target T, not ours), anchor D_abs, tau = D_abs − current_step:
  floor(T)   = garrison_T + prod_T * tau                # garrison at arrival
             + enemy_inbound_to_T arriving <= D_abs+tol # classified, before-deadline only
             + rho(tau) * reactive_enemy_mass_to_T(tau) # enemy PLANET garrison reachable to T within tau
             + margin
  cover(T)   = friendly mass already inbound to T arriving in [D_abs−tol, D_abs+tol]
  remaining  = relu(floor − cover)

DEFENSE (planet P, ours), anchor D_def, tau_def = D_def − current_step:
  floor(P)   = enemy_wave_to_P arriving <= D_def+tol + margin
  cover(P)   = garrison_P + prod_P * tau_def
             + friendly mass already inbound to P arriving <= D_def
  remaining  = relu(floor − cover)          # this IS hold_deficit; never net inbound twice
```
`reactive_enemy_mass_to_T(tau)` is computed from **current enemy planet garrisons** (the
ETA-binned feature, cumsum over bins with reach-ETA ≤ tau). Launched enemy fleets have already
left their garrison, so committed-elsewhere mass is excluded automatically (see §3.2).

### 3.2 classify_enemy_fleet(f, T, D_abs)  — exactly one of three
```
f → T,  arrival_step <= D_abs+tol   ⇒ raises capture_floor(T)       # it will be defending T when we land
f → T,  arrival_step  > D_abs+tol   ⇒ recapture_risk(T)             # post-capture hold problem, NOT capture floor
f → elsewhere              ⇒ committed_elsewhere            # already out of T's reactive mass (left its planet)
```
`committed_elsewhere` needs no floor term (the garrison-based reactive mass already dropped);
it surfaces only as the **opportunity feature** `enemy_committed_elsewhere` (§5) so the policy
can prioritize counterattacks while the enemy is overextended. No double-counting: each fleet
is classified once per `(T, D_abs)`.

### 3.3 choose_anchor — deterministic for v1
```
ATTACK(T):
  STICKY:   group friendly fleets inbound to T by arrival window (width tol). If an EARLIEST
            MATERIAL group exists (group mass >= max(min_material_ships, material_frac * floor(T,D_abs))
            — ignore sub-material trickle, e.g. a stray 1-ship fleet), D_abs = that group's
            arrival window. Continue it;
            do NOT re-select (memoryless churn guard). Later/​sub-material groups are subsequent
            waves or recapture reinforcement, not the anchor.
  FRESH:    else find the tightest feasible wave — the smallest tau such that
            { sources with ETA(safe_sendable) ≤ tau+tol, safe } have total safe_sendable ≥ floor(T,current_step+tau).
            D_abs = current_step + max source-ETA(safe_sendable) in that minimal bundle
            (⇒ the FARTHEST needed source is the anchor and is launchable now with full-safe).
            Far launches first, near waits.
  if no feasible wave for any candidate ⇒ no wave (target not attackable; see §6 target contract).

DEFENSE(P):
  D_def = earliest grouped absolute deadline D_abs (group enemy arrivals by tol) such that
          relu(floor(P,D_abs) − cover(P,D_abs)) > 0.
          # Same cumulative floor−cover as the floor definition (§3.1), NOT per-window mass —
          # two subcritical waves can cumulatively flip P. First step P falls if we do nothing;
          # later arrivals are recapture risk.
```

### 3.4 hold_value, hold_class, safe_sendable
```
hold_value(P)      = production(P) * min(remaining_steps, value_horizon)   # capped; deterministic; shared
reinforce_cost(P)  = remaining0(P)                                         # ships needed to fill the hold deficit
```

hold_class / safe_sendable are mutually recursive (HOLDABLE needs reachable_safe_mass, which
needs safe_sendable, which needs hold_class). Break it with ONE deterministic pass — no
fixed-point — so planner/features/eval agree:
```
  Pass 1 (own resources only):  tau_def = D_def − current_step
        remaining0 = relu(floor(P,D_def) − garrison − prod·tau_def − friendly_inbound)
        SAFE       if remaining0 ≈ 0
        CANDIDATE  otherwise
  Pass 2 (provisional safe_sendable):  SAFE → max(0, garrison − min_reserve) into the spare pool.
        For each CANDIDATE, compute optimistic_reachable_help(P) from all other planets' maximum
        currently legal send capacity that can arrive by D_def+tol.
        provisional DOOMED if remaining0 > optimistic_reachable_help(P)
                         OR hold_value(P) < reinforce_cost(P); its full garrison joins the pool.
        otherwise provisional CANDIDATE reserves full garrison and contributes 0 to the pool.
  Pass 3 (confirm, most-urgent first — ascending D_def, ties by planet id):
        for each remaining CANDIDATE:
          eligible pool sources = pool planets Q with ETA_Q→P(available_Q) <= tau_def+tol.
          claim greedily by (ETA_Q→P ascending, source planet id ascending), taking the minimum
          mass needed from each source and subtracting claimed mass from that source's pool capacity.
        HOLDABLE  if claimed ≥ remaining0   → keep the claim reserved for this defense wave
        DOOMED    otherwise                 → release any failed claim and add this planet's full garrison to the pool
```
The ascending-`D_def` order makes urgent defenses claim mass before opportunistic ones — and
makes the result a deterministic function of state.
```
safe_sendable(P)   = SAFE     : garrison − min_reserve
                     HOLDABLE : 0                           # receiver, not sender (v1; v1.5 may spare excess garrison)
                     DOOMED   : full garrison               # drain it; recycle into a counterattack
```
`HOLDABLE` is the **defense feasibility gate** (the mirror of `wave_feasible_by_D`); the defense
quota only runs on already-`HOLDABLE` planets. `DOOMED → drain full` is the recycle-into-
counterattack behavior the 61%-idle-reinforce diagnostic showed we lack.

### 3.5 ready-wave quota (attack and defense, same machinery)
```
remaining     = relu(floor − cover)                    # gap after already-committed mass
ready_sources = sources with ready_now(source, D_abs) AND eligible (see decode)
ready_safe    = Σ safe_sendable over ready_sources
my_quota      = remaining * my_safe_sendable / max(ready_safe, eps)
```
Pre-rounding invariant: when `ready_safe ≥ remaining`, `Σ my_quota = remaining` and each
`my_quota ≤ my_safe_sendable` — crosses the floor exactly, no overshoot, no source over-asked.
When `ready_safe < remaining`, all ready sources send full safe (partial now, rest joins in
later steps arriving at the same `D_abs`). **Bounds overcommit only in continuous pre-rounding
space**; ship-bin rounding (§7) can reintroduce overshoot, which is *logged*, not asserted.

### 3.6 wave_feasible_by_D(T, D_abs) = total_safe_mass_arrivable_by_D_abs ≥ floor(T, D_abs)   # attack commit gate

---

## 4. Architecture (fresh model — no residual)

Direct per-(source, target) heads. No "old slot head + target residual" (that was for Phase 4
checkpoint compatibility and is implicated in the inert-fire result).
```
pair_ctx[s, t]   = MLP(src_emb[s], tgt_emb[t], pairwise_features[s,t], wave_features[s,t])
target_logits[s, t]        = target_head(pair_ctx)
fire_logits[s, t]          = fire_head(pair_ctx)
ship_logits[s, t, bin]     = ship_head(pair_ctx)
```
Target width is `max_planets + 1`. Real planet targets use indices `0..max_planets-1`.
`NO_OP_IDX = max_planets` is a synthetic target column:
```
no_op target embedding       = learned per-source-compatible key (not a planet entity)
no_op pair/wave features     = zeros except target-valid/no-op indicator
target_mask[..., NO_OP_IDX]  = True for every valid source slot
fire label for NO_OP         = 0; decode never launches for NO_OP
ship loss for NO_OP          = masked out (same as non-fired slots)
```
PPO chain rule stays `p(target)·p(fire|t)·p(ship|t)`: for `NO_OP`, the sampled/stored fire
action is forced to `0`, its fire log-prob is included, and ship log-prob is excluded because
the slot did not fire. This keeps no-op target credit explicit without inventing a separate
action head.

**Deferred to v1.5 / v2** (do NOT build now):
- `wave_head[target]` (learned rush-vs-patient deadline). v1 uses deterministic `choose_anchor`.
- `commit_head[target]` (learned total-commit). The prefix/quota features already expose it.
- Canonical same-step prefix ordering (v1 uses inbound mass + quota; add only if over-commit logs demand).

---

## 5. Feature set (small, threshold-biased — expose discontinuities, not algebra)

```
target / wave level  (→ target choice & feasibility):
  floor_for_wave            total_safe_cover_by_wave        cover_ratio_by_wave
  enemy_reactive_mass_by_wave                               wave_feasible_by_D
  enemy_committed_elsewhere     # opportunity (deadline-aware)
  recapture_risk                # after-D enemy inbound to this target
  hold_class                    # for owned planets (SAFE/HOLDABLE/DOOMED, one-hot)

source-target level  (→ fire & ship):
  eta_fast (ETA@safe_sendable)   eta_slow (ETA@min_bin)   ready_now (some legal count lands in window)   # §3.0.1
  my_safe_sendable               already_inbound_wave_mass   remaining_after_inbound   ready_safe_mass
  my_quota                       crosses_if_all_ready_send   marginal_needed_from_me
```
(`eta_fast`/`eta_slow` give the achievable arrival range so the head can both judge readiness and
pick the count that lands on time. There is no single `eta_to_target`/`slack` — ETA is count-dependent.)
Dropped (linearly derivable / redundant): `cum_through_me`, `overkill_after`, `source_rank`,
`total_safe_cover` at the source level (kept only at the target level).

---

## 6. Decode / commit rules

**Target space + no-wave contract [P2].** A source's legal targets are
`{ feasible attack targets (choose_anchor ≠ no-wave) } ∪ { reachable HOLDABLE planets needing
reinforcement } ∪ { NO_OP }`. Infeasible attack targets are removed from `target_mask` (the
target head never spends a slot on an untakeable planet). `NO_OP` is an explicit reserved target
index meaning "do nothing this step" (fire=0); it gives no-fire slots a defined target so the
PPO target log-prob (which is summed over all valid slots) is well-posed. Decode never
argmaxes over infeasible targets.

```
ATTACK source s:
  t  = argmax target_logits[s] over the masked legal set
  if t == NO_OP: no launch.
  D_abs = choose_anchor(t)                      # sticky-material inbound, else tightest-feasible
  send iff:  wave_feasible_by_D(t, D_abs)       # don't start a doomed wave
             AND ready_now(s, D_abs)            # §3.0.1: a legal on-time bin exists
             AND marginal_needed_from_me > 0
  ship_target = min(safe_sendable(s), my_quota)
  chosen_bin  = §3.0.1 ship_target(s): smallest legal bin n with n ≥ ship_target AND on-time (≤ D_abs+tol).
                If my_quota > safe_sendable, ship_target=safe_sendable: send full safe capacity,
                log the ready-set underfill, and rely on other ready/later sources for the remaining gap.
                Residual rule (deterministic): if the ready set still under-crosses after each
                source rounds, the shortfall is assigned to ready sources in ascending source-id
                order, each ceiling to its next legal bin until floor is met or capacity exhausted.
  post-round validation: arrival ∈ [D_abs-tol, D_abs+tol] and Σ launched ≥ floor − cover (log misses).

DEFENSE source s reinforcing owned P (P is HOLDABLE):
  D_def = choose_anchor_defense(P)
  send iff:  hold_class(P) == HOLDABLE          # feasibility is in the classification
             AND ready_now(s, D_def)
             AND remaining(P) > 0
  ship_target = min(safe_sendable(s), my_quota_defense)
  chosen_bin  = same §3.0.1 legal-bin rule as attack, using D_def and the defense quota.
```
The target head **arbitrates** a source between attacking and reinforcing (one fleet per
source). v1 planner priority: **imminent HOLDABLE defense before opportunistic attack**.

---

## 7. BC planner (label generator) — same functions, per-step

- Calls the **same** `floor`, `choose_anchor`, `hold_class`, `safe_sendable`, quota functions.
- Targets the **reactive** floor (never the static garrison) — else labels teach undercommit.
- **Per-step recompute:** after each labeled launch, `remaining` shrinks (inbound grows);
  re-derive quotas next step. Do not pre-allocate the whole wave once.
- **Joint arbitration:** assign each source to attack-bundle or defense-bundle by the v1
  priority, then size within the chosen wave by quota, rounded to a legal bin (§6).
- Generates attack labels (offensive waves) AND defense labels (HOLDABLE reinforcement);
  DOOMED planets contribute full garrison as attack sources.

---

## 8. Hard invariants

```
I1  planner_floor == feature_floor == eval_floor        (one floor implementation, three callers)
I2  choose_anchor identical in planner and inference     (no planner-only lookahead)
I3  hold_value / hold_class / safe_sendable shared        (same three callers)
I4  each enemy fleet classified exactly once per (T, D_abs)   (no double-count; §3.2)
I5  tol is one named constant; the three intervals are derived from it, not independently tuned
I6  defense remaining = relu(floor − cover); friendly inbound netted once (in cover) only
```

---

## 9. Pre-training poisoning checks (run on the BC dataset BEFORE any training)

```
positive_reactive_cross_rate   high      # bundles size to the REACTIVE floor, not static
positive_static_cross_rate     (info)    # if high while reactive low ⇒ dataset poisoned
positive_arrival_spread_p90    <= 2*tol  # bundles arrive as a wave, not a trickle
ready_quota_error              small     # rounding/quota residual under control
overcommit_ratio (post-round)  bounded   # launched_wave_mass / floor(D_abs)
defense: held_when_holdable_rate high, reinforce_into_DOOMED_rate low
```
If reactive-cross is high but arrival-spread is wide, the planner is poisoned in *timing* even
if *mass* is clean — that is the exact failure the wave design exists to prevent. Fix the
planner, do not train.

---

## 10. Parity contract

All wave features computed identically in `torch_env._compute_pairwise` (GPU) and
`features.compute_pairwise_features` (numpy). The quota/prefix are **masked sums + a division**
(no sort, no cumsum) → parity-trivial; this is the whole reason we chose the ready-wave quota
over a canonical same-step prefix. Extend `tests/test_friendly_coverage.py` to cover every new
wave channel (the same sim-gap that has bitten before). `floor`/`choose_anchor`/`safe_sendable`
are shared Python called by both paths where possible; where the GPU path must re-implement,
the parity test is the gate.

---

## 10.5 Mask / overlay disposition [P2]

Phase 5 makes defense a **learned** target/fire/ship behavior, so the heuristic reinforcement
machinery must be retired, not left to overlay the learned policy differently at eval.
```
KEEP (legal-target masks — physics/contract, identical in train/eval/export):
  own planets legal as reinforce targets        NO_OP target index        source-selection (top-garrison slots)
  reverse-edge cooldown (if retained)            min-ship-bin
DISABLE for Phase 5:
  defensive_reinforce_overlay (action_mask.py)   — it FORCES reinforcement; defense is now a learned wave
  sufficient_commit_factor veto                  — OFF for the no-veto gate (§11); may return only as a
                                                   late-PPO safety net, never during BC or the gate
```
All mask/decode config is persisted in checkpoint metadata and auto-loaded by eval/export (the
Phase 4 contract), so a checkpoint is evaluated exactly as trained. `grep` the persisted flags
after export.

## 11. Gates

**Pre-PPO gate (cheap, CPU — the go/no-go):** the BC'd policy on **synthetic concentration
states**, with `sufficient_commit_factor = 0` (NO veto crutch):
```
oracle passes, noop fails (states are load-bearing)
attack: BC policy crosses the REACTIVE floor with synchronized arrival (spread ≤ 2*tol)
defense: holds HOLDABLE planets; drains + counterattacks on DOOMED
includes states where near sources must wait and far sources must launch first
```
If this fails, fix features/planner — do not spend a GPU-hour discovering it in PPO.

**Promotion gate (before considering the learned `wave_head`):**
```
positive_reactive_cross_rate high      no-veto undercommit rate down
positive_arrival_spread small          contested dm<50 cross up
out-massed rate drops                  held-out Ajay up (selection metric)
```
If these pass but the deterministic anchor makes bad rush-vs-patient calls → `wave_head` is the
next architectural step. If they fail, a learned head is premature.

---

## 12. v1 / v2 boundary

```
v1 (required):
  capture_floor + enemy-fleet classification (raises / committed-elsewhere / recapture)
  anchored offensive wave (sticky, launchable, tightest-feasible) + ready-wave quota
  hold_class for safe_sendable; DOOMED → drain full
  minimal defense wave for HOLDABLE (same floor−cover, same quota, earliest-threat anchor)
  MAX_OWNED=24, MAX_LANES=1 (more source planets, no same-source multi-move)
  direct per-source-target heads (no residual); bc.py per-target fix
  pre-train poisoning checks + pre-PPO no-veto gate (attack cross AND hold)

v1.5 / v2 (only if diagnostics demand):
  learned wave_head[target] (rush vs patient)        commit_head[target]
  canonical same-step prefix ordering                 richer hold_value / value model
  multi-wave defense scheduling                       refined attack/defense arbitration
```

---

## 13. Logging (PPO + eval)

```
attack:  reactive_cross_rate, ratio(inflight/floor), overcommit_ratio, arrival_spread_p90,
         ready_quota_error, wave-size distribution, synchronized-arrival commit ratio
defense: held_when_holdable, reinforce_into_DOOMED, drained_DOOMED_into_attack,
         hold_deficit_p50, recapture_loss_rate
shared:  out-massed%, dm<50 cross, planets@16/32/50/100, held-out Ajay
```

---

## 14. Build order

0. **Done:** replace the decisive-mass floor implementation. `torch_env._decisive_mass_fields`
   and eval `dm` now use deadline-filtered enemy inbound, ETA-threshold reactive planet mass,
   Phase 5 margin, and synchronized friendly wave mass.
1. **Done:** raise the one-move/source width to `MAX_OWNED=24` across train/eval/export and
   update source-selection parity tests. Still log `owned_count_gt24_rate`,
   `wave_sources_clipped_by_max_owned_rate`, and slot-17..24 label/use rates before long PPO.
2. **Done:** scalar shared reference for `cover` / `choose_anchor` / `hold_class`
   (single-pass, §3.4) / `safe_sendable` / ready-wave quota + the ETA contract (§3.0.1).
   Unit tests include the ETA↔count round-trip (`ship_target` lands in `[D_abs-tol,D_abs+tol]`).
3. **Done:** Wave features in both paths + parity test (§10). The ROI aux remains anchored
   to unchanged channels `12:15`; roi/contest weight-norm logging now reads those fixed
   columns instead of the last three pairwise cols. Added reach-bin (`15:21`) and wave
   (`21:41`) weight-norm logging plus eval ablation hooks.
4. **Done:** Direct per-source-target heads (slot prior + residual removed; the per-target
   `fire_scorer`/`ship_scorer` are now the heads) + a learned `no_op_head`. `NO_OP` target
   wired end-to-end: model target/fire/ship width `max_planets+1`, env `target_mask` NO_OP
   column (`= slot_valid`), env decode treats a NO_OP pick as no-launch, rollout sampling
   forces fire=0 at NO_OP, PPO chain rule unchanged (gather widths absorb the extra column).
   `bc.py` gathers fire/ship at the teacher's chosen target and labels NO_OP for valid
   no-launch slots. §10.5 overlay/veto: NO code removed (tested opt-in machinery used by
   Phase 4 experiments) — both `defensive_reinforce_k` (CLI default 0) and
   `sufficient_commit_factor` (ckpt-metadata default 0.0) are already off-by-default, and the
   checkpoint-as-trained eval contract means a Phase 5 ckpt evaluates without them; retirement
   = simply never enabling them in the Phase 5 BC/PPO/eval path (enforced when Steps 5-7 wire
   the run). NOTE: `export_agent.py` (its own standalone submission model) and the
   `check_phase4_parity.py` diagnostic still replicate the old slot prior — update at export
   time (Step 7).
5. **Done:** Sufficient-prefix **wave** planner (`wave_planner.py` §7) + poisoning checks
   (`build_wave_bc.py` §9) + `tests/test_wave_planner.py`. **AUDIT AMENDMENT (2026-06-19):** the
   §9 reactive-cross gate initially FAILED (0.14) — the tight two-sided arrival window starved the
   waves (sources couldn't deliver full mass on-time). The loss-mode audit (docs/phase5-blocked.md)
   showed staggered-arrival is ~1% of losses, so **synchronization was dropped**: the arrival
   window is now ONE-SIDED (arrive BY `tau+tol`, early arrival allowed) — changed consistently in
   `wave_primitives.ship_choice_for_quota`, `features.py`, and `torch_env._compute_pairwise`
   (parity preserved, test_friendly_coverage still 0.0000). With one-sided windows + a §6 commit
   gate (only launch a wave whose ready mass crosses), §9 now PASSES: reactive_cross 1.00,
   overcommit p50 1.03, ready_quota_error ~3, held_when_holdable 0.83, reinforce_into_DOOMED 0.
   `arrival_spread` is now INFO-only, not a gate.
6. Pre-PPO no-veto gate on synthetic states (§11). **Hard stop** until it passes.
7. From-scratch BC → PPO. Promotion gate (§11) before any `wave_head`.
