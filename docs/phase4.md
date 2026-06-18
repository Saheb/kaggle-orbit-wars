# Phase 4 — Per-target conditioning for fire/ship heads

**Purpose:** complete the `docs/bugs.md` §1 fix that was left half-done in 2026-05-24.
Target got per-(slot, target) conditioning; fire and ship were left on the slot-only
path. Two leaks found 2026-06-18 (documented in `docs/bugs.md` §3) are the predicted
symptoms: ship can't size to `garrison+1` (the +1 neutral trap), fire can't see
`enemy_contest` (opening paralysis). Both are one root cause — per-target features live
on the wrong branch.

This is an **architecture change**, not a reward/mask delta. It makes the three action
heads consistent: all read the same `[q_slot, k_target, pairwise_features]` inputs. It
does **not** build the full autoregressive action-list decoder (`docs/autoregressive-head.md`
Stage 0 parked AR — winners don't use enough within-turn multi-source floor-covering to
justify it). Phase 4 is the factored-policy version of the same insight: condition every
head on the per-target features, keep the factored one-fleet-per-source interface.

Origin: `docs/bugs.md` §3 (found 2026-06-18). Companion: `docs/autoregressive-head.md`
(the AR thesis whose "fire dissolves into stop" idea informed, but was *not* adopted
for the factored policy — see §2). Lineage context: `docs/phase3.md` (the
teacher-KL/league work this runs alongside) + `docs/training.md` Current State.

---

## 1. The two leaks this fixes (recap from `docs/bugs.md` §3)

**Leak A — the +1 neutral capture trap.** The game needs `sent > garrison` strictly to
flip a neutral. `sent == garrison` leaves it at 0 and neutral — a guaranteed waste.
The ship head picks oversized bins that clamp to src-garrison; when `src_garrison ==
neutral_garrison` the agent *cannot express* `garrison+1` from that source, and fires
anyway. Probe: 4/287 neutral launches across 16 games. The ship head can't see the
target's garrison → can't size to `garrison+1`.

**Leak B — fire-head opening paralysis.** The fire head learned a garrison-margin
threshold (fire when `src - tgt >= ~3`), not the correct `>= 1`. At an uncontested
prod=5 neutral, waiting from margin 1 to 3 costs ~10 ships of lost production for 2
extra survivors — pure waste from over-caution. Self-play doesn't punish it (the mirror
also waits). Probe: `fire_p` is flat to step index and flat to enemy pressure (a
500-ship inbound fleet moves `fire_p` from 0.003 to 0.000). The fire head is blind to
`enemy_contest` → keyed on the only strong signal it can see (src-garrison margin).

Both are "per-target feature on the wrong branch." `capture_cost` and `enemy_contest`
are in the pairwise bundle but feed `target_scorer` only (`model.py:241`); fire and ship
read `owned_enriched` (slot + softmax-pooled attention), which dilutes per-target signal
below the gradient's notice.

---

## 2. What this is NOT

- **Not the AR action-list decoder.** `docs/autoregressive-head.md` Stage 0 parked AR
  after finding winners don't use same-turn multi-source floor-covering (1.6% cross
  rate, list length p50=1). Phase 4 keeps the factored one-fleet-per-source interface.
  The AR doc's "fire dissolves into stop" idea was considered and rejected for the
  factored policy: dissolving fire into a target Categorical with a no-fire sentinel
  destroys the fire head's selectivity (a Categorical always argmaxes something), and
  without AR's sequential prefix state there's no basis for a strong `stop` logit →
  carpet-bomb Nash (the Rev41-45 graveyard). **Fire stays as a Bernoulli gate** — it's
  the selectivity that makes Isaiah-like play (3.6% launch rate) possible. We just make
  it per-target so it's no longer blind.
- **Not a reward/mask delta.** No new shaping term, no new veto beyond the
  `sufficient_commit_factor=1.0` mask already shipped (which stays as a training mask on
  top, to enforce the +1 floor while the ship head learns to size correctly).
- **Not top-k target retry.** Decode stays top-1: argmax target, fire gate at that
  target, ship bin at that target. If fire vetoes the top target, the slot no-fires
  (same as today). Top-k retry changes the log-prob factorization to ordered sampling;
  revisit only if veto-then-no-fire waste shows up as a lever.

---

## 3. The design — Option 1 (per-target fire, top-1 decode)

### Model (`model.py`)

Two new per-(slot, target) heads, mirroring `target_scorer`'s input pattern:

```python
# Fire: (B, MO, N_p) — "should this slot fire AT this target?"
q_fire = self.fire_q(owned_enriched).unsqueeze(2).expand(-1, -1, N_p, -1)   # (B, MO, N_p, D)
k_fire = self.fire_k(planet_emb_post).unsqueeze(1).expand(-1, max_owned, -1, -1)
fire_in = torch.cat([q_fire, k_fire, pairwise_features], dim=-1)            # (B, MO, N_p, D+D+F)
fire_scores = self.fire_scorer(fire_in).squeeze(-1)                         # (B, MO, N_p)

# Ship: (B, MO, N_p, num_ship_bins) — "what size if this slot fires at this target?"
# (reuse fire's q/k or add dedicated ship_q/ship_k; dedicated is cleaner for attribution)
ship_in = torch.cat([q_ship, k_ship, pairwise_features], dim=-1)
ship_scores = self.ship_scorer(ship_in)                                     # (B, MO, N_p, bins)
```

Six new parameter blocks: `fire_q/fire_k/fire_scorer`, `ship_q/ship_k/ship_scorer`.
Same shape pattern as the existing `tgt_q/tgt_k/target_scorer`. The old slot-level
`fire_head` and `ship_head` are retired (or kept as a residual-init prior — see §5).

`target_logits` unchanged (already per-target, already has pairwise).

### Decode (`action_mask.py` + `torch_env.py`) — top-1, fire stays the gate

```python
for slot in range(owned_count):
    tidx = target_logits[slot].argmax()                    # pick the objective (target already sees pairwise)
    if sigmoid(fire_scores[slot, tidx]) < fire_threshold:  # fire gate, now seeing the CHOSEN target
        continue                                           # no fire from this slot (selectivity preserved)
    bin = ship_scores[slot, tidx].argmax()                 # size for THIS target (sees its garrison+1)
    ships = _ship_bin_to_count(bin, max_ships[slot], mode)
    moves.append([src_id, intercept_angle(...), ships])
```

Fire is still a Bernoulli gate per slot — it just sees `enemy_contest[target]` and
`capture_cost[target]` now, so the margin can modulate with threat. Ship sizes relative
to the target's garrison, so `garrison+1` is expressible and the +1 trap dies.

### PPO log-prob (`ppo.py` + `train_torch.py`) — chain rule factorization

```
log_prob = Σ_slots [ log p(target_slot) · 1[fired] + log p(fire_slot | target_slot) + log p(ship_slot | target_slot) · 1[fired] ]
```

Per slot: `Categorical(target)` over planets (unchanged), `Bernoulli(fire | target)` at
the chosen target's fire_logit, `Categorical(ship | target)` at the chosen target's
ship_logits. Clean chain rule `p(target, fire, ship) = p(target)·p(fire|target)·p(ship|target)`.
PPO ratio unchanged; entropy is the joint entropy. `old_log_probs` storage shape
unchanged (fire/ship log-probs are scalars per slot, evaluated at the chosen target).

The IL-KL (`ppo.py:97-114`) needs the same per-target conditioning: fire KL at the
chosen target, ship KL at the chosen target. Target KL unchanged.

---

## 4. Why this order (target → fire → ship), and the rejected alternatives

**Target first** — the objective. The wall is contest-centric ("pick the objective,
then muster mass onto it" — `docs/autoregressive-head.md` §"Decoder order"). Target
already sees pairwise features; picking the objective first is correct.

**Fire second, conditioned on target** — fire is the selectivity gate. It sees the
chosen target's `enemy_contest`/`capture_cost` and decides "fire at *this* target or
not." Keeps fire as a Bernoulli (Isaiah's 3.6% launch rate depends on this gate; we must
not dissolve it).

**Ship third, conditioned on target** — size for the target you're actually firing at.
Sees `capture_cost = garrison+1`, can size to it.

Rejected:
- **Fire sentinel / "fire dissolves into stop"** (AR doc §"Fire dissolves into stop"):
  a Categorical over `[stop, target_1, ...]`. Correct for AR (sequential prefix state
  gives `stop` a basis), wrong for factored (no prefix state → always argmaxes a target
  → carpet-bomb). Keep fire as a gate.
- **Fire first, then target** (`p(fire)·p(target|fire)·p(ship|target)`): fire is now
  per-(slot,target), so "fire at all" has no single logit to use — circular. Fire must
  be conditioned on target, so target must come first.
- **Filter targets by fire_scores > thr, then argmax (Option 2 in earlier discussion):**
  circular — filtering by fire_scores requires evaluating fire at every target first,
  then re-ranking by fire alone throws away target_logits. Not logical. Drop.
- **Top-k target retry:** more expressive (try target Y if fire vetoes X), but changes
  log-prob to ordered sampling. Top-1 matches today's learned behavior; revisit only if
  veto-waste is a lever.

---

## 5. Warmstart — residual-init so step-0 == old behavior

New params (`fire_q/k/scorer`, `ship_q/k/scorer`) are uninitialized. Cold-start risk is
real (the docs' BC-warmstart lesson #9: frozen policy from partial ckpt). Three options,
ordered by safety:

**(b) Residual — RECOMMENDED.** Keep the old slot-level `fire_head(owned_enriched)` as a
prior; the new per-target head is a *residual*:
```python
slot_fire_logit = self.fire_head(owned_enriched)                  # old, broadcast to all targets
fire_scores = slot_fire_logit.unsqueeze(-1) + fire_scorer(...)    # residual starts at 0
```
Initialize `fire_scorer`'s last layer to zeros → `fire_scores[slot, target] == old
fire_logit[slot]` for all targets at step 0. Same for ship. Step-0 behavior is
byte-identical to the old model; PPO gradient flows into the residual and learns
per-target differentiation only where it pays. Cleanest attribution (the residual's
magnitude shows exactly when per-target signal activates).

**(a) Init from old heads** — `fire_scorer`'s last layer initialized so the per-target
logit approximates the old slot-level logit (broadcast). Same step-0 behavior as (b)
but without the explicit residual structure; harder to attribute.

**(c) BC warmstart** — train new heads on top-player replays. The docs warn BC craters
(rev54); skip unless (a)/(b) fail.

**Checkpoint compatibility:** old ckpts lack the new params. Two paths:
- **Strict:** from-scratch Stage B run (BC warmstart, new feature set) — cleanest but
  discards the revedge1 lineage.
- **Compat shim + residual (b):** `load_state_dict(strict=False)`; init new params from
  old heads (zeros for the residual). Lets us resume revedge1 4.72M + veto + per-target
  heads in one run. Step-0 == old model (residual=0), so the arch-change shock is
  bounded. This is the pragmatic path — the docs' "don't resume across arch changes"
  warning is about shape changes that alter step-0 behavior; residual-init (b) doesn't.

---

## 6. Scope

| File | Change |
|---|---|
| `model.py` | +6 Linear layers (fire/ship q/k/scorer), `forward` produces `(B,MO,N_p)` fire + `(B,MO,N_p,bins)` ship; old `fire_head`/`ship_head` kept as residual priors |
| `action_mask.py` | `actions_from_target_policy` decode → top-1 with per-target fire gate + per-target ship bin |
| `torch_env.py` | same decode change for training-time action sampling |
| `ppo.py` | log-prob factorization: `p(target)·p(fire\|target)·p(ship\|target)`; IL-KL same conditioning |
| `train_torch.py` | rollout log-prob uses per-target fire/ship at the chosen target; `old_log_probs` storage unchanged (scalars per slot) |
| `bc.py` | (if BC warmstart) fire/ship loss becomes per-target at the teacher's chosen target |
| `export_agent.py` | exported `forward` mirrors the new heads + top-1 decode |
| tests | new parity: per-target fire/ship shapes; **residual-init == old behavior** (byte-identical forward with residual=0); joint decode matches old decode when residual=0 |

---

## 7. Staged plan — one falsifiable gate per stage

### Stage A — parity + warmstart (cheap, local)
Build the new heads with residual-init (b). Prove:
1. `load_state_dict(strict=False)` on revedge1 4.72M loads cleanly (new params zeros).
2. With residual=0, `forward` is byte-identical to the old model (same fire_logits
   broadcast, same ship_logits, same target_logits) — parity test.
3. With residual=0, the eval panel matches the baseline (no-veto) panel within noise.
   → verify: parity test PASS + 16-game panel ≈ baseline.

Gate: if parity fails, the residual design is wrong — fix before any training.

### Stage B — short PPO smoke (cheap, GPU)
Resume revedge1 4.72M + `--sufficient-commit-factor 1.0` (the veto stays as a training
mask) + the new per-target heads (residual-init) for ~500k steps. Read:
1. `clip_frac` ≠ 0 and entropy not floored (not a frozen policy — lesson #9).
2. The fire/ship residuals' norms climb from zero (per-target signal is activating).
3. `dm NEUTRAL cross` rises (ship sizing to `garrison+1`), `cap/atk open<50` rises
   (fire gate opening at low-margin uncontested neutrals).
4. Held-out Ajay WR holds ≈ 23.8% baseline (no collapse from the arch change).

Gate: if clip_frac=0 / entropy floored → warmstart failed, revert to Stage A fix. If
Ajay collapses → the arch change shocked the policy despite residual-init → reconsider
the compat-shim path (from-scratch Stage B instead of resume).

### Stage C — full run (GPU, ~6M)
If Stage B passes: full run with the same config. Judge by the Phase 4 promotion
metrics (§8). One delta vs the revedge1 baseline: the per-target heads. The veto mask
is a co-delta but it's already LB-validated (+11.4pp panel) and is a mask (no learning
semantics) — acceptable to bundle, per the docs' "masks are safe to bundle" pattern.

---

## 8. Promotion metrics

Primary (the leaks this fixes):
```
failed-attack decomposition: single 0% stays, but the +1 trap (sent==garrison) -> 0
dm NEUTRAL cross up (ship sizing to garrison+1)
cap/atk open<50 up (fire gate opening at low-margin uncontested neutrals)
fire_p responds to enemy_contest (re-run the pressure probe — fire_p should MOVE)
```

Secondary (the wall, unchanged by this fix but must not regress):
```
hold-loss out-massed% flat-or-down (concentration wall; this fix doesn't target it)
Ajay WR up (the veto gave +11.4pp; per-target heads should add on top or hold)
Zach WR holds (~88-89% saturated)
```

Do not promote on:
```
training reward alone
self-play WR
fire_rate in isolation (could rise or fall — read it WITH cap/atk launch efficiency)
```

Panel-to-LB humility: Ajay panel ≠ LB-predictive (rev53b 10.9% Ajay → 933 LB < rev38
2.7% → 994). LB submission remains the promotion test for leaderboard claims. The
revedge1 4.72M + veto submission (2026-06-18, pending) is the first LB data point for
the veto alone; Phase 4's LB read comes after Stage C.

---

## 9. Risks / open questions

- **PPO factorization change:** `p(target)·p(fire|target)·p(ship|target)` vs the old
  `p(fire)·p(ship)·p(target)` (independent). The chain rule is mathematically clean but
  the *gradient* flows differently — fire/ship now get gradient through the target
  choice. Watch clip_frac / entropy in Stage B for signs the policy can't adapt.
- **Residual-init scale:** if the residual starts at exactly 0, the first PPO updates
  may push it slowly (vanishing gradient through the zero-init last layer). Mitigation:
  small-norm init (e.g. 0.01) instead of exact zero, accept a tiny step-0 deviation.
  Test in Stage A.
- **Fire-veto waste at top-1:** if the target head picks a target the fire head vetoes,
  the slot no-fires. If this is frequent, the policy is wasting slots. Measure
  `veto_rate` in Stage B; revisit top-k if it's high.
- **Per-target compute cost:** fire/ship are now `(B, MO, N_p)` not `(B, MO)` — a
  ~24× compute increase for those heads (small vs the transformer). Measure SPS in
  Stage B; the AR doc's throughput tripwire (SPS < 150) doesn't apply (this is
  one-shot, not sequential), but a >2× slowdown vs the factored baseline would hurt
  iteration economics.
- **Interaction with the veto mask:** `sufficient_commit_factor=1.0` blocks
  `sent <= garrison`. Once the ship head learns to size to `garrison+1` via the
  per-target features, the mask should rarely fire (the policy self-censors). If the
  mask is still firing frequently at 6M, the per-target signal didn't take — diagnose
  the residual norms. The mask is a safety net, not a crutch; the goal is the ship head
  learning the +1 rule natively.

---

## 10. Sequencing / gate

1. **GATE — parity (Stage A):** build, load revedge1 4.72M, prove residual=0 forward
   is byte-identical. No training until parity PASS.
2. **Stage B:** 500k PPO smoke, read clip_frac / residual norms / dm NEUTRAL cross /
   Ajay holds.
3. **Stage C:** full run to 6M, read promotion metrics + LB submission.

One delta per stage. The veto mask is the only co-delta (already LB-validated, safe to
bundle). Selection stays PURE: held-out Ajay WR + LB submission, never self-play WR or
shaped reward.
