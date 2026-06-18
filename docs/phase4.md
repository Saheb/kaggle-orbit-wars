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
- **Not a reward/mask delta.** No new shaping term. The `sufficient_commit_factor=1.0`
  veto mask already shipped as an eval/training mask (panel +11.4pp on revedge1, LB
  submission pending 2026-06-18 — *panel-validated, NOT LB-validated*); it stays
  available as a training mask on top of the arch change, but Stage B isolates the arch
  without it (§7) so the veto gain cannot mask an architecture regression.
- **Not top-k target retry.** Decode stays top-1: argmax target, fire gate at that
  target, ship bin at that target. If fire vetoes the top target, the slot no-fires
  (same as today). Top-k retry changes the log-prob factorization to ordered sampling;
  revisit only if veto-then-no-fire waste shows up as a lever.

---

## 3. The design — Option 1 (per-target fire, top-1 decode)

### Explicit shape names (avoid silent bugs from reusing `fire_logits`/`ship_logits`)

```
fire_logits_slot:    (B, MO)              # OLD, kept as residual prior (broadcast to all targets)
fire_logits_target:  (B, MO, N_p)        # NEW, per-(slot,target) — what forward returns
ship_logits_slot:    (B, MO, bins)       # OLD, kept as residual prior (broadcast)
ship_logits_target:  (B, MO, N_p, bins)  # NEW, per-(slot,target) — what forward returns
target_logits:       (B, MO, N_p)        # UNCHANGED
```

`forward` returns `fire_logits_target` and `ship_logits_target` (the per-target tensors).
The old slot-level `fire_logits_slot` / `ship_logits_slot` are internal to the residual
computation (§5) — they are NOT in the output dict, to force consumers to use the
per-target tensors. Any consumer that still indexes `outputs["fire_logits"]` as
`(B, MO)` will fail loudly on the shape change, which is the desired behavior (silent
bugs from ambiguous reuse are the failure mode to avoid).

### Model (`model.py`)

Two new per-(slot, target) heads, mirroring `target_scorer`'s input pattern:

```python
# fire_logits_target: (B, MO, N_p) — "should this slot fire AT this target?"
q_fire = self.fire_q(owned_enriched).unsqueeze(2).expand(-1, -1, N_p, -1)   # (B, MO, N_p, D)
k_fire = self.fire_k(planet_emb_post).unsqueeze(1).expand(-1, max_owned, -1, -1)
fire_in = torch.cat([q_fire, k_fire, pairwise_features], dim=-1)            # (B, MO, N_p, D+D+F)
fire_logits_target = self.fire_scorer(fire_in).squeeze(-1)                  # (B, MO, N_p)

# ship_logits_target: (B, MO, N_p, bins) — "what size if this slot fires at this target?"
q_ship = self.ship_q(owned_enriched).unsqueeze(2).expand(-1, -1, N_p, -1)
k_ship = self.ship_k(planet_emb_post).unsqueeze(1).expand(-1, max_owned, -1, -1)
ship_in = torch.cat([q_ship, k_ship, pairwise_features], dim=-1)
ship_logits_target = self.ship_scorer(ship_in)                             # (B, MO, N_p, bins)
```

Six new parameter blocks: `fire_q/fire_k/fire_scorer`, `ship_q/ship_k/ship_scorer`.
Same shape pattern as the existing `tgt_q/tgt_k/target_scorer`. Dedicated `ship_q/ship_k`
(not reused from fire) for clean attribution. The old slot-level `fire_head` and
`ship_head` are kept as residual-init priors (see §5), not retired — they hold the
step-0 behavior.

`target_logits` unchanged (already per-target, already has pairwise).

### Decode (`action_mask.py` + `torch_env.py`) — top-1, fire stays the gate

```python
for slot in range(owned_count):
    tidx = target_logits[slot].argmax()                              # pick the objective (target already sees pairwise)
    if sigmoid(fire_logits_target[slot, tidx]) < fire_threshold:     # fire gate, now seeing the CHOSEN target
        continue                                                     # no fire from this slot (selectivity preserved)
    bin = ship_logits_target[slot, tidx].argmax()                    # size for THIS target (sees its garrison+1)
    ships = _ship_bin_to_count(bin, max_ships[slot], mode)
    moves.append([src_id, intercept_angle(...), ships])
```

Fire is still a Bernoulli gate per slot — it just sees `enemy_contest[target]` and
`capture_cost[target]` now, so the margin can modulate with threat. Ship sizes relative
to the target's garrison, so `garrison+1` is expressible and the +1 trap dies.

### PPO log-prob (`ppo.py` + `train_torch.py`) — chain rule factorization

**Probabilistic contract (the part that must be precise):** the decode samples a target
for every valid slot, then fire conditioned on that target, then ship conditioned on
target only if fired. The joint distribution per slot is:

```
p(target=t, fire, ship | fired) = p(target=t) · p(fire | t) · p(ship | t)
p(target=t, fire=0)             = p(target=t) · p(fire=0 | t)        # no ship on no-fire
```

The target is a *latent* on no-fire slots (the environment never sees it), but it is
sampled, so it is part of the joint action PPO evaluates. **Target log-prob is included
for EVERY valid slot, not just fired slots.** This is a change from the current contract
(`ppo.py:182-184` gates target log-prob with `fired_slots`); that gating was harmless
under the old independent-fire head (fire didn't depend on target, so dropping target
log-prob on no-fire slots was a no-op for the ratio), but with per-target fire the
marginal `p(no_fire) = Σ_t p(t)·p(fire=0|t)` involves target probabilities — dropping
them makes the PPO ratio wrong. Including target log-prob everywhere is the simpler
contract (no marginalization); the alternative (marginalize no-fire over targets) is
more expensive and changes the gradient. Use the simple one.

```
log_prob_slot = log p(target_slot)                            # every valid slot (was: only fired)
             + log p(fire_slot | target_slot)                # every valid slot (Bernoulli at chosen target)
             + 1[fired] · log p(ship_slot | target_slot)     # only fired (no ship action on no-fire)
log_prob = Σ_slots log_prob_slot
```

Per slot: `Categorical(target)` over planets (unchanged), `Bernoulli(fire | target)` at
the chosen target's fire_logit, `Categorical(ship | target)` at the chosen target's
ship_logits. Clean chain rule `p(target, fire, ship) = p(target)·p(fire|target)·p(ship|target)`.
PPO ratio is the ratio of joint probabilities — correct under this contract. Entropy is
the joint entropy. `old_log_probs` storage shape unchanged (fire/ship log-probs are
scalars per slot, evaluated at the chosen target; target log-prob was already stored per
slot, it just now contributes on no-fire slots too).

The IL-KL (`ppo.py:97-114`) needs the same per-target conditioning: fire KL at the
chosen target, ship KL at the chosen target. Target KL unchanged (already per-slot).

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

## 5. Warmstart — residual-init so step-0 == old behavior (broadcast parity, not byte-identical)

New params (`fire_q/k/scorer`, `ship_q/k/scorer`) are uninitialized. Cold-start risk is
real (the docs' BC-warmstart lesson #9: frozen policy from partial ckpt). Three options,
ordered by safety:

**(b) Residual — RECOMMENDED.** Keep the old slot-level `fire_head(owned_enriched)` as a
prior; the new per-target head is a *residual*:
```python
fire_logits_slot   = self.fire_head(owned_enriched)                       # OLD (B, MO)
fire_logits_target = fire_logits_slot.unsqueeze(-1) + fire_scorer(...)    # (B, MO, N_p), residual starts at 0
# same for ship: ship_logits_slot broadcast + ship_scorer residual
```
Initialize `fire_scorer`'s (and `ship_scorer`'s) last layer to zeros →
`fire_logits_target[slot, target] == old fire_logits_slot[slot]` for ALL targets at
step 0. Same for ship. **Parity is "old slot logits broadcast identically across the
target dimension," NOT "byte-identical forward"** (the output shapes change from
`(B,MO)` → `(B,MO,N_p)` and `(B,MO,bins)` → `(B,MO,N_p,bins)`; raw forward-dict equality
is impossible and the wrong gate). The parity gate is: residual=0 ⇒ the *decode* and
*log-prob* match the old model on the same input (greedy actions identical, log-probs
identical under the §3 contract). PPO gradient flows into the residual and learns
per-target differentiation only where it pays. Cleanest attribution (the residual's
magnitude shows exactly when per-target signal activates).

**Zero-init caveat (P1 from review):** zeroing the residual's final layer means earlier
residual layers (`fire_q/fire_k`, the first layer of `fire_scorer`) get *zero gradient*
until the last layer moves off zero. This is usually acceptable (the last layer moves on
the first update, then earlier layers get gradient), but Stage B must explicitly track
whether `fire_q/k` and `ship_q/k` norms and per-target variance become nonzero. If they
stay at zero past ~50k steps, switch to small-norm init (e.g. last layer ~0.01) — a
tiny step-0 deviation in exchange for breaking the zero-gradient freeze.

**(a) Init from old heads** — `fire_scorer`'s last layer initialized so the per-target
logit approximates the old slot-level logit (broadcast). Same step-0 behavior as (b)
but without the explicit residual structure; harder to attribute.

**(c) BC warmstart** — train new heads on top-player replays. The docs warn BC craters
(rev54); skip unless (a)/(b) fail.

**Checkpoint compatibility:** old ckpts lack the new params. Two paths:
- **Strict:** from-scratch Stage B run (BC warmstart, new feature set) — cleanest but
  discards the revedge1 lineage.
- **Compat shim + residual (b):** `load_state_dict(strict=False)`; init new params with
  zeros for the residual. Lets us resume revedge1 4.72M + per-target heads in one run.
  Step-0 decode == old model (residual=0), so the arch-change shock is bounded. This is
  the pragmatic path — the docs' "don't resume across arch changes" warning is about
  shape changes that alter step-0 behavior; residual-init (b) doesn't.

---

## 6. Scope — every consumer of `fire_logits` / `ship_logits` must be updated

The shape change from `(B,MO)` / `(B,MO,bins)` to `(B,MO,N_p)` / `(B,MO,N_p,bins)`
touches every site that indexes these tensors. Returning only the per-target tensors
from `forward` (§3) forces each consumer to either index the chosen target or fail
loudly on the shape mismatch — no silent `(B,MO)`-shaped bugs.

| File | Site | Change |
|---|---|---|
| `model.py` | `EntityTransformer.__init__` + `forward` | +6 Linear layers (`fire_q/k/scorer`, `ship_q/k/scorer`); `forward` returns `fire_logits_target` `(B,MO,N_p)` + `ship_logits_target` `(B,MO,N_p,bins)`; old `fire_head`/`ship_head` kept as residual priors (broadcast) |
| `train_torch.py` | `sample_action_batched` (lines ~79-100) | sample target (unchanged), sample fire from `fire_logits_target[:, slot, sampled_target]`, sample ship from `ship_logits_target[:, slot, sampled_target, :]`; `old_log_probs` stores target log-prob for ALL valid slots (was: only fired), fire/ship log-probs at the chosen target |
| `ppo.py` | `compute_loss` (lines ~161-193) + `_il_kl_penalty` (lines ~97-114) | log-prob per §3 contract: `log p(target)` for every valid slot + `log p(fire\|target)` for every valid slot + `1[fired]·log p(ship\|target)`; IL-KL: fire KL at chosen target, ship KL at chosen target, target KL unchanged |
| `torch_env.py` | action decode in `step` (lines ~1840-1962) | top-1 decode: argmax target → fire gate at that target → ship bin at that target; `sufficient_commit_factor` veto now checks `ship_count` vs the chosen target's garrison (already per-target via `target_ships`) |
| `action_mask.py` | `actions_from_target_policy` (lines ~793-1013) | same top-1 decode as torch_env; `sufficient_commit_factor` veto at the chosen target |
| `eval.py` | `build_agent_fn` (lines ~212-323) + all probe/diagnostic sites that read `outputs["fire_logits"]` / `outputs["ship_logits"]` | index `fire_logits_target[:, slot, chosen_target]` / `ship_logits_target[:, slot, chosen_target, :]`; per-target `fire_p` diagnostic (richer — shows fire_p per target, not one slot-level number) |
| `export_agent.py` | embedded `EntityTransformer` + decode | mirrors the new `forward` (per-target heads + residual) and top-1 decode; export must produce byte-identical behavior to eval |
| `bc.py` | `train` (lines ~297-353) | (only if BC warmstart) fire/ship loss evaluated at the teacher's chosen target per sample; target loss unchanged |
| tests | new + existing | new parity: per-target fire/ship shapes; **residual=0 ⇒ decode + log-prob parity** (NOT byte-identical forward — shapes change); greedy actions identical when residual=0; existing `test_sufficient_commit`, `test_decisive_mass`, `test_source_selection_parity` must still pass |

**Files that should NOT need changes** (verify, don't assume): `features.py`
(feature computation is upstream of the heads), `opponent_pool.py` (uses `forward` outputs
via the same interface), `reinforce_cooldown.py` (operates on decoded moves, not logits).

---

## 7. Staged plan — separate architecture parity from veto value

**Discipline (from review):** prove the per-target heads with residual=0 reproduce old
behavior WITHOUT the veto, then add the veto for training/eval. Do NOT bundle the veto
into the architecture-parity run — the veto's +11.4pp panel gain could mask an
architecture regression. The veto is panel-validated, not LB-validated (the
revedge1+veto submission is pending as of 2026-06-18); do not treat it as a known-safe
floor to build on.

### Stage A — architecture parity, NO veto (cheap, local)
Build the new heads with residual-init (b). Prove, with `sufficient_commit_factor=0.0`:
1. `load_state_dict(strict=False)` on revedge1 4.72M loads cleanly (new params zeros).
2. With residual=0, greedy decode matches the old model on a fixed seed panel: same
   target picks, same fire decisions, same ship bins, same moves. (Parity = decode +
   log-prob equality, NOT raw forward-dict equality — shapes change, §5.)
3. With residual=0, a 16-game Ajay panel matches the baseline 23.8% within noise (no
   regression from the shape change alone).
   → verify: parity test PASS + 16-game panel ≈ 23.8% baseline (no veto).

Gate: if parity fails, the residual design is wrong — fix before any training. If the
no-veto panel regresses, the shape change broke something the residual isn't capturing.

### Stage B — short PPO smoke, NO veto (cheap, GPU)
Resume revedge1 4.72M + the new per-target heads (residual-init), `sufficient_commit_factor=0.0`,
for ~500k steps. **Isolate the architecture's effect with no mask confound.** Read:
1. `clip_frac` ≠ 0 and entropy not floored (not a frozen policy — lesson #9).
2. `fire_q/fire_k` and `ship_q/ship_k` norms climb from zero; per-target variance in
   `fire_logits_target` and `ship_logits_target` becomes nonzero (per-target signal is
   activating — the zero-init caveat from §5).
3. `dm NEUTRAL cross` rises (ship sizing to `garrison+1`), `cap/atk open<50` rises
   (fire gate opening at low-margin uncontested neutrals).
4. Held-out Ajay WR holds ≈ 23.8% baseline (no collapse from the arch change).

Gate: if clip_frac=0 / entropy floored → warmstart failed, revert to Stage A fix. If
residual norms stay at zero past ~50k steps → switch to small-norm init (§5 caveat). If
Ajay collapses → the arch change shocked the policy despite residual-init → reconsider
the compat-shim path (from-scratch Stage B instead of resume).

### Stage C — full run WITH veto (GPU, ~6M)
If Stage B passes (arch is sound without the mask): full run with
`--sufficient-commit-factor 1.0` + the new per-target heads. The veto is now a co-delta,
but Stage B already proved the arch alone doesn't regress — so any Stage C gain over
Stage B's end-state is attributable to the veto, and any gain over the revedge1+veto
panel (35.2%) is attributable to the per-target heads. Two deltas, but the attribution
is clean because Stage B isolated the arch.

Judge by the Phase 4 promotion metrics (§8). LB submission at the end.

---

## 8. Promotion metrics

Primary (the leaks this fixes):
```
failed-attack decomposition: single 0% stays, but the +1 trap (sent==garrison) -> 0
dm NEUTRAL cross up (ship sizing to garrison+1)
cap/atk open<50 up (fire gate opening at low-margin uncontested neutrals)
fire_p responds to enemy_contest (re-run the pressure probe — fire_p should MOVE)
residual norms: fire_q/k, ship_q/k, fire_scorer, ship_scorer nonzero + climbing (the
  per-target signal is activating — the zero-init caveat from §5; DEAD → warmstart failed)
```

Secondary (the wall, unchanged by this fix but must not regress):
```
hold-loss out-massed% flat-or-down (concentration wall; this fix doesn't target it)
Ajay WR up (Stage B no-veto should hold ≈ 23.8%; Stage C with veto should beat 35.2%)
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
revedge1 4.72M + veto submission (2026-06-18, **pending** — panel-validated, NOT
LB-validated) is the first LB data point for the veto alone; Phase 4's LB read comes
after Stage C. Do not claim LB validation until the submission scores.

---

## 9. Risks / open questions

- **PPO factorization change:** `p(target)·p(fire|target)·p(ship|target)` with target
  log-prob on ALL valid slots (§3), vs the old `p(fire)·p(ship)·p(target)` with target
  log-prob only on fired slots. The chain rule is mathematically clean but (a) the
  *gradient* flows differently — fire/ship now get gradient through the target choice,
  and (b) the target head now gets gradient from no-fire slots (where the target was a
  latent). Watch clip_frac / entropy in Stage B for signs the policy can't adapt.
- **Residual-init scale (zero-gradient freeze):** zeroing the residual's final layer
  means earlier residual layers (`fire_q/fire_k`, `ship_q/ship_k`, `fire_scorer`'s
  first layer) get *zero gradient* until the last layer moves off zero. Usually
  acceptable (last layer moves on the first update, then earlier layers get gradient),
  but if `fire_q/k` norms stay at zero past ~50k steps, switch to small-norm init
  (last layer ~0.01) — a tiny step-0 deviation in exchange for breaking the freeze.
  Track residual norms explicitly in Stage B (§8).
- **Fire-veto waste at top-1:** if the target head picks a target the fire head vetoes,
  the slot no-fires. If this is frequent, the policy is wasting slots. Measure
  `veto_rate` (target picked but fire_p < 0.5) in Stage B; revisit top-k if it's high.
- **Per-target compute cost:** fire/ship are now `(B, MO, N_p)` / `(B, MO, N_p, bins)`
  not `(B, MO)` / `(B, MO, bins)` — a ~24× compute increase for those heads (small vs
  the transformer, but the ship scorer is `D+D+F → D → bins` so it's
  `MO·N_p·bins·D²`-ish). Measure SPS in Stage B; the AR doc's throughput tripwire
  (SPS < 150) doesn't apply (this is one-shot, not sequential), but a >2× slowdown vs
  the factored baseline would hurt iteration economics.
- **Interaction with the veto mask:** `sufficient_commit_factor=1.0` blocks
  `sent <= garrison`. Once the ship head learns to size to `garrison+1` via the
  per-target features, the mask should rarely fire (the policy self-censors). If the
  mask is still firing frequently at 6M, the per-target signal didn't take — diagnose
  the residual norms. The mask is a safety net, not a crutch; the goal is the ship head
  learning the +1 rule natively. Stage B (no veto) tests whether the arch alone moves
  the +1 metric; Stage C (with veto) tests whether the mask still adds value on top.

---

## 10. Sequencing / gate

1. **GATE — Stage A (parity, NO veto):** build, load revedge1 4.72M, prove residual=0
   decode + log-prob match the old model (NOT byte-identical forward — shapes change,
   §5). 16-game Ajay panel ≈ 23.8% baseline. No training until parity PASS.
2. **Stage B (500k PPO, NO veto):** isolate the arch change. Read clip_frac / residual
   norms / dm NEUTRAL cross / Ajay holds ≈ 23.8%.
3. **Stage C (full run WITH veto):** add `--sufficient-commit-factor 1.0`. Read
   promotion metrics + LB submission.

One delta per stage. Stage B isolates the architecture (no mask); Stage C adds the veto
only after the arch is proven sound. Selection stays PURE: held-out Ajay WR + LB
submission, never self-play WR or shaped reward.
