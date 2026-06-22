# COMA Q-head — counterfactual credit assignment for the factored action

_Plan, 2026-06-22. Implements the lever from [`current_problem.md`](current_problem.md) CONCLUSION._

## ⭐⭐ STATUS (2026-06-22) — step 1 (offline gate) BUILT + RUN; verdict NEGATIVE on every self-play substrate

**Step 1 is done and the gate did not clear.** The Q-head, the offline action-sensitivity probe, the
per-opportunity probe, and a pure-replay dataset builder are all built, tested, and run (tools below). The
make-or-break question — *does a Q-head trained on returns price firing an idle spare as helping?* — comes
back **negative on every self-play substrate**, so COMA-as-planned should **not** go to GPU yet.

**The numbers** (`q_fire − q_idle` on **IDLE spare-source slots at real spare-fire opportunities vs Ajay**;
> 0 = "firing the idle spare helps" = the signal `delta_V≈0` lacked; % = fraction of that checkpoint's
returns-std, the only cross-checkpoint-comparable scale):

| checkpoint (returns-std) | idle-spare `q_fire−q_idle` | fired-spare | idle `A_i` | read |
|---|---|---|---|---|
| **hlr 2M** (0.315), 24-seed, n=29762 | **−0.0117 ± 0.0002** (−3.7%) | −0.0063 (−2.0%) | +0.0016 (**wrong sign**) | firing priced negative *everywhere* |
| **phase4e 3.6M** (~1.17), 4-game, n=1643 | **−0.0039 ± 0.0015** (−0.3% ≈ 0) | **+0.0574 (+4.9%)** | −0.0010 (right sign, ≈0) | fires credited, idle-counterfactual ≈0 |

**Mechanism — two faces of the same wall:**
- **hlr = the on-policy confound.** Firing priced negative everywhere, and *most* negative in the games we
  **WIN** (−0.0149) — because we win **by** the under-aggregating hold-and-grind, so the data says
  *idle→win* and the Q faithfully entrenches it. A centralized Q regressed on the fixed point's own data
  reproduces the fixed point. **This also kills the Jake-BC self-play *warmup* idea (USER):** self-play
  *data generation* is the confound source, independent of the policy — a BC-from-replays rolled out in
  self-play is confounded the same way; the warmup only fits a value head, it doesn't change the data.
- **phase4e = the OOD counterfactual.** A stronger policy correctly credits its **own** fires (+4.9%) but
  the idle-spare counterfactual is still ≈0 — the make-or-break risk realized: `Q_i^fire`/`Q_i^idle` on the
  not-taken branch are **pure generalization, never supervised**, and that is the 87%-idle population COMA
  must recruit. *Firing the spare is valued where the policy fires; it can't be valued where the policy
  never fires.*

**Pure-replay escape attempt (`build_replay_returns.py`, IN PROGRESS — result pending) + its ceiling.**
Train the Q-head on real **expert** games (Isaiah/Jake/…, mixed win/loss, terminal ±1 returns, never
self-play). The head fits outcomes (corr 0.757 — but that is the *overall* outcome fit, dominated by
game-phase / who's-ahead, **NOT** the idle-spare marginal; the gate isolates the marginal). **Validity
ceiling (USER):** experts' idle spares are idle *correctly*, so expert data lacks the
*wrongly-held-spare-that-lost-the-game* counterfactual that the wall actually **is**. ⇒ a **positive**
idle-spare read would CONFIRM the escape (expert value says our holds are mistakes); a **≈0 / negative** is
**ambiguous** (could be correct-to-hold, not a confound). **This test can confirm the escape, not cleanly
refute it.**

**The clean test (identified, not built): env-grounded counterfactual.** Take our loss states with an idle
spare, *force-fire it onto the contested target*, simulate forward, compare win rate vs the idle branch.
Measures "would firing the idle spare have won" **directly** — no Q-head, no confounded data, no
expert-correctness ambiguity. The falsification that sidesteps everything above.

**Tools (all in tree):** `model.py` Q-head (`q_counterfactual` + `_q_slot_tokens`; `tests/test_q_head.py`
4/4 — additive-pool-delta parity), `train_torch.py --dump-rollout-and-exit` (warmup-gated),
`q_head_offline_probe.py` (broad-population gate), `q_head_opportunity_gate.py` (per-opportunity gate vs
Ajay), `build_replay_returns.py` (raw replay → (state, action, terminal-return) batch). Gate logs under
`gpu_run_artifacts/qhead/eval_logs/`.

---

> **Design-review reconciliation (2026-06-22):** a parallel design proposed a "silence the slot"
> counterfactual (`Q(s,a) − Q(s, a₋ᵢ)`) with state/action pooled in separate streams and a target-index
> embedding table. Rejected on review: silence is **not** the COMA baseline (COMA marginalizes agent i's
> action under its *own policy* = the fire-marginalized form below) and gives **no gradient to the idle
> 87%**; separate pools drop the per-slot **state⊗action binding**; an index table embeds a positional
> planet id (a different planet each game) instead of the target's learned embedding. Adopted from it:
> **register the new Q-head keys** (load-bearing — see Diff footprint) and the **co-firing degeneracy**
> risk + cross-attention upgrade path (below).

## Why

The wall is a **credit-assignment fixed point**. `ppo.py` puts a single scalar GAE advantage on the
*summed* joint log-prob (`ppo.py:226` sums fire+ships+target over all slots; `ppo.py:243`
`surr = ratio * advantages`). So every source-slot's fire is credited with the **whole-state** advantage,
never its marginal contribution → `delta_V ≈ 0` on spare-fire → no gradient toward aggregating → the
under-aggregation Nash. idle-spare (`delta_V≈0`) and retention (`held@+15=47%`) are one loop: the fire's
advantage is diluted to ~0 by the policy's own ~53% lose-back.

Reward and data families are exhausted (see CONCLUSION). The untried, un-falsified lever is **COMA-style
counterfactual credit**: keep the joint surrogate (the stable coupling), but give each source-slot a
**per-slot counterfactual advantage** `A_i` from a **centralized** Q-critic — pricing each source's marginal
contribution, which is exactly what `delta_V≈0` shows is missing.

**Why this ≠ the shelved per-slot/VDN work** (which produced a degenerate Nash): that decomposed the
**value** into local per-slot values → independent optimizers → no coordination. COMA keeps **Q
centralized/joint** (credit = marginal to the *joint* outcome) and factorizes **only the baseline**.
Centralized-Q vs local-value is the whole difference, and it's the untested combination.

## The graft — a Q-head that mirrors the value head

The trunk already produces everything the Q-head needs, computed **once** and shared with the policy:
- `owned_enriched` (B, MO, D) — per-slot representation (`model.py:227`)
- `global_token` = `x[:, 0, :]` (B, D)

The existing value head is `concat[global_token(D), owned_pool(D)] → D → D/2 → 1` (`model.py:146-148`,
`owned_pool = mean(owned_enriched valid)`). The Q-head is its shape-twin, plus a per-slot **action
embedding**:

```
# new params (all small):
action_embed:  fire_bit(1) + target_emb(D = planet_emb_post[tgt_i]) + ship_bin_emb(32→d)  → a_emb_i
sa_mlp:        Linear(2D → D)                     # per-slot state-action token
q_fc, q_out:   Linear(2D → D), Linear(D → 1)      # twin of value_fc1/value_out

sa_i        = sa_mlp(concat[owned_enriched_i, a_emb_i])          # (B, MO, D)
action_pool = mean_valid(sa_i)                                  # (B, D)   mirrors owned_pool
Q(s, a)     = q_out(gelu(q_fc(concat[global_token, action_pool])))   # (B,)
```

**The additive pool is the efficiency trick.** `action_pool` is a mean of per-slot tokens, so a per-slot
counterfactual perturbs only *one* term:

```
pool_i^cf = action_pool + (sa_i^cf − sa_i) / n_valid
```

All MO counterfactuals = the base trunk pass (already done) + **one batched** `q_fc/q_out` MLP call over
`(B·MO, 2D)`. No 16× transformer forwards — the transformer encoding is reused. That's what keeps it from
blowing up over the factored 16-slot action.

## The counterfactual baseline — marginalize FIRE, not "silence"

A naive `A_i = Q(s,a) − Q(s, a with source_i silenced)` only credits slots that **fired**. But the wall is
the **idle** 87%, and silencing an already-idle slot is a no-op → `A_i = 0` for exactly the sources we need
to activate. It gives no gradient to *start* firing.

Use the **fire-marginalized** baseline (2 evals/slot, both via the cheap pool delta):

```
Q_i^fire = Q(s, a | slot i = its argmax fire)     # i fires (its sampled target/ship)
Q_i^idle = Q(s, a | slot i = idle)
p_i      = sigmoid(fire_logit_i)
baseline_i = p_i·Q_i^fire + (1 − p_i)·Q_i^idle
A_i      = Q(s, a) − baseline_i
```

The two cases are the whole point:
- **fired slot:** `A_i = (1 − p_i)·(Q_i^fire − Q_i^idle)` → positive when firing helped → reinforces it.
- **idle slot:** `A_i = −p_i·(Q_i^fire − Q_i^idle)` → **negative when firing would've helped → `∇log π(idle)·A_i`
  pushes P(idle) down → activates the source.**

That bidirectional term on idle sources is what `delta_V≈0` is missing and silence-only cannot supply.
v1 marginalizes **fire** (the wall); target/ship ride on the same `A_i` through the joint per-slot log-prob.
Per-head target/ship baselines are a later refinement.

## PPO integration — un-sum the joint surrogate

Today `ppo.py:226` sums all heads/slots into one log-prob × one scalar `A`. Go per-slot:

```
ratio_i = exp( (logp_fire_i + logp_tgt_i + logp_ship_i) − old_log_prob_i )    # per slot
surr_i  = min(ratio_i · A_i, clip(ratio_i, 1±ε) · A_i)
policy_loss = −(surr_i * slot_valid).sum(dim=1).mean()
```

Half the infra exists: per-slot fire log-probs and `slot_valid_2d` are already computed for the clip-frac
metric (`ppo.py:287-290`). Rollout already stores per-head `old_log_probs`; add per-slot `A_i` storage.

## Q training — keep V + GAE, add Q

Minimal disruption: **keep the V head and GAE** to produce the return target (`batch["returns"]`,
`ppo.py:250`), and add `Q_loss = MSE(Q(s,a), returns)`. V stays the bootstrap; Q is the action-conditional
critic the counterfactual reads from. (Full SARSA / drop-V is a later cleanup.)

## The make-or-break risk

On-policy data only ever shows Q the **taken** action → its return. `Q_i^fire` / `Q_i^idle` for the
not-taken branch are **pure generalization** — never directly supervised. **If the Q-head collapses toward V
(action-insensitive), `A_i → 0` and we are back at square one.** The additive per-slot token structure biases
toward sensitivity (each slot's action visibly enters its own term), but it is not guaranteed.

Second-order risk (from the CONCLUSION): even with a working Q-head, the recovered marginal is **small**
(47% retention, mirrored in self-play). COMA cleans the SNR; the bet is it **bootstraps** (better credit →
fire more → hold more → bigger signal). Necessary, maybe not sufficient.

### Co-firing degeneracy — the pool's blind spot
The per-slot tokens are **mean-pooled** into `action_pool`, which is what buys the cheap additive
counterfactual — but pooling is permutation-invariant, so **two slots firing at the same target are
largely indistinguishable from one slot firing "harder."** That is exactly the structure the wall is
about (multi-source pile-on onto one contested target), so a pool that washes it out is a real risk, not
cosmetic. v1 ships the pool anyway (the minimum that prices the `delta_V≈0` gap; the COMA bet is that the
marginal is small-but-recoverable). Guards:
- **Offline (step 1):** at a spare-fire opportunity, compare `Q(s, a + one idle spare forced to fire at
  the contested target)` vs `Q(s, a)`. If that "second-source" delta collapses toward the single-source
  delta, the pool can't see aggregation → escalate before any GPU.
- **Upgrade if degenerate:** swap the mean pool for **cross-attention from the state encoding to the 16
  per-slot action tokens** (re-attend per counterfactual; still 16 evals + one backbone pass, ~16× the
  attention cost only). Everything else — fire-marginalized baseline, state⊗action binding, per-slot
  surrogate — is unchanged.

### Instrument from step one
- `Q(s,a) − Q(s, all-idle)` materially ≠ 0 — the Q-head reacts to actions at all.
- mean `A_i` on spare-fire opportunities **> 0** — the specific signal `delta_V≈0` lacked.
- **co-firing delta** doesn't collapse to the single-source delta — the pool can see aggregation.
- `held@+15` on Ajay losses starts climbing — the loop unwinding.

If the first is ~0 after warmup, the Q-head isn't conditioning → add an auxiliary action-contrast loss
before burning a long run.

## Build order (cheapest falsification first)

1. **Q-head + offline action-sensitivity probe.** Add the Q-head to `model.py`; train it (Q-only, frozen
   policy) on **existing hlr 2M rollouts**, then measure `Q(s,a) − Q(s, all-idle)` and mean `A_i` on
   spare-fire states (reuse the `value_spare_diagnostic.py` rollout harness). **Gate:** action-sensitivity
   materially ≠ 0. If ~0, the design is dead before any GPU — stop or add the contrast loss.
   Also register `q_*` in `PHASE4_COMPAT_MISSING_KEYS` and instrument the **co-firing delta** in the probe.
   Q-target = the real `torch_env`+GAE `returns` (dump a rollout `batch`), *not* a kaggle-env terminal-only
   MC return — sparse/high-variance targets risk a false-negative gate.
2. **Per-slot advantage + surrogate** in `ppo.py` (un-sum; per-slot `A_i`). Unit-test that per-slot reduces
   to the old joint surrogate when `A_i` is the shared scalar.
3. **Resume run from hlr 2M.** Policy + V intact, Q-head fresh → short Q-only warmup (reuse critic-warmup
   path) before enabling the per-slot policy gradient. h12 pool, no anchor.

## Diff footprint
- `model.py`: +`q_fire_embed`/`q_ship_embed`/`q_tgt_proj` (per-slot action embed), +`q_sa_mlp`
  (state⊗action token), +`q_fc`/`q_out` (Q from `[global_token, action_pool]`), +`q_counterfactual()`
  (~55 lines). **Register the new `q_*` keys in `PHASE4_COMPAT_MISSING_KEYS`** (model.py:22) — else every
  resume/eval/export aborts (`_load_phase4_compatible` train_torch.py:71, `export_agent.py:505`,
  `eval.py:2388` all *raise* on an unregistered missing key). **Zero-init `q_out`** → Q≈0, `A_i≈0` at
  step 0 (no policy disruption pre-calibration; mirrors the Phase-4 residual zero-init).
- `ppo.py`: per-slot advantage + surrogate + `Q_loss` (~25 lines, mostly un-summing `:226`).
- `train_torch.py` rollout: compute per-slot `A_i` from the Q counterfactuals; store them (per-slot
  `old_log_probs` plumbing already exists).

## Success / kill criteria
- **Gate (offline, step 1):** Q action-sensitivity ≠ 0 — else the centralized-Q premise fails; do not run.
- **Success:** mean `A_i` on spare-fire > 0 **and** `held@+15` (Ajay losses) climbs **and** out-massed%
  bends below ~96% with reinforce held — the loop unwinding, not disengagement.
- **Kill:** `A_i` stays ~0 past warmup (Q collapsed to V), or fire/launch balloons without `held@+15`
  moving (spray, the joint-coordination broke — i.e. we re-created the per-slot degenerate Nash).
