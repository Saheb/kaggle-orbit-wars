# COMA Q-head — counterfactual credit assignment for the factored action

_Plan, 2026-06-22. Implements the lever from [`current_problem.md`](current_problem.md) CONCLUSION._

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

### Instrument from step one
- `Q(s,a) − Q(s, all-idle)` materially ≠ 0 — the Q-head reacts to actions at all.
- mean `A_i` on spare-fire opportunities **> 0** — the specific signal `delta_V≈0` lacked.
- `held@+15` on Ajay losses starts climbing — the loop unwinding.

If the first is ~0 after warmup, the Q-head isn't conditioning → add an auxiliary action-contrast loss
before burning a long run.

## Build order (cheapest falsification first)

1. **Q-head + offline action-sensitivity probe.** Add the Q-head to `model.py`; train it (Q-only, frozen
   policy) on **existing hlr 2M rollouts**, then measure `Q(s,a) − Q(s, all-idle)` and mean `A_i` on
   spare-fire states (reuse the `value_spare_diagnostic.py` rollout harness). **Gate:** action-sensitivity
   materially ≠ 0. If ~0, the design is dead before any GPU — stop or add the contrast loss.
2. **Per-slot advantage + surrogate** in `ppo.py` (un-sum; per-slot `A_i`). Unit-test that per-slot reduces
   to the old joint surrogate when `A_i` is the shared scalar.
3. **Resume run from hlr 2M.** Policy + V intact, Q-head fresh → short Q-only warmup (reuse critic-warmup
   path) before enabling the per-slot policy gradient. h12 pool, no anchor.

## Diff footprint
- `model.py`: +`action_embed`, +`sa_mlp`, +`q_fc/q_out`, +`Q(s,a)` / counterfactual method (~40 lines).
- `ppo.py`: per-slot advantage + surrogate + `Q_loss` (~25 lines, mostly un-summing `:226`).
- `train_torch.py` rollout: compute per-slot `A_i` from the Q counterfactuals; store them (per-slot
  `old_log_probs` plumbing already exists).

## Success / kill criteria
- **Gate (offline, step 1):** Q action-sensitivity ≠ 0 — else the centralized-Q premise fails; do not run.
- **Success:** mean `A_i` on spare-fire > 0 **and** `held@+15` (Ajay losses) climbs **and** out-massed%
  bends below ~96% with reinforce held — the loop unwinding, not disengagement.
- **Kill:** `A_i` stays ~0 past warmup (Q collapsed to V), or fire/launch balloons without `held@+15`
  moving (spray, the joint-coordination broke — i.e. we re-created the per-slot degenerate Nash).
