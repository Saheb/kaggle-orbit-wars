# COMA Q-head — counterfactual credit assignment for the factored action

_Plan, 2026-06-22. Implements the lever from [`current_problem.md`](current_problem.md) CONCLUSION._

## ⭐⭐ STATUS (2026-06-22) — COMA / multi-source AGGREGATION REFUTED (4 converging lines). Lever = EXPANSION-SPEED (planets@50) + retention/production-snowball, NOT pile-on. See CONCLUSION.

### CONCLUSION (2026-06-22) — aggregation is not the lever; expansion + retention is
Four converging lines refute COMA's premise (recruit the idle spares to pile on):
1. **Offline Q-head** prices firing the idle spare ≤0/≈0 on all 3 substrates (hlr −3.7%, phase4e ≈0, expert-replay ≈0 — corr(Q,return)=0.99 yet flat on the marginal).
2. **Env-grounded force-fire** (`force_fire_counterfactual.py`, force the policy to fire idle spares onto contested neutrals vs Ajay) is **net-negative** — maximal AND selective-holdable (holdability filter screened 88%, still 15.6%→0% win, material 190→73/130). Firing spares drains garrison; captures don't hold.
3. **Our own won/lost split vs h12** (correct aggregator, 4044 won / 2185 lost opp-rows, `value_spare_diagnostic.py`): pile-on 3.0% won vs 4.3% lost (gap −1.3pp) — we aggregate *more* when we LOSE.
4. **Winner replay** (Jake Will WON vs Isaiah, `leader-analysis/81327125.json`, `analyze_action_list_replays.py`): `aggTurn=0.039` (3.9%, ZERO in opening, max 2 sources). A top winner barely pools — matches the 171-replay probe (~0.08).

**What the winner actually does (the lever):** Jake — planets@50=**11** (our wall ~7), planets@150=20; ships@50=**184** → ships@150=**923** (Isaiah collapsed to 4 planets / 215 ships); avg_send **47** (vs 37); 76/162 turns with launches. Edge = **faster/sustained EXPANSION + HOLDABLE captures + bigger single sends (sufficiency) + production snowball** — size-to-floor + retention, NOT pile-on. Confirms the standing planets@50 wall and [[project_aggregation_probe]]'s "sufficiency+retention not presence."

**COMA verdict: CLOSED.** A correct centralized Q prices firing the idle spares ≤0 (the env agrees), so COMA would reinforce *idling* — it's not wrong, it's not the lever. Tools (`model.py` Q-head, `force_fire_counterfactual.py`, the `value_spare_diagnostic.py` pile-on split) remain for reference. The step-1 detail below is the record of how we got here.

---


**Step 1 is done and the gate did not clear.** The Q-head, the offline action-sensitivity probe, the
per-opportunity probe, and a pure-replay dataset builder are all built, tested, and run (tools below). The
make-or-break question — *does a Q-head trained on returns price firing an idle spare as helping?* — comes
back **negative on every self-play substrate AND ≈0 on the expert-replay escape** — the Q-head route is now
exhausted across all three data substrates, so COMA-as-planned should **not** go to GPU. The decisive next
step is the env-grounded force-fire test (below).

**The numbers** (`q_fire − q_idle` on **IDLE spare-source slots at real spare-fire opportunities vs Ajay**;
> 0 = "firing the idle spare helps" = the signal `delta_V≈0` lacked; % = fraction of that checkpoint's
returns-std, the only cross-checkpoint-comparable scale):

| checkpoint (returns-std) | idle-spare `q_fire−q_idle` | fired-spare | idle `A_i` | read |
|---|---|---|---|---|
| **hlr 2M** (0.315), 24-seed, n=29762 | **−0.0117 ± 0.0002** (−3.7%) | −0.0063 (−2.0%) | +0.0016 (**wrong sign**) | firing priced negative *everywhere* |
| **phase4e 3.6M** (~1.17), 4-game, n=1643 | **−0.0039 ± 0.0015** (−0.3% ≈ 0) | **+0.0574 (+4.9%)** | −0.0010 (right sign, ≈0) | fires credited, idle-counterfactual ≈0 |
| **expert-replay (bc_jake)** (0.449), 4g all-loss/~2 eff, n=493 | **−0.0006 ± 0.0020** (−0.1% ≈ 0) | **+0.0195 (+4.3%)** | −0.0008 (≈0) | escape didn't fire; ≈0 = ambiguous (experts hold correctly) |

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

**Pure-replay escape attempt (`build_replay_returns.py`) — RAN 2026-06-22, ≈0, escape did NOT fire.**
Q-head trained on real **expert** games (Jake replays: 21 games, 11/21 balanced wins, 20k samples, returns
std 0.449, never self-play); bc_jake policy played vs Ajay (4 games, all losses, ~2 effective). It fit
outcomes essentially perfectly — **corr(Q,return) 0.991** — yet **idle-spare `q_fire−q_idle` = −0.0006 ±
0.0020 (≈0)**, agg/pooling bucket −0.0019, fired-spares +0.0195 (fires still credited). **The sharpest
result in the whole step:** a value head that predicts *who wins* at corr 0.99 is **flat on whether firing
the idle spare helps** — the wall is not a value-fitting problem; the marginal signal isn't in the data,
however well outcomes are fit. **Validity ceiling (USER, pre-registered):** experts' idle spares are idle
*correctly*, so expert data lacks the *wrongly-held-spare-that-lost* counterfactual the wall **is** ⇒ a
**positive** read would have CONFIRMED the escape; a **≈0** is **AMBIGUOUS** (can't refute "firing would
help"). So the escape didn't fire **and couldn't have refuted** — by construction. (se is optimistic: ~2
effective games, slots correlated within game; but the point estimate is glued to 0 and ≈0 is ambiguous
regardless, so a seeds-24 rerun only tightens a CI around an inconclusive answer — low value. Log:
`gpu_run_artifacts/qhead/eval_logs/replay_gate.log`.)

**The clean test — env-grounded counterfactual (DONE 2026-06-22, `force_fire_counterfactual.py`): firing the
idle spares is NET-NEGATIVE.** Persistent aggregation overlay (policy + ALWAYS fire idle spares onto contested
neutrals — the af1+agg opportunity set, pooled cheapest-first to the floor, aimed with the policy's own
intercept aimer) vs the baseline policy, paired by (seed, seat) = common random numbers, hlr 2M vs Ajay,
32 games. **Result: win-rate 15.6% → 0.0%** (prize 0 / regress 5 — the overlay LOST all 5 games the baseline
WON and flipped NONE); **planets@50 6.84 → 5.97 (Δ −0.88)**; **material@50 190.7 → 73.1 (Δ −117.6)**. Firing
the "spares" bleeds garrison: vs a forward-projector our spares aren't actually spare (they're the defense
Ajay's inbound will require), the captures don't hold, the depleted sources fall. **Three consequences:**
(1) "we fail to fire spares" is **NOT the wall** — the spares are idle *correctly*; (2) this **retroactively
confirms the offline ≤0 reads as a TRUE signal, not the on-policy confound** — the env, with no confound,
independently says firing hurts; (3) **COMA's premise is undercut**, and COMA's own math agrees — with a Q
that correctly prices firing<idling, `A_i = −p·(Q_fire−Q_idle) > 0` on idle slots → COMA would reinforce
*idling*. So COMA isn't *wrong*, it's **not the lever**. **Caveat:** the overlay is MAXIMAL/indiscriminate
aggregation (66.8 fires/game); it refutes "fire all idle spares → win," not a hypothetical narrow *selective*
aggregation — but the offline marginal said ≤0 there too, so the evidence is heavily against. **Redirect:**
the wall is force-concentration / retention, not under-aggregation (`project_force_concentration_wall`).

**⚠ CORRECTION (USER, 2026-06-22):** the maximal overlay does NOT differentiate holdable from HOPELESS
captures (it never calls `_holdable_roi` — only reach/spare/capture-cost via `_spare_sources_for_neutral`) and
sprays across ~22 neutrals/turn sized to the *static* cost, so the captures don't hold. The net-negative was
nearly guaranteed by construction and does **NOT** refute SELECTIVE aggregation — consequences (2)/(3) above
are **PREMATURE**. The real test = a SELECTIVE overlay (`--roi-min 0`: only holdable neutrals via
`_holdable_roi`, best-ROI-first, sized to floor, sources used once).

**SELECTIVE RESULT (seeds=16, roi_min=0, hlr vs Ajay).** The holdability filter screened **88%** (370
holdable-piled of 3000 idle-spare opps) — so most idle spares ARE hopeless (negative reactive ROI), and the
maximal overlay was largely spraying at them (USER point confirmed). **But firing only the holdable 12% is
STILL net-negative:** win 15.6%→0.0% (prize 0 / regress 5), planets@50 6.84→5.88 (Δ −0.97), material@50
190.7→129.7 (Δ −61, less than maximal's −117). So "hopeless captures" is **ruled out** as the whole cause —
firing the holdable spares doesn't win either. **Remaining confound (USER):** this is vs **Ajay, who
out-masses hlr regardless** = a WIN-STARVED opponent, so net-negative here can't separate "aggregation is
bad" from "Ajay is unbeatable by hlr." ⇒ **the env-vs-Ajay test is the wrong instrument.** **PIVOT
(2026-06-22):** measure/learn aggregation against a **correct-aggregator opponent at matched difficulty**
(win-gradient, not win-starved) — which also supplies the **supervised `Q^fire`** the idle-step COMA
counterfactual lacks (the OOD-collapse: `Q_t^fire` on never-fired slots is unsupervised → A_i≈0). COMA
verdict: **leaning negative but NOT airtight**; the clean test is now learning from a demonstrator, not
beating Ajay. [[feedback_win_starvation]] [[project_h14_wingradient]] [[project_force_concentration_wall]]

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
