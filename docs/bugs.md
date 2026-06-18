# Bugs

Notable bugs found and fixed in the Orbit Wars stack. One section per bug.
Each entry should record: where it lived, how we noticed, root cause, fix,
and the evidence that ruled in/out competing hypotheses.

---

## 1. Target-head collapse — `model.py` (fixed 2026-05-24)

### Where
`orbit_wars_rl/model.py`, target head (formerly `self.target_head = nn.Linear(D, max_planets)`).

### How we noticed
Three consecutive BC runs against the new heuristic teacher failed the Phase-A gate:

| Commit | val_angle_red | val_target_red | val_target_top1 |
|---|---|---|---|
| `b5dff27` teacher.py | +0.08 | — | — |
| `b73b5e6` pairwise features | +0.08 | — | — |
| `104ba57` target-index head | +0.07 | +0.27 | 0.16 |

Pairwise features moved `angle_red` by **0.00**. Pivoting from angle-bin head to
target-index head pushed `target_top1` to only 0.16 (random over 13 planets ≈ 0.077).

A teacher-consistency audit (`orbit_wars_rl/audit_teacher.py`) then showed the
teacher itself had a clean ceiling: labels decoded 100% of the time, the
recovered label sat at rank-0 of the teacher's own score 99.6% of the time, and
score ties (<5% gap between top1 and top2) occurred on only 8.9% of decisions.
So the implied BC top1 ceiling was ~100%, while the model sat at 16% — an
84-point gap that could not be explained by teacher noise.

### Root cause
The "pairwise cross-attention" path *did* compute per-target scores — but
immediately softmax-pooled them into attention weights and produced a single
slot-level enriched vector. The target head then read only this slot vector:

```python
scores = (q @ k.transpose(-2, -1))   # (B, MO, N_p) ← per-target signal here
attn = F.softmax(scores, dim=-1)     #     ← softmax destroys absolute scores
enriched = (attn @ v)                #     ← collapses to (B, MO, D)
owned_entities = ln(owned_entities + out(enriched))

target_logits = self.target_head(owned_entities)   # Linear(D, max_planets)
```

`self.target_head` was a fixed `Linear(D, max_planets)` — no per-target
conditioning. To predict planet `j`, the model had to encode "the j-th column
of W is the right one" in a single D-vector. With max_planets=48 and only
~10 owned slots, the head had no architectural pathway to use per-target
features like distance, ETA, or production.

This also explains why `angle_red` stayed at +0.08 when pairwise features were
added: the angle head reads the same per-slot vector with the same structural
problem (bin softmax over 144 angles from a slot-only embedding).

### Fix
Score each `(slot, target)` pair from its own inputs:

```python
q_tgt = self.tgt_q(owned_entities).unsqueeze(2).expand(-1, -1, N_p, -1)
k_tgt = self.tgt_k(planet_emb_post).unsqueeze(1).expand(-1, max_owned, -1, -1)
scorer_in = torch.cat([q_tgt, k_tgt, pairwise_features], dim=-1)
target_logits = self.target_scorer(scorer_in).squeeze(-1)   # (B, MO, N_p)
```

`target_scorer` is a 2-layer MLP `(D+D+F_pair) → D → 1`. The cross-attn
enrichment of `owned_entities` is kept for fire/angle/ship heads. When
`use_pairwise=False`, the old `Linear(D, max_planets)` head is the fallback.

### Evidence
- Audit showed teacher ceiling ≈ 100%, ruling out "the labels are noisy".
- Adding pairwise features earlier moved `angle_red` by 0.00, consistent with a
  structural inability of the slot-only heads to use per-target signal.
- Forward-pass smoke test post-fix: `target_logits` shape `(B, MO, max_planets)`,
  no NaNs, ~393k params (+86k vs prior).

### Lessons
- "Cross-attention" with a softmax over targets discards exactly the per-target
  signal you wanted to use. For "pick one of N", score N candidates and softmax
  over the logits — don't pool first.
- Always check that a head's input shape actually carries the variable you want
  to predict over. A `Linear(D, N)` head is correct only when N is fixed and
  position-independent; for per-entity selection it's almost always wrong.
- Multiple failed BC iterations all showing the same +0.07–0.08 reduction on
  per-target heads is a structural signal, not a data signal. Audit the model
  architecture before iterating on features or teacher.

---

## 2. Misleading in-training eval — `train_torch.py` (removed 2026-05-24)

### Where
`orbit_wars_rl/train_torch.py`, `eval_vs_baseline` (current model vs frozen
snapshot of initial policy) and the early-stop / "★ improved" logging built
around it.

### How we noticed
Several pure-self-play runs reported steadily rising `vs initial` win rates
(e.g. 60% → 75% → 80% over 15M+ steps) while replay inspection of the same
checkpoints showed the policy had collapsed to a degenerate behavior — median
fleet size = 1 ship, ~5 planets captured per 5 games vs Suneet's ~40. Local
eval (`eval.py`) against raw Suneet/Zach/Rahul confirmed no real improvement.

### Root cause
`eval_vs_baseline` plays the current policy against a deep-copy of the *initial*
policy (snapshotted at training start). In mirror self-play where "do almost
nothing" is locally safe, the current policy can drift to a degenerate behavior
that still beats the unchanged initial policy via small asymmetric advantages
(or via the initial policy's own degeneracies being slightly worse). The
metric goes up while real strength goes down.

This also drove `best_eval_winrate`, the `★ improved` flag, the
`torch_eval_best.pt` checkpoint selection, and the `--early-stop-patience`
logic — so a false-positive eval cascaded into "the best checkpoint" being
the most degenerate one.

### Fix
Removed entirely:
- `eval_vs_baseline`, `eval_vs_heuristic`, `_act_deterministic`,
  `_heuristic_moves_to_action_tensor`, `_load_heuristic`.
- `baseline_model = copy.deepcopy(model)` snapshot.
- The in-loop eval block, `best_eval_winrate`, `no_improve_evals`,
  `eval_history`, `--eval-interval`, `--eval-games`, `--eval-heuristic`,
  `--eval-heuristic-games`, `--early-stop-patience` CLI flags.

The source of truth for "is the policy actually better" is now local
`eval.py` on downloaded checkpoints against raw Suneet/Zach/Rahul.
Checkpointing still happens at `--checkpoint-interval` so every artefact
is downloadable for offline eval.

### Lessons
- Vs-frozen-initial is not a strength metric in mirror self-play. Any
  in-training proxy must hit an *independent* opponent (a strong heuristic
  or a held-out pool member with known strength).
- An eval that doesn't catch policy collapse is worse than no eval — it
  manufactures confidence. Default to fewer, stronger signals over more,
  weaker ones.
- Don't let an unvalidated metric drive checkpoint selection or early stop.
  If you can't trust it for a green light, don't trust it for those either.

---

## 3. Fire/ship head blindness — `model.py` (found 2026-06-18, fix designed, implementation pending)

### Where
`orbit_wars_rl/model.py`, `fire_head` and `ship_head`. Both read only
`owned_enriched` — the slot embedding after pairwise cross-attention pooling — and are
structurally blind to the per-target pairwise features that the target head sees.

### How we noticed
Two independent leaks surfaced in the same session, both traced to this one root cause:

**Leak A — the +1 neutral capture trap (sizing).** Replay inspection of
`seed2476_seat0_BLUE_LOSS` (and a 16-game probe across peeler_c1 2M) found launches
sending *exactly* the neutral's garrison to a neutral planet. The game's capture rule
needs `sent > garrison` strictly (`planet[5] -= survivor; if planet[5] < 0: flip`), so
`sent == garrison` leaves the neutral at 0 and *stays neutral* — a guaranteed waste of
the launch. Probe (`/tmp/probe_neutral_exact.py`): 4 of 287 neutral launches across 16
games were `sent == garrison`, and in all 4 the source garrison also equalled the neutral
garrison, so the agent *could not express* `garrison+1` from that single source at that
step (highest bin ≤ src clamps down). Confirmed via the real env
(`kaggle_environments.envs.orbit_wars.orbit_wars.py:671-674`): sent=10 vs garrison 10 →
stays neutral at 0; sent=11 → captured with 1 ship.

The ship head picks oversized bins that clamp down — it cannot size relative to the
target's garrison because it cannot *see* the target's garrison. The `capture_cost =
ships + 1` feature exists in the pairwise bundle (`features.py:208`), but it enters the
model only via `target_scorer` (`model.py:241`), a separate branch the ship head never
reads.

**Leak B — fire-head opening paralysis (selectivity).** Probing the same seed at steps
2-3 (`/tmp/probe_firehead_threshold.py`): source garrison 14, neutral garrison 13, the
ship head picked bin 19 (clamps to 14, would capture, `14 > 13`). But `fire_p = 0.003`
→ no fire. One step later, source garrison 16, `fire_p = 0.999` → fire. The fire head
learned a **garrison-margin threshold** (fire when `src - tgt >= ~3`), not the
structurally correct `src - tgt >= 1`. Perturbing the source garrison at fixed step
confirmed: `fire_p` crosses 0.5 between src-garrison 15 and 16, independent of step
index.

A follow-up probe (`/tmp/probe_firehead_pressure.py`) perturbed enemy pressure at the
step-2 obs: adding a 500-ship enemy fleet inbound to the target moved `fire_p` from
0.003 to 0.000 — i.e. **the fire head is entirely blind to enemy pressure.** The
`enemy_contest` pairwise feature (ch14, per-target enemy fleet mass) and `enemy_pressure`
(planet ch13) both exist, but `enemy_contest` feeds `target_scorer` only, and
`enemy_pressure` reaches the fire head only after being diluted through a 24-way softmax
attention over all planets — attenuated below the gradient's notice.

### Root cause
The same structural blindness `docs/bugs.md` §1 diagnosed for the target head in
2026-05-24 — "the angle head reads the same per-slot vector with the same structural
problem" — was *explicitly left unfixed* for fire and ship (§1 fix text: "The cross-attn
enrichment of `owned_entities` is kept for fire/angle/ship heads"). The angle head was
later deleted; fire and ship never received the per-target conditioning upgrade.

Concretely (`model.py:268-269`):
```python
fire_logits = self.fire_head(owned_enriched).squeeze(-1)  # (B, max_owned)
ship_logits = self.ship_head(owned_enriched)              # (B, max_owned, num_ship_bins)
```
`owned_enriched` is the slot embedding after pairwise cross-attention has softmax-pooled
over all planets. Per-target features (`enemy_contest`, `capture_cost`, `roi_20/50`)
enter the model via a *separate* branch — `target_scorer([q_slot, k_target,
pairwise_features])` (`model.py:241`) — which fire and ship do not read. So:

- **Ship** can't see the target's garrison → can't size to `garrison+1` → picks oversized
  bins that clamp, hitting the `sent == garrison` trap when src==tgt garrison.
- **Fire** can't see `enemy_contest` → learned a garrison-only margin rule → opening
  paralysis (waits for margin ≥3 when the real floor is 1, because self-play never
  punishes waiting at uncontested neutrals).

This is the same bug class as §1, in the two heads §1 explicitly deferred. Two leaks,
one root cause: per-target features live on the wrong branch.

### Fix (designed 2026-06-18, see `docs/phase4.md` for the build)
Make fire and ship per-(slot, target), reading the same `[q_slot, k_target,
pairwise_features]` inputs as `target_scorer`. Decode stays top-1 (argmax target, then
fire gate at that target, then ship bin at that target) — preserves the fire head as the
selectivity gate (Isaiah fires 3.6% of the time; we must not dissolve fire into target
selection or the policy will carpet-bomb). Log-prob factorizes as the chain rule
`p(target) · p(fire|target) · p(ship|target)`. Warmstart via residual-init so step-0 ==
old behavior; the per-target differentiation starts at zero and PPO learns it.

Considered and rejected:
- **Fire sentinel / "fire dissolves into stop"** (from `docs/autoregressive-head.md`):
  dissolves fire into the target Categorical via a no-fire sentinel. Correct for the
  full AR decoder where sequential prefix state gives `stop` an informational basis, but
  in the factored one-shot policy there's no prefix state — a Categorical over
  `[stop, target_1, ...]` would always argmax a target, destroying the fire head's
  selectivity and re-introducing the carpet-bomb Nash (Rev41-45 graveyard). Keep fire as
  a Bernoulli gate, just make it per-target.
- **Top-k target retry after fire veto:** more expressive (try target Y if fire vetoes
  target X), but changes the log-prob factorization to ordered sampling. Top-1 (accept
  the veto) matches the policy's learned behavior and is safer for the first run; revisit
  if the veto-then-no-fire waste shows up as a lever.

### Evidence
- Controlled env test (`/tmp/probe_exact_cases.py`): `sent==garrison` → no capture;
  `sent==garrison+1` → capture. Matches `orbit_wars.py:671-674`.
- 16-game probe (`/tmp/probe_neutral_exact.py`): 4/287 neutral launches were
  `sent == garrison`, all 4 with `src_garrison == neutral_garrison` (the inexpressible
  +1 case).
- Fire-head threshold probe (`/tmp/probe_firehead_threshold.py`): `fire_p` is flat to
  step index (2/3/5/10/20/50 all ≤0.004 at src_garr=14) and thresholded on src-garrison
  (0.003 at 14, 0.039 at 15, 0.649 at 16, 0.997 at 18).
- Fire-head pressure probe (`/tmp/probe_firehead_pressure.py`): `fire_p` stays in
  0.000-0.035 under enemy-planet (garr 10/50/200 near target), enemy-fleet (20/100/500
  inbound to target), enemy-fleet inbound to source (50/200), and max-combined pressure
  — fire head is blind to all of it.
- `docs/bugs.md` §1 predicted this in 2026-05-24 (line 53-55): the angle head had the
  same structural problem; fire/ship were left on the slot-only path.

### Mask mitigation shipped (eval-only, 2026-06-18)
Before the architectural fix, a one-line eval/training mask closes Leak A:
`--sufficient-commit-factor 1.0` vetoes `sent <= garrison` on attacks — exactly the +1
rule for neutrals. Already supported in `eval.py`, `train_torch.py`, `torch_env.py`, and
`action_mask.py` (the `sufficient_commit_factor` parameter). Panel A/B vs Ajay:

| checkpoint | baseline | + veto (factor=1.0) | Δ |
|---|---|---|---|
| peeler_c1 2M | 21.5% | 30.9% | +9.4pp |
| revedge1 4.72M | 23.8% | **35.2%** | +11.4pp |
| boardc1 1M | 25.8% | 30.5% | +4.7pp |

Wall metrics (`out-massed%`, `garr@loss`) flat across all three — confirms the +1 bug
was an *independent* leak from the concentration wall, not a symptom of it. The mask is
a band-aid (the policy still *generates* the bad sizing; the mask blocks execution); the
architectural fix teaches the ship head to size correctly. Submitted to LB:
revedge1 4.72M + veto (sub pending, 2026-06-18).

### Lessons
- When a head's input shape doesn't carry the variable you want to predict over, no
  amount of training will teach it. The §1 audit identified the angle head's blindness
  and explicitly flagged fire/ship as the same class — that prediction held across
  months and 30+ runs. Finish structural fixes when they're identified, don't defer half.
- "The feature exists" is not "the head can use it." `capture_cost` and `enemy_contest`
  were both in the feature bundle, but on the wrong branch. Verify the data-flow path
  from feature → head input, not just the feature's existence.
- A learned conservatism that looks like a "tuned threshold" can be a blindness
  symptom. The fire head's margin=3 rule looked like a calibration choice; the pressure
  probe revealed it was the *only* signal the head could see. When a head is blind to
  the relevant context, it will anchor on whatever strong signal it *can* see — and that
  anchor will look reasonable in isolation.
