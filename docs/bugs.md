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
