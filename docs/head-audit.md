# Head audit — fire / target / ship coordination

Date: 2026-06-17

## Why this exists

We ran the defensive-reinforce overlay as an eval-time diagnostic, not as a proposed
submission policy. The overlay forcibly fills a planner-style defensive deficit on
threatened own planets, then logs what the network would have done from those same
source planets before the override.

The point is to locate the bottleneck across the action heads:

1. fire gate: should this source act at all?
2. target head: if acting, which planet is worth affecting?
3. ship head: if acting there, how many ships?

This matters because a lot of recent work has been downstream of the first two
decisions: decisive mass, hold floors, reinforce overlays, nearest-k source selection,
and reverse-edge cooldown all assume the policy has already decided to act toward a
strategically relevant planet. The audit says that assumption is often false.

## Experiment

Checkpoint:

```bash
gpu_run_artifacts/revedge1/checkpoints/torch_step_4718592_revedge1_20260616_095906.pt
```

Opponent:

```bash
opponents/candidate_ajay_1200.py
```

Command shape:

```bash
PYTHONPATH=. /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint gpu_run_artifacts/revedge1/checkpoints/torch_step_4718592_revedge1_20260616_095906.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --panel --target-decode \
  --defensive-reinforce-k 3 \
  --defensive-reinforce-max-targets 1
```

Logs:

- `gpu_run_artifacts/def_reinf_overlay/eval_revedge1_4718592_ajay_k3.log`
- `gpu_run_artifacts/def_reinf_overlay/eval_revedge1_4718592_ajay_k3_headaudit.log`

Result:

```text
Overall: 43/256 (16.8%)
```

This is below the recent revedge1/decmass neighborhood, so the hard overlay is not a
keeper. Its value is diagnostic.

## Overlay summary

From the head-audit panel:

```text
Defensive reinforce overlay:
  threatened 36130 · fillable 3753 · forced targets 3753 moves 5203 ships 165184
  deficit before/after 159074/2387 · hopeless 25622 · blocked cooldown/mask 3614
  original policy on forced sources: no-fire 79% · same-target 6% · other-own 4% · enemy 10% · neutral 1% · undersent(if fired) 2%
  head audit on forced sources: fire_p mean 0.22 (<0.1/68%, <0.3/75%, <0.5/79%)
     target rank avg 9.6 top1/top3/top5 14%/35%/48% · ship sufficient rank avg 1.7 top1/top3/top5 79%/92%/95%
     joint ready top1/top3 5%/13%
  replaced policy moves 1096 · dropped-for-cap 0
```

Important: this is conditional on overlay interventions. It is not a global report for
all policy decisions. It asks: when the diagnostic planner found a fillable defensive
move, where was that move inside the network's native heads?

## How to read each line

### Original policy on forced sources

This is the decoded behavior before the overlay.

On the exact source planets where the overlay forced a defensive move:

- `no-fire 79%`: the source would not have launched anything.
- `same-target 6%`: the policy already selected the same defensive target.
- `other-own 4%`: the policy reinforced a different own planet.
- `enemy 10%`: the policy attacked instead.
- `neutral 1%`: the policy targeted a neutral.
- `undersent(if fired) 2%`: when it did fire, insufficient ship count was rare.

Read: the normal policy rarely naturally selects these defensive moves.

### Fire probability

```text
fire_p mean 0.22 (<0.1/68%, <0.3/75%, <0.5/79%)
```

The fire head is a Bernoulli gate per owned source. In deterministic eval, the source
fires when `sigmoid(fire_logit) > 0.5`.

So this says the fire head was usually not close to firing. The `79% < 0.5` matches the
decoded `no-fire 79%`, which makes fire/triage the strongest visible veto in this
slice.

### Target rank

```text
target rank avg 9.6 top1/top3/top5 14%/35%/48%
```

This ignores whether the source fired and asks where the overlay's chosen defensive
planet ranked inside the target logits for that source.

- `top1 14%`: the needed defensive planet was the target head's best choice only 14%
  of the time.
- `top3 35%`: it was close sometimes, but not usually dominant.
- `top5 48%`: about half the time it was at least plausible.
- `avg 9.6`: usually not high enough for deterministic action selection.

Read: target ranking is also weak. Even if we forced the source to fire, the target
head usually would not pick the needed defensive planet.

### Ship sufficient rank

```text
ship sufficient rank avg 1.7 top1/top3/top5 79%/92%/95%
```

This is not exact ship imitation. It asks: among ship bins ranked by the ship head, how
high is the first bin that would send at least the overlay's forced amount?

Read: ship amount is comparatively healthy in this defensive slice. If fire and target
were correct, the ship head usually has a sufficient amount near the top.

### Joint ready

```text
joint ready top1/top3 5%/13%
```

This is the "do the heads work as a team?" metric.

`joint top1` requires all of:

- fire probability `>= 0.5`
- forced defensive target rank `<= 1`
- sufficient ship bin rank `<= 1`

Only 5% passed.

`joint top3` relaxes target and ship to top 3, but still requires fire probability
`>= 0.5`. Only 13% passed.

Read: the heads are not coordinated on this concept. The ship head often has a usable
answer, but fire and target do not jointly select it.

## Interpretation

The hard overlay regressed, so "always fill defensive deficits" is too blunt. Many of
these save attempts are strategically bad, hopeless, or over-defensive.

But the head audit still changes the diagnosis:

- The primary bottleneck is not "how many ships should this chosen move send?"
- The upstream bottleneck is "should this source act?" and "which planet is worth
  affecting?"
- We were spending too much effort on downstream mass/floor mechanics while assuming
  fire and target had already selected the right contest.

Working bottleneck order for this defensive slice:

1. fire / triage: strongest veto
2. target ranking: weak alignment with planner-style defensive targets
3. ship sizing: comparatively okay

This also reframes opening aggression. A less conservative fire head would increase
initial activity mechanically, but a global lower fire threshold risks spam/churn. The
target should be conditional aggression: fire more when the target head has a high-value
candidate, not fire more everywhere.

## What changed in our focus

Old emphasis:

- decisive mass rewards
- hold-floor rewards
- reinforce overlays
- nearest-k source selection
- ship-count/outmass fixes

Updated emphasis:

- fire/triage credit assignment
- target ranking quality
- head coordination
- outcome-conditioned labels for "worth acting" and "worth saving"

This does not mean prior work was useless. It ruled out important downstream stories:
action grammar presence, simple reward proxies, reverse-edge ping-pong, and raw ship
under-sizing. The audit now points upstream.

## Next useful experiments

### 1. Broader natural-policy head audit

Add an audit that does not depend on the defensive overlay. For each natural policy
decision, log:

- fire probability bucket
- chosen target owner/type
- target rank of planner/winner-style candidates
- ship sufficiency against simple floors
- eventual outcome class when observable

Goal: determine whether the same fire/target split appears in attacks and openings, not
only defensive reinforcement.

Implemented as an opt-in eval diagnostic:

```bash
PYTHONPATH=. /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint <checkpoint.pt> \
  --opponent opponents/candidate_ajay_1200.py \
  --panel --target-decode \
  --natural-head-audit
```

The audit is passive. It does not alter action selection. It reports fire probability,
decoded target owner, and head agreement with two lightweight planner-like candidates:

- `attack-cand`: best cheap attack candidate available to the source.
- `save-cand`: best threatened own planet the source can plausibly reinforce in time.

Read the same way as the overlay head audit:

- low `fire>=.5` means the fire gate vetoes the candidate.
- low target top1/top3 means the target head is not ranking that candidate.
- high `ship>=req` means ship sizing is probably not the bottleneck for that candidate.
- low joint top1/top3 means the heads do not coordinate on the full action.

Full-panel read on revedge1 4.72M vs Ajay:

```text
Log: gpu_run_artifacts/head_audit/eval_revedge1_4718592_ajay_natural_headaudit.log
Overall: 61/256 (23.8%)
hold-loss out-massed 96%

Natural head audit:
  all   slots 223283 fire_p 0.12 fired 9% (<0.5 91%)
        attack-cand 124328: fire>=.5 13% · target top1/3/5 18%/35%/46% · ship>=req top1/3 82%/93% · joint top1/3 3%/6%
        save-cand   11883: fire>=.5 19% · target top1/3/5 12%/31%/46% · ship>=req top1/3 88%/95% · joint top1/3 3%/9%
  <50   slots 42808 fire_p 0.12 fired 11% (<0.5 89%)
        attack-cand 10799: fire>=.5 30% · target top1/3/5 14%/33%/49% · ship>=req top1/3 90%/96% · joint top1/3 6%/14%
        save-cand   867: fire>=.5 28% · target top1/3/5 13%/31%/45% · ship>=req top1/3 86%/97% · joint top1/3 6%/13%
```

Read: the broad natural-policy audit agrees with the overlay audit. Ship sufficiency
is usually available near the top of the ship head, but fire and target rarely agree
with attack/save candidates. The opening is slightly less fire-gated (`attack-cand
fire>=.5 30%`) but still target-limited (`top1 14%`) and poorly coordinated (`joint
top1 6%`).

### Candidate-quality validation vs Producer-v2

The natural audit's lightweight candidates are not automatically ground truth. We
validated them against the latest local Producer-v2 candidate (`opponents/candidate_producer_v2.py`)
with:

```bash
PYTHONPATH=. /Users/saheb/home/.venv/bin/python orbit_wars_rl/validate_head_audit_candidates.py \
  --checkpoint gpu_run_artifacts/revedge1/checkpoints/torch_step_4718592_revedge1_20260616_095906.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 8 --max-step 120 --state-stride 3 --max-states 800 \
  --output-json gpu_run_artifacts/head_audit/candidate_validation_producerv2_8g.json \
  --output-md gpu_run_artifacts/head_audit/candidate_validation_producerv2_8g.md
```

Ajay-state sample, 288 recorded states:

```text
attack: producer_n=833 light_n=601 both_n=481
  producer missed by light 42.3%
  light extra vs producer 20.0%
  light target not in producer shortlist 35.3%
  when present: producer rank avg 1.4
  top1/top3/top5 over both cases: 46.2% / 63.2% / 64.7%

save: producer_n=266 light_n=64 both_n=40
  producer missed by light 85.0%
  light extra vs producer 37.5%
  light target not in producer shortlist 40.0%
  top1/top3/top5 over both cases: 50.0% / 60.0% / 60.0%
```

Producer-v2-opponent sample, 297 recorded states:

```text
attack: producer_n=738 light_n=603 both_n=433
  producer missed by light 41.3%
  light extra vs producer 28.2%
  light target not in producer shortlist 44.6%
  when present: producer rank avg 1.4
  top1/top3/top5 over both cases: 37.6% / 54.3% / 55.4%

save: producer_n=67 light_n=11 both_n=3
  producer missed by light 95.5%
  light extra vs producer 72.7%
  top1/top3/top5 over both cases: 100% / 100% / 100%  (n=3 only)
```

Verdict:

- The lightweight `attack-cand` is a useful rough diagnostic: when it overlaps
  Producer-v2's same-source candidate set, it is usually near the top, but it misses
  about 40% of Producer-v2 attack opportunities and often picks targets outside
  Producer-v2's valid shortlist.
- The lightweight `save-cand` is not a good Producer-v2 proxy. It misses most
  Producer-v2 same-source defensive candidates. Do not use it as a supervised label.
- The broad head-audit conclusion still has value because it agrees with the separate
  hard-overlay audit: ship sufficiency looks much healthier than fire/target agreement.
  But for training labels, use Producer-v2's real candidate ranking, not the lightweight
  save heuristic.

Next correction: add a Producer-v2-backed head audit or offline label builder that uses
the real Producer-v2 candidate enumeration for attack/save targets. The current
lightweight audit should remain a cheap smoke diagnostic, not the source of truth.

## Producer-v2 head-label dataset v0

Built the first Producer-v2-backed label dataset on our visited states:

```bash
PYTHONPATH=. /Users/saheb/home/.venv/bin/python orbit_wars_rl/build_producerv2_head_labels.py \
  --checkpoint gpu_run_artifacts/revedge1/checkpoints/torch_step_4718592_revedge1_20260616_095906.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 8 --max-step 120 --state-stride 3 --max-states 800 \
  --samples-out gpu_run_artifacts/head_audit/producerv2_head_labels_8g.pkl \
  --summary-out gpu_run_artifacts/head_audit/producerv2_head_labels_8g.json \
  --md-out gpu_run_artifacts/head_audit/producerv2_head_labels_8g.md
```

Artifacts:

- `gpu_run_artifacts/head_audit/producerv2_head_labels_8g.pkl` (19 MB)
- `gpu_run_artifacts/head_audit/producerv2_head_labels_8g.json`
- `gpu_run_artifacts/head_audit/producerv2_head_labels_8g.md`

Dataset shape:

```text
states/samples: 297
candidate labels: 434 sources = 263 attack + 171 save
selected labels: 547 sources = 317 attack + 230 save
selected decode failed: 4
multi-source conflicts after single-action projection: 0
```

Two label families are stored:

- `candidate_*`: Producer-v2's best same-source attack/save candidate above its ROI
  threshold (`score >= 1.5`), choosing the higher-scoring attack/save if both exist.
- `selected_*`: what a fresh Producer-v2 runtime actually emits on the same state,
  projected into our one-action-per-source interface. If Producer-v2 emits multiple
  moves from one source, keep the largest send. In this sample that conflict count was
  zero.

Baseline model audit against these real Producer-v2 labels:

```text
candidate attack n=263: fire>=.5 21.7% · target top1/3/5 34.6/53.6/62.7% · ship top1/3 60.1/81.0% · joint top1/3 9.1/17.9%
candidate save   n=171: fire>=.5 18.1% · target top1/3/5 18.1/37.4/48.0% · ship top1/3 57.9/73.7% · joint top1/3 2.9/12.3%
selected attack  n=317: fire>=.5 19.9% · target top1/3/5 18.0/29.3/36.9% · ship top1/3 60.9/78.9% · joint top1/3 5.7/10.7%
selected save    n=230: fire>=.5 16.5% · target top1/3/5 18.3/36.1/45.7% · ship top1/3 63.9/81.7% · joint top1/3 3.0/9.1%
```

Read: with real Producer-v2 labels, the diagnosis still holds but the ship-head story
is less rosy than the lightweight audit made it look. Ship top1 is ~58-64% and top3
~74-82%, still much healthier than fire readiness (~16-22%) and joint top1 (~3-9%).
Target is particularly weak on selected Producer-v2 actions. This supports a narrow
fire+target supervised/auxiliary objective, with ship loss optional and lower priority.

### 2. Outcome-conditioned triage labels

Build labels around whether an action would have been strategically valuable:

- good attack candidate: cheap, valuable, reachable, holdable or favorable trade
- good save candidate: threatened, saveable, valuable, not hopeless
- hopeless own planet: do not reinforce
- low-value distraction: do not fire

These are better targets than "fill every deficit" because the overlay showed blind
defense can hurt.

### 3. Auxiliary triage / target heads

Add auxiliary prediction heads on the shared encoder, trained with supervised labels,
without necessarily changing the action interface at first.

Candidate heads:

- per-owned-source `should_act_now`
- per-source/target `target_value`
- per-own-planet `saveable_threat`
- per-own-planet `hopeless_threat`

The purpose is representation shaping: teach the shared trunk the concepts that fire
and target need to coordinate on.

### 4. Policy-constrained inference tests

Before another long training run, test narrow eval-time constraints that require head
agreement instead of overriding everything:

- only boost fire when target rank/value is high
- only allow defensive reinforce when the target is predicted saveable
- suppress reinforce to hopeless targets
- compare head agreement in wins vs losses

Success criterion is not just WR. We need the structural metrics to move:

- opening `<50` useful fire / capture pressure
- target rank against valuable candidates
- joint ready rate
- out-massed percentage on lost captures
- reinforce mass wasted on lost/hopeless planets

## Current conclusion

The ad-hoc overlay did not produce a better policy, but it produced a better diagnosis.

The agent's heads are not yet acting like a coordinated decision system. In the audited
defensive slice, the ship head mostly knows how to send enough once a move is chosen.
The fire and target heads usually fail to select the move in the first place.

Next serious work should therefore target fire/triage and target ranking, with ship
sizing treated as secondary unless a new audit contradicts this.
