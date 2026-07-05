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

## Defensive overlay value gate

Correction: the first defensive-reinforce overlay was not blind "defend everything."
It already skipped already-safe targets, skipped hopeless targets, and respected
reinforce cooldown/masks. The missing condition was not "is this saveable?" but "is this
worth saving after accounting for the attack opportunity we are giving up?"

Added an opt-in value/opportunity gate:

```bash
--defensive-reinforce-value-margin <margin>
```

Unset preserves the original overlay. With `0`, the overlay only forces a save when:

```text
save_value - foregone_attack_value >= 0
```

Current approximation:

- `save_value`: target production over the defensive horizon, plus a small preserved
  garrison/inbound term, minus deficit and urgency cost.
- `foregone_attack_value`: sum of positive same-source attack-candidate scores for the
  nearest-k sources the overlay would consume.

This is still a heuristic, not a learned value model. Its purpose is to test whether
the missing concept is "net worthwhile save" rather than just "fillable save."

Full-panel Ajay comparison on revedge1 4.72M:

```text
control, no overlay:                  61/256 = 23.8%
ungated k=3 defensive overlay:        43/256 = 16.8%
value-gated k=3 overlay, margin 0:    59/256 = 23.0%
```

Value-gated overlay summary:

```text
threatened 35916 · fillable 2581 · forced targets 2581 moves 3349 ships 75556
hopeless 25339 · blocked cooldown/mask 2469
value gate: checked 5121 · skipped 2309 (45%) · avg save/opportunity/net 15.4/11.7/3.7
original policy on forced sources: no-fire 80% · same-target 7% · other-own 4% · enemy 8%
```

Ungated comparison:

```text
threatened 36130 · fillable 3753 · forced targets 3753 moves 5203 ships 165184
original policy on forced sources: no-fire 79% · same-target 6% · other-own 4% · enemy 10%
```

Read:

- The value gate cut forced targets by ~31% and forced ships by ~54% versus ungated.
- The value gate removed nearly all of the hard-overlay regression (`16.8% -> 23.0%`),
  but it did not clearly beat the no-overlay control (`23.8%`).
- The key label is not binary `saveable`; hopeless-skipping was already tested and was
  insufficient. The better target is `worth_saving_net_of_attack`.
- This is strong evidence that a save value/ROI concept is the right diagnostic target,
  but still weak evidence for an eval-time hard override. The safer use is a PPO
  auxiliary/regularizer so reward can reject bad saves instead of forcing them.

### Sufficiency-arm follow-up

The value-gated overlay only tests selection plus a one-shot snapshot floor. It does not
settle whether the remaining failure is:

- aggregate arrival sufficiency: the save was right, but under-massed;
- retention: the save arrived, but follow-up defense was not sustained;
- saves are not the lever: the attack-bias was roughly correct.

Added:

```bash
--defensive-reinforce-overfill <multiplier>
```

This leaves target selection and the value gate unchanged, then multiplies the forced
deficit after selection. The eval summary logs realized fill:

```text
realized fill: forced/requested X/Y (Zx) · full targets A/B
```

The null is valid only if requested overfill actually lands. If realized fill is much
below the requested arm, increase `k` before interpreting WR.

Pre-registered arms:

```text
k=3, value margin 0, overfill 1.25
k=3, value margin 0, overfill 1.50
k=5, value margin 0, overfill 2.00
```

Readout priority:

1. realized fill ratio: did the arm actually overfill?
2. reinforce mass to lost planets / forced-planet hold quality: did larger saves survive?
3. WR versus control: at 256 games, treat only roughly `>=28-29%` as meaningful positive
   signal over the `23-24%` control neighborhood.

Interpretation:

- mass-to-lost improves but WR ties: sufficiency is real, but too expensive; need cheaper
  or more selective save-value training, not a hard override.
- mass-to-lost stays flat: one-shot arrival mass is not enough; retention remains live.
- mass-to-lost improves and WR clears the bar: aggregate hold-mass is a real lever.
- mass-to-lost flat and WR regresses: defensive forcing is stealing tempo; shift back to
  attack/opening selection.

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

## Producer-v2 supervised head-label probe

Ran a narrow offline probe on the `selected_*` labels. This is not yet a proposed
submission checkpoint; it asks whether the existing heads can be moved toward real
Producer-v2 actions on our visited states.

The probe freezes the trunk and trains only the action heads:

```bash
PYTHONPATH=. /Users/saheb/home/.venv/bin/python orbit_wars_rl/train_producerv2_head_labels.py \
  --checkpoint gpu_run_artifacts/revedge1/checkpoints/torch_step_4718592_revedge1_20260616_095906.pt \
  --samples gpu_run_artifacts/head_audit/producerv2_head_labels_8g.pkl \
  --label-source selected --steps 400 --batch-size 32 --lr 0.0001 \
  --fire-pos-weight 32 --fire-coef 2 --trainable heads \
  --summary-out gpu_run_artifacts/head_audit/producerv2_head_ft_selected_heads_fire32_400.json
```

Comparison against selected Producer-v2 labels:

| run | fire>=.5 | target top1 | target top3 | ship top1 | ship top3 | joint top1 | joint top3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 18.5% | 18.1% | 32.2% | 62.2% | 80.1% | 4.6% | 10.1% |
| heads 200, fire weight 8 | 20.1% | 32.7% | 55.2% | 62.2% | 80.1% | 6.6% | 11.0% |
| heads 400, fire weight 32, fire coef 2 | 23.0% | 34.9% | 57.2% | 62.2% | 80.1% | 8.0% | 13.5% |

Read:

- Target ranking is very movable with a small supervised head-only objective.
- Fire readiness moves, but much less than target, even with strong positive weighting.
- Ship is unchanged because this probe did not train ship loss.
- Joint readiness improves, but remains low because fire is still a bottleneck.

This supports the new direction, but also warns against overselling the first probe.
The immediate next test should save a conservative head-tuned checkpoint and run eval
before deciding whether to involve the trunk, add ship loss, or convert this into an
auxiliary objective during PPO.

Saved-checkpoint evals, 64 games vs Ajay:

```text
control revedge1 4.72M:                 13/64 = 20.3%
heads 200, fire weight 8 checkpoint:     7/64 = 10.9%
heads 400, fire weight 32 checkpoint:    3/64 =  4.7%
```

The gameplay regression is visible in the diagnostics, not just the win count:

| run | fire frac | cap/attack | reinf share | peel-rate | reinforce mass to lost planets | hopeless reinforce share |
|---|---:|---:|---:|---:|---:|---:|
| control | 0.25 | 0.548 | 0.31 | 0.80 | 31% | 12% |
| heads 200 | 0.25 | 0.601 | 0.39 | 0.87 | 46% | 14% |
| heads 400 | 0.27 | 0.628 | 0.46 | 0.94 | 68% | 19% |

Read: direct selected-label head tuning improves offline overlap, but as a standalone
checkpoint it makes the policy worse. It increases capture/attack conversion but
destabilizes retention and sends far more reinforcement mass to planets that still die.
That points away from "just fine-tune the action heads and submit it" and toward one
of:

- use Producer-v2 labels as a small auxiliary loss during PPO, so the value/reward loop
  can reject bad imitations;
- train a separate triage/value head instead of directly overwriting fire/target;
- split selected labels by outcome/value, especially to distinguish useful save actions
  from Producer-v2 actions that are locally reasonable but globally bad in our policy's
  state distribution.

## Rank1 replay winner-vs-loser head audit

To remove the proxy-agent problem, added a replay-backed audit:

```bash
PYTHONPATH=. /Users/saheb/home/.venv/bin/python orbit_wars_rl/audit_replay_head_labels.py \
  --checkpoint gpu_run_artifacts/revedge1/checkpoints/torch_step_4718592_revedge1_20260616_095906.pt \
  --output-json gpu_run_artifacts/head_audit/replay_head_audit_rank1_1v1_revedge1_4718592.json \
  --output-md gpu_run_artifacts/head_audit/replay_head_audit_rank1_1v1_revedge1_4718592.md
```

This reads rank1 1v1 replays, uses `steps[t-1][seat].observation` as the state
for `steps[t][seat].action`, projects the replay action list into our current
one-action-per-source/top-16 interface, then forwards our policy heads on the replay
state. It compares winner moves against loser moves from the same replay corpus.

Artifacts:

- `gpu_run_artifacts/head_audit/replay_head_audit_rank1_1v1_revedge1_4718592.json`
- `gpu_run_artifacts/head_audit/replay_head_audit_rank1_1v1_revedge1_4718592.md`

Run quality:

```text
rank1 1v1 replays used: 116
errors: 0
```

Head agreement:

| side | labels | attack/save | fire>=.5 | target top1/3/5 | ship top1/3 | joint top1/3 | target rank avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| winner | 7282 | 5196/2086 | 24.0% | 15.8/32.4/42.7% | 81.0/94.3% | 4.6/9.5% | 10.5 |
| loser | 11140 | 8457/2683 | 10.4% | 10.4/23.1/31.4% | 72.0/91.5% | 1.8/3.6% | 13.0 |

This aggregate is between-distribution: winner states and loser states are different
boards. The winner fire edge can partly mean "our policy fires more on ahead/easier
boards," not necessarily "our policy prefers winner-quality moves." Treat the aggregate
winner-vs-loser gap as consistent with representation, not a clean magnitude estimate.

Opening slice:

| side | labels | fire>=.5 | target top1/3/5 | joint top1/3 |
|---|---:|---:|---:|---:|
| winner <50 | 1536 | 38.5% | 24.0/44.4/54.6% | 9.9/19.3% |
| loser <50 | 3417 | 15.3% | 13.0/29.5/39.5% | 3.1/6.2% |

Attack/save split, all phases:

| side | attack target top1/3 | save target top1/3 | attack fire>=.5 | save fire>=.5 |
|---|---:|---:|---:|---:|
| winner | 20.2/39.1% | 4.8/16.0% | 24.4% | 23.0% |
| loser | 11.4/24.6% | 6.9/18.3% | 12.0% | 5.3% |

Read: the winner preference is attack-led. On save targets, winner target top1/top3 is
not better than loser. That matches the earlier overlay/Producer-v2 finding: defensive
target selection is still the hard gap.

Same-state target baseline:

The audit also compares each replay move against a same-source baseline from the same
state: keep the replay source and ship count, but replace the target with the nearest
non-own target. This isolates target ranking from board/source/fire confounds. It does
not control the fire head because fire readiness is source-level and therefore identical
for the replay target and the same-source baseline.

| side | phase | replay target top1/3 | baseline target top1/3 | replay rank | baseline rank |
|---|---|---:|---:|---:|---:|
| winner | all | 15.8/32.4% | 12.7/28.8% | 10.5 | 9.7 |
| winner | open | 24.0/44.4% | 8.9/23.8% | 8.5 | 9.3 |
| winner | mid | 12.5/27.3% | 12.2/25.9% | 11.7 | 10.7 |
| winner | late | 16.4/33.7% | 17.4/40.3% | 9.6 | 7.6 |
| loser | all | 10.4/23.1% | 9.5/20.4% | 13.0 | 13.0 |

Same-state target baseline by move kind:

For save labels, this is not an apples-to-apples "winner save target vs other save
target" null. It is an opportunity-cost null: same source, same ships, winner's save
target versus nearest non-own attack target.

| side | phase | kind | labels | replay target top1/3 | baseline target top1/3 |
|---|---|---|---:|---:|---:|
| winner | all | attack | 5196 | 20.2/39.1% | 11.5/27.2% |
| winner | all | save | 2086 | 4.8/16.0% | 15.9/32.9% |
| winner | open | attack | 1425 | 25.3/46.2% | 9.3/24.5% |
| winner | open | save | 111 | 8.1/21.6% | 3.6/14.4% |
| winner | mid | attack | 2692 | 16.6/33.7% | 11.4/25.1% |
| winner | mid | save | 1341 | 4.1/14.7% | 13.9/27.5% |
| winner | late | attack | 1079 | 22.6/43.2% | 14.5/36.1% |
| winner | late | save | 634 | 5.8/17.7% | 22.4/47.4% |

Read: the within-state target signal is real for attacks, especially in the opening.
For winner saves, the target head ranks the nearest attack baseline above the actual
save target in aggregate, midgame, and late. The only positive save slice is tiny
opening save count (`111` labels). This is the most defensible version of the save
finding: on sources where rank1 winners saved, our target head strongly prefers the
nearest attack alternative. This is an attack-bias/save-aversion read, not a test of
"winner save target versus other legal save targets."

Same-state fire source contrast:

This compares fire readiness on replay-used source slots against unused owned source
slots from the same replay state. It controls the board, but not winner-vs-loser board
distribution.

| side | phase | used/unused sources | used fire>=.5 | unused fire>=.5 | used fire_p | unused fire_p |
|---|---|---:|---:|---:|---:|---:|
| winner | all | 7282/47988 | 24.0% | 18.0% | 0.256 | 0.203 |
| winner | open | 1536/6132 | 38.5% | 26.2% | 0.385 | 0.266 |
| winner | mid | 4033/29046 | 19.8% | 15.7% | 0.222 | 0.186 |
| winner | late | 1713/12810 | 20.8% | 19.4% | 0.221 | 0.212 |
| loser | all | 11140/27756 | 10.4% | 5.4% | 0.117 | 0.060 |
| loser | open | 3417/5618 | 15.3% | 7.6% | 0.170 | 0.082 |
| loser | mid | 6051/16674 | 9.0% | 5.5% | 0.103 | 0.063 |
| loser | late | 1672/5464 | 5.0% | 2.5% | 0.057 | 0.029 |

Read: fire is not pure board state. Within the same board, the replay-used sources are
more fire-ready than unused owned sources. Some unused sources are correctly held
garrison, so this is a conservative baseline for the source-level fire signal. But the
aggregate winner-vs-loser fire gap still remains partly board-confounded: winner boards
are ahead/easier boards, and even unused winner sources fire much more readily than
unused loser sources.

Projection loss:

| side | owned moves | projected | move keep | mass keep | source not top16 move/mass | same-source lost move/mass | >16-turn |
|---|---:|---:|---:|---:|---:|---:|---:|
| winner | 8818 | 7282 | 82.6% | 94.8% | 14.7/3.9% | 1.0/0.6% | 0.0% |
| loser | 12039 | 11140 | 92.5% | 97.0% | 0.3/0.0% | 6.4/2.5% | 0.0% |

Phase detail for projection:

| side | phase | move keep | mass keep | source not top16 move/mass | same-source lost move/mass |
|---|---|---:|---:|---:|---:|
| winner | open | 98.8% | 99.4% | 0.0/0.0% | 0.2/0.1% |
| winner | mid | 81.7% | 95.8% | 15.3/2.8% | 1.2/0.6% |
| winner | late | 73.6% | 92.0% | 23.2/6.7% | 1.0/0.7% |
| loser | open | 92.6% | 98.4% | 0.0/0.0% | 6.8/1.1% |
| loser | mid | 93.0% | 97.2% | 0.5/0.1% | 5.6/2.1% |
| loser | late | 90.9% | 95.0% | 0.0/0.0% | 8.6/4.7% |

Read:

- The aggregate winner-vs-loser gap is real, but board-confounded. Phrase it as
  "consistent with some representation" rather than a clean causal estimate.
- The within-state target control says the target head prefers rank1 winner attack
  targets, especially in the opening. Mid/late aggregate target preference is weak once
  save labels are included.
- Defensive/save target ranking is the sharpest negative result. Winner save targets
  are usually ranked below the same-source nearest attack baseline, so the target head
  is not naturally choosing defensive concentration.
- The fire head has some source-level selectivity within a board, but the winner-loser
  fire gap is still not a clean move-quality estimate.
- The absolute joint rates are still low. Even where signal exists, the heads rarely
  assemble the full action together.
- Rank1 winners are more selective than losers: fewer replay moves overall, but our
  heads score their attack/opening moves higher.
- Same-source multi-move is not the winner ceiling in this corpus: only 1.0% of winner
  owned moves, 0.6% of winner ship mass, is lost to same-source projection.
- The top-16 source selector drops many winner moves by count, especially late, but much
  less mass: 14.7% of moves and 3.9% of ship mass overall; late 23.2% of moves and 6.7%
  of mass. This is a real source-selection issue, but not a huge mass ceiling.

Conclusion from this audit:

This points away from pure global representation failure, but only on the attack/opening
side. The stronger conclusion is narrower:

- attack/opening target concepts are partly present but weakly selected;
- defensive/save target concepts are not present in a useful way;
- direct action-head overwrite remains unsafe;
- the better path is a phase- and intent-conditioned auxiliary/triage signal inside PPO,
  with special focus on save target selection and hopeless/cheap-save discrimination.

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
