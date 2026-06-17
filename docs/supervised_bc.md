# Supervised BC Parallel Track

Goal: test the top-10 suggestion directly, outside the active self-play loop.
Train a standalone policy from strong replay decisions, rebalance away from idle
frames, then evaluate/export it exactly like any other checkpoint.

## Hypothesis

Replay-supervised learning can produce a competent policy faster than PPO because
the target signal is stable and dense. The failure mode to avoid is label
collapse: most replay ticks are no-op, and most owned slots inside a tick are also
no-fire. A naive dataset teaches the globally common action, not good play.

## Dataset

First rank replay winners by measurable quality. This is the high-precision
"good play" gate: conversion, expansion, retention, and launch-waste metrics must
look like strong play before we clone the decisions.

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/score_good_play_replays.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --scores-out /tmp/supervised_bc/good_play_scores.json \
  --samples-out /tmp/supervised_bc/good_play_balanced.pkl \
  --require-known-winner \
  --noop-keep-prob 0.05 \
  --fire-repeat 2
```

Read `good_play_scores.json` before training. Each accepted or rejected replay
has `score`, `reasons`, `hard_fails`, and the underlying metrics.
Drop `--require-known-winner` only for an explicit broad-cohort ablation; the
first supervised run should bias toward precision over coverage.

Opening-only curriculum variant:

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/score_good_play_replays.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --scores-out /tmp/supervised_bc/good_play_known_opening50_scores.json \
  --samples-out /tmp/supervised_bc/good_play_known_opening50.pkl \
  --require-known-winner \
  --steps-max 50 \
  --noop-keep-prob 0.02 \
  --fire-repeat 4
```

Retention/reinforcement curriculum variant:

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/score_good_play_replays.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --scores-out /tmp/supervised_bc/good_play_known_retention50_r3_scores.json \
  --samples-out /tmp/supervised_bc/good_play_known_retention50_r3.pkl \
  --require-known-winner \
  --steps-min 50 \
  --noop-keep-prob 0.02 \
  --fire-repeat 2 \
  --reinforce-repeat 3
```

Contest/hold curriculum variant:

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/score_good_play_replays.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --scores-out /tmp/supervised_bc/good_play_known_contest16_140_w30_scores.json \
  --samples-out /tmp/supervised_bc/good_play_known_contest16_140_w30.pkl \
  --require-known-winner \
  --steps-min 16 \
  --steps-max 140 \
  --contest-window 30 \
  --noop-keep-prob 0.02 \
  --fire-repeat 3 \
  --reinforce-repeat 4
```

Answer-inbound action-filter variant:

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/score_good_play_replays.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --scores-out /tmp/supervised_bc/good_play_known_answer_inbound16_140_scores.json \
  --samples-out /tmp/supervised_bc/good_play_known_answer_inbound16_140.pkl \
  --require-known-winner \
  --steps-min 16 \
  --steps-max 140 \
  --answer-inbound-only \
  --noop-keep-prob 0.0 \
  --fire-repeat 4 \
  --reinforce-repeat 4
```

Soft answer-inbound weighting variant. This preserves the full teacher action
label but repeats frames where at least one move starts from or targets a
threatened owned planet:

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/score_good_play_replays.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --scores-out /tmp/supervised_bc/good_play_known_answer_weighted16_140_scores.json \
  --samples-out /tmp/supervised_bc/good_play_known_answer_weighted16_140.pkl \
  --require-known-winner \
  --steps-min 16 \
  --steps-max 140 \
  --contest-window 30 \
  --noop-keep-prob 0.02 \
  --fire-repeat 3 \
  --reinforce-repeat 4 \
  --answer-inbound-repeat 6
```

Low-lost-cap version of the same slice:

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/score_good_play_replays.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --scores-out /tmp/supervised_bc/good_play_known_lowlost045_answer_weighted16_140_scores.json \
  --samples-out /tmp/supervised_bc/good_play_known_lowlost045_answer_weighted16_140.pkl \
  --require-known-winner \
  --max-lost-cap 0.45 \
  --steps-min 16 \
  --steps-max 140 \
  --contest-window 30 \
  --noop-keep-prob 0.02 \
  --fire-repeat 3 \
  --reinforce-repeat 4 \
  --answer-inbound-repeat 6
```

Synthetic-defense variant. This keeps replay labels but also appends a generated
reinforce move from a rear owned planet to a threatened owned planet when the
projected garrison is insufficient:

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/score_good_play_replays.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --scores-out /tmp/supervised_bc/good_play_known_lowlost045_synthdef_mild16_140_scores.json \
  --samples-out /tmp/supervised_bc/good_play_known_lowlost045_synthdef_mild16_140.pkl \
  --require-known-winner \
  --max-lost-cap 0.45 \
  --steps-min 16 \
  --steps-max 140 \
  --contest-window 30 \
  --noop-keep-prob 0.02 \
  --fire-repeat 3 \
  --reinforce-repeat 4 \
  --answer-inbound-repeat 4 \
  --synthetic-defense-repeat 4
```

The lower-level builder can still be used when you want a manual filter, for
example winners from a specific player:

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/build_supervised_bc.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --noop-keep-prob 0.05 \
  --fire-repeat 2 \
  --winner-name "Isaiah" \
  --samples-out /tmp/supervised_bc/isaiah_winners_balanced.pkl \
  --summary-out /tmp/supervised_bc/isaiah_winners_balanced_summary.json
```

Read the summary before training:

- `decision_frame_rate_seen`: how sparse real action frames were in the raw replay stream.
- `decision_sample_share`: post-rebalance share of samples containing at least one launch.
- `fire_slot_rate`: post-rebalance positive fire-label rate across valid owned slots.
- `reinforce_labels`: post-rebalance own-target labels; these only matter if the
  BC checkpoint is trained and saved with `--allow-reinforce`.
- `contest_frames_seen`, `contest_recent_capture`, `contest_enemy_inbound`: when
  `--contest-window` is set, these explain why a frame was kept.
- `answer_frames_kept`, `answer_source_threatened`,
  `answer_target_threatened`: when `--answer-inbound-only` is set, these show
  how many labels were kept because they start from or target a threatened
  owned planet.
- `answer_frames_weighted`, `answer_moves_weighted`: when
  `--answer-inbound-repeat` is set, these show how many full-label frames were
  repeated because they contained at least one inbound-answer move.
- `synthetic_defense_frames`, `synthetic_defense_moves`,
  `synthetic_defense_ships`: when `--synthetic-defense-repeat` is set, these
  count generated rear-to-front reinforce labels for threatened owned planets.
- `threat_pos_rate`: when `--threat-horizon` is set, this is the fraction of
  owned slots that lose ownership within the future horizon.
- `subjects`: whose winning decisions dominate the dataset.
- `subject_samples`, `subject_decision_samples`: whose labels dominate the
  final repeated training set. These are the counters to inspect before a
  training run.
- `selection`: when `--max-accepted-per-subject` is set, this records how many
  accepted replay seats were eligible, selected, and skipped by the per-subject
  cap. Use it to avoid cloning one strong player's style as if it were the whole
  concept of good play.
- `samples_skipped_subject_sample_cap`: when `--max-samples-per-subject` is set,
  this records repeated samples dropped after a subject reached its final sample
  budget.

If `fire_slot_rate` is still tiny, increase `--fire-repeat` and/or train with a
higher `--fire-pos-weight`. If `subjects` is too concentrated, use
`--max-accepted-per-subject`, `--max-samples-per-subject`, or `--winner-name`
filters to build separate per-player datasets and mix them deliberately.

## Train

Pure supervised baseline: omit `--init-checkpoint`. This is the cleanest test of
whether replay supervision alone can train the policy.

For this standalone track, `bc.py` rejects PPO/RL learner checkpoints passed via
`--init-checkpoint` unless `--allow-rl-init` is explicitly set. Curriculum
training from an earlier supervised BC checkpoint is still allowed; that remains
supervised-only because the predecessor was trained from replay labels, not RL.

```bash
tmux new-session -d -s supervised_bc_top \
  'cd /Users/saheb/.codex/worktrees/341b/kaggle-orbit-wars && \
   exec stdbuf -oL -eL /Users/saheb/home/.venv/bin/python -u orbit_wars_rl/bc.py \
     --samples /tmp/supervised_bc/good_play_balanced.pkl \
     --steps 300 \
     --lr 1e-4 \
     --fire-pos-weight 1.0 \
     --save seed_checkpoints/supervised_top_winners_bc.pt \
     2>&1 | tee /tmp/supervised_bc/top_winners_train.log'
```

## Evaluate

Use the normal held-out gates. Quick eval is only a smoke check; the decision
requires panels and then LB submission if panels are non-embarrassing.

```bash
CUDA_VISIBLE_DEVICES="" /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint seed_checkpoints/supervised_top_winners_bc.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --panel --target-decode --fire-threshold 0.35

CUDA_VISIBLE_DEVICES="" /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint seed_checkpoints/supervised_top_winners_bc.pt \
  --opponent opponents/candidate_zach_public.py \
  --panel --target-decode --fire-threshold 0.35
```

Selection remains outcome-first: held-out win-rate, conversion/hoard line, and
behavioral degeneracy checks. BC validation loss is only a training sanity check.

## First Local Results

Using `/tmp/supervised_bc/good_play_known_balanced.pkl` from the known-winner
gate.

Pure scratch:

- `300 steps`, `fire_pos_weight=1.0`, no init checkpoint: offline target top1
  `0.22`, target gate failed.
- Same checkpoint at threshold `0.35`: no-op collapse in gameplay (`0 attack
  launches/game`, Zach `0/16`).
- Same checkpoint at threshold `0.25`: fires, but badly (`1/16` Zach,
  `cap/atk-launch 0.039`, `109 attack launches/game`).

Opening-only curriculum from scratch:

- Dataset: `/tmp/supervised_bc/good_play_known_opening50.pkl`, known winners,
  first 50 steps only, `9,785` samples, `99.3%` decision samples after
  rebalance, `fire_slot_rate 0.218`.
- `1000 steps`, `fire_pos_weight=1.0`, no init checkpoint:
  target gate passed (`target_red +0.36`, top1 `0.34`, top3 `0.58`).
- Random quick-8 passed at both thresholds:
  - threshold `0.35`: `8/8`, `cap/atk-launch 0.165`, `140 attack launches/game`;
  - threshold `0.25`: `8/8`, `cap/atk-launch 0.116`, `176 attack launches/game`.
- Zach quick-16: `5/16` at both thresholds.
  - threshold `0.35`: `cap/atk-launch 0.204`, `148 attack launches/game`,
    `lost-cap 0.76`;
  - threshold `0.25`: `cap/atk-launch 0.172`, `178 attack launches/game`,
    `lost-cap 0.72`.
- Ajay quick-16: `0/16` at both thresholds. It expands early (`planets@50=5`)
  but collapses by step 100 (`planets@100=0-1`) with `lost-cap 0.96-0.98` and
  reinforcement share `0.03`.

Retention/reinforcement curriculum:

- Dataset `/tmp/supervised_bc/good_play_known_retention50.pkl`: known winners,
  steps `>=50`, `35,974` samples with aggressive own-target upweighting
  (`reinforce_repeat=8`). It produced `37,312` repeated reinforcement labels.
- Fine-tune from the scratch opening BC checkpoint with `--allow-reinforce`:
  `/tmp/supervised_bc/supervised_curriculum_opening_retention50_800_firew1_reinf.pt`.
  Offline gate passed (`target_red +0.36`, top1 `0.32`, top3 `0.56`).
- Ungated eval showed the model learned reinforcement but too much of it:
  random `8/8`, Zach `4/16`, Ajay `0/16`, reinforcement share `0.65-0.74`.
- With discipline gates (`--reinforce-gate-min-planets 3 --reinforce-forward-only
  --reinforce-garrison-floor 10`), reinforcement rate became saner but did not
  solve retention: Zach `4/16`, Ajay `0/16`, Ajay `lost-cap 0.99`.

Mixed opening + lower-weight retention:

- Dataset `/tmp/supervised_bc/good_play_known_retention50_r3.pkl`: same slice,
  lower own-target upweighting (`reinforce_repeat=3`), `19,084` samples and
  `13,992` repeated reinforcement labels.
- Fine-tuned from scratch opening BC using both opening and retention datasets:
  `/tmp/supervised_bc/supervised_curriculum_mix_opening_retention50_r3_700_reinf.pt`.
  Offline gate passed (`target_red +0.35`, top1 `0.34`, top3 `0.55`).
- Disciplined quick eval: random `8/8`, Zach `5/16`, Ajay `0/16`.
  Zach retention improved versus opening-only (`lost-cap 0.69` vs `0.76`, median
  hold `37st` vs `20st`), but Ajay still collapses (`lost-cap 0.98`).
- Readout: supervised replay BC is learning the requested behavior class, but the
  label mix is not yet teaching contest/hold decisions that survive Ajay.

Contest-window opening curriculum:

- Dataset `/tmp/supervised_bc/good_play_known_contest16_140_w30.pkl`: known
  winners, steps `16..140`, only states with a recent captured planet
  (`<=30` steps old) or enemy fleet inbound to an owned planet. It produced
  `17,931` samples, `9,919` contest frames, `8,435` enemy-inbound frames, and
  `4,413` recent-capture frames.
- Fine-tuned from scratch opening BC using opening + contest datasets:
  `/tmp/supervised_bc/supervised_curriculum_opening_contest16_140_w30_700_reinf.pt`.
  Offline gate passed (`target_red +0.36`, top1 `0.37`, top3 `0.56`).
- Disciplined quick eval: random `8/8`, Zach `5/16`, Ajay `0/16`.
  Zach retention nudged up again (`lost-cap 0.67`, planets@50 `7`), and Ajay
  threshold `0.35` improved only marginally (`lost-cap 0.96`), still no wins.
- Ungated quick eval: Zach `6/16`, Ajay `0/16`. This is the best supervised-only
  Zach quick result so far, but ungated reinforcement is too high (`0.46-0.54`)
  and does not survive Ajay.

Answer-inbound action-filter curriculum:

- Dataset `/tmp/supervised_bc/good_play_known_answer_inbound16_140.pkl`: known
  winners, steps `16..140`, only enemy-inbound frames where at least one teacher
  move starts from or targets a threatened owned planet. Non-answer moves are
  removed from the BC label. It produced `8,188` samples from `2,047` kept
  answer frames, with `1,761` source-threatened and `1,469` target-threatened
  answer labels.
- Fine-tuned from scratch opening BC using opening + answer-inbound datasets:
  `/tmp/supervised_bc/supervised_curriculum_opening_answer_inbound16_140_600_reinf.pt`.
  Offline gate passed (`target_red +0.32`, top1 `0.33`, top3 `0.53`).
- Quick eval regressed: disciplined random `8/8`, Zach `3/16`, Ajay `0/16`;
  ungated Zach `5/16`. The label filter is too narrow and appears to damage
  broader attack/hold behavior despite fitting offline.

Soft answer-inbound weighting curriculum:

- Dataset `/tmp/supervised_bc/good_play_known_answer_weighted16_140.pkl`: known
  winners, steps `16..140`, contest-window states, and `--answer-inbound-repeat
  6`. Unlike the hard filter, it preserved the full action label. It produced
  `22,602` samples, `2,047` weighted answer frames, `2,796` weighted answer
  moves, and `fire_slot_rate 0.159`.
- Fine-tuned from scratch opening BC using opening + soft answer-weighted data:
  `/tmp/supervised_bc/supervised_curriculum_opening_answer_weighted16_140_700_reinf.pt`.
  Offline gate passed (`target_red +0.35`, top1 `0.34`, top3 `0.54`).
- Disciplined quick eval: random `8/8`, Zach `5/16`, Ajay `0/16`, Ajay
  `lost-cap 0.99`. Ungated quick eval: Zach `6/16`, Ajay `0/16`, but
  reinforcement share rose to `0.46-0.54`. Soft weighting avoids the hard-filter
  regression but still does not solve Ajay retention.

Low-lost-cap soft answer-weighted curriculum:

- Dataset
  `/tmp/supervised_bc/good_play_known_lowlost045_answer_weighted16_140.pkl`:
  same state/answer weighting but with `--max-lost-cap 0.45`. It accepted `43`
  known-winner replays (`Jake Will 25`, `TonyK 11`, `Isaiah 6`,
  `typeIIIfairy 1`) and produced `8,922` samples, `785` weighted answer frames,
  `1,170` weighted answer moves, and `fire_slot_rate 0.155`.
- Fine-tuned from scratch opening BC using opening + low-lost-cap soft answer
  data:
  `/tmp/supervised_bc/supervised_curriculum_opening_lowlost045_answer_weighted16_140_600_reinf.pt`.
  Offline gate passed strongly for this track (`target_red +0.37`, top1 `0.37`,
  top3 `0.58`).
- Disciplined quick eval: random `8/8`, Zach `6/16`, Ajay `0/16`.
  Zach retention improved relative to the broad soft-weighted run (`lost-cap
  0.73` vs `0.80`) and matched the best supervised-only Zach quick result, but
  Ajay still collapsed (`lost-cap 0.98`, planets@100 `1`). Ungated decode
  over-reinforced (`reinf_share 0.46-0.50`) and did not help. Lowering the fire
  threshold to `0.25` increased launch volume but regressed Zach to `5/16` and
  kept Ajay at `0/16`.

Synthetic-defense curriculum:

- Aggressive dataset
  `/tmp/supervised_bc/good_play_known_lowlost045_synthdef16_140.pkl`: same
  low-lost-cap replay cohort plus synthetic defensive reinforce labels with
  `--synthetic-defense-repeat 8`. It produced `17,043` samples, `1,767`
  synthetic-defense frames, `2,705` synthetic-defense moves, and `73,559`
  synthetic-defense ships.
- Training from the scratch opening BC checkpoint:
  `/tmp/supervised_bc/supervised_curriculum_opening_lowlost045_synthdef16_140_700_reinf.pt`.
  Offline target gate failed after best-weight restore (`top1 0.27`, top3
  `0.50`). Quick eval regressed: random `8/8`, Zach `3/16`, Ajay `0/16`.
  The synthetic labels were too heavy/noisy as a replacement curriculum.
- Milder dataset
  `/tmp/supervised_bc/good_play_known_lowlost045_synthdef_mild16_140.pkl`: same
  detected opportunities, lower `--synthetic-defense-repeat 4`. It produced
  `9,975` samples with the same `1,767` synthetic-defense frames.
- Short fine-tune from the best supervised low-lost soft-answer checkpoint:
  `/tmp/supervised_bc/supervised_ft_lowlost045_synthdef_mild16_140_300_reinf.pt`.
  Offline gate passed (`target_red +0.32`, top1 `0.31`, top3 `0.53`).
  Disciplined quick eval: random `8/8`, Zach `7/16`, Ajay `0/16`. This was the
  best supervised-only Zach quick result before subject-cap tuning, with Zach `lost-cap 0.67` and
  a healthier reinforcement ramp than the ungated variants. Ajay still collapsed
  (`lost-cap 0.97`, planets@100 `1`). Relaxing garrison floor to `0` increased
  reinforcement but worsened Zach to `6/16` and kept Ajay `0/16`.
- Soft subject-cap ablation:
  `/tmp/supervised_bc/good_play_known_lowlost045_synthdef_mild16_140_softcap20.pkl`
  used `--max-accepted-per-subject 20` and no final sample cap. It selected
  `38/43` accepted replay seats and produced `8,705` samples with final subject
  distribution `Jake Will 4470`, `TonyK 2771`, `Isaiah 1061`,
  `typeIIIfairy 403`. Fine-tune:
  `/tmp/supervised_bc/supervised_ft_lowlost045_synthdef_mild16_140_softcap20_300_reinf.pt`
  failed the honest shuffled offline target gate (`target_red +0.30`, top1
  `0.28`, top3 `0.50`) but improved disciplined quick eval to Zach `8/16`,
  Ajay `0/16`. Zach retention was `lost-cap 0.74`, cap/attack `0.318`,
  reinforcement share `0.13`. Readout: a soft cap helps Zach slightly, but the
  same Ajay collapse remains (`lost-cap 0.97`, planets@100 `1`). This is the
  current best supervised-only Zach quick result, not a promotion candidate.
- Balanced-subject ablation:
  `/tmp/supervised_bc/good_play_known_lowlost045_synthdef_mild16_140_balanced.pkl`
  used `--max-accepted-per-subject 12 --max-samples-per-subject 3000`. It
  selected `30/43` accepted replay seats and reduced the final repeated sample
  distribution to `TonyK 2769`, `Jake Will 2699`, `Isaiah 1063`,
  `typeIIIfairy 403` (`6,934` samples total, `fire_slot_rate 0.211`).
  Fine-tune:
  `/tmp/supervised_bc/supervised_ft_lowlost045_synthdef_mild16_140_balanced_300_reinf.pt`
  from the same low-lost soft-answer checkpoint failed the honest shuffled
  offline target gate (`target_red +0.29`, top1 `0.27`, top3 `0.49`) and
  regressed in quick eval: Zach `2/16`, Ajay `0/16`. Readout: this cap setting
  improves teacher diversity but appears to underfit/damage the synthetic-defense
  action distribution. Do not promote it over the unbalanced mild checkpoint;
  if revisiting balancing, use a softer cap or oversample minority subjects
  instead of cutting the majority signal this aggressively.

Threat-head auxiliary ablation:

- Implementation notes:
  - `bc.py` now shuffles concatenated sample files before train/validation
    splitting. Previous multi-file offline gates used the first 10% of the
    concatenated list, which was usually all opening samples. Gameplay evals
    remain the deciding evidence; old offline target gates were too
    opening-heavy.
  - `model.py` has an optional `threat_head` that predicts per-owned-planet
    `P(lost within K steps)`.
  - `eval.py --threat-target-bias X` can add `X * P(lost soon)` to own-target
    logits before target decode.
- Dataset
  `/tmp/supervised_bc/good_play_known_lowlost045_synthdef_mild_threat24.pkl`:
  low-lost-cap mild synthetic-defense data plus `--threat-horizon 24`. It
  produced `9,975` samples, `29,086` threat-labeled owned slots, `2,135`
  positive slots, and `threat_pos_rate 0.073`.
- Fine-tune:
  `/tmp/supervised_bc/supervised_ft_lowlost045_synthdef_threat24_300_reinf.pt`
  from the best mild synthetic checkpoint with `--threat-loss-weight 0.2` and
  `--threat-pos-weight 8.0`.
- Honest shuffled validation failed the action target gate (`target_red +0.29`,
  top1 `0.28`, top3 `0.48`) while the threat head learned a real signal
  (`threat_acc 0.75`, `threat_pos_rate 0.079`). Gameplay regressed:
  no-bias Zach `4/16`, Ajay `0/16`; with `--threat-target-bias 1.0`, Zach
  `6/16`, Ajay `0/16`.
- Readout: the threat label is learnable, but this first auxiliary formulation
  disrupts action quality and the simple target-logit bias does not solve Ajay.
  Future threat work should decouple action-head preservation better, for
  example freezing most action heads/backbone while training threat, or using
  the threat score as a postprocessor only after calibrating it.
- Frozen-backbone threat-head follow-up:
  `/tmp/supervised_bc/supervised_threathead_only_lowlost045_h24_1000_reinf.pt`
  trained only `threat_head.weight` and `threat_head.bias` from the mild
  synthetic-defense checkpoint. It learned the label better (`threat_acc 0.88`,
  `threat_loss 0.42`) and preserved action behavior with no bias: Zach `7/16`,
  Ajay `0/16`. Adding `--threat-target-bias 0.5` regressed Zach to `6/16` and
  kept Ajay `0/16`. Readout: the frozen threat head is a cleaner diagnostic and
  safe to carry, but the naive own-target logit bias is not the right inference
  use.

Policy-teacher BC:

- `orbit_wars_rl/build_policy_teacher_bc.py` builds supervised samples by
  running a deterministic teacher policy on replay observations. This is a
  separate non-RL signal from replay cloning: labels come from the teacher's
  fresh action on `obs[t-1]`, not from the replay's recorded `action[t]`.
- Ajay-teacher smoke:
  `/tmp/supervised_bc/ajay_teacher_smoke.pkl` from one replay verified the
  real `opponents/candidate_ajay_1200.py` path: `83` samples, decision-frame
  rate `0.65`, fire-slot rate `0.398`, and reinforce labels present.
- 10-replay Ajay-teacher dataset:
  `/tmp/supervised_bc/ajay_teacher_winner10_s100.pkl`, built with
  `--max-replays 10 --steps-max 100`, produced `1,246` samples, decision-frame
  rate `0.748`, fire-slot rate `0.361`, and subject mix `Isaiah 8`, `Boey 1`,
  `Jake Will 1`.
- Fine-tune from the scratch opening BC checkpoint:
  `/tmp/supervised_bc/supervised_opening50_ajay_teacher10_s100_300_reinf.pt`
  failed the offline target gate (`target_red +0.22`, top1 `0.17`, top3
  `0.39`). Readout: the teacher-action pipeline works, but a tiny 10-replay
  Ajay-teacher set is too small or too heterogeneous to teach the target head.
  Next scaling step should be a larger, curated teacher-state set or a narrower
  state slice, not promotion of this checkpoint.
- All-seat scale-up:
  `/tmp/supervised_bc/ajay_teacher_all30_s120.pkl` used `--seat-mode all`,
  `--max-replays 30`, and `--steps-max 120`. It produced `9,329` samples,
  decision-frame rate `0.772`, fire-slot rate `0.416`, and labels from `60`
  seat sequences. Fine-tune:
  `/tmp/supervised_bc/supervised_opening50_ajay_teacher_all30_s120_600_reinf.pt`
  also failed the offline target gate (`target_red +0.22`, top1 `0.19`, top3
  `0.38`) despite much better ship-bin fit (`ship_red +0.54`). Readout: direct
  full-action imitation of Ajay/Producer-style teacher plans is not limited only
  by sample count. The target labels need filtering/decomposition: likely
  single-source target-only labels, high-confidence teacher decisions, or a
  separate source/ship/target factorization before gameplay eval is worthwhile.
- Split-move target-only ablation:
  `/tmp/supervised_bc/ajay_teacher_all30_s120_splitmoves.pkl` used the same
  all-seat slice plus `--split-moves`, producing `27,931` one-launch samples and
  `27,820` target labels. Training only `tgt_q`, `tgt_k`, and
  `target_scorer` from the scratch opening BC checkpoint saved
  `/tmp/supervised_bc/supervised_opening50_ajay_teacher_all30_s120_splitmoves_targetonly_600_reinf.pt`
  but still failed the offline target gate (`target_red +0.23`, top1 `0.20`,
  top3 `0.43`). This is better than full-action all-seat top3 but still not
  gameplay-worthy. Readout: splitting multi-action plans is necessary but not
  sufficient; the next teacher-label variant should filter for high-confidence
  teacher moves or train a ranking/preference target instead of flat classifying
  every emitted target.
- Filtered policy-teacher ablation:
  `--split-moves --max-teacher-moves-per-frame 1 --target-owner not-own`
  produced only `897` samples, so it was too strict. Relaxing to
  `--max-teacher-moves-per-frame 2 --target-owner not-own` produced `2,019`
  samples from the same 30-replay all-seat slice, with decision-sample share
  `1.0`, fire-slot rate `0.145`, and `773` own-target split moves filtered out.
  Target-only fine-tune saved
  `/tmp/supervised_bc/supervised_opening50_ajay_teacher_all30_s120_max2_notown_targetonly_600_reinf.pt`
  but failed the gate (`target_red +0.34`, top1 `0.30`, top3 `0.47`) after
  best-loss restore. Step `400` briefly reached top3 `0.55`, but this did not
  survive checkpoint selection. Readout: confidence filtering helps the target
  head more than raw split-move labels, but the flat single-positive target
  classification objective is likely the bottleneck now. Next attempt should
  train target preferences/rankings or save/select by target metric rather than
  total BC loss.
- Metric-selected/source-filtered variant:
  `bc.py --select-metric val_target_top3` now allows target-only BC runs to
  restore the best validation target checkpoint instead of the best total BC
  loss. Adding `--max-teacher-moves-per-source 1` to the dataset builder removed
  only `8` moves across `4` frames, producing `2,011` samples in
  `/tmp/supervised_bc/ajay_teacher_all30_s120_max2_source1_notown.pkl`; this
  mostly falsifies same-source contradictory labels as the main bottleneck on
  this slice. Training saved
  `/tmp/supervised_bc/supervised_opening50_ajay_teacher_all30_s120_max2_source1_notown_targettop3_600_reinf.pt`
  and passed the offline target gate (`target_red +0.32`, top1 `0.32`, top3
  `0.55`). Quick gameplay still failed Ajay (`0/16`) and was ordinary against
  Zach (`5/16`). Behavior readout: Ajay `caps/game 10.8`, `lost-cap 0.97`,
  `reinf_share 0.03`; Zach `caps/game 26.0`, `lost-cap 0.74`, `reinf_share
  0.12`. Conclusion: the filtered teacher target head is now learnable, but
  the live policy is still dominated by retention/reinforcement failure rather
  than initial target selection.
- Own-target retention teacher variant:
  Ajay-teacher own-target labels are abundant after step 16. The 30-replay
  all-seat slice `/tmp/supervised_bc/ajay_teacher_all30_s16_140_own.pkl` used
  `--target-owner own --split-moves --reinforce-repeat 3` and produced `28,152`
  repeated samples from `9,384` own-target teacher moves (`fire_slot_rate
  0.093`). Mixed with the scratch opening dataset and initialized from the
  supervised target-gate checkpoint, it passed offline (`target_red +0.42`,
  top1 `0.33`, top3 `0.62`) but overfit toward reinforcement/hoarding: Zach
  `1/16`, Ajay `0/16`, Zach `caps/game 4.9`, Ajay `caps/game 4.0`,
  `reinf_share 0.32/0.41`.
- Balanced own-target retention variant:
  Rebuilding the same slice without repeat
  `/tmp/supervised_bc/ajay_teacher_all30_s16_140_own_r1.pkl` produced `9,384`
  samples, roughly matching the `9,785` opening samples. The mixed checkpoint
  `/tmp/supervised_bc/supervised_opening50_targettop3_ajay_own30_s16_140_r1_600_reinf.pt`
  also passed offline (`target_red +0.37`, top1 `0.31`, top3 `0.58`) but still
  collapsed attack tempo: Zach `1/16`, Ajay `0/16`, Zach `caps/game 6.1`,
  Ajay `caps/game 4.7`, `reinf_share 0.24/0.31`. Inference gate
  `--reinforce-gate-min-planets 7` reduced early reinforcement (`reinf_share
  0.03/0.02`) but did not restore attacks (Zach `2/16`, Ajay `0/16`,
  caps/game `6.6/5.2`). Readout: own-target supervised labels are learnable, but
  updating the main action heads on them suppresses expansion. The next
  supervised retention attempt should be lower-ratio, source/target-only, or a
  separate reinforcement bias/postprocessor instead of full-head BC on
  own-target-only positives.
- Low-ratio target-only retention variant:
  A 2,000-sample subset of the no-repeat own-target set mixed with the `9,785`
  opening samples and trained only `tgt_q`, `tgt_k`, and `target_scorer`:
  `/tmp/supervised_bc/supervised_opening50_targettop3_ajay_own30_s16_140_r1_sub2k_targetonly_500_reinf.pt`.
  Offline gate passed (`target_red +0.34`, top1 `0.32`, top3 `0.54`), but live
  behavior still over-targeted own planets. Ungated quick eval: Zach `1/16`,
  Ajay `0/16`; Zach retained attack volume (`caps/game 26.0`) but
  `reinf_share 0.62` and `lost-cap 0.93`, while Ajay had `caps/game 9.1`,
  `reinf_share 0.38`, `lost-cap 0.99`.
- Reinforce target bias postprocessor:
  `eval.py --reinforce-target-bias` adds a fixed scalar to all own-target logits
  before target decode. On the low-ratio target-only retention checkpoint,
  `-1.0` gave the best quick behavior among tested biases but still did not beat
  the earlier no-retention target checkpoint: Zach `3/16`, Ajay `0/16`,
  Zach `reinf_share 0.24`, Ajay `reinf_share 0.12`, lost-cap `0.86/0.98`.
  `-2.0` pushed reinforcement too low again (`0.13/0.04`) and scored Zach
  `2/16`, Ajay `0/16`. Readout: own-target labels currently shift the
  own-vs-attack prior, but not the defensive timing/source/ship sizing needed to
  hold planets. The next supervised retention attempt needs labels conditioned on
  imminent inbound threats, not generic teacher own-target moves.
- Threat-conditioned own-target retention:
  `build_policy_teacher_bc.py --inbound-threat-horizon 30` keeps only decoded
  own-target teacher samples whose target has an enemy fleet arriving within 30
  steps. On the same 30-replay all-seat slice this produced `2,123` samples:
  `4,854` frames had an inbound threat, `5,206` own-target samples were skipped
  because the own target was not threatened, and `2,123` threat-target samples
  were kept. Target-only training from the supervised target checkpoint saved
  `/tmp/supervised_bc/supervised_opening50_targettop3_ajay_own_threat30_targetonly_500_reinf.pt`
  and passed offline (`target_red +0.33`, top1 `0.33`, top3 `0.53`).
  Ungated quick eval: Zach `4/16`, Ajay `0/16`, with too much reinforcement
  (`reinf_share 0.46/0.37`) and Ajay `lost-cap 0.99`. With
  `--reinforce-target-bias -1.0`, Zach stayed `4/16` and had a much more
  plausible ramp (`reinf_share 0.29`, lost-cap `0.72`), but Ajay remained
  `0/16` (`reinf_share 0.20`, lost-cap `0.97`). Readout: threat-conditioned
  teacher labels are a better retention signal than generic own-target labels,
  but still do not teach enough source/ship sizing to hold planets against Ajay.
  Next supervised direction should label "how many ships from which rear source"
  for threatened planets, likely via synthetic defense or a separate sizing head,
  instead of only changing the target prior.
- Synthetic source/ship defense labels:
  `orbit_wars_rl/build_synthetic_defense_bc.py` builds pure synthetic defense
  samples from replay states: when an owned planet has enemy inbound and
  projected garrison is insufficient, choose a rear owned source and send enough
  ships to cover the threat. The first 30-replay all-seat slice
  `/tmp/supervised_bc/synthetic_defense_all30_s16_180_g10_need5_4k.pkl` capped
  at `4,000` samples, with `4,000` synthetic moves and `162,919` support ships
  total. Mixed with opening labels and trained heads-only (`fire_head`,
  `ship_head`, `tgt_q`, `tgt_k`, `target_scorer`) from the supervised target
  checkpoint, it saved
  `/tmp/supervised_bc/supervised_opening50_targettop3_synthdef4k_heads_600_reinf.pt`.
  Offline target gate narrowly failed (`target_red +0.30`, top1 `0.29`, top3
  `0.50`). Ungated quick eval: Zach `3/16`, Ajay `0/16`; Ajay lost-cap improved
  slightly to `0.95` but attack volume dropped (`caps/game 7.7`). With
  `--reinforce-target-bias -1.0`, Zach recovered to `4/16` and reinforcement
  looked saner (`reinf_share 0.24`, lost-cap `0.75`), but Ajay stayed `0/16`
  and lost-cap regressed to `0.98`. Readout: explicit source/ship defense labels
  can move retention metrics, but a 4k heads-only mix is still too heavy or too
  late-reactive; it trades away expansion before it can hold against Ajay.
  Smaller ratios, stricter high-value threat states, or a separate deterministic
  defense overlay may be better than training the main heads on all synthetic
  defense events.
- Small-ratio synthetic defense:
  A 500-sample subset of the same synthetic-defense set
  `/tmp/supervised_bc/synthetic_defense_all30_s16_180_g10_need5_sub500.pkl`
  was mixed with the `9,785` opening labels and trained only `ship_head`,
  `tgt_q`, `tgt_k`, and `target_scorer`, preserving the supervised checkpoint's
  fire head. Checkpoint:
  `/tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt`.
  Offline gate passed (`target_red +0.33`, top1 `0.33`, top3 `0.53`).
  Ungated quick eval: Zach `4/16`, Ajay `0/16`; attack volume was preserved
  better than the 4k run but reinforcement was still high. With
  `--reinforce-target-bias -1.0`, Zach improved to `6/16` with strong material
  (`avg_material 5846`, `caps/game 30.5`, `planets@100 10`, `reinf_share
  0.18`, `lost-cap 0.71`), but Ajay remained `0/16` (`caps/game 10.3`,
  `lost-cap 0.98`). Readout: a small synthetic source/ship dose plus decode
  bias is the best supervised synthetic-defense Zach result so far, but Ajay
  still requires a stronger anti-recapture mechanism. This checkpoint is useful
  for Zach sanity but not a leaderboard/panel candidate.

- Outcome-aware hold-success synthetic defense:
  `build_synthetic_defense_bc.py` now supports `--recent-capture-window` and
  `--hold-success-horizon`. These constrain synthetic defense targets to owned
  planets that were recently captured and are not lost within the future
  horizon, making the label source closer to "defense in states that actually
  worked" rather than all threatened planets. The first dataset:
  `/tmp/supervised_bc/synthetic_defense_holdsuccess_all30_s16_180_w40_h30_1k.pkl`
  used the same first 30 replay/all-seat slice with steps `16..180`,
  `--recent-capture-window 40`, `--hold-success-horizon 30`, `garrison_floor=10`,
  and `min_need=5`. It was intentionally selective: only `362` samples, with
  `814` candidate targets rejected for future loss and `37,142` ineligible
  target skips.
- Hold-success ship/target-only training:
  Mixed the `362` hold-success samples with the `9,785` opening labels and
  trained only `ship_head`, `tgt_q`, `tgt_k`, and `target_scorer` from the
  supervised target checkpoint. Checkpoint:
  `/tmp/supervised_bc/supervised_opening50_targettop3_holdsuccess_synthdef362_shiptarget_500_reinf.pt`.
  Offline gate passed (`target_red +0.32`, top1 `0.31`, top3 `0.50`). With
  `--reinforce-target-bias -1.0`, Zach quick eval was `6/16`, matching the
  previous 500-sample synthetic result but not improving it (`caps/game 29.1`,
  `reinf_share 0.18`, `lost-cap 0.74`). Ajay stayed `0/16` with the same
  collapse signature (`caps/game 10.6`, `planets@100 1`, `lost-cap 0.98`).
- Hold-success fire+ship+target training:
  Same data, but allowed `fire_head` to train as well. Checkpoint:
  `/tmp/supervised_bc/supervised_opening50_targettop3_holdsuccess_synthdef362_heads_500_reinf.pt`.
  Offline gate also passed (`target_red +0.32`, top1 `0.31`, top3 `0.50`).
  With `--reinforce-target-bias -1.0`, Zach remained `6/16` (`caps/game 26.5`,
  `reinf_share 0.18`, `lost-cap 0.75`) and Ajay remained `0/16`
  (`caps/game 10.9`, `planets@100 0`, `lost-cap 0.98`).
  Readout: filtering synthetic defense labels by future hold success is a
  cleaner trust signal, but the current formulation still does not teach the
  anti-recapture mechanism Ajay punishes. The bottleneck is likely action
  timing/source allocation under immediate counterattack, not just target choice
  or ship sizing on isolated defensive examples.
- Defense-overlay inference ablation:
  `eval.py --defense-overlay` now appends at most a few synthetic rear-source
  support moves after model decode, using the same projected-garrison rule as
  synthetic-defense labels. It is disabled by default and exists to test whether
  the failure is the learned action heads versus the defensive rule itself.
  On the best small synthetic checkpoint
  `/tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt`
  with `--reinforce-target-bias -1.0`, the original overlay setting
  (`recent_capture_window=40`, `garrison_floor=10`, `min_need=5`) overfired:
  Zach fell from `6/16` to `4/16`, `reinf_share` rose to `0.36`, and lost-cap
  worsened to `0.79`. A stricter setting (`window=20`, `floor=20`, `min_need=8`)
  scored Zach `5/16`, Ajay `0/16`, and moved Ajay lost-cap only to `0.96`.
  The strictest tested setting (`window=20`, `floor=30`, `min_need=10`) restored
  Zach to `6/16` but still left Ajay at `0/16` with `planets@100 1` and
  `lost-cap 0.96`. Readout: a deterministic defense overlay can slightly reduce
  recapture rate, but it still trades off attack tempo and does not solve Ajay.
  The next useful version needs either better timing/counterattack prediction or
  a learned selector for when the overlay is worth firing, not a blanket
  projected-garrison trigger.
- Defense-overlay selector:
  `build_defense_selector.py` mines candidate overlay moves from replay states
  and trains a small supervised selector from local tactical features. The label
  is whether the threatened owned target survives the next `30` steps. On the
  first 30-replay/all-seat slice it produced `569` candidates with validation
  AUC `0.81`. Scaling to all local `/tmp/fresh_validate` + `/tmp/snowball`
  replays produced
  `/tmp/supervised_bc/defense_selector_all_s16_180_h30_g10_need5_age40.pt` from
  `7,806` candidates, positive survival rate `0.554`, validation AUC `0.886`,
  and validation accuracy `0.794` vs base `0.557`.
- Selector-gated overlay gameplay:
  Using the best small synthetic checkpoint
  `/tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt`,
  `--reinforce-target-bias -1.0`, loose overlay parameters
  (`window=40`, `floor=10`, `min_need=5`), and selector threshold `0.7` restored
  Zach to `6/16` with better material and `lost-cap 0.68`, but Ajay stayed
  `0/16` with `lost-cap 0.98`. Lowering the selector threshold to `0.5`
  improved Zach to `7/16` (`lost-cap 0.66`, `reinf_share 0.19`) but Ajay still
  stayed `0/16`, `planets@100 1`, `lost-cap 0.98`. Threshold `0.3` also scored
  Zach `7/16` and moved Ajay lost-cap to `0.96`, but Ajay still collapsed by
  step 100 and material was zero. Readout: the selector is a real offline
  survival predictor and improves Zach retention/quick win rate, but the label
  direction is not the right rescue objective for Ajay. Predicting "will survive"
  filters safe supports; Ajay likely needs a risk/intervention label: states
  where support changes a likely recapture, not merely states that look
  survivable in strong-player replays.
- Selector risk-mode check:
  `eval.py --defense-overlay-selector-mode risk` inverts the selector gate and
  fires when predicted survival is below the threshold. This tested whether the
  survival model could be reused as a recapture-risk detector. It did not work
  well enough to justify Ajay runs: threshold `0.5` dropped Zach to `4/16`
  (`reinf_share 0.31`, `lost-cap 0.78`), while stricter threshold `0.3` scored
  Zach `5/16` (`reinf_share 0.28`, `lost-cap 0.69`). Readout: naive high-risk
  firing mostly reintroduces raw-overlay overreinforcement. The next selector
  label should be intervention-specific, not just inverted survival probability.
- Paired intervention-label selector:
  `collect_defense_interventions.py` runs the supervised BC policy against an
  opponent, finds defense-overlay opportunities, branches the live Kaggle env by
  deep copy, and compares a baseline rollout against a rollout with the support
  move appended. This labels whether the overlay actually changes target
  ownership after a short horizon, rather than asking whether a similar replay
  state survived. Initial Ajay slices showed the label exists but is sparse:
  `s0_16_h30_50` had `10/50` helped and `1/50` hurt; `s100_16_h30_50` had
  `0/50` helped and `3/50` hurt; the combined `s0_64_h30_200` set had
  `17/200` helped and `7/200` hurt. A small selector trained on the 200-record
  set reached validation AUC `0.829`, but positives were only `8.5%` overall.
  Gameplay with this intervention selector preserved the Zach sanity gate
  (`7/16`, `lost-cap 0.64`, `reinf_share 0.21`) but Ajay remained `0/16`
  (`lost-cap 0.96`, `planets@100 1`). Readout: this is the right kind of label
  direction, but the current record count and one-move overlay are too weak to
  change Ajay outcomes. Scaling labels is only worth doing if the collector is
  made more selective toward early recapture states or the action space expands
  beyond one rear-source support move.
- Hold-advantage intervention labels:
  The intervention collector now also records target owner traces for the base
  and support branches, plus `base_owned_steps`, `support_owned_steps`,
  `hold_delta`, first-loss steps, and `hold_advantage`. This turns the label
  from only "owned at the final horizon" into "did this support buy more owned
  ticks during the horizon." On seed slice `s200_48_h30_120`, horizon-helped was
  very sparse (`3/120`), but hold-advantage was less sparse (`20/120`). A
  selector trained with `--label hold_advantage` reached validation AUC `0.775`
  at positive rate `0.167`, but thresholded gameplay overfired support:
  threshold `0.5` scored Zach `5/16` with `reinf_share 0.34`; threshold `0.7`
  also scored Zach `5/16` with `reinf_share 0.31`. No Ajay eval was run because
  the Zach gate regressed below the earlier survival-selector result. Readout:
  hold-time labels are a better data signal than final-horizon labels, but this
  small slice is not enough to produce a usable inference gate. The next version
  should either collect a larger early-contest corpus or train/rank by expected
  `hold_delta` while explicitly penalizing `hurt`, rather than using a raw
  binary hold-advantage threshold.
- ETA-aware intervention features:
  The defense selector feature schema now includes support travel timing:
  `support_eta`, `eta_margin = enemy_min_eta - support_eta`, and
  `support_arrives_before`. Inference remains backward compatible with old
  16-feature selectors by truncating the new feature vector to the saved
  selector length. A continuous `hold_delta` regressor was also added with
  `train_intervention_selector.py --objective regression` and linear selector
  activation. On the old traced slice, penalized `hold_delta_minus_hurt` failed
  offline (`val_corr -0.18`, positive AUC `0.49`), while raw `hold_delta`
  showed only weak signal (`val_corr 0.10`, positive AUC `0.77`) and scored
  Zach `7/16`, Ajay `0/16`.
  A fresh ETA-aware slice `s300_48_h30_120` had denser labels (`7/120` helped,
  `26/120` hold-advantage). The ETA-aware binary hold-advantage selector was
  much cleaner offline (`val_auc 0.926`, `val_acc 0.792` vs base `0.75`).
  Gameplay at threshold `0.5` scored Zach `6/16`; threshold `0.7` recovered Zach
  to `7/16` with `planets@100 10`, `end 9.9`, `lost-cap 0.66`, and
  `reinf_share 0.21`. Ajay still stayed `0/16` with `lost-cap 0.97` and
  collapse by step 100. Readout: support-arrival timing is a real feature and
  should remain in the intervention data, but one synthetic support move is
  still insufficient against Ajay. The next action-space change should consider
  multiple support sources or earlier preemptive garrisoning, because the
  current overlay often reacts after Ajay's counterplay is already decisive.
- Multi-source intervention actions:
  `defensive_overlay_moves(..., multi_source_per_target=True)` now allows
  multiple rear sources to support the same threatened recent capture when
  `max_moves > 1`. Default behavior remains unchanged. Eval exposes this as
  `--defense-overlay-multi-source-per-target`; the paired-intervention collector
  exposes `--support-max-moves` and `--multi-source-per-target`. This is meant
  to move the action closer to a planner allocation instead of a one-move
  patch. Directly using the prior ETA-aware selector with two support moves kept
  Zach at `7/16` and improved Zach retention (`lost-cap 0.63`, end planets
  `10.2`) but Ajay was still `0/16` (`lost-cap 0.96`).
  A fresh multi-source counterfactual slice
  `/tmp/supervised_bc/intervention_ajay_s400_32_h30_80_multisource.pkl`
  showed a much healthier intervention signal: `18/80` final helped,
  `5/80` hurt, and `32/80` hold-advantage. A selector trained on those labels
  reached validation AUC `0.794` and accuracy `0.8125` vs base `0.5625`.
  Gameplay with `max_moves=2`, multi-source enabled, and threshold `0.5` scored
  Zach `9/16`, with `lost-cap 0.57`, end planets `12.6`, and `avg_material
  8105`. Ajay remained `0/16` and collapsed by step 100 (`lost-cap 0.98`).
  Readout: multi-source support is the first clear supervised intervention
  improvement on Zach and gives much denser counterfactual labels, but it still
  does not beat Ajay/Producer-style planning. The next supervised track should
  use the planner itself as the teacher signal: decompose Producer/Ajay actions
  into source-target-ship allocations or preferences, especially around
  contested captures, rather than only adding reactive defense after our BC
  policy has already committed.
- Contest-checkpoint intervention selector:
  Starting from the best pure replay checkpoint
  `/tmp/supervised_bc/jake_live_parquet_contest_w40_heads_600.pt`, a smaller
  Ajay multi-source counterfactual slice collected `40` records with `5/40`
  final helped, `1/40` hurt, and `9/40` hold-advantage. The selector had noisy
  offline signal (`val_auc 0.917` on an eight-record validation split), but live
  gating over-defended: Zach regressed to `3/8` at thresholds `0.5` and `0.7`,
  and Ajay stayed `0/8` with `lost-cap 0.97`. Readout: the contest-fine-tuned
  policy should remain the best supervised-only checkpoint. This overlay is not
  a promotion candidate; larger intervention collection should first get
  partial flushing/resume safety and better risk/hurt labels.
- Producer planner-candidate BC:
  `build_producer_planner_bc.py` builds ordinary BC samples directly from
  Producer/Ajay planner candidate rankings via `_enumerate_attack_candidates`.
  This avoids the previous full-agent teacher failure mode where emitted
  multi-action plans were decoded as a flat action label. Each sample is a
  single scored source-target-ship candidate, with optional filters for
  `score_min`, `top_k`, target owner, steps, and max samples. Smoke dataset
  `/tmp/supervised_bc/producer_planner_top1_s1_80_300.pkl` used all seats from
  the first 10 local replays, `top_k=1`, `score_min=1.5`, steps `1..80`, and
  produced `300` samples with `300` target labels, average Producer score
  `47.0`, `125` neutral targets, `22` own targets, and only `8` reinforce
  labels. A small heads-only fine-tune from the supervised target checkpoint
  mixed these 300 samples with the opening set and selected by
  `val_target_top3`; it barely passed the offline target gate (`top3 0.5015`,
  `top1 0.306`) and scored Zach `6/16`, Ajay not run. Readout: direct planner
  candidate labels are now buildable and decode cleanly, but the first tiny
  top-1 mix is too weak and may underrepresent defensive/regroup decisions. The
  next Producer-supervised run should scale this builder and filter for the
  states Ajay punishes: contested captures, own-target defensive candidates, or
  high-score top-k preferences instead of only the single best candidate.
- Defensive Producer planner-candidate BC:
  The planner builder now supports `--recent-capture-window` and
  `--inbound-threat-horizon`, applied to the candidate target itself. A broad
  own-target planner slice
  `/tmp/supervised_bc/producer_planner_own_top4_s16_160_800.pkl` used
  `target_owner=own`, `top_k=4`, steps `16..160`, and produced `805` samples
  with `668` reinforce labels and average candidate score `60.7`. It passed the
  offline target gate after a heads-only fine-tune (`top3 0.540`) but regressed
  live Zach badly: `1/8`, `lost-cap 0.91`. Adding the multi-source overlay did
  not rescue it (`1/8`, `lost-cap 0.93`). A stricter slice
  `/tmp/supervised_bc/producer_planner_own_recent40_threat30_top4_500.pkl`
  required own-target candidates to be both recent captures and inbound
  threatened; it produced `500` samples with `450` reinforce labels, but the
  heads-only fine-tune failed the generic offline target gate (`top3 0.496`) and
  was not evaluated. Readout: the planner-candidate builder can now isolate the
  right tactical states, but cross-entropy on own-target positives still shifts
  the live policy toward poor allocation. The next planner-supervised attempt
  should avoid overwriting the main target prior: either train a separate
  planner-rerank/postprocessor, add a preference/ranking objective over
  Producer candidates, or validate on a held-out contested-state metric instead
  of the mixed opening-heavy target-top3 gate.
- Producer planner overlay and distilled reranker:
  `eval.py --producer-overlay` appends high-confidence Producer planner
  candidates after the BC target-decode move. This is a diagnostic bridge, not a
  standalone submitted supervised policy: it still enumerates Producer-style
  candidates at inference. Raw Producer-score ordering was strong against Zach
  (`7/8`, `lost-cap 0.48`) but failed Ajay (`0/8`, `lost-cap 0.99`). Restricting
  raw overlay to own targets also failed Ajay (`0/8`), and composing raw overlay
  with the multi-source defense selector was still `0/8`.

  `build_producer_reranker.py` is the separate supervised path. It trains a
  lightweight scorer over source-target-ship candidates to imitate Producer's
  candidate ranking without putting those labels into the main policy heads.
  The smoke reranker
  `/tmp/supervised_bc/producer_reranker_smoke.pt` used only four replays
  (`600` records, `103` states) and reached validation group `top1=0.80`,
  `top3=1.00`. As an overlay scorer it won Zach `4/4` and then Ajay `1/8`
  (`seed=2`, avg material `204.8`, `lost-cap 0.89`). This is the first nonzero
  Ajay gate from the supervised branch, but still far from competent.

  Scaling broad all-seat data to
  `/tmp/supervised_bc/producer_reranker_s30_8k.pt` produced good offline
  metrics (`8000` records, `1012` states, val `top1=0.723`, `top3=0.906`,
  AUC `0.853`) and Zach `8/8`, but Ajay regressed to `0/8`. Filtering to Isaiah
  winning seats with
  `/tmp/supervised_bc/producer_reranker_isaiah_winner_s80_8k.pt` also kept Zach
  `8/8` but stayed Ajay `0/8`. Readout: the architecture direction is better
  than CE into the main heads, but label/data selection matters more than size.
  Next reranker work should preserve the tiny-smoke behavior while adding only
  states with proven retention quality, or train/evaluate specifically on
  Ajay-contested hold decisions rather than generic Producer top-1 ranking.

  The reranker builder now exposes replay-quality filters (`min_replay_score`,
  `max_lost_cap`, `min_median_hold`, `min_cap_attack`, `min_planets50`) and
  state/candidate filters for recent-capture and inbound-threat contexts. A
  strict retention-quality slice
  `/tmp/supervised_bc/producer_reranker_retention_winner_s120_3k.pt` kept only
  `3` Isaiah winning seats from the first `120` local replays (`avg lost_cap
  0.32`), yielding `1816` records over `259` recent-capture states. Offline
  group ranking was strong (`val top1=0.804`, `top3=0.980`, AUC `0.935`).
  Live eval: Zach `7/8`, Ajay `0/8`; restricting overlay candidates to own
  targets was also Ajay `0/8`. Adding `producer_score_tanh` as an optional
  feature produced
  `/tmp/supervised_bc/producer_reranker_retention_scorefeat_s120_3k.pt`, but it
  was still Ajay `0/8` and regressed retention (`lost-cap 1.00`). Readout:
  low-lost-cap replay selection is necessary but not sufficient. The one
  positive signal remains the four-replay smoke reranker; allowing two appended
  planner moves preserved Ajay `1/8` and improved avg material (`300.9` vs
  `204.8`) and game length, while a high raw-score floor (`score_min=20`) killed
  the win (`0/8`). Trace files
  `/tmp/supervised_bc/trace_smoke_ajay_seed2_win.json` and
  `/tmp/supervised_bc/trace_retention_ajay_seed2_loss.json` show the actual
  split: the smoke win uses permissive low-score neutral expansion before step
  `44`, then switches into mostly enemy/own pressure. A scheduled overlay
  (`score_min=1.5` early, `score_min=20` from step `44`) preserved the seed-2
  win, improved that seed's conversion (`cap/atk-launch 0.361`) and game length
  (`195` steps), and kept Zach `8/8`, but the 8-game Ajay gate remained `1/8`.
  The scheduled seed-2 trace had `14/14` early overlay moves to neutral targets;
  late moves were `49` enemy and `50` own-target, with only `1` neutral and avg
  Producer score `66`. Readout: the current useful pattern is phase-conditioned
  allocation, not simply "more good replay data" or a single static score floor.

RL-init diagnostic upper bound:

This is intentionally outside the standalone supervised track. It requires
`bc.py --allow-rl-init` now, so it cannot be confused with pure replay BC.

- `300 steps`, `fire_pos_weight=1.0`, `fire_threshold=0.35`: Zach quick-16
  `9/16`, with `cap/atk-launch 0.431` and `45 attack launches/game`.
- Same checkpoint at threshold `0.25`: Zach quick-16 `11/16`, but regressed into
  launch spam (`373 attack launches/game`, `cap/atk-launch 0.079`).
- `1000 steps`, `fire_pos_weight=1.0`, threshold `0.35`: validation improved
  but gameplay worsened (`8/16` Zach, `234 attack launches/game`).
- Ajay quick-16 remained `0/16`; the failure is mid-game retention after early
  expansion, not initial representation learning.

Top-two replay scaling path:

The small curated datasets above were useful for debugging target labels and
Ajay failure modes, but they are not the right scale for pure BC. A top player
reported using about `47k` replays / `28M` states after pulling every seat from
each game. For this branch, constrain labels to Jake Will and Isaiah only; do
not train on broad all-seat or average-winner data unless doing an explicit
negative ablation.

Fetch a broad score-sorted daily slice, retain only top-two games, then build
compact frame shards. Use `--format frame` for scale; tensor shards are useful
only for small debugging because they are about an order of magnitude larger.

```bash
PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/fetch_best_player_replays.py \
  --last-days 30 \
  --n-per-day 2000 \
  --player-name "Jake Will" \
  --player-name "Isaiah @ Tufa Labs" \
  --out-dir /tmp/orbit_top2_replays \
  --cache-dir /tmp/ow_manifests \
  --max-kept 5000 \
  --retry-attempts 3 \
  --cache-flush-every 100

PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/build_best_player_bc_shards.py \
  --replay-dir /tmp/orbit_top2_replays \
  --player-name "Jake Will" \
  --player-name "Isaiah @ Tufa Labs" \
  --require-win \
  --noop-keep-prob 0.02 \
  --samples-per-shard 50000 \
  --format frame \
  --out-dir /tmp/supervised_bc/top2_win_frame_shards
```

Train from scratch with shard streaming so the trainer does not concatenate the
whole dataset into RAM:

```bash
PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/top2_win_frame_shards/manifest.json \
  --stream-shards \
  --steps 50000 \
  --eval-every 1000 \
  --max-val-samples 8192 \
  --allow-reinforce \
  --select-metric val_target_top3 \
  --save seed_checkpoints/supervised_top2_win_bc.pt
```

Local smoke on the old archived top-replay slice used only Jake/Isaiah winning
seats and produced `1900` samples from `45` games (`Jake Will 24`, `Isaiah @
Tufa Labs 21`), with `98%` decision samples after no-op downsampling. A
5-gradient-step streaming trainer smoke loaded the shard manifest directly and
saved `/tmp/supervised_bc/top2_stream_smoke.pt`; validation was intentionally
weak (`target_top3 0.138`) because this was an infrastructure check, not a real
training run. A live Kaggle acquisition smoke on `2026-06-13` scanned the top
`50` score-sorted 2p episodes and kept `5/5` Jake Will games with avg scores
`1738.997` to `1749.297`. Sharding those fresh replays with `--require-win`
selected `4` Jake winning seats and wrote `184` samples to
`/tmp/supervised_bc/top2_live_smoke_shards/manifest.json`. The equivalent
compact frame shard
`/tmp/supervised_bc/top2_live_frame_smoke_shards/samples_00000.pkl` was `815K`
instead of `11M` for the tensor shard and trained successfully through
`bc.py --stream-shards` in a 5-step smoke
(`/tmp/supervised_bc/top2_frame_stream_smoke.pt`).

Scale-1 run from live top-player data:

- Fetch:
  `/tmp/orbit_top2_replays_scale1` used `--last-days 7 --n-per-day 300
  --max-kept 100` and stopped on `2026-06-13`: `106` episodes considered,
  `106` downloaded, `100` kept, `6` nonmatching. All kept matches were
  `Jake Will`; avg-score range in the kept top slice was roughly `1749` down to
  `1637`.
- Compact shard:
  `/tmp/supervised_bc/top2_scale1_frame_shards/manifest.json` kept `86` Jake
  winning seats after `--require-win`, producing `6738` compact frame samples in
  two shards. `decision_sample_share=0.976`, `fire_slot_rate=0.134`, and
  `replays_without_matching_player=14` are Jake games that were not strict wins
  under the current extraction rule.
- Scratch BC:
  `/tmp/supervised_bc/top2_scale1_frame_400.pt`, trained for `400` streaming
  steps from compact frame shards. The streaming trainer now saves the best
  checkpoint at every improved eval, which made this long local run
  interrupt-safe. Final validation: `val_target_top3=0.507`, `target_top1=0.269`,
  `fire_red=0.590`, `ship_red=0.444`, `target_red=0.284`.
- Gameplay:
  default threshold `0.5`: random `8/8`, Zach `2/8`, Ajay `0/8`.
  Fire threshold `0.4` improved Zach to `4/8`, with planets@50 `6`, end planets
  `7.4`, `cap/atk-launch=0.365`, and `lost-cap=0.69`, but Ajay stayed `0/8`
  with `lost-cap=0.98`. Fire threshold `0.6` stayed Zach `2/8`; the older
  reinforcement gate (`gate>=3`, forward-only, garrison floor `10`) also stayed
  Zach `2/8`.

Readout: top-player-only supervised BC from scratch is now clearly learning
playable behavior without any RL checkpoint: after only `86` winning seats it
beats random and reaches a noisy `4/8` Zach gate with a threshold tweak. The
remaining gap is not no-op collapse or basic target decoding; it is retention
against strong planners. Next data run should scale the same compact route and
either include all Jake/Isaiah seats with outcome weighting or add contested
hold/recapture labels rather than only strict winning-seat imitation.

Outcome-weighted and Isaiah ablations:

- Outcome-weighted Jake all-seat run:
  `/tmp/supervised_bc/top2_scale1_allseats_win2_frame_shards/manifest.json`
  kept all `100` Jake seats from the same scale-1 replay set, repeating wins
  twice and non-wins once. It produced `14,569` compact samples. Scratch BC
  checkpoint `/tmp/supervised_bc/top2_scale1_allseats_win2_frame_400.pt`
  reached validation `val_target_top3=0.506`, close to the strict-win run, but
  gameplay was worse: Zach `3/8` at thresholds `0.4` and `0.5`, Ajay `0/8`.
  Readout: adding Jake non-wins with simple outcome weighting did not improve
  over strict winning-seat imitation.
- Isaiah acquisition via the archived parquet index:
  `/tmp/orbit_parquet/player_episodes.parquet` contains `11,852` player-seat
  rows from `2026-05-20`, including `595` Isaiah rows (`265` wins) and `691`
  Jake rows (`322` wins). Downloading `100` Isaiah wins through
  `archive/bc_pipeline/fetch_top_replays.py` wrote
  `/tmp/orbit_isaiah_parquet_wins`. Sharding those wins produced
  `/tmp/supervised_bc/isaiah_parquet_win_frame_shards/manifest.json` with
  `10,249` compact samples, `decision_sample_share=0.930`, and
  `fire_slot_rate=0.113`.
- Mixed Jake+Isaiah strict-win scratch BC:
  `/tmp/supervised_bc/top2_jake_isaiah_win_frame_500.pt` trained from the Jake
  strict-win manifest plus Isaiah strict-win manifest. It was weaker than
  Jake-only: random `8/8`, Zach `2/8`, Ajay `0/8` at threshold `0.4`.
  Validation was also noisy because, at the time of this run, `--stream-shards`
  reserved whole shard files for validation; this mixed run validated on only
  `249` samples from a tiny Isaiah shard. The trainer now samples validation
  records across shards and skips those records during streaming training.
- Fixed-validation mixed Jake+Isaiah rerun:
  `/tmp/supervised_bc/top2_jake_isaiah_win_frame_fixedval_700.pt` used the same
  Jake + Isaiah manifests but with the fixed cross-shard validation split. It
  trained against `1,699` validation records from all `5` shards and reached
  `val_target_top3=0.500`, `target_top1=0.280`. Gameplay improved over the old
  mixed run but still did not beat strict Jake-only: random `8/8`, Zach `3/8`,
  Ajay `0/8` at threshold `0.4`. Zach retention was better (`lost-cap=0.63`),
  but thresholds `0.35`, `0.45`, and `0.5` did not improve the win gate.
- Isaiah-only scratch BC:
  `/tmp/supervised_bc/isaiah_parquet_win_frame_500.pt` was interrupted after
  step `400`, but the best checkpoint had already been saved
  (`val_target_top3=0.353`). Gameplay: random `8/8`, Zach `1/8`, Ajay `0/8`
  at threshold `0.4`.
- Larger strict-Jake scale-up:
  Downloaded `200/200` Jake winning episodes from the same `2026-05-20` parquet
  index into `/tmp/orbit_jake_parquet_wins_200`. Compact sharding selected all
  `200` Jake winning seats and produced
  `/tmp/supervised_bc/jake_parquet_win200_frame_shards/manifest.json` with
  `17,950` samples, `decision_sample_share=0.976`, and
  `fire_slot_rate=0.127`. Training from scratch on this manifest plus the live
  Jake strict-win manifest (`24,688` samples total) saved
  `/tmp/supervised_bc/jake_live_parquet_win_frame_1000.pt`. Fixed cross-shard
  validation improved materially: `val_target_top3=0.561`, `target_top1=0.327`,
  `fire_red=0.629`, `ship_red=0.595`, `target_red=0.332`.
  Gameplay at threshold `0.4`: random `8/8`, Zach `4/8` and `7/16`, Ajay
  `0/8`. The Zach `8`-game line matched the smaller strict-Jake win count but
  improved retention (`lost-cap=0.61` vs `0.69`); the 16-game check was `7/16`.
  Thresholds `0.3`, `0.35`, `0.45`, and `0.5` were worse on Zach.
- Full available Jake winner slice:
  The same `2026-05-20` parquet index contains `322` Jake wins. Downloading all
  of them into the existing `/tmp/orbit_jake_parquet_wins_200` directory and
  rebuilding `/tmp/supervised_bc/jake_parquet_win322_frame_shards/manifest.json`
  produced `27,952` compact samples with almost identical label distribution to
  the 200-win subset (`decision_sample_share=0.977`, `fire_slot_rate=0.128`).
  Fine-tuning from the 200-win supervised checkpoint saved
  `/tmp/supervised_bc/jake_live_parquet_win322_frame_ft800.pt` and improved
  offline validation to `val_target_top3=0.5686`, `target_top1=0.331`,
  `fire_red=0.639`, `ship_red=0.652`, `target_red=0.340`. Gameplay did not
  improve: random `8/8`, Zach `3/8`, Ajay `0/8` at threshold `0.4`; Ajay
  remained a retention collapse (`lost-cap=0.98`).
- Replay-only contest/hold fine-tune:
  Built `/tmp/supervised_bc/jake_parquet_win200_contest16_160_w40.pkl` from the
  same `200` Jake wins using `score_good_play_replays.py` with `--max-lost-cap
  0.80`, steps `16..160`, and `--contest-window 40`. This accepted `159` Jake
  seats and produced `16,130` tensor samples, `decision_sample_share=0.991`,
  `fire_slot_rate=0.129`, `5,170` reinforcement frames, and `15,384`
  enemy-inbound contest frames. It preserves real replay actions; no synthetic
  defense labels are appended.
- Heads-only contest fine-tune:
  `/tmp/supervised_bc/jake_live_parquet_contest_w40_heads_600.pt` initialized
  from the larger strict-Jake checkpoint, trained only `fire_head`, `ship_head`,
  `tgt_q`, `tgt_k`, and `target_scorer`, and mixed base Jake data with the
  contest slice at low LR (`5e-5`). Validation improved to
  `val_target_top3=0.586`, `target_top1=0.354`, `fire_red=0.640`,
  `ship_red=0.632`, `target_red=0.357`.
  Gameplay at threshold `0.4`: random `8/8`, Zach `6/8` and `10/16`, Ajay
  `0/8`. Zach retention improved sharply (`lost-cap=0.53`, cap/attack
  `0.417`, planets@100 `9`), making this the current best supervised-only Zach
  checkpoint. Ajay threshold checks at `0.35`, `0.4`, and `0.45` were all
  `0/8`; the best Ajay lost-cap among them was still only `0.96`.
- Expanded Jake contest fine-tune:
  Rebuilding the contest-window slice over all `322` Jake wins accepted `255`
  seats and produced `/tmp/supervised_bc/jake_parquet_win322_contest16_160_w40.pkl`
  with `25,523` samples, `23,796` enemy-inbound contest frames,
  `14,416` recent-capture frames, and `decision_sample_share=0.991`.
  Heads-only fine-tuning from the Jake-322 strict checkpoint saved
  `/tmp/supervised_bc/jake_live_parquet_win322_contest_w40_heads_600.pt` with
  `val_target_top3=0.5833`, below the older 200-win contest checkpoint's
  `0.586`. Gameplay stayed below the current best: random `8/8`, Zach `5/8`,
  Ajay `0/8` at threshold `0.4`; Zach threshold sweep was `3/8` at `0.35`,
  `5/8` at `0.45`, and `3/8` at `0.5`.
- Fraction ship-head experiment:
  vkhydras described a per-planet output shape of launch `(P)`, target pointer
  `(P,P)`, and fraction bucket `(P,K)`. Our architecture was already per owned
  planet for fire/target/ship, but the ship head used absolute `SHIP_COUNTS`
  buckets. `bc.py --ship-bin-mode fraction` now relabels compact frame samples
  into 10 bins for `10%..100%` of source ships, saves `ship_bin_mode=fraction`
  checkpoint metadata, and eval decodes the fraction bins automatically.
  A scratch fraction model on Jake-322 frame shards
  `/tmp/supervised_bc/jake_live_parquet_win322_fraction_scratch_1000.pt`
  learned ship sizing easily (`val_ship_red=0.870`) but had weaker target fit
  (`val_target_top3=0.527`). Gameplay: random `8/8`, Zach `4/8`, Ajay `0/8`.
  A compatible-init transfer from the best absolute contest checkpoint skipped
  only the shape-mismatched `ship_head`, preserved target fit
  (`val_target_top3=0.565`), and trained the 10-bin ship head to
  `val_ship_red=0.412`; gameplay still regressed to random `8/8`, Zach `3/8`,
  Ajay `0/8`. Readout: the output shape now matches the top-player comment more
  closely, but fraction ship buckets alone do not fix the retention problem or
  beat the absolute-bucket contest checkpoint.
- Contest-checkpoint intervention selector:
  Collecting a small Ajay multi-source intervention slice from this checkpoint
  produced offline signal (`5/40` final helped, `9/40` hold-advantage), but
  live overlay gating regressed Zach to `3/8` and left Ajay at `0/8`. This is
  not a keeper; it confirms that naive reactive reinforcement disturbs the
  replay-cloned policy more than it helps against Ajay.
- Held-capture outcome-aware replay weighting:
  `score_good_play_replays.py` now supports
  `--held-capture-window`, `--hold-success-horizon`,
  `--held-capture-repeat`, and `--held-capture-only`. This is a replay-only
  trust filter: find owned planets captured recently and not lost within a
  future horizon, then upweight or keep only real replay actions that source
  from or target those held captures. The broad repeat dataset
  `/tmp/supervised_bc/jake_parquet_win322_heldcap_w40_h30_r6.pkl` used Jake's
  `322` downloaded wins, accepted `255` seats, and wrote `48,458` samples with
  `5,100` held-capture-weighted frames. Heads-only fine-tune from the best
  contest checkpoint saved
  `/tmp/supervised_bc/jake_contest_heldcap_w40_h30_r6_heads_500.pt`, but
  validation regressed (`val_target_top3=0.574` vs `0.586`) and gameplay
  regressed: Zach `3/8`, Ajay `0/8`, Ajay `lost-cap=1.00`, `reinf_share=0.41`.
  The hard-filter version
  `/tmp/supervised_bc/jake_parquet_win322_heldcap_only_w40_h30.pkl` wrote
  `10,200` samples from the same `5,100` held-capture frames, dropping `9,316`
  recent-capture frames where the action did not involve the held planet.
  Training only `ship_head`, `tgt_q`, `tgt_k`, and `target_scorer` saved
  `/tmp/supervised_bc/jake_contest_heldcap_only_w40_h30_shiptarget_400.pt`;
  it also failed to improve: `val_target_top3=0.574`, Zach `4/8`, Ajay `0/8`,
  Ajay `lost-cap=1.00`, `reinf_share=0.42`. A decode bias
  `--reinforce-target-bias -1.0` reduced Ajay reinforcement to `0.20`, but Ajay
  stayed `0/8` and Zach fell to `3/8`. Readout: future-held capture filtering
  is the right kind of replay-only trust proxy, but ordinary CE on those frames
  still over-shifts the own-target prior and does not teach the source/timing
  allocation Ajay punishes.
- Replay-action candidate reranker:
  `build_replay_reranker.py` trains the same lightweight candidate scorer format
  used by `eval.py --producer-reranker-checkpoint`, but positives are decoded
  top-player replay launches and negatives are plausible same-state candidates
  from the same source and target-owner mode. The first implementation allowed
  negatives from a different target-owner class, which made own-target ranking
  artificially easy; this is now fixed and covered by
  `tests/test_build_replay_reranker.py`.
  The corrected not-own Jake ranker
  `/tmp/supervised_bc/replay_reranker_jake_notown_s16_120_2k_v2.pt` used `80`
  Jake winning replays, steps `16..120`, and produced `1,952` records over
  `663` replay-positive groups. Offline validation stayed meaningful but not
  trivial (`AUC 0.703`, group top1 `0.629`, top3 `0.985`). Used as a
  one-move not-own Producer-candidate overlay on the best contest checkpoint, it
  scored Zach `8/8` with excellent conversion (`cap/atk-launch 0.572`,
  `lost-cap 0.37`, planets@100 `16`), the strongest supervised-branch Zach
  quick gate so far. Ajay stayed `0/8`: capture volume improved
  (`caps/game 19.9`, planets@50 `6`), but retention remained the failure
  (`lost-cap 1.00`, median hold `9st`).
  A two-move not-own overlay preserved Zach `8/8` but did not help Ajay
  (`0/8`, `lost-cap 0.99`). The corrected own-target recent-capture ranker was
  sparse (`135` records, `67` groups from `160` wins) and did not solve
  retention: Zach `6/8`, Ajay `0/8`, Ajay `lost-cap 0.99`.
  Readout: replay-derived ranking over candidates is a better supervised
  objective than more CE weighting for expansion/pressure, but the current
  candidate overlay still lacks a hold/intervention objective. It should not be
  treated as a standalone submission path yet because runtime candidate
  enumeration is still Producer-style; the supervised contribution is ranking,
  not full policy generation.

Current top-two readout: strict Jake winning-seat imitation is still the best
new replay-cloning branch, and contest-weighted Jake fine-tuning is the best
pure neural supervised checkpoint so far (`10/16` Zach quick gate, Ajay `0/8`).
The replay-action candidate reranker is the strongest supervised-branch Zach
diagnostic (`8/8`), but it is an overlay/ranking bridge rather than a pure
policy. Scaling strict Jake data improves offline fit, but adding the older
same-day Jake wins does not improve gameplay. Adding replay-only contest states
improves Zach retention without synthetic labels, but the smaller 200-win
contest slice still beats the expanded 322-win contest slice. Fraction ship
buckets match the top-player shape description better and improve ship-label
fit, but they do not improve gameplay without a better contested-target/hold
signal. Isaiah data from the old parquet index is learnable enough to beat
random, but the current label format/model fit is weaker than Jake and does not
fix retention. The remaining blocker is still Ajay recapture/hold behavior, not
no-op collapse or basic expansion.

Implication: pure replay-supervised learning is possible without an RL base. The
full-game scratch dataset failed through no-op/spam, but the opening-only
curriculum learned non-random playable behavior from scratch. The current
bottleneck is not initial movement; it is durable capture conversion, especially
reinforcement and holding planets against Ajay. The next serious attempt should
select or weight better contested decisions inside top-player data, not simply
add more same-day Jake wins or add reactive planner overlays.

Next levers:

- Calibrate reinforcement labels by empire size instead of using a single global
  repeat. Current variants swing from too much reinforcement ungated to too little
  after discipline gates.
- Mine states around contested captures/losses, not just all steps `>=50`;
  Ajay failure is a recapture contest problem, not generic late-game behavior.
  Contest-window data, hard answer filtering, soft answer weighting, and
  low-lost-cap outcome filtering all improve or preserve Zach but still fail
  Ajay. A direct synthetic-defense label improved Zach when applied mildly, but
  still failed Ajay. A first threat-head auxiliary learned the threat label but
  hurt action quality. Simple volume from older same-day Jake wins improved
  offline fit but not gameplay, so the next version needs better
  target/source/ship sizing, stronger quality weighting, better calibration, or a separate
  postprocessor that does not disturb the action BC heads.
- Select or weight samples by retention metrics: low lost-cap, higher median hold,
  and healthy garrison fraction after step 50.
- Evaluate checkpoints by behavior gates (`attack launches/game`,
  `cap/atk-launch`, `lost-cap`) before spending panel time.
