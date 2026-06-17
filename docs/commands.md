# Orbit Wars — Command Reference

Sorted by frequency. Copy-paste ready. No guessing paths.

---

## ⚠️ Before running anything locally — activate the venv

```bash
# One-time per terminal session
source /Users/saheb/home/.venv/bin/activate

# Verify: should print /Users/saheb/home/.venv/bin/python
which python
```

`python` alone will not work on this Mac (Python 3.14 is system default, wrong version,
and `nohup python ...` will fail with "No such file or directory"). Always activate first
OR use the full path `/Users/saheb/home/.venv/bin/python`.

---

## 1. Monitor a running training run

```bash
# Tail the latest log on remote (replace IP)
ssh -i ~/.ssh/samosa-key.pem -o StrictHostKeyChecking=no ubuntu@<PUB_IP> \
  "ls -t ~/orbit_wars_rl/train_gpu_phase1_*.log 2>/dev/null | head -1 | xargs tail -5"

# Last 10 iter lines only (fire[0], srcs_multi, fleet_size)
ssh -i ~/.ssh/samosa-key.pem -o StrictHostKeyChecking=no ubuntu@<PUB_IP> \
  "ls -t ~/orbit_wars_rl/train_gpu_phase1_*.log 2>/dev/null | head -1 | xargs grep '^iter' | tail -10"

# Check tmux session is alive
ssh -i ~/.ssh/samosa-key.pem -o StrictHostKeyChecking=no ubuntu@<PUB_IP> "tmux list-sessions"

# GPU utilisation
ssh -i ~/.ssh/samosa-key.pem -o StrictHostKeyChecking=no ubuntu@<PUB_IP> \
  "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
```

> ℹ️ Run name varies by revision. If the above finds nothing, try:
> `ls -t ~/orbit_wars_rl/*.log 2>/dev/null | head -5` to find the active log.

---

## 2. Check locally synced checkpoints

```bash
ls -lht gpu_run_artifacts/hellburner_spot/checkpoints/*.pt | head -10

# Latest watcher heartbeat (look for rsync lines)
tail -20 gpu_run_artifacts/hellburner_spot/logs/watcher_rev7.log
```

---

## 3. Panel eval

### Full panel (256 games, ~40 min locally)

```bash
source /Users/saheb/home/.venv/bin/activate

# Ajay (primary metric — harder than Zach)
python orbit_wars_rl/eval.py \
  --checkpoint <path>.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --panel --target-decode \
  > /tmp/eval_ajay.log 2>&1 &

# Zach (secondary — saturating ~88-89%)
python orbit_wars_rl/eval.py \
  --checkpoint <path>.pt \
  --opponent opponents/candidate_zach_public.py \
  --panel --target-decode \
  > /tmp/eval_zach.log 2>&1 &

wait
grep "Overall" /tmp/eval_ajay.log /tmp/eval_zach.log
```

### Quick eval (16 games, ~2 min — for trend tracking)

```bash
python orbit_wars_rl/eval.py \
  --checkpoint <path>.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 16 --target-decode
```

> ⚠️ `orbit_lite` package must be at `opponents/orbit_lite/` for Ajay/Producer to work.
> ⚠️ Always `--target-decode` for Phase 1. Opponent paths relative to repo root.

### Cross-checkpoint eval (cycling / forgetting detector)

Evals a checkpoint vs HELD-OUT fast heuristics **and our own best past selves**
(exported to `.py`). A drop vs a fixed heuristic = absolute regression; a drop vs
a *past self* = forgetting/cycling — the failure self-play win-rate is blind to.
Run every ~1M steps during training.

```bash
bash gpu_run_artifacts/cross_eval/run_cross_eval.sh <checkpoint.pt> [games=48]
```

Opponent set (all held-out, none in the training pool): zach, 1166-peak,
1043-simple (floor), and exported `self_rev38` / `self_rev53b`.

Re-export a past self as an opponent (needs same feature arch = pairwise-15; rev38+):
```bash
python orbit_wars_rl/export_agent.py --checkpoint <best>.pt \
  --output opponents/ourbest/<name>.py --target-decode
```

### Pool diversity — preseed cross-run best selves

Fold our best *cross-run* checkpoints into the training pool as fast GPU `self`
members (more strategic diversity than dense same-run snapshots → less cycling).
Symlinked under `gpu_run_artifacts/preseed_pool/` (rev38, rev53b — pairwise-15).
Add to a training launch (keep it a *separate* delta — don't bundle with a
method test like VDN):
```bash
  --preseed-pool ../preseed_pool   # appended as 'self' members; step parsed from filename
```
> ⚠️ Older checkpoints (rev31/rev32b = pairwise-12) won't load under the current
> pairwise-15 model — they need the old config to export/preseed (follow-up).

### Producer-style ranking audit on replay losses

```bash
source /Users/saheb/home/.venv/bin/activate

python orbit_wars_rl/analyze_producer_ranking.py \
  --checkpoint gpu_run_artifacts/jarvis/checkpoints/<checkpoint>.pt \
  --replay-dir /tmp/ajay_seed_replays \
  --player-slot 0 \
  --step-limit 40 \
  --output-json /tmp/producer_ranking.json \
  --output-md /tmp/producer_ranking.md
```

This compares, per launch:
- our decoded target
- our target-head top target
- producer-style shortlist best target (`H=18` in 2P)

Use it to check whether failures come from:
- target ranking
- source/shortlist mismatch
- commitment / no valid producer candidate

### Producer whole-action ranking audit

```bash
source /Users/saheb/home/.venv/bin/activate

python orbit_wars_rl/analyze_producer_action_ranking.py \
  --replay-dir /tmp/ajay_seed_replays \
  --player-slot 0 \
  --step-limit 40 \
  --output-json /tmp/producer_action_ranking.json \
  --output-md /tmp/producer_action_ranking.md
```

This compares replay launches against Producer-best **whole actions**:
- best `(source, target, ships)` candidate in the state
- whether replay source matches producer-best source
- whether replay target matches producer-best target
- rank/score of the replay move under the same action scorer

### Producer-target BC dataset for target-head supervision

```bash
source /Users/saheb/home/.venv/bin/activate

python orbit_wars_rl/build_producer_target_bc.py \
  --replay-dir /tmp/sub53359633_eps \
  --player-name Saheb \
  --step-limit 40 \
  --mismatch-repeat 4 \
  --samples-out /tmp/producer_target_bc.pkl \
  --summary-out /tmp/producer_target_bc_summary.json
```

Then fine-tune only the target modules:

```bash
source /Users/saheb/home/.venv/bin/activate

python orbit_wars_rl/bc.py \
  --samples /tmp/producer_target_bc.pkl \
  --init-checkpoint seed_checkpoints/rev31_31M_resume.pt \
  --allow-rl-init \
  --trainable-param tgt_q \
  --trainable-param tgt_k \
  --trainable-param target_scorer \
  --steps 300 \
  --save orbit_wars_rl/checkpoints/bc_producer_target_smoke.pt
```

### Replay-supervised standalone BC track

Parallel non-RL path: score strong replay winners for measurable good play,
downsample idle frames, then train/eval a standalone BC checkpoint. Full workflow:
[`docs/supervised_bc.md`](supervised_bc.md).

Rule for this section: omit `--init-checkpoint` for the first model. Later
curriculum stages may initialize from an earlier supervised BC checkpoint, but
`bc.py` now refuses PPO/RL checkpoints unless `--allow-rl-init` is passed for an
explicit diagnostic outside this track.

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/score_good_play_replays.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --scores-out /tmp/supervised_bc/good_play_scores.json \
  --samples-out /tmp/supervised_bc/good_play_balanced.pkl \
  --require-known-winner \
  --noop-keep-prob 0.05 \
  --fire-repeat 2

/Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_balanced.pkl \
  --steps 300 \
  --lr 1e-4 \
  --fire-pos-weight 1.0 \
  --save seed_checkpoints/supervised_top_winners_bc.pt
```

Opening-only scratch curriculum, which is the first pure supervised variant that
produced playable behavior without an RL init:

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

/Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_known_opening50.pkl \
  --steps 1000 \
  --lr 1e-4 \
  --fire-pos-weight 1.0 \
  --save /tmp/supervised_bc/supervised_opening50_scratch_1000_firew1.pt
```

Mixed opening + retention curriculum, still supervised-only:

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

/Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_known_opening50.pkl \
  --samples /tmp/supervised_bc/good_play_known_retention50_r3.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_opening50_scratch_1000_firew1.pt \
  --steps 700 \
  --lr 7e-5 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --save /tmp/supervised_bc/supervised_curriculum_mix_opening_retention50_r3_700_reinf.pt

CUDA_VISIBLE_DEVICES="" /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_curriculum_mix_opening_retention50_r3_700_reinf.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 16 --target-decode --fire-threshold 0.35 \
  --reinforce-gate-min-planets 3 --reinforce-forward-only --reinforce-garrison-floor 10
```

Contest-window curriculum for the Ajay recapture failure:

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

/Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_known_opening50.pkl \
  --samples /tmp/supervised_bc/good_play_known_contest16_140_w30.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_opening50_scratch_1000_firew1.pt \
  --steps 700 \
  --lr 7e-5 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --save /tmp/supervised_bc/supervised_curriculum_opening_contest16_140_w30_700_reinf.pt
```

Soft answer-inbound weighting. Unlike `--answer-inbound-only`, this repeats
enemy-inbound answer frames while preserving all concurrent teacher moves:

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

/Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_known_opening50.pkl \
  --samples /tmp/supervised_bc/good_play_known_answer_weighted16_140.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_opening50_scratch_1000_firew1.pt \
  --steps 700 \
  --lr 7e-5 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --save /tmp/supervised_bc/supervised_curriculum_opening_answer_weighted16_140_700_reinf.pt
```

Low-lost-cap soft answer-inbound weighting. Current quick result: random `8/8`,
Zach `6/16`, Ajay `0/16` with disciplined reinforcement decode.

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

/Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_known_opening50.pkl \
  --samples /tmp/supervised_bc/good_play_known_lowlost045_answer_weighted16_140.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_opening50_scratch_1000_firew1.pt \
  --steps 600 \
  --lr 7e-5 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --save /tmp/supervised_bc/supervised_curriculum_opening_lowlost045_answer_weighted16_140_600_reinf.pt
```

Synthetic defensive reinforce labels. Uncapped baseline: random `8/8`, Zach
`7/16`, Ajay `0/16` with disciplined reinforcement decode.

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

/Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_known_opening50.pkl \
  --samples /tmp/supervised_bc/good_play_known_lowlost045_synthdef_mild16_140.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_curriculum_opening_lowlost045_answer_weighted16_140_600_reinf.pt \
  --steps 300 \
  --lr 3e-5 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --save /tmp/supervised_bc/supervised_ft_lowlost045_synthdef_mild16_140_300_reinf.pt

CUDA_VISIBLE_DEVICES="" /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_ft_lowlost045_synthdef_mild16_140_300_reinf.pt \
  --opponent opponents/candidate_zach_public.py \
  --games 16 --target-decode --fire-threshold 0.35 \
  --reinforce-gate-min-planets 3 --reinforce-forward-only --reinforce-garrison-floor 10
```

Soft subject-cap ablation. Current best supervised Zach quick result: offline
target gate still failed (`target_red +0.30`, top1 `0.28`, top3 `0.50`), Zach
`8/16`, Ajay `0/16`. Useful for Zach/retention exploration, not a promotion
candidate.

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/score_good_play_replays.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --scores-out /tmp/supervised_bc/good_play_known_lowlost045_synthdef_mild16_140_softcap20_scores.json \
  --samples-out /tmp/supervised_bc/good_play_known_lowlost045_synthdef_mild16_140_softcap20.pkl \
  --require-known-winner \
  --max-lost-cap 0.45 \
  --max-accepted-per-subject 20 \
  --steps-min 16 \
  --steps-max 140 \
  --contest-window 30 \
  --noop-keep-prob 0.02 \
  --fire-repeat 3 \
  --reinforce-repeat 4 \
  --answer-inbound-repeat 4 \
  --synthetic-defense-repeat 4

/Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_known_opening50.pkl \
  --samples /tmp/supervised_bc/good_play_known_lowlost045_synthdef_mild16_140_softcap20.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_curriculum_opening_lowlost045_answer_weighted16_140_600_reinf.pt \
  --steps 300 \
  --lr 3e-5 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --save /tmp/supervised_bc/supervised_ft_lowlost045_synthdef_mild16_140_softcap20_300_reinf.pt

CUDA_VISIBLE_DEVICES="" /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_ft_lowlost045_synthdef_mild16_140_softcap20_300_reinf.pt \
  --opponent opponents/candidate_zach_public.py \
  --games 16 --target-decode --fire-threshold 0.35 \
  --reinforce-gate-min-planets 3 --reinforce-forward-only --reinforce-garrison-floor 10
```

Hard balanced-subject ablation. Result: offline target gate failed (`target_red
+0.29`, top1 `0.27`, top3 `0.49`), Zach `2/16`, Ajay `0/16`; keep for
reproducibility, not as the current best recipe.

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/score_good_play_replays.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --scores-out /tmp/supervised_bc/good_play_known_lowlost045_synthdef_mild16_140_balanced_scores.json \
  --samples-out /tmp/supervised_bc/good_play_known_lowlost045_synthdef_mild16_140_balanced.pkl \
  --require-known-winner \
  --max-lost-cap 0.45 \
  --max-accepted-per-subject 12 \
  --max-samples-per-subject 3000 \
  --steps-min 16 \
  --steps-max 140 \
  --contest-window 30 \
  --noop-keep-prob 0.02 \
  --fire-repeat 3 \
  --reinforce-repeat 4 \
  --answer-inbound-repeat 4 \
  --synthetic-defense-repeat 4

/Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_known_opening50.pkl \
  --samples /tmp/supervised_bc/good_play_known_lowlost045_synthdef_mild16_140_balanced.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_curriculum_opening_lowlost045_answer_weighted16_140_600_reinf.pt \
  --steps 300 \
  --lr 3e-5 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --save /tmp/supervised_bc/supervised_ft_lowlost045_synthdef_mild16_140_balanced_300_reinf.pt
```

Threat-head auxiliary ablation. Current result: threat label learns
(`val_threat_acc 0.75`) but gameplay regressed; Zach `4/16` no bias, `6/16`
with `--threat-target-bias 1.0`, Ajay `0/16`.

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/score_good_play_replays.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --scores-out /tmp/supervised_bc/good_play_known_lowlost045_synthdef_mild_threat24_scores.json \
  --samples-out /tmp/supervised_bc/good_play_known_lowlost045_synthdef_mild_threat24.pkl \
  --require-known-winner \
  --max-lost-cap 0.45 \
  --steps-min 16 \
  --steps-max 140 \
  --contest-window 30 \
  --noop-keep-prob 0.02 \
  --fire-repeat 3 \
  --reinforce-repeat 4 \
  --answer-inbound-repeat 4 \
  --synthetic-defense-repeat 4 \
  --threat-horizon 24

/Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_known_opening50.pkl \
  --samples /tmp/supervised_bc/good_play_known_lowlost045_synthdef_mild_threat24.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_ft_lowlost045_synthdef_mild16_140_300_reinf.pt \
  --steps 300 \
  --lr 2e-5 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --threat-loss-weight 0.2 \
  --threat-pos-weight 8.0 \
  --save /tmp/supervised_bc/supervised_ft_lowlost045_synthdef_threat24_300_reinf.pt

CUDA_VISIBLE_DEVICES="" /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_ft_lowlost045_synthdef_threat24_300_reinf.pt \
  --opponent opponents/candidate_zach_public.py \
  --games 16 --target-decode --fire-threshold 0.35 \
  --reinforce-gate-min-planets 3 --reinforce-forward-only --reinforce-garrison-floor 10 \
  --threat-target-bias 1.0
```

Frozen threat-head-only variant. This preserves the mild synthetic-defense
action policy (`7/16` Zach with no bias) but `--threat-target-bias 0.5` still
regressed Zach to `6/16` and Ajay stayed `0/16`.

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_known_lowlost045_synthdef_mild_threat24.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_ft_lowlost045_synthdef_mild16_140_300_reinf.pt \
  --trainable-param threat_head \
  --steps 1000 \
  --lr 1e-3 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --threat-loss-weight 1.0 \
  --threat-pos-weight 8.0 \
  --save /tmp/supervised_bc/supervised_threathead_only_lowlost045_h24_1000_reinf.pt

CUDA_VISIBLE_DEVICES="" /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_threathead_only_lowlost045_h24_1000_reinf.pt \
  --opponent opponents/candidate_zach_public.py \
  --games 16 --target-decode --fire-threshold 0.35 \
  --reinforce-gate-min-planets 3 --reinforce-forward-only --reinforce-garrison-floor 10 \
  --threat-target-bias 0.5
```

Narrow answer-inbound ablation. This regressed in quick eval, but keep the
command for reproducibility:

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

/Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_known_opening50.pkl \
  --samples /tmp/supervised_bc/good_play_known_answer_inbound16_140.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_opening50_scratch_1000_firew1.pt \
  --steps 600 \
  --lr 7e-5 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --save /tmp/supervised_bc/supervised_curriculum_opening_answer_inbound16_140_600_reinf.pt
```

Policy-teacher BC path. This asks a deterministic teacher policy for fresh
actions on replay observations, instead of cloning the replay action. The
10-replay Ajay-teacher smoke below produced `1,246` samples but failed the
offline target gate after fine-tune (`target_red +0.22`, top1 `0.17`, top3
`0.39`), so treat it as pipeline validation before scaling/curating.

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/build_policy_teacher_bc.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --max-replays 10 \
  --teacher-agent opponents/candidate_ajay_1200.py \
  --seat-mode winner \
  --steps-max 100 \
  --noop-keep-prob 0.0 \
  --action-repeat 1 \
  --reinforce-repeat 2 \
  --samples-out /tmp/supervised_bc/ajay_teacher_winner10_s100.pkl \
  --summary-out /tmp/supervised_bc/ajay_teacher_winner10_s100_summary.json

/Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/ajay_teacher_winner10_s100.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_opening50_scratch_1000_firew1.pt \
  --steps 300 \
  --lr 5e-5 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --save /tmp/supervised_bc/supervised_opening50_ajay_teacher10_s100_300_reinf.pt
```

All-seat scale-up also failed the target gate (`target_red +0.22`, top1 `0.19`,
top3 `0.38`) despite `9,329` samples. This suggests direct full-action teacher
imitation needs filtered target-only or factorized labels before gameplay eval.

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/build_policy_teacher_bc.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --max-replays 30 \
  --teacher-agent opponents/candidate_ajay_1200.py \
  --seat-mode all \
  --steps-max 120 \
  --noop-keep-prob 0.0 \
  --action-repeat 1 \
  --reinforce-repeat 2 \
  --samples-out /tmp/supervised_bc/ajay_teacher_all30_s120.pkl \
  --summary-out /tmp/supervised_bc/ajay_teacher_all30_s120_summary.json

/Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/ajay_teacher_all30_s120.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_opening50_scratch_1000_firew1.pt \
  --steps 600 \
  --lr 5e-5 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --save /tmp/supervised_bc/supervised_opening50_ajay_teacher_all30_s120_600_reinf.pt
```

Split-move target-only ablation. This decomposes each teacher frame into one
single-launch sample and trains only the target-ranking parameters. It improved
top3 over full-action labels but still failed the gate (`target_red +0.23`,
top1 `0.20`, top3 `0.43`), so keep as evidence for the next filtered/ranking
variant.

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/build_policy_teacher_bc.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --max-replays 30 \
  --teacher-agent opponents/candidate_ajay_1200.py \
  --seat-mode all \
  --steps-max 120 \
  --noop-keep-prob 0.0 \
  --action-repeat 1 \
  --reinforce-repeat 2 \
  --split-moves \
  --samples-out /tmp/supervised_bc/ajay_teacher_all30_s120_splitmoves.pkl \
  --summary-out /tmp/supervised_bc/ajay_teacher_all30_s120_splitmoves_summary.json

/Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/ajay_teacher_all30_s120_splitmoves.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_opening50_scratch_1000_firew1.pt \
  --trainable-param tgt_q \
  --trainable-param tgt_k \
  --trainable-param target_scorer \
  --steps 600 \
  --lr 1e-4 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --save /tmp/supervised_bc/supervised_opening50_ajay_teacher_all30_s120_splitmoves_targetonly_600_reinf.pt
```

Filtered split-move ablation. The single-move-only filter produced just `897`
samples, so the more useful run keeps frames with at most two teacher moves and
filters out own-target labels. Selecting by total BC loss still failed the final
target gate (`target_red +0.34`, top1 `0.30`, top3 `0.47`). Selecting by
`val_target_top3` plus a one-move-per-source filter passed the offline gate
(`target_red +0.32`, top1 `0.32`, top3 `0.55`) but still failed live Ajay
quick eval (`0/16`, `lost-cap 0.97`, `reinf_share 0.03`).

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/build_policy_teacher_bc.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --max-replays 30 \
  --teacher-agent opponents/candidate_ajay_1200.py \
  --seat-mode all \
  --steps-max 120 \
  --noop-keep-prob 0.0 \
  --action-repeat 1 \
  --reinforce-repeat 2 \
  --split-moves \
  --max-teacher-moves-per-frame 2 \
  --max-teacher-moves-per-source 1 \
  --target-owner not-own \
  --samples-out /tmp/supervised_bc/ajay_teacher_all30_s120_max2_source1_notown.pkl \
  --summary-out /tmp/supervised_bc/ajay_teacher_all30_s120_max2_source1_notown_summary.json

/Users/saheb/home/.venv/bin/python -u orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/ajay_teacher_all30_s120_max2_source1_notown.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_opening50_scratch_1000_firew1.pt \
  --trainable-param tgt_q \
  --trainable-param tgt_k \
  --trainable-param target_scorer \
  --steps 600 \
  --lr 1e-4 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --select-metric val_target_top3 \
  --save /tmp/supervised_bc/supervised_opening50_ajay_teacher_all30_s120_max2_source1_notown_targettop3_600_reinf.pt
```

Own-target policy-teacher retention ablation. These labels are learnable but
not promotion candidates: both repeated and no-repeat mixes suppress attack
tempo. The no-repeat mix is the less extreme version and still scored Zach
`1/16`, Ajay `0/16`; `--reinforce-gate-min-planets 7` only improved Zach to
`2/16`.

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/build_policy_teacher_bc.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --max-replays 30 \
  --teacher-agent opponents/candidate_ajay_1200.py \
  --seat-mode all \
  --steps-min 16 \
  --steps-max 140 \
  --noop-keep-prob 0.0 \
  --action-repeat 1 \
  --reinforce-repeat 1 \
  --split-moves \
  --target-owner own \
  --samples-out /tmp/supervised_bc/ajay_teacher_all30_s16_140_own_r1.pkl \
  --summary-out /tmp/supervised_bc/ajay_teacher_all30_s16_140_own_r1_summary.json

/Users/saheb/home/.venv/bin/python -u orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_known_opening50.pkl \
  --samples /tmp/supervised_bc/ajay_teacher_all30_s16_140_own_r1.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_opening50_ajay_teacher_all30_s120_max2_source1_notown_targettop3_600_reinf.pt \
  --steps 600 \
  --lr 5e-5 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --select-metric val_target_top3 \
  --save /tmp/supervised_bc/supervised_opening50_targettop3_ajay_own30_s16_140_r1_600_reinf.pt

CUDA_VISIBLE_DEVICES="" /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_ajay_own30_s16_140_r1_600_reinf.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 16 \
  --target-decode \
  --reinforce-gate-min-planets 7
```

Low-ratio target-only retention plus decode bias. This preserved attack volume
better than full-head retention, but still did not improve Ajay. Best quick
bias tested was `--reinforce-target-bias -1.0`: Zach `3/16`, Ajay `0/16`.

```bash
/Users/saheb/home/.venv/bin/python -c "import pickle, random; src='/tmp/supervised_bc/ajay_teacher_all30_s16_140_own_r1.pkl'; dst='/tmp/supervised_bc/ajay_teacher_all30_s16_140_own_r1_sub2k.pkl'; rng=random.Random(17); samples=pickle.load(open(src,'rb')); pickle.dump(rng.sample(samples, min(2000, len(samples))), open(dst,'wb'))"

/Users/saheb/home/.venv/bin/python -u orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_known_opening50.pkl \
  --samples /tmp/supervised_bc/ajay_teacher_all30_s16_140_own_r1_sub2k.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_opening50_ajay_teacher_all30_s120_max2_source1_notown_targettop3_600_reinf.pt \
  --trainable-param tgt_q \
  --trainable-param tgt_k \
  --trainable-param target_scorer \
  --steps 500 \
  --lr 5e-5 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --select-metric val_target_top3 \
  --save /tmp/supervised_bc/supervised_opening50_targettop3_ajay_own30_s16_140_r1_sub2k_targetonly_500_reinf.pt

CUDA_VISIBLE_DEVICES="" /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_ajay_own30_s16_140_r1_sub2k_targetonly_500_reinf.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 16 \
  --target-decode \
  --reinforce-target-bias -1.0
```

Threat-conditioned own-target retention. This keeps only teacher own-target
labels where the target has enemy inbound within 30 steps. It is better
calibrated than generic own-target labels but still not Ajay-capable: best quick
decode tested was `--reinforce-target-bias -1.0`, scoring Zach `4/16`, Ajay
`0/16`.

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/build_policy_teacher_bc.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --max-replays 30 \
  --teacher-agent opponents/candidate_ajay_1200.py \
  --seat-mode all \
  --steps-min 16 \
  --steps-max 160 \
  --noop-keep-prob 0.0 \
  --action-repeat 1 \
  --reinforce-repeat 1 \
  --split-moves \
  --target-owner own \
  --inbound-threat-horizon 30 \
  --samples-out /tmp/supervised_bc/ajay_teacher_all30_s16_160_own_threat30.pkl \
  --summary-out /tmp/supervised_bc/ajay_teacher_all30_s16_160_own_threat30_summary.json

/Users/saheb/home/.venv/bin/python -u orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_known_opening50.pkl \
  --samples /tmp/supervised_bc/ajay_teacher_all30_s16_160_own_threat30.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_opening50_ajay_teacher_all30_s120_max2_source1_notown_targettop3_600_reinf.pt \
  --trainable-param tgt_q \
  --trainable-param tgt_k \
  --trainable-param target_scorer \
  --steps 500 \
  --lr 5e-5 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --select-metric val_target_top3 \
  --save /tmp/supervised_bc/supervised_opening50_targettop3_ajay_own_threat30_targetonly_500_reinf.pt

CUDA_VISIBLE_DEVICES="" /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_ajay_own_threat30_targetonly_500_reinf.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 16 \
  --target-decode \
  --reinforce-target-bias -1.0
```

Synthetic source/ship defense labels. This directly labels rear-source support
and ship count for threatened planets. First run is not a promotion candidate:
ungated Zach `3/16`, Ajay `0/16`; with `--reinforce-target-bias -1.0`, Zach
`4/16`, Ajay `0/16`.

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/build_synthetic_defense_bc.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --max-replays 30 \
  --seat-mode all \
  --steps-min 16 \
  --steps-max 180 \
  --action-repeat 1 \
  --garrison-floor 10 \
  --min-need 5 \
  --max-samples 4000 \
  --samples-out /tmp/supervised_bc/synthetic_defense_all30_s16_180_g10_need5_4k.pkl \
  --summary-out /tmp/supervised_bc/synthetic_defense_all30_s16_180_g10_need5_4k_summary.json

/Users/saheb/home/.venv/bin/python -u orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_known_opening50.pkl \
  --samples /tmp/supervised_bc/synthetic_defense_all30_s16_180_g10_need5_4k.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_opening50_ajay_teacher_all30_s120_max2_source1_notown_targettop3_600_reinf.pt \
  --trainable-param fire_head \
  --trainable-param ship_head \
  --trainable-param tgt_q \
  --trainable-param tgt_k \
  --trainable-param target_scorer \
  --steps 600 \
  --lr 5e-5 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --select-metric val_target_top3 \
  --save /tmp/supervised_bc/supervised_opening50_targettop3_synthdef4k_heads_600_reinf.pt

CUDA_VISIBLE_DEVICES="" /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_synthdef4k_heads_600_reinf.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 16 \
  --target-decode \
  --reinforce-target-bias -1.0
```

Small-ratio synthetic defense. This preserves the fire head and trains only ship
and target parameters on a 500-sample synthetic-defense dose. Best decode tested:
`--reinforce-target-bias -1.0`, Zach `6/16`, Ajay `0/16`.

```bash
/Users/saheb/home/.venv/bin/python -c "import pickle, random; src='/tmp/supervised_bc/synthetic_defense_all30_s16_180_g10_need5_4k.pkl'; dst='/tmp/supervised_bc/synthetic_defense_all30_s16_180_g10_need5_sub500.pkl'; rng=random.Random(23); samples=pickle.load(open(src,'rb')); pickle.dump(rng.sample(samples, min(500, len(samples))), open(dst,'wb'))"

/Users/saheb/home/.venv/bin/python -u orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_known_opening50.pkl \
  --samples /tmp/supervised_bc/synthetic_defense_all30_s16_180_g10_need5_sub500.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_opening50_ajay_teacher_all30_s120_max2_source1_notown_targettop3_600_reinf.pt \
  --trainable-param ship_head \
  --trainable-param tgt_q \
  --trainable-param tgt_k \
  --trainable-param target_scorer \
  --steps 500 \
  --lr 5e-5 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --select-metric val_target_top3 \
  --save /tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt

CUDA_VISIBLE_DEVICES="" /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt \
  --opponent opponents/candidate_zach_public.py \
  --games 16 \
  --target-decode \
  --reinforce-target-bias -1.0
```

Outcome-aware hold-success synthetic defense. This is a stricter variant of the
same synthetic source/ship idea: only label targets that were recently captured
and are not lost in the next 30 steps. First tested run produced only `362`
samples and matched, but did not beat, the 500-sample synthetic result: Zach
`6/16`, Ajay `0/16` with `--reinforce-target-bias -1.0`.

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/build_synthetic_defense_bc.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --max-replays 30 \
  --seat-mode all \
  --steps-min 16 \
  --steps-max 180 \
  --garrison-floor 10 \
  --min-need 5 \
  --recent-capture-window 40 \
  --hold-success-horizon 30 \
  --max-samples 1000 \
  --samples-out /tmp/supervised_bc/synthetic_defense_holdsuccess_all30_s16_180_w40_h30_1k.pkl \
  --summary-out /tmp/supervised_bc/synthetic_defense_holdsuccess_all30_s16_180_w40_h30_1k_summary.json

/Users/saheb/home/.venv/bin/python -u orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_known_opening50.pkl \
  --samples /tmp/supervised_bc/synthetic_defense_holdsuccess_all30_s16_180_w40_h30_1k.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_opening50_ajay_teacher_all30_s120_max2_source1_notown_targettop3_600_reinf.pt \
  --trainable-param fire_head \
  --trainable-param ship_head \
  --trainable-param tgt_q \
  --trainable-param tgt_k \
  --trainable-param target_scorer \
  --steps 500 \
  --lr 5e-5 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --select-metric val_target_top3 \
  --save /tmp/supervised_bc/supervised_opening50_targettop3_holdsuccess_synthdef362_heads_500_reinf.pt

CUDA_VISIBLE_DEVICES="" /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_holdsuccess_synthdef362_heads_500_reinf.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 16 \
  --target-decode \
  --reinforce-target-bias -1.0
```

Defense-overlay inference ablation. Disabled by default. This appends
post-decode synthetic rear-source support moves to recently captured threatened
owned planets. It is diagnostic, not a promotion recipe: `window=20`,
`floor=30`, `min_need=10` restored Zach to `6/16` but Ajay stayed `0/16`
(`lost-cap 0.96`).

```bash
CUDA_VISIBLE_DEVICES="" /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 16 \
  --target-decode \
  --reinforce-target-bias -1.0 \
  --defense-overlay \
  --defense-overlay-recent-capture-window 20 \
  --defense-overlay-garrison-floor 30 \
  --defense-overlay-min-need 10 \
  --defense-overlay-max-moves 1
```

Defense-overlay selector. This trains a small supervised survival selector for
overlay candidates from replay outcomes. Full local corpus result:
`7,806` candidates, validation AUC `0.886`. Best quick Zach setting so far:
threshold `0.5`, Zach `7/16`; Ajay remains `0/16`.

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/build_defense_selector.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --seat-mode all \
  --steps-min 16 \
  --steps-max 180 \
  --hold-horizon 30 \
  --garrison-floor 10 \
  --min-need 5 \
  --max-target-age 40 \
  --train-steps 3000 \
  --lr 0.03 \
  --records-out /tmp/supervised_bc/defense_selector_all_s16_180_h30_g10_need5_age40.pkl \
  --selector-out /tmp/supervised_bc/defense_selector_all_s16_180_h30_g10_need5_age40.pt \
  --summary-out /tmp/supervised_bc/defense_selector_all_s16_180_h30_g10_need5_age40_summary.json

CUDA_VISIBLE_DEVICES="" /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt \
  --opponent opponents/candidate_zach_public.py \
  --games 16 \
  --target-decode \
  --reinforce-target-bias -1.0 \
  --defense-overlay \
  --defense-overlay-recent-capture-window 40 \
  --defense-overlay-garrison-floor 10 \
  --defense-overlay-min-need 5 \
  --defense-overlay-max-moves 1 \
  --defense-overlay-selector-checkpoint /tmp/supervised_bc/defense_selector_all_s16_180_h30_g10_need5_age40.pt \
  --defense-overlay-selector-threshold 0.5
```

Risk-mode selector check. This inverts the survival selector and fires on low
predicted survival. It regressed Zach (`4/16` at threshold `0.5`, `5/16` at
threshold `0.3`), so do not use this as the current best setting.

```bash
CUDA_VISIBLE_DEVICES="" /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt \
  --opponent opponents/candidate_zach_public.py \
  --games 16 \
  --target-decode \
  --reinforce-target-bias -1.0 \
  --defense-overlay \
  --defense-overlay-recent-capture-window 40 \
  --defense-overlay-garrison-floor 10 \
  --defense-overlay-min-need 5 \
  --defense-overlay-max-moves 1 \
  --defense-overlay-selector-checkpoint /tmp/supervised_bc/defense_selector_all_s16_180_h30_g10_need5_age40.pt \
  --defense-overlay-selector-threshold 0.3 \
  --defense-overlay-selector-mode risk
```

Paired intervention selector. This collects labels by branching live eval states:
baseline action versus baseline plus one defense-overlay support move. It is
still supervised learning, but the label is intervention outcome instead of
replay survival. Current 200-record Ajay slice has validation AUC `0.829`, Zach
`7/16`, Ajay `0/16`, so this is a diagnostic track, not a promotion candidate.

```bash
/Users/saheb/home/.venv/bin/python orbit_wars_rl/collect_defense_interventions.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 64 \
  --seed-start 0 \
  --horizon 30 \
  --recent-capture-window 40 \
  --garrison-floor 10 \
  --min-need 5 \
  --max-records 200 \
  --reinforce-target-bias -1.0 \
  --records-out /tmp/supervised_bc/intervention_ajay_s0_64_h30_200.pkl \
  --summary-out /tmp/supervised_bc/intervention_ajay_s0_64_h30_200_summary.json

/Users/saheb/home/.venv/bin/python orbit_wars_rl/train_intervention_selector.py \
  --records /tmp/supervised_bc/intervention_ajay_s0_64_h30_200.pkl \
  --steps 2000 \
  --lr 0.03 \
  --selector-out /tmp/supervised_bc/intervention_selector_ajay_s0_64_h30_200.pt \
  --summary-out /tmp/supervised_bc/intervention_selector_ajay_s0_64_h30_200_summary.json

CUDA_VISIBLE_DEVICES="" /Users/saheb/home/.venv/bin/python orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt \
  --opponent opponents/candidate_zach_public.py \
  --games 16 \
  --target-decode \
  --reinforce-target-bias -1.0 \
  --defense-overlay \
  --defense-overlay-recent-capture-window 40 \
  --defense-overlay-garrison-floor 10 \
  --defense-overlay-min-need 5 \
  --defense-overlay-max-moves 1 \
  --defense-overlay-selector-checkpoint /tmp/supervised_bc/intervention_selector_ajay_s0_64_h30_200.pt \
  --defense-overlay-selector-threshold 0.5 \
  --defense-overlay-selector-mode survive
```

Top-two supervised replay scaling. Use this as the main pure-BC data path now:
fetch broad daily score slices, retain only Jake Will / Isaiah games, write
compact frame shards for their winning-seat labels, then stream shards into
`bc.py`.

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

Smoke result on the old 200-replay local archive:
`/tmp/supervised_bc/top2_smoke_shards/manifest.json` contains `1900` samples
from `45` Jake/Isaiah winning games. `bc.py --stream-shards` loaded the manifest
and saved `/tmp/supervised_bc/top2_stream_smoke.pt` in a 5-step infrastructure
check. Live Kaggle smoke on `2026-06-13`: scanning the top `50` score-sorted 2p
episodes kept `5/5` Jake Will games; `--require-win` sharding selected `4`
winning seats and wrote `184` samples to
`/tmp/supervised_bc/top2_live_smoke_shards/manifest.json`. The compact
`--format frame` shard for the same records was `815K` versus `11M` for the
tensor shard, and `bc.py --stream-shards` trained from it in a 5-step smoke,
saving `/tmp/supervised_bc/top2_frame_stream_smoke.pt`.

Scale-1 live data/checkpoint:

```bash
PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/fetch_best_player_replays.py \
  --last-days 7 \
  --n-per-day 300 \
  --player-name "Jake Will" \
  --player-name "Isaiah @ Tufa Labs" \
  --out-dir /tmp/orbit_top2_replays_scale1 \
  --cache-dir /tmp/ow_manifests \
  --max-kept 100 \
  --retry-attempts 3 \
  --retry-sleep 0.5 \
  --cache-flush-every 25

PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/build_best_player_bc_shards.py \
  --replay-dir /tmp/orbit_top2_replays_scale1 \
  --player-name "Jake Will" \
  --player-name "Isaiah @ Tufa Labs" \
  --require-win \
  --noop-keep-prob 0.02 \
  --samples-per-shard 5000 \
  --format frame \
  --out-dir /tmp/supervised_bc/top2_scale1_frame_shards

PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/top2_scale1_frame_shards/manifest.json \
  --stream-shards \
  --steps 400 \
  --eval-every 100 \
  --max-val-samples 1024 \
  --allow-reinforce \
  --select-metric val_target_top3 \
  --save /tmp/supervised_bc/top2_scale1_frame_400.pt
```

Scale-1 result: fetch kept `100` top-player replays from `106` considered
episodes on `2026-06-13`, all `Jake Will`. Strict-win compact sharding selected
`86` seats and `6738` samples. The `400`-step scratch checkpoint reached
validation `target_top1=0.269`, `target_top3=0.507`, `fire_red=0.590`,
`ship_red=0.444`. Gameplay: default threshold `0.5` random `8/8`, Zach `2/8`,
Ajay `0/8`; `--fire-threshold 0.4` improved Zach to `4/8` but Ajay stayed
`0/8` with `lost-cap=0.98`.

Isaiah acquisition from the archived player index. Use this when broad recent
score scanning is too slow or yields only Jake games:

```bash
mkdir -p /tmp/orbit_parquet
/Users/saheb/home/.venv/bin/kaggle datasets download \
  nbridelancetb/orbit-wars-replay-parquet \
  -f player_episodes.parquet \
  -p /tmp/orbit_parquet \
  --force

/Users/saheb/home/.venv/bin/python archive/bc_pipeline/fetch_top_replays.py \
  --parquet /tmp/orbit_parquet/player_episodes.parquet \
  --agents "Isaiah @ Tufa Labs" \
  --n-episodes 100 \
  --replay-dir /tmp/orbit_isaiah_parquet_wins \
  --download-only \
  --delay 0.1

PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/build_best_player_bc_shards.py \
  --replay-dir /tmp/orbit_isaiah_parquet_wins \
  --player-name "Isaiah @ Tufa Labs" \
  --require-win \
  --noop-keep-prob 0.02 \
  --samples-per-shard 5000 \
  --format frame \
  --out-dir /tmp/supervised_bc/isaiah_parquet_win_frame_shards
```

Current top-two ablation results:

- Outcome-weighted Jake all-seat run:
  `/tmp/supervised_bc/top2_scale1_allseats_win2_frame_400.pt`, Zach `3/8`,
  Ajay `0/8`.
- Mixed strict Jake+Isaiah:
  `/tmp/supervised_bc/top2_jake_isaiah_win_frame_500.pt`, random `8/8`, Zach
  `2/8`, Ajay `0/8` at `--fire-threshold 0.4`.
- Fixed-validation mixed strict Jake+Isaiah:
  `/tmp/supervised_bc/top2_jake_isaiah_win_frame_fixedval_700.pt`, random
  `8/8`, Zach `3/8`, Ajay `0/8` at `--fire-threshold 0.4`.
- Isaiah-only:
  `/tmp/supervised_bc/isaiah_parquet_win_frame_500.pt`, random `8/8`, Zach
  `1/8`, Ajay `0/8` at `--fire-threshold 0.4`.

Larger strict-Jake scale-up from the same parquet index:

```bash
/Users/saheb/home/.venv/bin/python archive/bc_pipeline/fetch_top_replays.py \
  --parquet /tmp/orbit_parquet/player_episodes.parquet \
  --agents "Jake Will" \
  --n-episodes 200 \
  --replay-dir /tmp/orbit_jake_parquet_wins_200 \
  --download-only \
  --delay 0.05

PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/build_best_player_bc_shards.py \
  --replay-dir /tmp/orbit_jake_parquet_wins_200 \
  --player-name "Jake Will" \
  --require-win \
  --noop-keep-prob 0.02 \
  --samples-per-shard 5000 \
  --format frame \
  --out-dir /tmp/supervised_bc/jake_parquet_win200_frame_shards

PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/top2_scale1_frame_shards/manifest.json \
  --samples /tmp/supervised_bc/jake_parquet_win200_frame_shards/manifest.json \
  --stream-shards \
  --steps 1000 \
  --eval-every 100 \
  --max-val-samples 4096 \
  --allow-reinforce \
  --select-metric val_target_top3 \
  --save /tmp/supervised_bc/jake_live_parquet_win_frame_1000.pt
```

Result:
`/tmp/supervised_bc/jake_parquet_win200_frame_shards/manifest.json` has
`17,950` compact samples from `200` Jake wins. Combined with the live strict
Jake manifest, `/tmp/supervised_bc/jake_live_parquet_win_frame_1000.pt` reached
`val_target_top3=0.561`, random `8/8`, Zach `4/8` and `7/16`, Ajay `0/8` at
`--fire-threshold 0.4`. Thresholds `0.3`, `0.35`, `0.45`, and `0.5` were worse
on Zach.

Full available Jake winner slice from the same parquet index. The directory
name is historical; it now contains all `322` Jake wins from the `2026-05-20`
index:

```bash
/Users/saheb/home/.venv/bin/python archive/bc_pipeline/fetch_top_replays.py \
  --parquet /tmp/orbit_parquet/player_episodes.parquet \
  --agents "Jake Will" \
  --n-episodes 400 \
  --replay-dir /tmp/orbit_jake_parquet_wins_200 \
  --download-only \
  --delay 0.02

PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/build_best_player_bc_shards.py \
  --replay-dir /tmp/orbit_jake_parquet_wins_200 \
  --player-name "Jake Will" \
  --require-win \
  --noop-keep-prob 0.02 \
  --samples-per-shard 5000 \
  --format frame \
  --out-dir /tmp/supervised_bc/jake_parquet_win322_frame_shards

PYTHONUNBUFFERED=1 PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/top2_scale1_frame_shards/manifest.json \
  --samples /tmp/supervised_bc/jake_parquet_win322_frame_shards/manifest.json \
  --stream-shards \
  --init-checkpoint /tmp/supervised_bc/jake_live_parquet_win_frame_1000.pt \
  --steps 800 \
  --lr 5e-5 \
  --eval-every 100 \
  --max-val-samples 4096 \
  --allow-reinforce \
  --select-metric val_target_top3 \
  --save /tmp/supervised_bc/jake_live_parquet_win322_frame_ft800.pt
```

Result:
`/tmp/supervised_bc/jake_parquet_win322_frame_shards/manifest.json` has
`27,952` compact samples from all `322` Jake wins, with
`decision_sample_share=0.977` and `fire_slot_rate=0.128`. Fine-tuning from the
200-win supervised checkpoint improved offline target fit to
`val_target_top3=0.5686`, but gameplay did not improve: random `8/8`, Zach
`3/8`, Ajay `0/8` at threshold `0.4`.

Replay-only contest/hold fine-tune from strong Jake games. This keeps real
replay actions and only reweights the state distribution around recent captures
or enemy inbound threats:

```bash
PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/score_good_play_replays.py \
  --replay-dir /tmp/orbit_jake_parquet_wins_200 \
  --winner-name "Jake Will" \
  --strong-name "Jake Will" \
  --require-known-winner \
  --min-score 0 \
  --max-lost-cap 0.80 \
  --steps-min 16 \
  --steps-max 160 \
  --contest-window 40 \
  --noop-keep-prob 0.02 \
  --fire-repeat 1 \
  --reinforce-repeat 2 \
  --scores-out /tmp/supervised_bc/jake_parquet_win200_contest16_160_w40_scores.json \
  --samples-out /tmp/supervised_bc/jake_parquet_win200_contest16_160_w40.pkl

PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/top2_scale1_frame_shards/manifest.json \
  --samples /tmp/supervised_bc/jake_parquet_win200_frame_shards/manifest.json \
  --samples /tmp/supervised_bc/jake_parquet_win200_contest16_160_w40.pkl \
  --stream-shards \
  --init-checkpoint /tmp/supervised_bc/jake_live_parquet_win_frame_1000.pt \
  --steps 600 \
  --lr 5e-5 \
  --eval-every 100 \
  --max-val-samples 4096 \
  --allow-reinforce \
  --select-metric val_target_top3 \
  --trainable-param fire_head \
  --trainable-param ship_head \
  --trainable-param tgt_q \
  --trainable-param tgt_k \
  --trainable-param target_scorer \
  --save /tmp/supervised_bc/jake_live_parquet_contest_w40_heads_600.pt
```

Result:
`/tmp/supervised_bc/jake_parquet_win200_contest16_160_w40.pkl` has `16,130`
samples from `159` accepted Jake seats, with `15,384` enemy-inbound contest
frames and `5,170` reinforcement frames. The heads-only fine-tune reached
`val_target_top3=0.586`, random `8/8`, Zach `6/8` and `10/16`, Ajay `0/8` at
`--fire-threshold 0.4`. Ajay threshold checks at `0.35` and `0.45` were also
`0/8`. This is the current best supervised-only Zach checkpoint, not an Ajay
promotion candidate.

Expanded Jake contest fine-tune over all `322` Jake wins:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/score_good_play_replays.py \
  --replay-dir /tmp/orbit_jake_parquet_wins_200 \
  --winner-name "Jake Will" \
  --strong-name "Jake Will" \
  --require-known-winner \
  --min-score 0 \
  --max-lost-cap 0.80 \
  --steps-min 16 \
  --steps-max 160 \
  --contest-window 40 \
  --noop-keep-prob 0.02 \
  --fire-repeat 1 \
  --reinforce-repeat 2 \
  --scores-out /tmp/supervised_bc/jake_parquet_win322_contest16_160_w40_scores.json \
  --samples-out /tmp/supervised_bc/jake_parquet_win322_contest16_160_w40.pkl

PYTHONUNBUFFERED=1 PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/top2_scale1_frame_shards/manifest.json \
  --samples /tmp/supervised_bc/jake_parquet_win322_frame_shards/manifest.json \
  --samples /tmp/supervised_bc/jake_parquet_win322_contest16_160_w40.pkl \
  --stream-shards \
  --init-checkpoint /tmp/supervised_bc/jake_live_parquet_win322_frame_ft800.pt \
  --steps 600 \
  --lr 5e-5 \
  --eval-every 100 \
  --max-val-samples 4096 \
  --allow-reinforce \
  --select-metric val_target_top3 \
  --trainable-param fire_head \
  --trainable-param ship_head \
  --trainable-param tgt_q \
  --trainable-param tgt_k \
  --trainable-param target_scorer \
  --save /tmp/supervised_bc/jake_live_parquet_win322_contest_w40_heads_600.pt
```

Result:
The expanded contest slice accepted `255` Jake seats and wrote `25,523`
samples, including `23,796` enemy-inbound contest frames. Heads-only fine-tune
reached `val_target_top3=0.5833`, below the `200`-win contest checkpoint's
`0.586`. Gameplay: random `8/8`, Zach `5/8`, Ajay `0/8` at threshold `0.4`;
Zach threshold sweep was `3/8` at `0.35`, `5/8` at `0.45`, and `3/8` at `0.5`.
Readout: adding the older same-day Jake wins improves offline fit but does not
beat the smaller/cleaner contest checkpoint in gameplay.

Fraction ship-head experiment, matching vkhydras's per-planet
`launch (P), target (P,P), frac (P,K)` comment more closely. The model was
already per-owned-planet for launch/target/ship; this adds `--ship-bin-mode
fraction`, relabels compact frame samples as 10%..100% of source ships, and
saves checkpoint metadata so eval decodes fraction bins.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/top2_scale1_frame_shards/manifest.json \
  --samples /tmp/supervised_bc/jake_parquet_win322_frame_shards/manifest.json \
  --stream-shards \
  --ship-bin-mode fraction \
  --steps 1000 \
  --lr 1e-4 \
  --eval-every 100 \
  --max-val-samples 4096 \
  --allow-reinforce \
  --select-metric val_target_top3 \
  --save /tmp/supervised_bc/jake_live_parquet_win322_fraction_scratch_1000.pt
```

Result:
Fraction scratch learned ship sizing easily (`val_ship_red=0.870`) but target fit
lagged absolute buckets (`val_target_top3=0.527` vs `0.569` for strict
Jake-322 absolute). Gameplay was playable but not better: random `8/8`, Zach
`4/8`, Ajay `0/8` at threshold `0.4`.

Compatible-init fraction ship-head transfer. This keeps the best absolute
contest checkpoint's trunk/fire/target weights, skips only the shape-mismatched
`ship_head`, and trains the new 10-bin fraction ship head:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/top2_scale1_frame_shards/manifest.json \
  --samples /tmp/supervised_bc/jake_parquet_win322_frame_shards/manifest.json \
  --stream-shards \
  --ship-bin-mode fraction \
  --partial-init-compatible \
  --init-checkpoint /tmp/supervised_bc/jake_live_parquet_contest_w40_heads_600.pt \
  --steps 500 \
  --lr 1e-4 \
  --eval-every 100 \
  --max-val-samples 4096 \
  --allow-reinforce \
  --select-metric val_ship_red \
  --trainable-param ship_head \
  --save /tmp/supervised_bc/jake_contest_abs_target_fraction_ship_head_500.pt
```

Result:
Partial init loaded `68` tensors and skipped only `ship_head.weight/bias`.
Final validation preserved target fit (`val_target_top3=0.565`) and trained the
fraction ship head to `val_ship_red=0.412`, but gameplay regressed: random
`8/8`, Zach `3/8`, Ajay `0/8` at threshold `0.4`. Readout: the output shape is
now aligned with the top-player comment, but fraction ship buckets alone do not
fix Ajay retention or beat the absolute-bucket contest checkpoint.

Held-capture outcome-aware replay weighting. This stays replay-only: identify
recently captured owned planets that are not lost within a future horizon, then
repeat or filter frames where Jake's real replay action sources from or targets
those held captures.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/score_good_play_replays.py \
  --replay-dir /tmp/orbit_jake_parquet_wins_200 \
  --winner-name "Jake Will" \
  --strong-name "Jake Will" \
  --require-known-winner \
  --min-score 0 \
  --max-lost-cap 0.80 \
  --steps-min 16 \
  --steps-max 160 \
  --contest-window 40 \
  --noop-keep-prob 0.02 \
  --fire-repeat 1 \
  --reinforce-repeat 2 \
  --held-capture-window 40 \
  --hold-success-horizon 30 \
  --held-capture-repeat 6 \
  --scores-out /tmp/supervised_bc/jake_parquet_win322_heldcap_w40_h30_r6_scores.json \
  --samples-out /tmp/supervised_bc/jake_parquet_win322_heldcap_w40_h30_r6.pkl

PYTHONUNBUFFERED=1 PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/top2_scale1_frame_shards/manifest.json \
  --samples /tmp/supervised_bc/jake_parquet_win322_frame_shards/manifest.json \
  --samples /tmp/supervised_bc/jake_parquet_win322_heldcap_w40_h30_r6.pkl \
  --stream-shards \
  --init-checkpoint /tmp/supervised_bc/jake_live_parquet_contest_w40_heads_600.pt \
  --steps 500 \
  --lr 3e-5 \
  --eval-every 100 \
  --max-val-samples 4096 \
  --allow-reinforce \
  --select-metric val_target_top3 \
  --trainable-param fire_head \
  --trainable-param ship_head \
  --trainable-param tgt_q \
  --trainable-param tgt_k \
  --trainable-param target_scorer \
  --save /tmp/supervised_bc/jake_contest_heldcap_w40_h30_r6_heads_500.pt
```

Result:
The broad repeat slice wrote `48,458` samples from `255` Jake wins, with
`5,100` held-capture-weighted frames and only `100` future-loss rejections.
Offline validation regressed versus the best contest checkpoint
(`val_target_top3=0.574` vs `0.586`). Gameplay also regressed: Zach `3/8`,
Ajay `0/8`, Ajay `lost-cap=1.00`, `reinf_share=0.41`.

Hard held-capture filter. This keeps only the moves involving held captures and
trains only ship/target heads, leaving the best checkpoint's fire head intact:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/score_good_play_replays.py \
  --replay-dir /tmp/orbit_jake_parquet_wins_200 \
  --winner-name "Jake Will" \
  --strong-name "Jake Will" \
  --require-known-winner \
  --min-score 0 \
  --max-lost-cap 0.80 \
  --steps-min 16 \
  --steps-max 160 \
  --contest-window 40 \
  --noop-keep-prob 0.0 \
  --fire-repeat 2 \
  --reinforce-repeat 2 \
  --held-capture-window 40 \
  --hold-success-horizon 30 \
  --held-capture-only \
  --scores-out /tmp/supervised_bc/jake_parquet_win322_heldcap_only_w40_h30_scores.json \
  --samples-out /tmp/supervised_bc/jake_parquet_win322_heldcap_only_w40_h30.pkl

PYTHONUNBUFFERED=1 PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/top2_scale1_frame_shards/manifest.json \
  --samples /tmp/supervised_bc/jake_parquet_win322_frame_shards/manifest.json \
  --samples /tmp/supervised_bc/jake_parquet_win322_heldcap_only_w40_h30.pkl \
  --stream-shards \
  --init-checkpoint /tmp/supervised_bc/jake_live_parquet_contest_w40_heads_600.pt \
  --steps 400 \
  --lr 3e-5 \
  --eval-every 100 \
  --max-val-samples 4096 \
  --allow-reinforce \
  --select-metric val_target_top3 \
  --trainable-param ship_head \
  --trainable-param tgt_q \
  --trainable-param tgt_k \
  --trainable-param target_scorer \
  --save /tmp/supervised_bc/jake_contest_heldcap_only_w40_h30_shiptarget_400.pt
```

Result:
The hard slice wrote `10,200` samples from `5,100` held-capture frames, dropping
`9,316` recent-capture frames whose action did not involve the held planet. It
still did not beat the best checkpoint: `val_target_top3=0.574`, Zach `4/8`,
Ajay `0/8`, Ajay `lost-cap=1.00`, `reinf_share=0.42`. Adding
`--reinforce-target-bias -1.0` reduced Ajay reinforcement to `0.20` but stayed
Ajay `0/8` and hurt Zach to `3/8`. Readout: future-held capture filtering is a
better trust proxy than raw contest weighting, but cross-entropy on those frames
still shifts the own-target prior without teaching enough timing/source sizing
to survive Ajay.

Replay-action candidate reranker. This changes the objective: positives are
Jake's decoded replay launches, negatives are plausible same-source candidate
alternatives from the same target-owner mode. It saves the same lightweight
reranker checkpoint format consumed by `eval.py --producer-reranker-checkpoint`.
Runtime candidate enumeration is still Producer-style, so this is an overlay
diagnostic/bridge rather than a pure submitted policy.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/build_replay_reranker.py \
  --replay-dir /tmp/orbit_jake_parquet_wins_200 \
  --max-replays 80 \
  --seat-mode winner \
  --winner-name-filter "Jake Will" \
  --steps-min 16 \
  --steps-max 120 \
  --target-owner not-own \
  --negatives-per-positive 8 \
  --score-slack 10 \
  --max-records 2000 \
  --train-steps 2000 \
  --records-out /tmp/supervised_bc/replay_reranker_jake_notown_s16_120_2k_v2_records.pkl \
  --reranker-out /tmp/supervised_bc/replay_reranker_jake_notown_s16_120_2k_v2.pt \
  --summary-out /tmp/supervised_bc/replay_reranker_jake_notown_s16_120_2k_v2_summary.json

CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/jake_live_parquet_contest_w40_heads_600.pt \
  --opponent opponents/candidate_zach_public.py \
  --games 8 \
  --target-decode \
  --fire-threshold 0.4 \
  --producer-overlay \
  --producer-overlay-max-moves 1 \
  --producer-overlay-score-min 1.5 \
  --producer-overlay-target-owner not-own \
  --producer-reranker-checkpoint /tmp/supervised_bc/replay_reranker_jake_notown_s16_120_2k_v2.pt
```

Result:
The corrected not-own replay ranker built `1,952` records over `663`
replay-positive groups from `80` Jake wins. Offline validation: AUC `0.703`,
group top1 `0.629`, top3 `0.985`. One-move overlay on the best contest
checkpoint scored Zach `8/8` with `cap/atk-launch 0.572`, `lost-cap 0.37`, and
planets@100 `16`. Ajay remained `0/8`; the overlay improved capture volume
(`caps/game 19.9`) but every capture was still stripped (`lost-cap 1.00`,
median hold `9st`). Two not-own overlay moves preserved Zach `8/8` but still
left Ajay `0/8`.

Own-target recent-capture reranker, corrected to use own-target negatives too:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/build_replay_reranker.py \
  --replay-dir /tmp/orbit_jake_parquet_wins_200 \
  --max-replays 160 \
  --seat-mode winner \
  --winner-name-filter "Jake Will" \
  --steps-min 16 \
  --steps-max 160 \
  --target-owner own \
  --candidate-recent-capture-window 40 \
  --negatives-per-positive 8 \
  --score-slack 10 \
  --max-records 2000 \
  --train-steps 2000 \
  --records-out /tmp/supervised_bc/replay_reranker_jake_own_recent40_s16_160_2k_v2_records.pkl \
  --reranker-out /tmp/supervised_bc/replay_reranker_jake_own_recent40_s16_160_2k_v2.pt \
  --summary-out /tmp/supervised_bc/replay_reranker_jake_own_recent40_s16_160_2k_v2_summary.json
```

Result:
The corrected own-target recent-capture slice was sparse: `135` records over
`67` groups from `160` Jake wins. As an own-target overlay it scored Zach `6/8`
and Ajay `0/8`; Ajay reinforcement rose to `0.44` but lost-cap stayed `0.99`.
Readout: replay-derived candidate ranking is promising for expansion/pressure,
but the retention side needs a stronger intervention/hold objective, not just
own-target replay imitation.

Readout: strict Jake wins plus replay-only contest weighting is the best new
replay-cloning path (`10/16` Zach, Ajay `0/8`). `bc.py --stream-shards` now
samples validation records across shards and skips those records during
training; older mixed-manifest validation metrics used a file-level split and
can be noisy.

Contest-checkpoint intervention selector diagnostic. This starts from the best
pure replay checkpoint instead of the older synthetic-defense checkpoint. It is
useful as a negative control: the counterfactual label exists offline, but the
live overlay over-defends and regresses Zach.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/collect_defense_interventions.py \
  --checkpoint /tmp/supervised_bc/jake_live_parquet_contest_w40_heads_600.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 24 \
  --seed-start 0 \
  --horizon 20 \
  --recent-capture-window 40 \
  --garrison-floor 10 \
  --min-need 5 \
  --support-max-moves 2 \
  --multi-source-per-target \
  --max-records 40 \
  --reinforce-target-bias 0.0 \
  --flush-every 10 \
  --records-out /tmp/supervised_bc/intervention_ajay_contestbc_s0_24_h20_40_multisource.pkl \
  --summary-out /tmp/supervised_bc/intervention_ajay_contestbc_s0_24_h20_40_multisource_summary.json

PYTHONUNBUFFERED=1 PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/train_intervention_selector.py \
  --records /tmp/supervised_bc/intervention_ajay_contestbc_s0_24_h20_40_multisource.pkl \
  --label hold_advantage \
  --steps 2000 \
  --lr 0.03 \
  --selector-out /tmp/supervised_bc/intervention_selector_ajay_contestbc_s0_24_h20_40_multisource_holdadv.pt \
  --summary-out /tmp/supervised_bc/intervention_selector_ajay_contestbc_s0_24_h20_40_multisource_holdadv_summary.json
```

Observed result: collection wrote `40` records with `5/40` final helped,
`1/40` hurt, and `9/40` hold-advantage. The selector showed noisy offline
signal (`val_auc 0.917` on only eight validation records), but inference
regressed Zach to `3/8` at thresholds `0.5` and `0.7`, and Ajay stayed `0/8`
with `lost-cap 0.97`. Do not promote this overlay; use it only as evidence that
larger counterfactual data needs tighter action/risk labels before live gating.

Hold-advantage intervention labels. The collector also records owner traces and
`hold_delta`; train with `--label hold_advantage` to predict whether support
buys more owned ticks during the horizon. This improves label density
(`20/120` positives vs `3/120` final-horizon helped on the first traced slice),
but the first selector regressed Zach to `5/16` at thresholds `0.5` and `0.7`,
so treat it as data evidence, not the current best inference gate.

```bash
PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/collect_defense_interventions.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 48 \
  --seed-start 200 \
  --horizon 30 \
  --recent-capture-window 40 \
  --garrison-floor 10 \
  --min-need 5 \
  --max-records 120 \
  --reinforce-target-bias -1.0 \
  --records-out /tmp/supervised_bc/intervention_ajay_s200_48_h30_120_trace.pkl \
  --summary-out /tmp/supervised_bc/intervention_ajay_s200_48_h30_120_trace_summary.json

PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/train_intervention_selector.py \
  --records /tmp/supervised_bc/intervention_ajay_s200_48_h30_120_trace.pkl \
  --label hold_advantage \
  --steps 2000 \
  --lr 0.03 \
  --selector-out /tmp/supervised_bc/intervention_selector_ajay_s200_48_h30_120_holdadv.pt \
  --summary-out /tmp/supervised_bc/intervention_selector_ajay_s200_48_h30_120_holdadv_summary.json

CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt \
  --opponent opponents/candidate_zach_public.py \
  --games 16 \
  --target-decode \
  --reinforce-target-bias -1.0 \
  --defense-overlay \
  --defense-overlay-recent-capture-window 40 \
  --defense-overlay-garrison-floor 10 \
  --defense-overlay-min-need 5 \
  --defense-overlay-max-moves 1 \
  --defense-overlay-selector-checkpoint /tmp/supervised_bc/intervention_selector_ajay_s200_48_h30_120_holdadv.pt \
  --defense-overlay-selector-threshold 0.7 \
  --defense-overlay-selector-mode survive
```

ETA-aware hold-advantage selector. Current best diagnostic version adds
`support_eta`, `eta_margin`, and `support_arrives_before` to the selector
features. Offline AUC improved to `0.926`; quick Zach recovered to `7/16` at
threshold `0.7`, but Ajay remained `0/16`, so this is not a promotion candidate.

```bash
PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/collect_defense_interventions.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 48 \
  --seed-start 300 \
  --horizon 30 \
  --recent-capture-window 40 \
  --garrison-floor 10 \
  --min-need 5 \
  --max-records 120 \
  --reinforce-target-bias -1.0 \
  --records-out /tmp/supervised_bc/intervention_ajay_s300_48_h30_120_trace_eta.pkl \
  --summary-out /tmp/supervised_bc/intervention_ajay_s300_48_h30_120_trace_eta_summary.json

PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/train_intervention_selector.py \
  --records /tmp/supervised_bc/intervention_ajay_s300_48_h30_120_trace_eta.pkl \
  --label hold_advantage \
  --steps 3000 \
  --lr 0.03 \
  --selector-out /tmp/supervised_bc/intervention_selector_ajay_s300_48_h30_120_eta_holdadv.pt \
  --summary-out /tmp/supervised_bc/intervention_selector_ajay_s300_48_h30_120_eta_holdadv_summary.json

CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt \
  --opponent opponents/candidate_zach_public.py \
  --games 16 \
  --target-decode \
  --reinforce-target-bias -1.0 \
  --defense-overlay \
  --defense-overlay-recent-capture-window 40 \
  --defense-overlay-garrison-floor 10 \
  --defense-overlay-min-need 5 \
  --defense-overlay-max-moves 1 \
  --defense-overlay-selector-checkpoint /tmp/supervised_bc/intervention_selector_ajay_s300_48_h30_120_eta_holdadv.pt \
  --defense-overlay-selector-threshold 0.7 \
  --defense-overlay-selector-mode survive
```

Multi-source intervention selector. This allows up to two rear sources to defend
the same recently captured target. It is the strongest supervised intervention
result so far against Zach: `9/16`, `lost-cap 0.57`, end planets `12.6`. Ajay
still stays `0/16`, so treat it as a real supervised improvement but not a
leaderboard candidate.

```bash
PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/collect_defense_interventions.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 32 \
  --seed-start 400 \
  --horizon 30 \
  --recent-capture-window 40 \
  --garrison-floor 10 \
  --min-need 5 \
  --support-max-moves 2 \
  --multi-source-per-target \
  --max-records 80 \
  --reinforce-target-bias -1.0 \
  --records-out /tmp/supervised_bc/intervention_ajay_s400_32_h30_80_multisource.pkl \
  --summary-out /tmp/supervised_bc/intervention_ajay_s400_32_h30_80_multisource_summary.json

PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/train_intervention_selector.py \
  --records /tmp/supervised_bc/intervention_ajay_s400_32_h30_80_multisource.pkl \
  --label hold_advantage \
  --steps 3000 \
  --lr 0.03 \
  --selector-out /tmp/supervised_bc/intervention_selector_ajay_s400_32_h30_80_multisource_holdadv.pt \
  --summary-out /tmp/supervised_bc/intervention_selector_ajay_s400_32_h30_80_multisource_holdadv_summary.json

CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt \
  --opponent opponents/candidate_zach_public.py \
  --games 16 \
  --target-decode \
  --reinforce-target-bias -1.0 \
  --defense-overlay \
  --defense-overlay-recent-capture-window 40 \
  --defense-overlay-garrison-floor 10 \
  --defense-overlay-min-need 5 \
  --defense-overlay-max-moves 2 \
  --defense-overlay-multi-source-per-target \
  --defense-overlay-selector-checkpoint /tmp/supervised_bc/intervention_selector_ajay_s400_32_h30_80_multisource_holdadv.pt \
  --defense-overlay-selector-threshold 0.5 \
  --defense-overlay-selector-mode survive
```

Producer planner-candidate BC. This builds labels from Producer/Ajay planner
candidate rankings instead of cloning emitted replay/teacher actions. First
300-sample top-1 smoke decoded cleanly but only scored Zach `6/16` after a
small heads-only fine-tune, so scale/curate before treating it as a candidate.

```bash
PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/build_producer_planner_bc.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --max-replays 10 \
  --seat-mode all \
  --steps-min 1 \
  --steps-max 80 \
  --top-k 1 \
  --score-min 1.5 \
  --target-owner any \
  --max-samples 300 \
  --samples-out /tmp/supervised_bc/producer_planner_top1_s1_80_300.pkl \
  --summary-out /tmp/supervised_bc/producer_planner_top1_s1_80_300_summary.json

PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python -u \
  orbit_wars_rl/bc.py \
  --samples /tmp/supervised_bc/good_play_known_opening50.pkl \
  --samples /tmp/supervised_bc/producer_planner_top1_s1_80_300.pkl \
  --init-checkpoint /tmp/supervised_bc/supervised_opening50_ajay_teacher_all30_s120_max2_source1_notown_targettop3_600_reinf.pt \
  --trainable-param ship_head \
  --trainable-param tgt_q \
  --trainable-param tgt_k \
  --trainable-param target_scorer \
  --steps 250 \
  --lr 5e-5 \
  --fire-pos-weight 1.0 \
  --allow-reinforce \
  --select-metric val_target_top3 \
  --save /tmp/supervised_bc/supervised_opening50_targettop3_producer_planner_top1_300_shiptarget_250_reinf.pt

CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_producer_planner_top1_300_shiptarget_250_reinf.pt \
  --opponent opponents/candidate_zach_public.py \
  --games 16 \
  --target-decode \
  --reinforce-target-bias -1.0
```

Defensive Producer planner labels. These isolate own-target planner candidates,
optionally requiring the target to be a recent capture and inbound-threatened.
The broad own-target slice passed offline but regressed Zach (`1/8`), while the
strict recent+threat slice failed the generic target gate. Use these commands as
builder recipes; the next step should be a ranking/postprocessor objective, not
more CE on the main target head.

```bash
PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/build_producer_planner_bc.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --max-replays 40 \
  --seat-mode all \
  --steps-min 16 \
  --steps-max 160 \
  --top-k 4 \
  --score-min 1.5 \
  --target-owner own \
  --reinforce-repeat 2 \
  --max-samples 800 \
  --samples-out /tmp/supervised_bc/producer_planner_own_top4_s16_160_800.pkl \
  --summary-out /tmp/supervised_bc/producer_planner_own_top4_s16_160_800_summary.json

PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/build_producer_planner_bc.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --max-replays 80 \
  --seat-mode all \
  --steps-min 16 \
  --steps-max 180 \
  --top-k 4 \
  --score-min 1.5 \
  --target-owner own \
  --recent-capture-window 40 \
  --inbound-threat-horizon 30 \
  --reinforce-repeat 2 \
  --max-samples 500 \
  --samples-out /tmp/supervised_bc/producer_planner_own_recent40_threat30_top4_500.pkl \
  --summary-out /tmp/supervised_bc/producer_planner_own_recent40_threat30_top4_500_summary.json
```

Producer candidate reranker. This is the current planner-supervised path: train
a separate lightweight scorer over Producer candidate features, then use it only
to order `--producer-overlay` candidates. It avoids writing Producer labels into
the main BC heads. The four-replay smoke checkpoint was the first supervised
variant to win an Ajay gate game (`1/8`); larger all-seat and Isaiah-winner
rerankers kept Zach `8/8` but were Ajay `0/8`.

```bash
PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/build_producer_reranker.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --max-replays 4 \
  --seat-mode all \
  --steps-min 1 \
  --steps-max 60 \
  --max-candidates-per-state 12 \
  --positive-top-k 1 \
  --score-min 1.5 \
  --max-records 600 \
  --train-steps 800 \
  --records-out /tmp/supervised_bc/producer_reranker_smoke_records.pkl \
  --reranker-out /tmp/supervised_bc/producer_reranker_smoke.pt \
  --summary-out /tmp/supervised_bc/producer_reranker_smoke_summary.json

CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 8 \
  --target-decode \
  --reinforce-target-bias -1.0 \
  --producer-overlay \
  --producer-overlay-max-moves 1 \
  --producer-overlay-score-min 1.5 \
  --producer-overlay-target-owner any \
  --producer-reranker-checkpoint /tmp/supervised_bc/producer_reranker_smoke.pt
```

Trace and scheduled-filter diagnostic. The smoke reranker win on Ajay seed `2`
depends on permissive neutral expansion early and high-confidence pressure later.
Use `--producer-overlay-trace-json` to inspect selected candidates. The scheduled
filter below preserves the seed-2 win and keeps Zach `8/8`, but the 8-game Ajay
gate is still only `1/8`, so this is a next anchor, not a competent agent.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 1 \
  --seed-start 2 \
  --target-decode \
  --reinforce-target-bias -1.0 \
  --producer-overlay \
  --producer-overlay-max-moves 2 \
  --producer-overlay-score-min 1.5 \
  --producer-overlay-target-owner any \
  --producer-overlay-late-step 44 \
  --producer-overlay-late-score-min 20 \
  --producer-reranker-checkpoint /tmp/supervised_bc/producer_reranker_smoke.pt \
  --producer-overlay-trace-json /tmp/supervised_bc/trace_smoke_sched44_ajay_seed2.json \
  --producer-overlay-trace-top-k 6

CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 8 \
  --target-decode \
  --reinforce-target-bias -1.0 \
  --producer-overlay \
  --producer-overlay-max-moves 2 \
  --producer-overlay-score-min 1.5 \
  --producer-overlay-target-owner any \
  --producer-overlay-late-step 44 \
  --producer-overlay-late-score-min 20 \
  --producer-reranker-checkpoint /tmp/supervised_bc/producer_reranker_smoke.pt
```

Scaled reranker ablations:

```bash
PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/build_producer_reranker.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --max-replays 30 \
  --seat-mode all \
  --steps-min 1 \
  --steps-max 160 \
  --max-candidates-per-state 16 \
  --positive-top-k 1 \
  --score-min 1.5 \
  --max-records 8000 \
  --train-steps 3000 \
  --records-out /tmp/supervised_bc/producer_reranker_s30_8k_records.pkl \
  --reranker-out /tmp/supervised_bc/producer_reranker_s30_8k.pt \
  --summary-out /tmp/supervised_bc/producer_reranker_s30_8k_summary.json

PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/build_producer_reranker.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --max-replays 80 \
  --seat-mode winner \
  --winner-name-filter Isaiah \
  --steps-min 1 \
  --steps-max 160 \
  --max-candidates-per-state 16 \
  --positive-top-k 1 \
  --score-min 1.5 \
  --max-records 8000 \
  --train-steps 3000 \
  --records-out /tmp/supervised_bc/producer_reranker_isaiah_winner_s80_8k_records.pkl \
  --reranker-out /tmp/supervised_bc/producer_reranker_isaiah_winner_s80_8k.pt \
  --summary-out /tmp/supervised_bc/producer_reranker_isaiah_winner_s80_8k_summary.json
```

Retention-quality reranker slice. This uses the same good-play replay metrics as
the replay BC filter, then emits Producer-candidate ranking records only from
recent-capture states in low-lost-cap winning seats. Result: strong offline
ranking and Zach `7/8`, but Ajay stayed `0/8`, so this is a diagnostic data path
rather than a promotion candidate.

```bash
PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/build_producer_reranker.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --max-replays 120 \
  --seat-mode winner \
  --min-replay-score 7.0 \
  --max-lost-cap 0.45 \
  --min-median-hold 20 \
  --min-cap-attack 0.30 \
  --min-planets50 5 \
  --steps-min 16 \
  --steps-max 180 \
  --state-recent-capture-window 40 \
  --max-candidates-per-state 12 \
  --positive-top-k 1 \
  --score-min 1.5 \
  --max-records 3000 \
  --train-steps 1500 \
  --records-out /tmp/supervised_bc/producer_reranker_retention_winner_s120_3k_records.pkl \
  --reranker-out /tmp/supervised_bc/producer_reranker_retention_winner_s120_3k.pt \
  --summary-out /tmp/supervised_bc/producer_reranker_retention_winner_s120_3k_summary.json

CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/eval.py \
  --checkpoint /tmp/supervised_bc/supervised_opening50_targettop3_synthdef500_shiptarget_500_reinf.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 8 \
  --target-decode \
  --reinforce-target-bias -1.0 \
  --producer-overlay \
  --producer-overlay-max-moves 2 \
  --producer-overlay-score-min 1.5 \
  --producer-overlay-target-owner any \
  --producer-reranker-checkpoint /tmp/supervised_bc/producer_reranker_retention_winner_s120_3k.pt
```

Score-feature ablation. `producer_score_tanh` is now appended to new reranker
feature vectors, while eval truncates/pads features so older reranker
checkpoints remain loadable. This ablation did not help Ajay, but keep the
feature for future candidate-ranking experiments.

```bash
PYTHONPATH=.:orbit_wars_rl /Users/saheb/home/.venv/bin/python \
  orbit_wars_rl/build_producer_reranker.py \
  --replay-dir /tmp/fresh_validate \
  --replay-dir /tmp/snowball \
  --max-replays 120 \
  --seat-mode winner \
  --min-replay-score 7.0 \
  --max-lost-cap 0.45 \
  --min-median-hold 20 \
  --min-cap-attack 0.30 \
  --min-planets50 5 \
  --steps-min 16 \
  --steps-max 180 \
  --state-recent-capture-window 40 \
  --max-candidates-per-state 12 \
  --positive-top-k 1 \
  --score-min 1.5 \
  --max-records 3000 \
  --train-steps 1500 \
  --records-out /tmp/supervised_bc/producer_reranker_retention_scorefeat_s120_3k_records.pkl \
  --reranker-out /tmp/supervised_bc/producer_reranker_retention_scorefeat_s120_3k.pt \
  --summary-out /tmp/supervised_bc/producer_reranker_retention_scorefeat_s120_3k_summary.json
```

### Full panel eval (legacy — 256 games, ~40 min/opponent on EC2; much slower on Mac CPU)

**Activate venv first** (see top section), then run from repo root:

```bash
# Run all three opponents — use & to run in parallel, save logs
source /Users/saheb/home/.venv/bin/activate

python orbit_wars_rl/eval.py \
  --checkpoint gpu_run_artifacts/hellburner_spot/checkpoints/<name>.pt \
  --opponent opponents/candidate_hellburner.py \
  --panel --target-decode \
  2>&1 | tee gpu_run_artifacts/hellburner_spot/panels/eval_hellburner.log &

python orbit_wars_rl/eval.py \
  --checkpoint gpu_run_artifacts/hellburner_spot/checkpoints/<name>.pt \
  --opponent opponents/candidate_zach_public.py \
  --panel --target-decode \
  2>&1 | tee gpu_run_artifacts/hellburner_spot/panels/eval_zach.log &

python orbit_wars_rl/eval.py \
  --checkpoint gpu_run_artifacts/hellburner_spot/checkpoints/<name>.pt \
  --opponent opponents/candidate_suneet_lb1200.py \
  --panel --target-decode \
  2>&1 | tee gpu_run_artifacts/hellburner_spot/panels/eval_suneet.log &

wait  # block until all three finish
```

**Check progress while running:**
```bash
tail -3 gpu_run_artifacts/hellburner_spot/panels/eval_hellburner.log
tail -3 gpu_run_artifacts/hellburner_spot/panels/eval_zach.log
tail -3 gpu_run_artifacts/hellburner_spot/panels/eval_suneet.log
```

**Read results when done** (look for `Overall:` line):
```bash
grep "Overall" gpu_run_artifacts/hellburner_spot/panels/eval_hellburner.log
grep "Overall" gpu_run_artifacts/hellburner_spot/panels/eval_zach.log
grep "Overall" gpu_run_artifacts/hellburner_spot/panels/eval_suneet.log
```

> ⚠️ Opponent paths are relative to **repo root**, not `orbit_wars_rl/`. Path is `opponents/candidate_*.py`.
> ⚠️ Always `--target-decode` for Phase 1 checkpoints (absolute 32-bin mode).
> ⚠️ Do NOT use `nohup python ...` — `python` alone fails. Either activate venv first or use full path.

---

## 4. EC2 instance management

```bash
# List running/stopped instances
aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=orbit-wars" "Name=instance-state-name,Values=running,stopped" \
  --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceType,PublicIpAddress]' \
  --output table

# Terminate (only after rsync — see section 5)
aws ec2 terminate-instances --instance-ids <id>

# Check state after terminate
aws ec2 describe-instances --instance-ids <id> \
  --query 'Reservations[0].Instances[0].State.Name' --output text
```

> ⚠️ Never terminate without pulling checkpoints first (section 5).
> ⚠️ Never leave in `stopped` state — EBS still bills. Always terminate.

---

## 5. Final rsync before terminating an instance

```bash
# SSH in first and confirm checkpoints exist
ssh -i ~/.ssh/samosa-key.pem ubuntu@<PUB_IP> "ls ~/orbit_wars_rl/checkpoints/*.pt | wc -l"

# Pull all checkpoints
rsync -az -e "ssh -i ~/.ssh/samosa-key.pem -o StrictHostKeyChecking=no" \
  ubuntu@<PUB_IP>:~/orbit_wars_rl/checkpoints/ \
  gpu_run_artifacts/hellburner_spot/checkpoints/

# Pull training log (using include/exclude to avoid pulling entire directory)
rsync -az -e "ssh -i ~/.ssh/samosa-key.pem -o StrictHostKeyChecking=no" \
  --include='train_gpu_*.log' --exclude='*' \
  ubuntu@<PUB_IP>:~/orbit_wars_rl/ \
  gpu_run_artifacts/hellburner_spot/logs/

# Confirm checkpoint count locally
ls gpu_run_artifacts/hellburner_spot/checkpoints/*.pt | wc -l

# Now safe to terminate
aws ec2 terminate-instances --instance-ids <id>
```

---

## 6. Launch a new training run

```bash
# Always use the launch script — never raw aws ec2 run-instances
bash gpu_run_artifacts/hellburner_spot/launch_phase1_rev7.sh

# The launch script handles: EC2 launch, rsync, W&B auth, seed checkpoint upload, tmux start
```

**After launch — start local watcher** (syncs checkpoints every 3 min):
```bash
nohup bash gpu_run_artifacts/hellburner_spot/watch_phase1.sh \
  > gpu_run_artifacts/hellburner_spot/logs/watcher_rev7.log 2>&1 &
echo "Watcher PID: $!"
```

> ⚠️ Edit `watch_phase1.sh` first — hardcode `INSTANCE_ID` and `PUB_IP` from the launch output.
> ⚠️ Use `nohup bash ...` not `nohup python ...`. The watcher is a shell script.

---

## 7. Kill/check local background processes

```bash
# Check what eval jobs are running (shows checkpoint name and opponent)
ps aux | grep eval.py | grep -v grep | awk '{print $2, $11, $14, $16}'

# Check what watchers are running
ps aux | grep "run_panel_eval_watcher\|watch_phase1" | grep -v grep | awk '{print $2, $11}'

# Kill all panel eval watchers (use when changing opponent paths or run version)
ps aux | grep "run_panel_eval_watcher" | grep -v grep | awk '{print $2}' | xargs kill

# Kill evals using wrong opponent paths (e.g., old candidate_*.py not opponents/)
ps aux | grep "eval.py" | grep -v grep | grep -v "opponents/" | awk '{print $2}' | xargs kill
```

### Long local jobs: use `tmux`, not plain `nohup ... &`

For long-running local Python jobs such as BC training, prefer a dedicated
`tmux` session. On this machine, detached shell jobs can disappear without
leaving a useful log, while `tmux` keeps a real terminal attached to the
process.

```bash
# Start a persistent tmux session for BC training
tmux new-session -d -s bc_isaiah_hober_5k \
  'cd /Users/saheb/home/kaggle-orbit-wars && \
   exec stdbuf -oL -eL orbit_wars_rl/.venv/bin/python -u orbit_wars_rl/bc.py \
     --samples /tmp/targeted_bc/isaiah_hober_pressure_merged.pkl \
     --steps 5000 \
     --save orbit_wars_rl/checkpoints/bc_isaiah_hober_pressure_5k.pt \
     2>&1 | tee /tmp/targeted_bc/bc_isaiah_hober_pressure_5k.log'

# Confirm session exists
tmux list-sessions | grep bc_isaiah_hober_5k

# Attach to the live job
tmux attach -t bc_isaiah_hober_5k

# Read the live log without attaching
tail -f /tmp/targeted_bc/bc_isaiah_hober_pressure_5k.log

# Stop the job from another shell
tmux send-keys -t bc_isaiah_hober_5k C-c

# Remove the finished session
tmux kill-session -t bc_isaiah_hober_5k
```

> ⚠️ For local BC/training, treat `tmux` as the default. Use plain background
> shell jobs only for small helpers/watchers.

---

## 8. Diagnose a silent eval failure

If eval logs show only startup noise (INFO: OpenSpiel...) and no `panel progress:` lines:

```bash
# Check the .failed files the panel watcher creates
ls gpu_run_artifacts/hellburner_spot/panels/*.failed | head -5
cat gpu_run_artifacts/hellburner_spot/panels/<name>.vs_hellburner.panel.txt.failed | grep -E "Error|Traceback|FileNotFoundError"
```

Common causes:
| Symptom | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: candidate_hellburner.py` | Old path, before `opponents/` move | Change to `opponents/candidate_hellburner.py` |
| `nohup: python: No such file or directory` | `python` not found, wrong venv | Activate venv: `source /Users/saheb/home/.venv/bin/activate` |
| eval hangs with 0 progress for >10 min | other CPU-heavy processes competing | Check `ps aux \| grep eval.py`; kill duplicates |
| W&B auth failure on remote | key not passed to tmux session | Re-run `wandb login --relogin <key>` on the remote |

---

## 9. Review a submission's loss replays

### Kaggle CLI 2.1.2+ — episodes / replays / logs (native subcommands)

The simplest way to pull a submission's games — no API tokens or custom scripts needed
(`kaggle --version` ≥ 2.0.2). The agent index matches `info.TeamNames` order in the replay.

```bash
# List episodes for a submission (find SUBMISSION_ID via `kaggle competitions submissions orbit-wars`)
kaggle competitions episodes <SUBMISSION_ID>            # table
kaggle competitions episodes <SUBMISSION_ID> -v         # CSV (col 1 = episode id) — for scripting

# Download a replay JSON (for analysis or the viewer)
kaggle competitions replay <EPISODE_ID>                 # -> episode-<id>-replay.json in cwd
kaggle competitions replay <EPISODE_ID> -p ./replays    # into a dir

# Download YOUR agent's logs (index 0..3; you can only fetch your own seat)
kaggle competitions logs <EPISODE_ID> 0 -p ./logs
```

**Quick 2p-vs-4p win-rate from a submission's replays** (the FFA-baseline check, 2026-06-08):
```bash
mkdir -p /tmp/sub_replays
kaggle competitions episodes <SUBMISSION_ID> -v 2>/dev/null | tail -n +2 | cut -d',' -f1 \
  | while read ep; do kaggle competitions replay "$ep" -p /tmp/sub_replays >/dev/null 2>&1; done
orbit_wars_rl/.venv/bin/python - <<'PY'
import json, glob
from collections import Counter
pc=Counter(); n2=w2=n4=0; w4=0
for f in glob.glob('/tmp/sub_replays/episode-*-replay.json'):
    d=json.load(open(f)); names=d['info']['TeamNames']
    if 'Saheb' not in names: continue
    i=names.index('Saheb'); rew=d['rewards']; win=rew[i]==max(rew) and rew.count(max(rew))==1
    if len(names)==2: n2+=1; w2+=win
    elif len(names)==4:
        n4+=1; w4+=win
        sc=[0.0]*4                              # final placement from last-step planet ships
        for p in d['steps'][-1][0]['observation']['planets']:
            o=int(p[1]);  sc[o]+= p[5] if 0<=o<4 else 0
        pc[1+sum(s>sc[i] for s in sc)]+=1
print(f"2p {w2}/{n2} ({100*w2/max(n2,1):.0f}%, par 50%) | 4p {w4}/{n4} ({100*w4/max(n4,1):.0f}%, par 25%) | 4p placement {dict(pc)}")
PY
```
> Win = our reward is the unique max (orbit_wars rewards are ±1). Mind the baselines: 2p par = 50%, 4p par = 25%.
> ⚠️ The `4p placement` above is **naive (final-score)** and is an ARTIFACT — FFA ends with one survivor, so all eliminated players tie at score 0 and look "2nd." For real 4p placement, rank by **elimination order** (last step each player still owned a planet/fleet), not the final-score snapshot. See `project_ffa_not_the_gap`.

### One command: fetch loss replays and run the target audit

Use this as the default daily workflow after a submission has enough episodes.

```bash
orbit_wars_rl/.venv/bin/python orbit_wars_rl/review_submission_targets.py \
  --submission-id 53359633 \
  --checkpoint seed_checkpoints/rev31_31M_resume.pt \
  --player-name Saheb
```

Outputs go to:
```text
/tmp/sub53359633_review/
  episodes.json          # full Kaggle manifest for the submission
  selected_losses.json   # metadata for the loss slice actually audited
  replays/*.json         # downloaded replay payloads
  target_audit.json      # machine-readable audit
  target_audit.md        # readable report
```

Useful flags:
```bash
# only 2-player losses
orbit_wars_rl/.venv/bin/python orbit_wars_rl/review_submission_targets.py \
  --submission-id 53359633 \
  --checkpoint seed_checkpoints/rev31_31M_resume.pt \
  --player-name Saheb \
  --only-two-player

# cap review size and focus on the opening
orbit_wars_rl/.venv/bin/python orbit_wars_rl/review_submission_targets.py \
  --submission-id 53359633 \
  --checkpoint seed_checkpoints/rev31_31M_resume.pt \
  --player-name Saheb \
  --max-episodes 10 \
  --step-limit 40
```

What the wrapper does:
- fetches the richer Kaggle `EpisodeService/ListEpisodes` manifest
- selects losses for the requested submission
- downloads replay JSONs with retry/backoff
- runs the same target audit used for manual replay review

### Audit target selection with the actual checkpoint

Use this when the replay viewer makes a move look like an aiming miss or a bad
planet choice and you want to know what the submitted policy actually did.

```bash
source /Users/saheb/home/.venv/bin/activate

python orbit_wars_rl/audit_submission_targets.py \
  --checkpoint gpu_run_artifacts/h100_rev31/checkpoints/torch_step_10485760_rev31_20260603_153146.pt \
  --replay-dir /tmp/sub53336058_eps \
  --player-name Saheb \
  --output-json /tmp/rev31_target_audit.json \
  --output-md /tmp/rev31_target_audit.md
```

Read the saved summary:
```bash
sed -n '1,160p' /tmp/rev31_target_audit.md
```

Audit just a few episodes:
```bash
python orbit_wars_rl/audit_submission_targets.py \
  --checkpoint gpu_run_artifacts/h100_rev31/checkpoints/torch_step_10485760_rev31_20260603_153146.pt \
  --replay-dir /tmp/sub53336058_eps \
  --episode-id 78686024 \
  --episode-id 78680551 \
  --episode-id 78656975 \
  --player-name Saheb \
  --output-md /tmp/rev31_target_audit_focus.md
```

The audit reports, per launch:
- decoded chosen target
- nearest / cheapest / best-tempo alternatives
- weakest nearby / highest-production nearby alternatives
- open-space / decode-fail count
- angle error vs the decoded target intercept

This is the quick way to separate:
- actual aiming/intercept misses
- target-priority mistakes
- “looks wrong in viewer, but the model deliberately chose that farther planet”

### Compare tempo metrics across checkpoints on fixed Ajay seeds

Use this when you want the exact comparison we just did: generate fresh replays
for a fixed opponent/seed slice, audit them, and print checkpoint-to-checkpoint
tempo metrics in one command.

```bash
orbit_wars_rl/.venv/bin/python orbit_wars_rl/compare_tempo_checkpoints.py \
  --checkpoints \
    gpu_run_artifacts/jarvis_rev33/checkpoints/torch_step_1572864_rev33_20260604_144227.pt \
    gpu_run_artifacts/jarvis_rev33/checkpoints/torch_step_3145728_rev33_20260604_144227.pt \
    gpu_run_artifacts/jarvis_rev33/checkpoints/torch_step_4194304_rev33_20260604_144227.pt \
    gpu_run_artifacts/jarvis_rev33/checkpoints/torch_step_5242880_rev33_20260604_144227.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --target-decode \
  --step-limit 40 \
  --output-dir /tmp/rev33_ajay_compare
```

Outputs:
- per-checkpoint replay JSONs under `/tmp/rev33_ajay_compare/<checkpoint-stem>/replays`
- per-checkpoint audit files:
  - `audit.json`
  - `audit.md`
- merged comparison summary:
  - `/tmp/rev33_ajay_compare/summary.json`
  - `/tmp/rev33_ajay_compare/summary.md`

The comparison summary includes:
- `farther_than_nearest_rate`
- `mean_eta_gap_vs_nearest`
- `mean_distance_gap_vs_nearest`
- `tempo_match_rate`
- `nearest_match_rate`
- `mean_first_capture_step`
- `invalid_raw_argmax`

### Build a conversion-focused auxiliary BC dataset

Use this when target ranking has improved but the opening still captures too
late because it spends too many ships before the first productive capture.

This extracts teacher openings that:
- face early pressure
- get the first extra planet early
- keep pre-capture ship spend restrained

It can also mix in the earlier target-relabel failure dataset so the resulting
auxiliary BC still carries the tempo-target correction.

```bash
orbit_wars_rl/.venv/bin/python orbit_wars_rl/build_conversion_bc.py \
  --teacher-replay-dir /tmp/orbit_episodes \
  --teacher-agent "Isaiah @ Tufa Labs" \
  --teacher-agent "Hober Malloc" \
  --teacher-agent "Ebi" \
  --teacher-steps-max 40 \
  --teacher-require-opponent-first-fire-by 12 \
  --teacher-max-first-capture-step 14 \
  --teacher-max-ships-before-first-capture 36 \
  --teacher-max-launches-before-first-capture 3 \
  --teacher-stop-at-first-capture \
  --failure-audit-json /tmp/sub53359633_target_audit.json \
  --failure-audit-json /tmp/ajay_panel_seat0_audit.json \
  --failure-relabel-mode tempo \
  --failure-min-eta-regret 4 \
  --failure-steps-max 40 \
  --samples-out /tmp/targeted_bc/conversion_mix_loose.pkl \
  --summary-out /tmp/targeted_bc/conversion_mix_loose_summary.json
```

**Calibration (full June-3 dataset, loose thresholds):**
- 316 teacher replays kept → 436 teacher samples
- 817 failure relabels
- Total: 1,253 samples → `seed_checkpoints/conversion_mix_loose.pkl`

**Strict thresholds** (first-capture ≤12, ships ≤28, launches ≤2) yield only ~9 samples — too tight.
**Loose thresholds** (first-capture ≤14, ships ≤36, launches ≤3) yield 1,253 — use these.

What it writes:
- standard BC sample pickle for `bc.py`
- JSON summary with:
  - kept replay count
  - teacher sample count
  - failure relabel count
  - filter-drop counts

### Low-level notes

Auth split still matters if you debug this manually:

| Endpoint | Auth method |
|----------|-------------|
| `EpisodeService/ListEpisodes` | Basic auth (`~/.kaggle/kaggle.json`) |
| `v1/competitions/episodes/{id}/replay` | Bearer token (`~/.kaggle/access_token`) |

The replay endpoint intermittently EOFs. Always use retry/backoff when
downloading many episodes.

---

## 10. Export a checkpoint as a submission agent

**⚠️ Phase 1 checkpoints require `--target-decode`. Forgetting it produces an agent that scores ~87 on LB (loses everything).**

```bash
source orbit_wars_rl/.venv/bin/activate

# Phase 1 checkpoints (32-bin absolute, action_decode=target) — ALWAYS use --target-decode
python3 orbit_wars_rl/export_agent.py \
  --checkpoint gpu_run_artifacts/hellburner_spot/checkpoints/<name>.pt \
  --output submission_agent.py \
  --target-decode

# Old architecture checkpoints (10-bin fraction) — no --target-decode
python3 orbit_wars_rl/export_agent.py \
  --checkpoint <old_arch_checkpoint>.pt \
  --output submission_agent.py
```

**Verify before submitting:**
```bash
source orbit_wars_rl/.venv/bin/activate
python3 -c "
import kaggle_environments as ke, re
code = open('submission_agent.py').read()
print('target_decode:', re.search(r'_TARGET_DECODE = (\w+)', code).group(1))
print('signature:', re.search(r'def agent\([^)]+\)', code).group(0))
g = {}; exec(code, g); agent = g['agent']
wins = sum(ke.make('orbit_wars').run([agent,'random'])[-1][0]['reward'] > 0 for _ in range(5))
print(f'vs random: {wins}/5')
"
```

**Then submit:**
```bash
kaggle competitions submit orbit-wars -f submission_agent.py -m "description"
```

**Known bugs fixed (2026-05-31):**
1. `def agent(obs)` → `def agent(obs, cfg=None)` — was crashing silently on every LB step
2. Missing `--target-decode` → used angle decode, scored 87 on LB vs 894 expected

---

## 10. Run unit tests

```bash
source /Users/saheb/home/.venv/bin/activate
python -m pytest orbit_wars_rl/tests/ -x -q
```

---

## Key files — what does what

| File | Role |
|------|------|
| `orbit_wars_rl/train_torch.py` | Main training entry point |
| `orbit_wars_rl/eval.py` | Full 256-game panel eval (source of truth) |
| `orbit_wars_rl/quick_eval.py` | Sanity check only — **never use for decisions** |
| `orbit_wars_rl/torch_env.py` | Vectorised GPU env, ship bin decode, SHIP_COUNTS |
| `orbit_wars_rl/features.py` | Feature extraction (Phase 1: planet=20, fleet=13, global=11, pairwise=12) |
| `orbit_wars_rl/model.py` | Entity transformer |
| `orbit_wars_rl/ppo.py` | PPO update + IL regularisation |
| `orbit_wars_rl/opponent_pool.py` | Self-play pool + PFSP sampling |
| `orbit_wars_rl/export_agent.py` | Export checkpoint → submission .py |
| `orbit_wars_rl/bc.py` | BC loss (lazily imported by ppo.py when --il-lambda set) |
| `orbit_wars_rl/build_producer_target_bc.py` | Build producer-labeled target-only BC dataset from replay launches |
| `orbit_wars_rl/build_conversion_bc.py` | Build conversion-focused BC dataset (fast-capture teacher + failure relabels) |
| `orbit_wars_rl/compare_tempo_checkpoints.py` | Compare first-capture/conversion metrics across checkpoints on fixed Ajay seeds |
| `orbit_wars_rl/eval_joint_opening.py` | Opening-only live prototype: joint action scorer picks early moves, base policy handles the rest |
| `orbit_wars_rl/step_firep.py` | Compare FireP at steps 0-3 across multiple checkpoints (opening aggression) |
| `opponents/orbit_lite/` | Ajay/Producer dependency — intercept aiming, fleet routing (must be present) |
| `orbit_wars_rl/env.py` | Old kaggle_environments wrapper (used by validate_training + tests) |
| `orbit_wars_rl/eval_panel.py` | 128-seed stratified panel used inside eval.py |
| `gpu_run_artifacts/launch_gpu.sh` | **Only** way to launch EC2 — bakes in terminate-on-shutdown |
| `gpu_run_artifacts/hellburner_spot/` | All active training scripts, logs, checkpoints |
| `seed_checkpoints/phase1_resume.pt` | Current Phase 1 resume point (rev5 6M peak, HB=38.7%) |
| `seed_checkpoints/bc_phase1_warmstart.pt` | BC warmstart (IL reference, not used in Rev7+) |
| `opponents/candidate_hellburner.py` | Primary HB opponent |
| `opponents/candidate_zach_public.py` | Primary Zach opponent |
| `opponents/candidate_suneet_lb1200.py` | Primary Suneet opponent |

---

## Common mistakes to avoid

| Mistake | Fix |
|---------|-----|
| RL-agent external opponent (141208) without `--heuristic-workers 2` | Auto-default uses `cpu_count-1=7` workers; each loads a neural net → all 8 CPUs saturated, GPU drops to 0%, training deadlocks. **Always add `--heuristic-workers 2` for RL-agent opponents.** If already hung: `gcloud compute instances reset` (SSH unresponsive). |
| `python train_torch.py` without activating venv | `source /Users/saheb/home/.venv/bin/activate` first, or use full path |
| `nohup python ...` on Mac | Fails silently — activate venv first so `python` resolves |
| `--opponent candidate_hellburner.py` | Path is `opponents/candidate_hellburner.py` |
| `--external-opponents candidate_hellburner.py` | Path is `opponents/candidate_hellburner.py` |
| Forgetting `--panel` in eval | Without it you get a single game, not 256 |
| Forgetting `--target-decode` in eval | Phase 1 checkpoints need this; wrong mode = garbage results |
| Trusting `quick_eval` results | Always run full panel before making decisions |
| Launching with `aws ec2 run-instances` raw | Use `launch_gpu.sh` — raw launch defaults to stop-not-terminate |
| Leaving instance in stopped state | Stopped still bills for EBS; terminate after pulling everything |
| Panel watcher running with old opponent paths | Kill watcher (`ps aux \| grep run_panel_eval_watcher ... xargs kill`), fix paths, restart |
| Running many evals in parallel on Mac | CPU contention → each eval takes 10× longer; limit to 3 at most |
| Eval OOM on training instance | Training occupies GPU; eval auto-detects CUDA and OOMs. Always prefix: `CUDA_VISIBLE_DEVICES="" python3 orbit_wars_rl/eval.py ...` on any instance running training |
