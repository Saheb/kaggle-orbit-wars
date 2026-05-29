#!/usr/bin/env bash
# Phase 1 — fresh retrain from BC warmstart with all Phase 1 features.
#
# Changes vs prior runs:
#   - Fresh model (random weights → BC warmstart, NOT resuming old checkpoint)
#   - Phase 1 feature bundle: planet=20, fleet=13, global=11, pairwise=12,
#     max_owned=16, value head concat(global_token, owned_pool)
#   - 1 external opponent in pool (Hellburner only) — Zach/Suneet excluded (14 workers on 8 vCPU g5.2xlarge → 0% GPU, 229 SPS; Phase 0 got ~2300 SPS with HB only)
#   - ship_bin_mode=absolute (32 bins), min_ship_bin=0 (no masking needed for
#     fresh model — no cold-start 1-ship collapse since BC initialises well)
#
# rev3 change (single delta vs rev2):
#   - il-decay-frac: 0.5 → 1.0
#     Rev2 anchor hit ~zero at 15M steps (half of 30M). avgfleet drifted 7→100+
#     by 18M steps; panel peak was 9M (Zach 12.9%, Suneet 9.8%, HB 0.8%), then
#     regression as anchor expired. With decay-frac=1.0 the cosine decays over
#     the FULL 30M run — anchor is 2× stronger at 9M, still meaningful at 20M.
#     Hypothesis: policy can build real game-sense through 15-20M steps while
#     constrained from carpet-bombing, then finishes with less-anchored RL polish.
#
# rev5 change (single delta vs rev4):
#   - Resume from torch_step_3047424_20260529_104020.pt (15.7M effective total)
#     — best panel checkpoint from rev4 (HB=1.2%, Zach=12.1%, Suneet=13.3%)
#   - Add --srcs-multi-penalty 0.001 --srcs-multi-threshold 2.0
#     Rev4 collapse postmortem: srcs_multi was clean (1.3–2.8) through 6M resume
#     steps, then broke suddenly at ~6.5M (4.4 → 5.5 → 5.6+ in a few iters).
#     The 3M checkpoint (srcs_multi=2.41) is 3.5M steps before the cliff.
#     Penalty: 0.001 per excess source per step. At srcs_multi=3 (1 over threshold),
#     ~0.30 penalty/game vs ±1 win/loss — meaningful but not overwhelming.
#     Threshold=2.0: firing from ≤2 sources simultaneously is free.
#
# LR: 3e-4 peak with cosine decay over 2x the training horizon (60M schedule,
# 30M actual) → ~50% decay by end.
# total-steps=30M: run until manually killed when plateau detected.
set -euo pipefail

cd "$HOME/orbit_wars_rl"
source /opt/pytorch/bin/activate
pip install -q kaggle-environments

TS="$(date '+%Y%m%d_%H%M%S')"
LOG="train_gpu_phase1_${TS}.log"

echo "Run: phase1 rev5  timestamp: $TS  log: $LOG"

PYTHONUNBUFFERED=1 python train_torch.py \
  --resume ../seed_checkpoints/phase1_resume.pt \
  --total-steps 30000000 \
  --lr-schedule-steps 60000000 \
  --learning-rate 0.0003 \
  --num-envs 512 --rollout-steps 64 --num-minibatches 32 \
  --ppo-epochs 2 \
  --checkpoint-interval 1000000 \
  --pool-checkpoint-interval 500000 --pool-max-size 20 \
  --pool-mode mixed --pool-fraction 0.75 \
  --external-opponents ../candidate_hellburner.py \
  --pool-external-fraction 0.05 \
  --win-margin-coeff 0.5 \
  --action-decode target \
  --pool-pfsp-min-games 30 \
  --pool-mastered-threshold 0.99 \
  --pool-mastered-min-games 500000 \
  --il-lambda 0.01 \
  --il-decay-frac 1.0 \
  --il-ref ../seed_checkpoints/bc_phase1_warmstart.pt \
  --srcs-multi-penalty 0.001 \
  --srcs-multi-threshold 2.0 \
  --terminate-on-done \
  2>&1 | tee "$LOG"
