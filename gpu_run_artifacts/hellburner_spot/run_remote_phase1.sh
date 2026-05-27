#!/usr/bin/env bash
# Phase 1 — fresh retrain from BC warmstart with all Phase 1 features.
#
# Changes vs prior runs:
#   - Fresh model (random weights → BC warmstart, NOT resuming old checkpoint)
#   - Phase 1 feature bundle: planet=20, fleet=13, global=11, pairwise=12,
#     max_owned=16, value head concat(global_token, owned_pool)
#   - 2 external opponents in pool (Hellburner, Zach) — Suneet excluded (slows training)
#   - ship_bin_mode=absolute (32 bins), min_ship_bin=0 (no masking needed for
#     fresh model — no cold-start 1-ship collapse since BC initialises well)
#
# Hypothesis: richer features (fleet destination, connectivity, ships-at-arrival,
# enemy fleet split) + larger owned-planet cap + better value head → policy
# learns better target selection and defense, lifting Hellburner score above
# 55.5% while maintaining Zach/Suneet >75%.
#
# Pool: Hellburner + Zach externals + self-checkpoints via PFSP.
# external_fraction=0.25 keeps ~25% of samples on externals (balanced across
# both), 75% on self-play PFSP. Same recipe that produced 55.5% HB.
#
# LR: fresh model needs warmer LR than a resume. 3e-4 peak with cosine decay
# over 2x the training horizon (12M schedule, 6M actual) → ~50% decay by end.
set -euo pipefail

cd "$HOME/orbit_wars_rl"
source /opt/pytorch/bin/activate
pip install -q kaggle-environments

TS="$(date '+%Y%m%d_%H%M%S')"
LOG="train_gpu_phase1_${TS}.log"

echo "Run: phase1  timestamp: $TS  log: $LOG"

PYTHONUNBUFFERED=1 python train_torch.py \
  --resume ../seed_checkpoints/bc_phase1_warmstart.pt \
  --total-steps 6000000 \
  --lr-schedule-steps 12000000 \
  --learning-rate 0.0003 \
  --num-envs 512 --rollout-steps 64 --num-minibatches 32 \
  --ppo-epochs 2 \
  --checkpoint-interval 1000000 \
  --pool-checkpoint-interval 500000 --pool-max-size 20 \
  --pool-mode mixed --pool-fraction 0.75 \
  --external-opponents ../candidate_hellburner.py,../candidate_zach_public.py \
  --pool-external-fraction 0.25 \
  --win-margin-coeff 0.5 \
  --action-decode target \
  --pool-pfsp-min-games 30 \
  --pool-mastered-threshold 0.99 \
  --pool-mastered-min-games 500000 \
  --terminate-on-done \
  2>&1 | tee "$LOG"
