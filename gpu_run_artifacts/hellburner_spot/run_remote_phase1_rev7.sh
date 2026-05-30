#!/usr/bin/env bash
# Phase 1 Rev7 — short-run chain, no IL anchor.
#
# Strategy: replicate the 123203/141208 chain structure on Phase 1.
# Run short (2-3M steps), kill before passivity sets in, chain runs.
#
# Single delta vs rev5: remove IL anchor entirely.
# Rationale:
#   - Rev5 peaked at 6M then regressed to passivity (fire[0] 0.35→0.25) by 12M
#   - Passivity is NOT from srcs-multi-penalty (srcs_multi stayed 1.2-1.8, clean)
#   - IL anchor (il-lambda=0.01) pulls policy toward BC warmstart on every update
#     BC is methodical/conservative → anchor amplifies self-play's passivity drift
#   - 141208 chain had NO IL and produced the 55.5% best result
#   - Foundation (62M, no IL, no penalty) also got passive (fire[0]=0.09) — but
#     that was 62M unconstrained. Short runs (≤3M) don't run long enough to collapse.
#
# Kill signals (check every checkpoint, in priority order):
#   1. fire[0] declining 3 consecutive checkpoints → kill immediately
#   2. fire[0] < 0.25 at any checkpoint → kill immediately
#   3. srcs_multi > 4.0 or fire[0] > 0.55 → kill immediately (collapse)
#   4. Hard cap: kill at 3M steps regardless
#
# Everything else identical to rev5:
#   - Resume: torch_step_6094848_20260529_160908.pt (rev5 6M peak = 38.7% HB)
#   - pool-fraction=0.75, pool-external-fraction=0.05 (minimal HB, mostly self-play)
#   - lr=3e-4 cosine over 60M schedule
#   - srcs-multi-penalty=0.001, threshold=2.0 (collapse guard, unchanged)
#   - checkpoint every 1M steps
set -euo pipefail

cd "$HOME/orbit_wars_rl"
source /opt/pytorch/bin/activate
pip install -q kaggle-environments

TS="$(date '+%Y%m%d_%H%M%S')"
LOG="train_gpu_phase1_rev7_${TS}.log"

echo "Run: phase1 rev7 (no IL)  timestamp: $TS  log: $LOG"

PYTHONUNBUFFERED=1 python orbit_wars_rl/train_torch.py \
  --resume seed_checkpoints/phase1_resume.pt \
  --total-steps 30000000 \
  --lr-schedule-steps 60000000 \
  --learning-rate 0.0003 \
  --num-envs 512 --rollout-steps 64 --num-minibatches 32 \
  --ppo-epochs 2 \
  --checkpoint-interval 1000000 \
  --pool-checkpoint-interval 500000 --pool-max-size 20 \
  --pool-mode mixed --pool-fraction 0.75 \
  --external-opponents candidate_hellburner.py \
  --pool-external-fraction 0.05 \
  --win-margin-coeff 0.5 \
  --action-decode target \
  --pool-pfsp-min-games 30 \
  --pool-mastered-threshold 0.99 \
  --pool-mastered-min-games 500000 \
  --srcs-multi-penalty 0.001 \
  --srcs-multi-threshold 2.0 \
  --terminate-on-done \
  2>&1 | tee "$LOG"
