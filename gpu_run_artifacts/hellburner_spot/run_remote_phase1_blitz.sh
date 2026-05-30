#!/usr/bin/env bash
# Phase 1 blitz — replicate the 141208 run's HB-blitz approach on Phase 1.
#
# The 141208 run (old architecture, fraction mode, 10 bins) hit 55.5% HB at exactly
# the 1M step mark starting from a 44.5% checkpoint, then declined monotonically.
# Peak came from 90% HB exposure + no IL anchor + conservative lr=3e-5.
#
# This run applies the same blitz recipe to Phase 1:
#   - Resume: torch_step_6094848_20260529_160908.pt (rev5 6M peak = 38.7% HB)
#   - pool-fraction 0.9, pool-external-fraction 1.0 → ~90% HB exposure (vs 4% in rev5)
#   - lr 3e-5 (conservative, matching 141208; vs 3e-4 in rev5)
#   - lr-schedule-steps 4M (2× total-steps, same ratio as Phase 1 standard)
#   - No IL anchor (141208 had none; IL was pulling toward BC, constraining HB learning)
#   - total-steps 2M — short window; 141208 peaked at exactly 1M, never recovered after
#   - checkpoint-interval 500K — catch the peak early if it arrives before 1M
#   - srcs-multi-penalty kept for safety (141208 was short enough it didn't need it)
#
# Single deltas vs rev5: pool-external-fraction, pool-fraction, lr, remove IL
# (multiple changes, but all are intentionally replicating a known-good config)
set -euo pipefail

cd "$HOME/orbit_wars_rl"
source /opt/pytorch/bin/activate
pip install -q kaggle-environments

TS="$(date '+%Y%m%d_%H%M%S')"
LOG="train_gpu_phase1_blitz_${TS}.log"

echo "Run: phase1 blitz  timestamp: $TS  log: $LOG"

PYTHONUNBUFFERED=1 python orbit_wars_rl/train_torch.py \
  --resume seed_checkpoints/phase1_resume.pt \
  --total-steps 2000000 \
  --lr-schedule-steps 4000000 \
  --learning-rate 0.00003 \
  --num-envs 512 --rollout-steps 64 --num-minibatches 32 \
  --ppo-epochs 2 \
  --checkpoint-interval 500000 \
  --pool-checkpoint-interval 500000 --pool-max-size 20 \
  --pool-mode mixed --pool-fraction 0.9 \
  --external-opponents candidate_hellburner.py \
  --pool-external-fraction 1.0 \
  --win-margin-coeff 0.5 \
  --action-decode target \
  --pool-pfsp-min-games 30 \
  --pool-mastered-threshold 0.99 \
  --pool-mastered-min-games 500000 \
  --srcs-multi-penalty 0.001 \
  --srcs-multi-threshold 2.0 \
  --terminate-on-done \
  2>&1 | tee "$LOG"
