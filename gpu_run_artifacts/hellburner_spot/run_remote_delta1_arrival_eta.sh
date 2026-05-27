#!/usr/bin/env bash
# delta1_arrival_eta — rotation-aware pairwise features
#
# Delta: features.py pairwise feats 0-3 (direction sin/cos, dist, 1/eta) now
#        use predicted ARRIVAL position for orbiting targets instead of current
#        position.  Diagnosis: seed-10 replay showed 17-turn transit because
#        model targeted a "close" rotating planet that had rotated away by
#        arrival; HB captured nearby static planets in 5-7 turns and snowballed.
#
# Hypothesis: correcting arrival ETA/direction helps the model learn to prefer
#             nearby targets in the opening on rotating-planet maps, closing the
#             early production gap vs Hellburner.
#
# Base: torch_step_1015808_20260526_141208.pt (55.5% HB, 80.1% Suneet, 74.2% Zach)
# Pool: balanced (external_fraction=0.25, pool_fraction=0.75, PFSP)
# LR: slow cosine (schedule_steps=12M, total=6M → ~50% decay by end)
set -euo pipefail

cd "$HOME/orbit_wars_rl"
source /opt/pytorch/bin/activate

TS="$(date '+%Y%m%d_%H%M%S')"
LOG="train_gpu_delta1_arrival_eta_${TS}.log"

echo "Run: delta1_arrival_eta  timestamp: $TS  log: $LOG"

PYTHONUNBUFFERED=1 python train_torch.py \
  --resume ../seed_checkpoints/torch_step_1015808_20260526_141208.pt \
  --total-steps 6000000 \
  --lr-schedule-steps 12000000 \
  --num-envs 512 --rollout-steps 64 --num-minibatches 32 \
  --ppo-epochs 2 --learning-rate 0.00002 \
  --checkpoint-interval 1000000 \
  --pool-checkpoint-interval 500000 --pool-max-size 20 \
  --pool-mode mixed --pool-fraction 0.75 \
  --external-opponents ../candidate_hellburner.py \
  --pool-external-fraction 0.25 \
  --win-margin-coeff 0.5 \
  --action-decode target \
  --min-ship-bin 1 \
  --pool-pfsp-min-games 30 \
  --pool-mastered-threshold 0.99 \
  --pool-mastered-min-games 500000 \
  --terminate-on-done \
  2>&1 | tee "$LOG"
