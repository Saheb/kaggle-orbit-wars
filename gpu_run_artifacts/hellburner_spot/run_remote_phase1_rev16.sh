#!/usr/bin/env bash
# Phase 1 Rev16 — DEFENSE: add --defense-coef to break the late-game collapse.
#
# Context: rev15 2M (795.7 LB, new Phase-1 best) still collapses in 9/10 2-player
# LB losses. The collapse is a late-game behavioural switch: mid-game ship-bin
# median ~32 (decent), late-game median ~1 (1-ship probes). The trigger is falling
# behind on planets around step 50-150, after which the agent completely gives up.
#
# Single delta from rev15: add --defense-coef 0.02
# Mechanism: each step a planet is lost, prod_lost is subtracted from rewards
# (terminal shaping, symmetric with expansion-coef). This creates a direct gradient
# AGAINST the collapse spiral: losing a planet → negative signal → agent learns to
# send real force to defend rather than defaulting to 1-ship probes.
#
# Resume: rev15 2M (torch_step_2031616_rev15_20260531_211208.pt, 795.7 LB).
# Pool:   pool_step_2031616_rev15_20260531_211208.pt (auto-loaded as pool_phase1_resume.pt).
# Success: 2-player LB losses where late-game ship-bin median > 10 (not 1-ship collapse),
#          and LB score > 795.7.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$HOME/orbit_wars_rl"
TS="$(date '+%Y%m%d_%H%M%S')"
LOG="train_gpu_phase1_rev16_${TS}.log"
echo "Run: phase1 rev16 (expansion+win-margin+defense on rev15-2M)  ts: $TS  log: $LOG"

PYTHONUNBUFFERED=1 python3 orbit_wars_rl/train_torch.py \
  --resume seed_checkpoints/phase1_resume.pt \
  --run-name rev16 \
  --total-steps 6000000 \
  --lr-schedule-steps 200000000 \
  --learning-rate 0.0003 \
  --num-envs 64 --rollout-steps 512 --num-minibatches 16 \
  --ppo-epochs 2 \
  --checkpoint-interval 1000000 \
  --pool-checkpoint-interval 500000 --pool-max-size 40 \
  --pool-mode mixed --pool-fraction 0.75 \
  --win-margin-coeff 0.5 \
  --expansion-coef 0.01 \
  --defense-coef 0.02 \
  --action-decode target \
  --pool-pfsp-min-games 30 \
  --pool-mastered-threshold 0.99 \
  --pool-mastered-min-games 500000 \
  --srcs-multi-penalty 0.001 \
  --srcs-multi-threshold 2.0 \
  --max-grad-norm 10.0 \
  --entropy-coef-fire 0.02 --entropy-coef-angle 0.03 --entropy-coef-ships 0.02 \
  2>&1 | tee "$LOG"

echo "Training exited"
