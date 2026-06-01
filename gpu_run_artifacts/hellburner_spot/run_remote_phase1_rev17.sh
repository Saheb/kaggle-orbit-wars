#!/usr/bin/env bash
# Phase 1 Rev17 — HANDICAP CURRICULUM: force the agent to practise fighting from behind.
#
# Context: rev16 (defense-coef) failed — halved ship0 but lost to rev15-2M across all
# checkpoints (1M: 46%, 2M: 46%, 3M: 37%). Defense penalty taught hoarding, not aggression.
#
# Root cause of LB losses (confirmed via 30 two-player LB replays, rev15-2M @ 795.7):
# 9/10 losses = late-game bimodal collapse. Agent commits decent force mid-game (med_bin ~32)
# but switches to 1-ship probes past step 150 when behind. Pure symmetric self-play never
# generates the "behind but still viable" gradient because the losing side is eliminated before
# the policy can learn to fight back.
#
# Fix: --handicap-frac 0.3 — 30% of games start with player 0 having 5 ships instead of 10.
# This directly creates the "losing position at step 0" state the agent must learn to handle.
# No reward change — just curriculum via asymmetric starts. Single delta from rev15 recipe.
#
# Resume: rev15 2M (torch_step_2031616_rev15, 795.7 LB — our best base).
# Pool:   pool_step_2031616_rev15 (auto-loaded as pool_phase1_resume.pt).
# Success: panel vs rev15-2M > 52%, ship0 stays low, late-game force doesn't collapse.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$HOME/orbit_wars_rl"
TS="$(date '+%Y%m%d_%H%M%S')"
LOG="train_gpu_phase1_rev17_${TS}.log"
echo "Run: phase1 rev17 (handicap curriculum 30% on rev15-2M)  ts: $TS  log: $LOG"

PYTHONUNBUFFERED=1 python3 orbit_wars_rl/train_torch.py \
  --resume seed_checkpoints/phase1_resume.pt \
  --run-name rev17 \
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
  --handicap-frac 0.3 \
  --handicap-ships 5 \
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
