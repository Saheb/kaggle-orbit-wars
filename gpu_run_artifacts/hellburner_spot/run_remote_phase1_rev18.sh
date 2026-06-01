#!/usr/bin/env bash
# Phase 1 Rev18 — MIXED AGGRESSIVE OPPONENT: break the symmetric passive Nash equilibrium.
#
# Context: rev15-rev17 all converge to fire_rate~0.15 and passive play. LB replay analysis
# shows 141208 (894 LB) fires 0.56 in wins and ends games at step 203. Phase 1 always times
# out at step 500. Pure self-play creates a symmetric "accumulate, don't commit" equilibrium
# because both sides use the same policy and neither applies pressure.
#
# Fix: --external-opponents pointing to 141208 at 25% of pool games. 141208 fires every
# other step — our policy MUST respond (counter-pressure or defend under attack). Self-play
# covers the remaining 75% so generalisation isn't destroyed (the 90% HB blitz failure).
# No reward shaping — the gradient comes from actually playing against aggression.
#
# Single delta from rev15 recipe: add --external-opponents + --pool-external-fraction 0.25.
# Resume: rev15 2M (torch_step_2031616_rev15, 795.7 LB). Pool: pool_phase1_resume.pt.
# Success: fire_rate rises above 0.20, games start ending before step 500, Zach holds >=69%.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$HOME/orbit_wars_rl"
TS="$(date '+%Y%m%d_%H%M%S')"
LOG="train_gpu_phase1_rev18_${TS}.log"
echo "Run: phase1 rev18 (141208 external opp 25% on rev15-2M)  ts: $TS  log: $LOG"

PYTHONUNBUFFERED=1 python3 orbit_wars_rl/train_torch.py \
  --resume seed_checkpoints/phase1_resume.pt \
  --run-name rev18 \
  --total-steps 6000000 \
  --lr-schedule-steps 200000000 \
  --learning-rate 0.0003 \
  --num-envs 64 --rollout-steps 512 --num-minibatches 16 \
  --ppo-epochs 2 \
  --checkpoint-interval 1000000 \
  --pool-checkpoint-interval 500000 --pool-max-size 40 \
  --pool-mode mixed --pool-fraction 0.75 \
  --external-opponents opponents/archive/main_rl_141208.py \
  --pool-external-fraction 0.25 \
  --win-margin-coeff 0.5 \
  --expansion-coef 0.01 \
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
