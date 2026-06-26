#!/usr/bin/env bash
# Parallel finish of the stgpr1 0.5M cross-eval (zach + hellburner already done).
# Each --panel is single-threaded (1 core @ ~99%); 9 concurrent fits a 10-core box.
# Same full 256-game both-seats methodology as the rest of the sweep.
set -uo pipefail
PREPHANTOM=/Users/saheb/home/orbit-prephantom
MAIN=/Users/saheb/home/kaggle-orbit-wars
AUDIT=/Users/saheb/home/orbit-audit
PY=$MAIN/orbit_wars_rl/.venv/bin/python
CKPT=$AUDIT/gpu_run_artifacts/stgpr1/checkpoints/torch_step_524288_stgpr1_20260623_133835.pt
OUT=$MAIN/results/submitted_cross_eval_stgpr1_parallel.log
cd "$PREPHANTOM"

OPPONENTS=(
  "h1166_peak:opponents/orbit-wars-heuristic-bots/08_v13_3_R8_full_stack_lb1166_PEAK_HEURISTIC.py"
  "h1043_simple:opponents/orbit-wars-heuristic-bots/02_v12_6d_lb1043_simple_clean.py"
  "self_rev38:opponents/ourbest/rev38_5M.py"
  "self_rev53b:opponents/ourbest/rev53b_10M.py"
  "self_rev31:opponents/ourbest/rev31.py"
  "self_rev32b:opponents/ourbest/rev32b.py"
  "pool_lb1152:opponents/orbit-wars-heuristic-bots/14_main_k_v2_lb1152_LAST_HEURISTIC.py"
  "pool_lb1138:opponents/orbit-wars-heuristic-bots/11_v14_1n_lb1138_doom_evac_mega_hammer.py"
  "pool_lb1084:opponents/orbit-wars-heuristic-bots/03_v12_7m_lb1084_4p_relative_gap_hammer.py"
)

echo "=== stgpr1 remaining panels (parallel) started $(date) ===" | tee "$OUT"
pids=()
for oe in "${OPPONENTS[@]}"; do
  oname="${oe%%:*}"; opp="$MAIN/${oe##*:}"
  log="/tmp/xeval2_stgpr1_05_${oname}.log"
  OMP_NUM_THREADS=1 $PY orbit_wars_rl/eval.py --checkpoint "$CKPT" --opponent "$opp" \
      --panel --target-decode > "$log" 2>&1 &
  pids+=($!)
  echo "launched $oname (pid $!)" | tee -a "$OUT"
done

echo "waiting on ${#pids[@]} panels..." | tee -a "$OUT"
wait
echo "" | tee -a "$OUT"
printf "%-14s %8s  %8s  %8s  %8s\n" "opponent" "overall" "seat0" "seat1" "asym" | tee -a "$OUT"
for oe in "${OPPONENTS[@]}"; do
  oname="${oe%%:*}"; log="/tmp/xeval2_stgpr1_05_${oname}.log"
  ov=$(grep -E "^Overall:" "$log" | tail -1 | sed -E 's/.*\(([0-9.]+%)\).*/\1/')
  s0=$(grep -E "seat 0:" "$log" | tail -1 | sed -E 's/.*\(([0-9.]+%)\).*/\1/')
  s1=$(grep -E "seat 1:" "$log" | tail -1 | sed -E 's/.*\(([0-9.]+%)\).*/\1/')
  asym=$(grep -E "asymmetry" "$log" | tail -1 | sed -E 's/.*: ([+-][0-9.]+pp).*/\1/')
  printf "%-14s %8s  %8s  %8s  %8s\n" "$oname" "${ov:-ERR}" "${s0:-?}" "${s1:-?}" "${asym:-?}" | tee -a "$OUT"
done
echo "=== done $(date) ===" | tee -a "$OUT"
