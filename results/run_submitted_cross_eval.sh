#!/usr/bin/env bash
# Broad cross-eval of the two SUBMITTED 2p agents (presres1 0.5M, stgpr1 0.5M) against the
# held-out opponent set: public heuristics + our own best past selves.
#
# Run from the orbit-prephantom worktree (commit 8f78555) — the code lineage these checkpoints
# were trained on, BEFORE the phantom-neutral-production feature fix. Evaluating them on main
# (b3f0b52, phantom fix applied) would make their trained features OOD. Checkpoints live in the
# orbit-audit worktree; opponents + venv come from the main kaggle-orbit-wars repo.
#
# Full 256-game both-seats panel per opponent (matches the cert methodology) → ~8 min/opp.
set -uo pipefail
PREPHANTOM=/Users/saheb/home/orbit-prephantom
MAIN=/Users/saheb/home/kaggle-orbit-wars
AUDIT=/Users/saheb/home/orbit-audit
PY=$MAIN/orbit_wars_rl/.venv/bin/python
OUT=$MAIN/results/submitted_cross_eval.log
cd "$PREPHANTOM"

# agent label -> checkpoint
AGENTS=(
  "presres1_05:$AUDIT/gpu_run_artifacts/presres1/checkpoints/torch_step_524288_presres1_20260623_115802.pt"
  "stgpr1_05:$AUDIT/gpu_run_artifacts/stgpr1/checkpoints/torch_step_524288_stgpr1_20260623_133835.pt"
)
# opponent label -> path (relative to $MAIN)
OPPONENTS=(
  "zach:opponents/candidate_zach_public.py"
  "hellburner:opponents/candidate_hellburner.py"
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

echo "=== submitted-agent cross-eval (256-game both-seats panels) started $(date) ===" | tee "$OUT"
for ae in "${AGENTS[@]}"; do
  aname="${ae%%:*}"; ckpt="${ae##*:}"
  echo "" | tee -a "$OUT"
  echo "########## agent=$aname  ckpt=$(basename "$ckpt") ##########" | tee -a "$OUT"
  printf "%-14s %8s  %8s  %8s  %8s\n" "opponent" "overall" "seat0" "seat1" "asym" | tee -a "$OUT"
  for oe in "${OPPONENTS[@]}"; do
    oname="${oe%%:*}"; opp="$MAIN/${oe##*:}"
    log="/tmp/xeval2_${aname}_${oname}.log"
    $PY orbit_wars_rl/eval.py --checkpoint "$ckpt" --opponent "$opp" \
        --panel --target-decode > "$log" 2>&1
    ov=$(grep -E "^Overall:" "$log" | tail -1 | sed -E 's/.*\(([0-9.]+%)\).*/\1/')
    s0=$(grep -E "seat 0:" "$log" | tail -1 | sed -E 's/.*\(([0-9.]+%)\).*/\1/')
    s1=$(grep -E "seat 1:" "$log" | tail -1 | sed -E 's/.*\(([0-9.]+%)\).*/\1/')
    asym=$(grep -E "asymmetry" "$log" | tail -1 | sed -E 's/.*: ([+-][0-9.]+pp).*/\1/')
    printf "%-14s %8s  %8s  %8s  %8s\n" "$oname" "${ov:-ERR}" "${s0:-?}" "${s1:-?}" "${asym:-?}" | tee -a "$OUT"
  done
done
echo "" | tee -a "$OUT"
echo "=== done $(date) ===" | tee -a "$OUT"
