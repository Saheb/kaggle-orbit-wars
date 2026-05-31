#!/usr/bin/env bash
# run_eval.sh — verified eval wrapper. Exports checkpoint if needed, then evals.
#
# Usage:
#   bash orbit_wars_rl/run_eval.sh <checkpoint.pt> <opponent> [--games N]
#
# Opponent can be:
#   - a .pt checkpoint path (auto-exported to gpu_run_artifacts/agents/)
#   - an already-exported .py agent path
#   - "141208"  (shorthand for the archived 141208 agent)
#   - "zach"    (shorthand for opponents/candidate_zach_public.py)
#   - "random"
#
# Examples:
#   bash orbit_wars_rl/run_eval.sh gpu_run_artifacts/hellburner_spot/checkpoints/torch_step_2031616_20260531_065423.pt 141208
#   bash orbit_wars_rl/run_eval.sh <ckpt.pt> zach --games 64
#   bash orbit_wars_rl/run_eval.sh <ckpt.pt> <other_ckpt.pt>
set -euo pipefail

CKPT="$1"
OPPONENT_ARG="$2"
GAMES="${4:-64}"  # default 64 games
AGENTS_DIR="$(dirname "$0")/../gpu_run_artifacts/agents"
mkdir -p "$AGENTS_DIR"

# ── Resolve opponent ──────────────────────────────────────────────────────────
case "$OPPONENT_ARG" in
  141208)
    OPPONENT="$(dirname "$0")/../.codex/worktrees/296f/kaggle-orbit-wars/archive/main_rl_141208.py"
    # Check in current repo too
    [[ ! -f "$OPPONENT" ]] && OPPONENT="$(find "$(dirname "$0")/.." -name "main_rl_141208.py" ! -path "*/venv/*" 2>/dev/null | head -1)"
    ;;
  zach)    OPPONENT="opponents/candidate_zach_public.py" ;;
  suneet)  OPPONENT="opponents/candidate_suneet_lb1200.py" ;;
  hb)      OPPONENT="opponents/candidate_hellburner.py" ;;
  random)  OPPONENT="random" ;;
  *.pt)
    # Export the opponent checkpoint
    OPP_BASE="$(basename "$OPPONENT_ARG" .pt)"
    OPP_PY="$AGENTS_DIR/${OPP_BASE}.py"
    if [[ ! -f "$OPP_PY" ]]; then
      echo "Exporting opponent $OPPONENT_ARG → $OPP_PY"
      source orbit_wars_rl/.venv/bin/activate 2>/dev/null || true
      python3 orbit_wars_rl/export_agent.py --checkpoint "$OPPONENT_ARG" --output "$OPP_PY" --target-decode
    fi
    OPPONENT="$OPP_PY"
    ;;
  *.py)    OPPONENT="$OPPONENT_ARG" ;;
  *)       echo "Unknown opponent: $OPPONENT_ARG"; exit 1 ;;
esac

# ── Verify our checkpoint ─────────────────────────────────────────────────────
echo "=== Checkpoint info ==="
source orbit_wars_rl/.venv/bin/activate 2>/dev/null || true
python3 orbit_wars_rl/ckpt_info.py "$CKPT"

# ── Export our checkpoint if .pt ─────────────────────────────────────────────
CKPT_BASE="$(basename "$CKPT" .pt)"
if [[ "$OPPONENT" == "$AGENTS_DIR/${CKPT_BASE}.py" ]]; then
  echo "ERROR: checkpoint and opponent are the same file"; exit 1
fi

echo ""
echo "=== Eval: $CKPT_BASE vs $(basename "$OPPONENT" .py) ($GAMES games) ==="
python3 orbit_wars_rl/eval.py \
  --checkpoint "$CKPT" \
  --opponent "$OPPONENT" \
  --games "$GAMES" \
  --target-decode \
  2>&1 | grep -v "INFO\|WARNING\|Loading"
