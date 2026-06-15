#!/usr/bin/env bash
# ONE controller for ALL run watchers (sync + held-out eval), on ANY platform.
# Solves the recurring "stale prior-run watcher still pointed at the old folder" bug two ways:
#   1. `start` ALWAYS tears down every existing watcher before launching new ones.
#   2. each watcher self-terminates when it is no longer the active run — it checks the marker
#      file gpu_run_artifacts/.active_run every cycle and exits if RUN changed. So even a watcher
#      that escapes the pkill dies on its own at the next launch.
#
# Usage:
#   run_watchers.sh start <run> <platform> <target> [held_out_opp]
#       platform = jarvis | gcp | custom
#         jarvis : target = IP            (ssh -i ~/.ssh/jarvis-labs-key root@IP, /home paths)
#         gcp    : target = config-ssh alias (ssh <alias>, ~/orbit_wars_rl paths)
#         custom : target unused; set env RSYNC_SSH / HOST / REMOTE_LOG_DIR / REMOTE_CKPT_DIR yourself
#   run_watchers.sh stop                  # kill all watchers
#   run_watchers.sh status                # active run + live procs
#
#   <run> is BOTH the local artifact folder (gpu_run_artifacts/<run>/) and, by default, the
#   substring matched in remote log/checkpoint filenames. For a multi-arm A/B whose arms use
#   DIFFERENT run-names (e.g. gate2 + gate3) under one umbrella folder, set env MATCH to the common
#   prefix so ONE watcher syncs/evals both arms:
#       MATCH=gate run_watchers.sh start gate_ab jarvis <ip>   # folder=gate_ab, matches gate2/gate3
#   For a gate A/B, also set EVAL_GATE_FROM_RUNNAME=1 so each ckpt is evaled with its OWN
#   --reinforce-gate-min-planets (parsed from the <MATCH><N> token in the filename) — eval must match
#   training, so gate2 ckpts get the gate=2 mask and gate3 ckpts the gate=3 mask automatically.
#
# held_out_opp default = candidate_ajay_1200 (deb is usually IN training, so not held-out).
# Eval masks default to the current locked design (gate3/floor0/NO forward-only) — override via env
# REINFORCE_MASKS if a run trains different masks (eval MUST match training).
# Only _sync is platform-specific; _eval operates on local synced files and is platform-independent.
set -uo pipefail
ROOT=/Users/saheb/home/kaggle-orbit-wars
PY="$ROOT/orbit_wars_rl/.venv/bin/python"
MARKER="$ROOT/gpu_run_artifacts/.active_run"
POLL="${POLL:-120}"

_still_active() { [ "$(head -1 "$MARKER" 2>/dev/null | awk '{print $1}')" = "$1" ]; }

stop_all() {
  pkill -f "run_watchers.sh _" 2>/dev/null && echo "stopped managed watchers" || echo "no managed watchers running"
  # also catch legacy ad-hoc per-run watchers from before this controller existed
  pkill -f "gpu_run_artifacts/.*sync_watcher.sh" 2>/dev/null && echo "stopped legacy sync_watcher" || true
  pkill -f "gpu_run_artifacts/.*_watch.sh" 2>/dev/null && echo "stopped legacy eval watchers" || true
}

# resolve a platform preset into the per-run .watch_env (sourced by the loops)
write_env() {  # write_env <run> <platform> <target> <opp>
  local RUN="$1" PLAT="$2" TGT="$3" OPP="$4" ENVF="$ROOT/gpu_run_artifacts/$1/.watch_env"
  mkdir -p "$ROOT/gpu_run_artifacts/$RUN"
  local RSYNC_SSH HOST RLOG RCKPT
  case "$PLAT" in
    jarvis)
      RSYNC_SSH="ssh -i $HOME/.ssh/jarvis-labs-key -o StrictHostKeyChecking=no -o ConnectTimeout=10"
      # checkpoints save to /home/checkpoints (cwd-relative); /home/orbit_wars_rl/checkpoints is a
      # SYMLINK to it. rsync -L on the top-level symlinked dir nests files under checkpoints/ and the
      # include/exclude filter then drops them → sync the REAL dir flat instead.
      HOST="root@$TGT"; RLOG="/home/"; RCKPT="/home/checkpoints/" ;;
    gcp)
      RSYNC_SSH="ssh -o ConnectTimeout=10"
      HOST="$TGT"; RLOG="~/orbit_wars_rl/"; RCKPT="~/orbit_wars_rl/checkpoints/" ;;
    custom)
      RSYNC_SSH="${RSYNC_SSH:?custom platform needs RSYNC_SSH env}"
      HOST="${HOST:?need HOST env}"; RLOG="${REMOTE_LOG_DIR:?need REMOTE_LOG_DIR}"; RCKPT="${REMOTE_CKPT_DIR:?need REMOTE_CKPT_DIR}" ;;
    *) echo "unknown platform '$PLAT' (jarvis|gcp|custom)"; exit 1 ;;
  esac
  cat > "$ENVF" <<EOF
RUN='$RUN'
MATCH='${MATCH:-$RUN}'
PLATFORM='$PLAT'
TARGET='$TGT'
RSYNC_SSH='$RSYNC_SSH'
HOST='$HOST'
REMOTE_LOG_DIR='$RLOG'
REMOTE_CKPT_DIR='$RCKPT'
OPP='$OPP'
REINFORCE_MASKS='${REINFORCE_MASKS:---reinforce-gate-min-planets 2 --reinforce-garrison-floor 0}'
EVAL_GATE_FROM_RUNNAME='${EVAL_GATE_FROM_RUNNAME:-}'
EOF
}

# ----- backgrounded loops (internal subcommands) ----------------------------
_sync() {  # _sync <run>
  local RUN="$1"; . "$ROOT/gpu_run_artifacts/$1/.watch_env"
  local DST="$ROOT/gpu_run_artifacts/$1"; mkdir -p "$DST/logs" "$DST/checkpoints"
  while _still_active "$RUN"; do
    rsync -az -e "$RSYNC_SSH" --include="train_gpu_phase1_${MATCH}*.log" --exclude='*' \
      "$HOST:$REMOTE_LOG_DIR" "$DST/logs/" 2>/dev/null
    rsync -azL -e "$RSYNC_SSH" \
      --include="torch_step_*${MATCH}*.pt" --include="pool_step_*${MATCH}*.pt" \
      --include="torch_best_${MATCH}*.pt" --exclude='*' \
      "$HOST:$REMOTE_CKPT_DIR" "$DST/checkpoints/" 2>/dev/null
    sleep "$POLL"
  done
  echo "[sync] $RUN no longer active — exiting" >> "$DST/watchers.log"
}

_eval() {  # _eval <run> [opp_override]   (platform-independent — local files only)
  local RUN="$1"; . "$ROOT/gpu_run_artifacts/$1/.watch_env"
  # An explicit opp arg overrides .watch_env's OPP, so a SECOND eval loop (e.g. Ajay) can run
  # alongside the primary (zach) under the same run/marker, each writing its own eval_<opp>.csv.
  local OPP="${2:-$OPP}"
  local DIR="$ROOT/gpu_run_artifacts/$1/checkpoints" LOGDIR="$ROOT/gpu_run_artifacts/$1/eval_logs"
  local OUT="$ROOT/gpu_run_artifacts/$1/eval_$(basename "${OPP%.py}" | sed 's/candidate_//').csv"
  mkdir -p "$LOGDIR"; cd "$ROOT"
  [ -f "$OUT" ] || echo "utc_time,step,win_rate,seat0_wr,seat1_wr,open_capatk_WON,mid_capatk_WON,peelrate_WON,planets100_WON,reinf_step_early,reinf_step_mid,reinf_dir_fwd,games,checkpoint" > "$OUT"
  while _still_active "$RUN"; do
    while IFS= read -r ckpt; do
      [ -n "$ckpt" ] || continue
      _still_active "$RUN" || break
      base=$(basename "$ckpt")
      grep -q ",$base$" "$OUT" 2>/dev/null && continue
      step=$(echo "$base" | sed -E 's/torch_step_([0-9]+)_.*/\1/')
      # elog is OPPONENT-specific: with two eval loops (zach + ajay) sharing this LOGDIR, an
      # opponent-agnostic name would collide/race on every checkpoint both loops eval.
      elog="$LOGDIR/eval_${base%.pt}__$(basename "${OPP%.py}" | sed 's/candidate_//').log"
      # Per-arm eval mask: for a gate A/B whose arms are named <MATCH><N> (gate2/gate3), eval each
      # ckpt with its OWN gate-min-planets (eval must match training). Opt-in via EVAL_GATE_FROM_RUNNAME.
      masks="$REINFORCE_MASKS"
      if [ -n "$EVAL_GATE_FROM_RUNNAME" ] && [[ "$base" =~ ${MATCH}([0-9]+) ]]; then
        masks=$(echo "$REINFORCE_MASKS" | sed -E "s/--reinforce-gate-min-planets [0-9]+/--reinforce-gate-min-planets ${BASH_REMATCH[1]}/")
      fi
      $PY orbit_wars_rl/eval.py --checkpoint "$ckpt" --opponent "$OPP" \
          --panel --target-decode $masks > "$elog" 2>&1 || true
      wr=$(grep -E "^Overall:"  "$elog" | tail -1 | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
      s0=$(grep -E "^  seat 0:" "$elog" | tail -1 | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
      s1=$(grep -E "^  seat 1:" "$elog" | tail -1 | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
      oc=$(grep -E "WON\(.*cap/atk open" "$elog" | tail -1 | sed -E 's#.*open<50 ([0-9.]+).*#\1#')
      mc=$(grep -E "WON\(.*cap/atk open" "$elog" | tail -1 | sed -E 's#.*mid50-100 ([0-9.]+).*#\1#')
      p100=$(grep -E "WON\(.*cap/atk open" "$elog" | tail -1 | sed -E 's#.*WON\([0-9]+g\) [0-9]+/[0-9]+/[0-9]+/([0-9]+).*#\1#')
      pr=$(grep -E "WON\(.*peel-rate" "$elog" | tail -1 | sed -E 's#.*WON\([0-9]+g\) peel-rate ([0-9.]+).*#\1#')
      rse=$(grep -E "reinf by step" "$elog" | tail -1 | sed -E 's#.*reinf by step +<50:([0-9.]+).*#\1#')
      rsm=$(grep -E "reinf by step" "$elog" | tail -1 | sed -E 's#.*reinf by step +<50:[0-9.]+ +50-100:([0-9.]+).*#\1#')
      rdf=$(grep -E "reinf direction" "$elog" | tail -1 | sed -E 's#.*fwd ([0-9]+)%.*#\1#')
      echo "$(date -u +%FT%TZ),${step},${wr:-ERR},${s0:-ERR},${s1:-ERR},${oc:-NA},${mc:-NA},${pr:-NA},${p100:-NA},${rse:-NA},${rsm:-NA},${rdf:-NA},256,${base}" >> "$OUT"
    done < <(find "$DIR" -maxdepth 1 -name "torch_step_*${MATCH}*.pt" -mmin +2 2>/dev/null | sort -rV)
    sleep "$POLL"
  done
  echo "[eval] $RUN no longer active — exiting" >> "$ROOT/gpu_run_artifacts/$RUN/watchers.log"
}

case "${1:-}" in
  start)
    RUN="${2:?usage: start <run> <platform> <target> [opp]}"; PLAT="${3:?need platform jarvis|gcp|custom}"
    TGT="${4:?need target (IP for jarvis, ssh-alias for gcp)}"; OPP="${5:-opponents/candidate_ajay_1200.py}"
    write_env "$RUN" "$PLAT" "$TGT" "$OPP"
    stop_all; sleep 1
    echo "$RUN $PLAT $TGT" > "$MARKER"
    nohup bash "$0" _sync "$RUN" >/dev/null 2>&1 &
    nohup bash "$0" _eval "$RUN" >/dev/null 2>&1 &
    echo "started watchers: run=$RUN platform=$PLAT target=$TGT held-out-eval=$OPP"
    ;;
  add-eval)
    # Add a SECOND held-out eval loop for another opponent to the CURRENTLY ACTIVE run,
    # WITHOUT tearing down the existing watchers (start does stop_all — that would kill the
    # in-flight panel). Self-terminates via the same .active_run marker, so no stale-watcher
    # risk. Usage: add-eval <run> <opp.py> [from-latest]
    #   from-latest: (re)create the opp CSV and seed every existing ckpt EXCEPT the newest as
    #   already-done, so only the latest + future checkpoints get evaled (slow panels like Ajay
    #   should not backfill the whole history).
    RUN="${2:?usage: add-eval <run> <opp> [from-latest]}"; OPP_ARG="${3:?need opponent path}"; MODE="${4:-}"
    _still_active "$RUN" || { echo "run '$RUN' is not the active run (marker='$(cat "$MARKER" 2>/dev/null)') — start it first"; exit 1; }
    . "$ROOT/gpu_run_artifacts/$RUN/.watch_env"   # for MATCH; NB this also sets OPP=primary —
    OPP="$OPP_ARG"                                  # so override with our arg AFTER sourcing.
    OUT="$ROOT/gpu_run_artifacts/$RUN/eval_$(basename "${OPP%.py}" | sed 's/candidate_//').csv"
    DIR="$ROOT/gpu_run_artifacts/$RUN/checkpoints"
    if [ "$MODE" = "from-latest" ]; then
      echo "utc_time,step,win_rate,seat0_wr,seat1_wr,open_capatk_WON,mid_capatk_WON,peelrate_WON,planets100_WON,reinf_step_early,reinf_step_mid,reinf_dir_fwd,games,checkpoint" > "$OUT"
      newest=$(find "$DIR" -maxdepth 1 -name "torch_step_*${MATCH}*.pt" 2>/dev/null | sort -V | tail -1)
      while IFS= read -r ck; do
        [ -n "$ck" ] || continue
        [ "$ck" = "$newest" ] && continue
        b=$(basename "$ck"); s=$(echo "$b" | sed -E 's/torch_step_([0-9]+)_.*/\1/')
        echo "SEED,${s},skip,,,,,,,,,,0,${b}" >> "$OUT"
      done < <(find "$DIR" -maxdepth 1 -name "torch_step_*${MATCH}*.pt" 2>/dev/null)
      echo "seeded $(basename "$OUT") to start from latest: $(basename "${newest:-<none>}")"
    fi
    nohup bash "$0" _eval "$RUN" "$OPP" >/dev/null 2>&1 &
    echo "added eval loop: run=$RUN opp=$OPP -> $(basename "$OUT")"
    ;;
  stop)    : > "$MARKER"; stop_all ;;
  status)
    echo "active run: $(cat "$MARKER" 2>/dev/null || echo none)"
    echo "live watcher procs:"; pgrep -af "run_watchers.sh _" || echo "  (none)"
    ;;
  _sync)  _sync "$2" ;;
  _eval)  _eval "$2" "${3:-}" ;;
  *) echo "usage: $0 {start <run> <platform> <target> [opp] | add-eval <run> <opp> [from-latest] | stop | status}"; exit 1 ;;
esac
