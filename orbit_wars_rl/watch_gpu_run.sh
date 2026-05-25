#!/usr/bin/env bash
# Polls the GPU training instance, rsyncs checkpoints + logs to local every
# 3 min. When training ends, does a final sync and terminates the instance.
#
# Usage:
#   ./watch_gpu_run.sh <instance-id> <public-ip>

set -u

INSTANCE_ID="${1:-i-09f7f32c2e044ea05}"
PUB_IP="${2:-52.200.49.179}"
KEY="$HOME/.ssh/samosa-key.pem"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_DIR="${LOCAL_DIR:-$REPO_ROOT/gpu_run_artifacts}"
REMOTE_USER="ubuntu"

mkdir -p "$LOCAL_DIR/checkpoints" "$LOCAL_DIR/logs"

echo "Watching instance $INSTANCE_ID at $PUB_IP"
echo "Local sync target: $LOCAL_DIR"
echo "Polling every 3 min. Will terminate on detection of 'Training complete' or 'Early stop'."
echo "---"

ITER=0
while true; do
    ITER=$((ITER + 1))
    TS=$(date '+%Y-%m-%d %H:%M:%S')

    # Pull checkpoints — capture errors instead of swallowing them
    ckpt_err=$(rsync -az --partial \
      -e "ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=5" \
      ${REMOTE_USER}@${PUB_IP}:~/orbit_wars_rl/checkpoints/ \
      "$LOCAL_DIR/checkpoints/" 2>&1 >/dev/null)
    ckpt_rc=$?

    # Pull the training log — matches both legacy train_gpu.log and
    # timestamped train_gpu_YYYYMMDD_HHMMSS.log from newer runs.
    log_err=$(rsync -az --partial \
      -e "ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=5" \
      --include='train_gpu*.log' --exclude='*' \
      ${REMOTE_USER}@${PUB_IP}:~/orbit_wars_rl/ \
      "$LOCAL_DIR/logs/" 2>&1 >/dev/null)
    log_rc=$?

    # Pick the most-recently-modified local log file for status checks.
    ACTIVE_LOG=$(ls -t "$LOCAL_DIR/logs"/train_gpu*.log 2>/dev/null | head -1)

    # Print last iter line
    last=$(grep -E "^iter|^  ★|Early stop|Training complete" "$ACTIVE_LOG" 2>/dev/null | tail -1)
    n_ckpt=$(ls "$LOCAL_DIR/checkpoints"/*.pt 2>/dev/null | wc -l | xargs)
    if [ $ckpt_rc -ne 0 ] || [ $log_rc -ne 0 ]; then
        # rsync failed — surface the error loudly so we notice silent breakages
        # (e.g. SSH blocked by security group when local IP changes)
        echo "[$TS] iter $ITER  ⚠ SYNC FAILED  ckpt_rc=$ckpt_rc  log_rc=$log_rc"
        echo "    ckpt_err: $ckpt_err"
        echo "    log_err:  $log_err"
    else
        echo "[$TS] iter $ITER  ckpts_local=$n_ckpt  last: $last"
    fi

    # Stop condition: log contains "Training complete" or "Early stop:" near the end
    if grep -qE "Training complete|Early stop:" "$ACTIVE_LOG" 2>/dev/null; then
        echo "---"
        echo "Detected training end. Final sync, then terminating instance."
        sleep 5  # give the script a moment to flush final checkpoint
        rsync -az --partial \
          -e "ssh -i $KEY" \
          ${REMOTE_USER}@${PUB_IP}:~/orbit_wars_rl/checkpoints/ \
          "$LOCAL_DIR/checkpoints/"
        rsync -az --partial \
          -e "ssh -i $KEY" \
          --include='train_gpu*.log' --exclude='*' \
          ${REMOTE_USER}@${PUB_IP}:~/orbit_wars_rl/ \
          "$LOCAL_DIR/logs/"
        echo "Final sync done. Terminating $INSTANCE_ID..."
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" \
          --query 'TerminatingInstances[0].[InstanceId,CurrentState.Name]' \
          --output text
        echo "Instance terminated. Artifacts in $LOCAL_DIR"
        exit 0
    fi

    sleep 180   # 3 minutes
done
