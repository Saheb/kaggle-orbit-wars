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
LOCAL_DIR="$HOME/home/kaggle-orbit-wars/gpu_run_artifacts"
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

    # Pull checkpoints (--mkpath ensures dir exists; size-only is fast)
    rsync -az --partial \
      -e "ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
      ${REMOTE_USER}@${PUB_IP}:~/orbit_wars_rl/checkpoints/ \
      "$LOCAL_DIR/checkpoints/" 2>/dev/null

    # Pull the training log
    rsync -az --partial \
      -e "ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
      ${REMOTE_USER}@${PUB_IP}:~/orbit_wars_rl/train_gpu.log \
      "$LOCAL_DIR/logs/" 2>/dev/null

    # Print last iter line
    last=$(grep -E "^iter|^  ★|Early stop|Training complete" "$LOCAL_DIR/logs/train_gpu.log" 2>/dev/null | tail -1)
    n_ckpt=$(ls "$LOCAL_DIR/checkpoints"/*.pt 2>/dev/null | wc -l | xargs)
    echo "[$TS] iter $ITER  ckpts_local=$n_ckpt  last: $last"

    # Stop condition: log contains "Training complete" or "Early stop:" near the end
    if grep -qE "Training complete|Early stop:" "$LOCAL_DIR/logs/train_gpu.log" 2>/dev/null; then
        echo "---"
        echo "Detected training end. Final sync, then terminating instance."
        sleep 5  # give the script a moment to flush final checkpoint
        rsync -az --partial \
          -e "ssh -i $KEY" \
          ${REMOTE_USER}@${PUB_IP}:~/orbit_wars_rl/checkpoints/ \
          "$LOCAL_DIR/checkpoints/"
        rsync -az --partial \
          -e "ssh -i $KEY" \
          ${REMOTE_USER}@${PUB_IP}:~/orbit_wars_rl/train_gpu.log \
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
