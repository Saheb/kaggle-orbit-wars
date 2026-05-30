#!/usr/bin/env bash
# Watcher for Phase 1 run. Update INSTANCE_ID and PUB_IP after launch.
set +euo pipefail

INSTANCE_ID="i-0a5129fba17cf3de6"
PUB_IP="3.236.231.55"
KEY="$HOME/.ssh/samosa-key.pem"
ROOT="/Users/saheb/.codex/worktrees/296f/kaggle-orbit-wars"
ART="$ROOT/gpu_run_artifacts/hellburner_spot"

mkdir -p "$ART/checkpoints" "$ART/logs"

final_sync() {
  local ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$ts] FINAL SYNC — pulling all checkpoints and logs"
  rsync -az --partial --timeout=60 --contimeout=15 \
    -e "ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15" \
    ubuntu@"$PUB_IP":~/orbit_wars_rl/checkpoints/ \
    "$ART/checkpoints/" && \
    echo "[$ts] checkpoints: $(ls "$ART/checkpoints/"*.pt 2>/dev/null | wc -l | xargs) .pt files"
  rsync -az --partial --timeout=60 --contimeout=15 \
    --include='train_gpu_*.log' --exclude='*' \
    -e "ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15" \
    ubuntu@"$PUB_IP":~/orbit_wars_rl/ \
    "$ART/logs/"
  echo "[$ts] Final sync done."
}

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"

  INSTANCE_STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "unknown")

  if [[ "$INSTANCE_STATE" == "stopping" || "$INSTANCE_STATE" == "stopped" || "$INSTANCE_STATE" == "terminated" ]]; then
    echo "[$ts] Instance state=$INSTANCE_STATE — triggering final sync"
    final_sync
    echo "[$ts] Done. Exiting watcher."
    exit 0
  fi

  rsync -az --partial --timeout=30 --contimeout=10 \
    -e "ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=5" \
    ubuntu@"$PUB_IP":~/orbit_wars_rl/checkpoints/ \
    "$ART/checkpoints/"
  ckpt_rc=$?

  rsync -az --partial --timeout=30 --contimeout=10 \
    --include='train_gpu_*.log' --exclude='*' \
    -e "ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=5" \
    ubuntu@"$PUB_IP":~/orbit_wars_rl/ \
    "$ART/logs/"
  log_rc=$?

  log="$(ls -t "$ART/logs/train_gpu_phase1_rev7_"*.log "$ART/logs/train_gpu_phase1_"*.log 2>/dev/null | head -1)"
  last="$(grep -E '^iter|saved checkpoint|Training complete' "$log" 2>/dev/null | tail -1)"
  n_ckpt="$(find "$ART/checkpoints" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l | xargs)"
  echo "[$ts] state=$INSTANCE_STATE ckpt_rc=$ckpt_rc log_rc=$log_rc ckpts=$n_ckpt last: $last"

  sleep 60
done
