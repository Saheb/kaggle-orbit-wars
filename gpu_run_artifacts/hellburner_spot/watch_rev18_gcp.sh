#!/usr/bin/env bash
# Watcher for Rev16 GCP run — syncs checkpoints every 3 min.
# Edit INSTANCE and ZONE before running.
# Usage: nohup bash watch_rev18_gcp.sh > logs/watcher_rev18.log 2>&1 &

INSTANCE="orbit-wars-training"
ZONE="us-east1-b"
PROJECT="orbit-wars-rl"
SSH_ALIAS="${INSTANCE}.${ZONE}.${PROJECT}"
LOCAL_CKPT="$(cd "$(dirname "$0")" && pwd)/checkpoints"
LOCAL_LOGS="$(cd "$(dirname "$0")" && pwd)/logs"

mkdir -p "$LOCAL_CKPT" "$LOCAL_LOGS"

echo "[$(date)] Watcher started for rev18. Instance: $INSTANCE @ $ZONE"

while true; do
  rsync -az --no-progress \
    "${SSH_ALIAS}:~/orbit_wars_rl/checkpoints/" "$LOCAL_CKPT/" 2>/dev/null && \
    echo "[$(date)] checkpoints synced: $(ls "$LOCAL_CKPT"/*rev18*.pt 2>/dev/null | wc -l) rev18 files" || \
    echo "[$(date)] rsync failed (instance may have shut down)"

  rsync -az --no-progress --include='train_gpu_phase1_rev18_*.log' --exclude='*' \
    "${SSH_ALIAS}:~/orbit_wars_rl/" "$LOCAL_LOGS/" 2>/dev/null || true

  sleep 180
done
