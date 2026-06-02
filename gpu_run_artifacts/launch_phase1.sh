#!/usr/bin/env bash
# ============================================================
# launch_phase1.sh — one-shot Phase 1 GPU launch
#
# Steps:
#   1. Launch instance via launch_gpu.sh (terminate-on-done baked in)
#   2. Wait for SSH readiness
#   3. rsync orbit_wars_rl code + candidate files
#   4. Upload bc_phase1_warmstart.pt to ~/seed_checkpoints/
#   5. Start training in a screen session
#   6. Print updated watch_phase1.sh sed command
#
# Usage:
#   bash gpu_run_artifacts/launch_phase1.sh [--spot]
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ART="$ROOT/gpu_run_artifacts/hellburner_spot"
KEY="$HOME/.ssh/samosa-key.pem"
EXTRA_ARGS=()

for arg in "$@"; do
  EXTRA_ARGS+=("$arg")
done

# Ensure array is non-empty for bash -u compatibility
[ ${#EXTRA_ARGS[@]} -eq 0 ] && LAUNCH_CMD=(bash "$ROOT/gpu_run_artifacts/launch_gpu.sh") || LAUNCH_CMD=(bash "$ROOT/gpu_run_artifacts/launch_gpu.sh" "${EXTRA_ARGS[@]}")

echo "=== Step 1: Launch GPU instance ==="
LAUNCH_OUT=$("${LAUNCH_CMD[@]}" 2>&1)
echo "$LAUNCH_OUT"

INSTANCE_ID=$(echo "$LAUNCH_OUT" | grep "^Instance ID:" | awk '{print $3}')
PUB_IP=$(echo "$LAUNCH_OUT" | grep "^Ready:" | awk '{print $3}' | cut -d'@' -f1)

# Fallback parse
if [ -z "$INSTANCE_ID" ]; then
  INSTANCE_ID=$(echo "$LAUNCH_OUT" | grep -oE 'i-[0-9a-f]+' | head -1)
fi
if [ -z "$PUB_IP" ]; then
  PUB_IP=$(echo "$LAUNCH_OUT" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | tail -1)
fi

echo ""
echo "INSTANCE_ID=$INSTANCE_ID  PUB_IP=$PUB_IP"

echo ""
echo "=== Step 2: Wait for SSH ==="
for i in $(seq 1 30); do
  if ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
       ubuntu@"$PUB_IP" 'echo ssh_ok' 2>/dev/null | grep -q ssh_ok; then
    echo "SSH ready after $((i * 10))s"
    break
  fi
  echo "  attempt $i/30 — waiting 10s..."
  sleep 10
done

echo ""
echo "=== Step 3: rsync code + candidates ==="
RSYNC_EXCLUDES=(
  --exclude='.git' --exclude='.venv' --exclude='.claude'
  --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pt'
  --exclude='archive' --exclude='leader-replays' --exclude='kernels'
  --exclude='checkpoints' --exclude='seed_checkpoints'
  --exclude='gpu_run_artifacts/hellburner_spot/checkpoints'
  --exclude='gpu_run_artifacts/hellburner_spot/panels'
  --exclude='gpu_run_artifacts/hellburner_spot/logs'
  --exclude='orbit_wars_rl/episode_data'
  --exclude='orbit_wars_rl/replays' --exclude='orbit_wars_rl/replays_4p_heuristic'
  --exclude='orbit_wars_rl/episode_index' --exclude='orbit_wars_rl/*.pkl'
  --exclude='submission_hybrid*.py' --exclude='submission_rev*.py'
  --exclude='submission_agent.py'
)

# ⚠️ Pre-flight size check — abort if > 50MB
MAX_MB=50
TRANSFER_MB=$(rsync --dry-run --stats "${RSYNC_EXCLUDES[@]}" "$ROOT/" /tmp/dummy/ 2>/dev/null | \
  awk '/Total transferred file size/{gsub(/,/,"",$NF); printf "%d", $NF/1024/1024}')
TRANSFER_MB=${TRANSFER_MB:-0}
echo "Estimated transfer: ~${TRANSFER_MB}MB"
if [ "$TRANSFER_MB" -gt "$MAX_MB" ]; then
  echo ""
  echo "❌ ERROR: rsync transfer size ${TRANSFER_MB}MB exceeds ${MAX_MB}MB limit."
  echo "   Fix RSYNC_EXCLUDES in this script before launching."
  echo "   Top offenders:"
  du -sh "$ROOT"/* 2>/dev/null | sort -rh | head -10
  exit 1
fi

rsync -az "${RSYNC_EXCLUDES[@]}" \
  -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$ROOT/" ubuntu@"$PUB_IP":~/orbit_wars_rl/
echo "Code uploaded (~${TRANSFER_MB}MB)"

echo ""
echo "=== Step 4: Upload BC warmstart ==="
ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@"$PUB_IP" 'mkdir -p ~/seed_checkpoints'
rsync -az \
  -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$ROOT/checkpoints/bc_phase1_warmstart.pt" \
  ubuntu@"$PUB_IP":~/seed_checkpoints/bc_phase1_warmstart.pt
# Also copy as phase1_resume.pt (the name --resume expects) so both
# --resume and --il-ref can reference their respective roles from the same file.
ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@"$PUB_IP" \
  'cp ~/seed_checkpoints/bc_phase1_warmstart.pt ~/seed_checkpoints/phase1_resume.pt'
echo "BC warmstart uploaded (bc_phase1_warmstart.pt + phase1_resume.pt)."

echo ""
echo "=== Step 5: Upload + start training in screen ==="
rsync -az \
  -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$ART/run_remote_phase1.sh" \
  ubuntu@"$PUB_IP":~/run_remote_phase1.sh
ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@"$PUB_IP" \
  'chmod +x ~/run_remote_phase1.sh && screen -dmS phase1 bash ~/run_remote_phase1.sh'
echo "Training started in screen session 'phase1'."
echo "To attach: ssh -i $KEY ubuntu@$PUB_IP -t 'screen -r phase1'"

echo ""
echo "=== Step 6: Update watch_phase1.sh ==="
WATCH="$ART/watch_phase1.sh"
# Replace any existing INSTANCE_ID / PUB_IP value (not just the initial placeholder)
sed -i '' "s|INSTANCE_ID=\"[^\"]*\"|INSTANCE_ID=\"$INSTANCE_ID\"|" "$WATCH"
sed -i '' "s|PUB_IP=\"[^\"]*\"|PUB_IP=\"$PUB_IP\"|"               "$WATCH"
echo "watch_phase1.sh updated with INSTANCE_ID=$INSTANCE_ID PUB_IP=$PUB_IP"

echo ""
echo "=== Starting local checkpoint watcher ==="
WATCHER_LOG="$ART/logs/watcher_aws_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ART/logs" "$ART/checkpoints"
nohup bash -c "
  while true; do
    rsync -az -e 'ssh -i $KEY -o StrictHostKeyChecking=no' \
      ubuntu@${PUB_IP}:~/orbit_wars_rl/checkpoints/ '$ART/checkpoints/' 2>/dev/null
    rsync -az -e 'ssh -i $KEY -o StrictHostKeyChecking=no' \
      --include='train_gpu_phase1_*.log' --exclude='*' \
      ubuntu@${PUB_IP}:~/orbit_wars_rl/ '$ART/logs/' 2>/dev/null
    sleep 180
  done
" > "$WATCHER_LOG" 2>&1 &
echo "Watcher PID: $!  log: $WATCHER_LOG"

echo ""
echo "==================================================================="
echo "Phase 1 launch complete!"
echo "  Instance : $INSTANCE_ID @ $PUB_IP"
echo "  Training : screen session 'phase1' on remote"
echo "  Watcher  : running (PID above), syncing every 3 min → $ART/logs/"
echo "  Attach   : ssh -i $KEY ubuntu@$PUB_IP -t 'screen -r phase1'"
echo "==================================================================="
