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
# orbit_wars_rl directory
rsync -az --exclude '__pycache__' --exclude '*.pyc' --exclude 'checkpoints/' \
  -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$ROOT/orbit_wars_rl/" \
  ubuntu@"$PUB_IP":~/orbit_wars_rl/

# Candidate files (opponents)
rsync -az \
  -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$ROOT/candidate_hellburner.py" \
  "$ROOT/candidate_zach_public.py" \
  ubuntu@"$PUB_IP":~/

echo ""
echo "=== Step 4: Upload BC warmstart ==="
ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@"$PUB_IP" 'mkdir -p ~/seed_checkpoints'
rsync -az \
  -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$ROOT/checkpoints/bc_phase1_warmstart.pt" \
  ubuntu@"$PUB_IP":~/seed_checkpoints/bc_phase1_warmstart.pt
echo "BC warmstart uploaded."

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
sed -i '' \
  "s|INSTANCE_ID=\"REPLACE_AFTER_LAUNCH\"|INSTANCE_ID=\"$INSTANCE_ID\"|" \
  "$WATCH"
sed -i '' \
  "s|PUB_IP=\"REPLACE_AFTER_LAUNCH\"|PUB_IP=\"$PUB_IP\"|" \
  "$WATCH"
echo "watch_phase1.sh updated with INSTANCE_ID=$INSTANCE_ID PUB_IP=$PUB_IP"

echo ""
echo "==================================================================="
echo "Phase 1 launch complete!"
echo "  Instance : $INSTANCE_ID @ $PUB_IP"
echo "  Training : screen session 'phase1' on remote"
echo "  Watcher  : bash $ART/watch_phase1.sh &"
echo "  Attach   : ssh -i $KEY ubuntu@$PUB_IP -t 'screen -r phase1'"
echo "==================================================================="
