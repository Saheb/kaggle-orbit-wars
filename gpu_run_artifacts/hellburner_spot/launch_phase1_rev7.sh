#!/usr/bin/env bash
# Launch Phase 1 Rev7 — no IL anchor, short-run chain approach.
#
# Resume: torch_step_6094848_20260529_160908.pt (rev5 6M peak = 38.7% HB, 54.3% Zach, 61.7% Suneet)
# Kill manually at fire[0] decline signal or 3M steps hard cap.
#
# Usage: bash launch_phase1_rev7.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ART="$ROOT/gpu_run_artifacts/hellburner_spot"
KEY="$HOME/.ssh/samosa-key.pem"

SEED_CKPT="$ART/checkpoints/torch_step_6094848_20260529_160908.pt"
if [ ! -f "$SEED_CKPT" ]; then
  echo "ERROR: seed checkpoint not found: $SEED_CKPT"
  exit 1
fi

echo "=== Checking for existing running instances ==="
aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=orbit-wars" "Name=instance-state-name,Values=running,stopped" \
  --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceType,PublicIpAddress]' \
  --output table

echo ""
echo "=== Launching on-demand g5.2xlarge ==="
LAUNCH_OUT=$(bash "$ROOT/gpu_run_artifacts/launch_gpu.sh")
echo "$LAUNCH_OUT"
INSTANCE_ID=$(echo "$LAUNCH_OUT" | grep "^Instance ID:" | awk '{print $3}')
PUB_IP=$(echo "$LAUNCH_OUT"     | grep "^Ready:"       | awk '{print $4}')
echo "Instance ID: $INSTANCE_ID"
echo "Public IP:   $PUB_IP"

echo "Waiting 60s for SSH to become available..."
sleep 60

SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=30"

echo "=== Uploading repo ==="
rsync -az --exclude='.git' \
  --exclude='gpu_run_artifacts/hellburner_spot/checkpoints' \
  --exclude='gpu_run_artifacts/hellburner_spot/panels' \
  --exclude='gpu_run_artifacts/hellburner_spot/logs' \
  -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$ROOT/" ubuntu@"$PUB_IP":~/orbit_wars_rl/

echo "=== Uploading seed checkpoint as phase1_resume.pt ==="
$SSH ubuntu@"$PUB_IP" "mkdir -p ~/orbit_wars_rl/seed_checkpoints"
scp -i "$KEY" -o StrictHostKeyChecking=no \
  "$SEED_CKPT" ubuntu@"$PUB_IP":~/orbit_wars_rl/seed_checkpoints/phase1_resume.pt

echo "=== Starting training in tmux ==="
$SSH ubuntu@"$PUB_IP" "
  cd ~/orbit_wars_rl
  tmux new-session -d -s training 'bash gpu_run_artifacts/hellburner_spot/run_remote_phase1_rev7.sh'
  sleep 5
  tmux list-sessions
  echo '--- startup ---'
  ls -t train_gpu_phase1_rev7_*.log 2>/dev/null | head -1 | xargs tail -3 2>/dev/null || echo '(starting...)'
"

echo ""
echo "=== Rev7 launched ==="
echo "Instance: $INSTANCE_ID @ $PUB_IP"
echo "Resume: torch_step_6094848_20260529_160908.pt (HB=38.7%, Zach=54.3%, Suneet=61.7%)"
echo "Delta: IL anchor removed (no --il-lambda)"
echo ""
echo "KILL SIGNALS — watch fire[0] at each 1M checkpoint:"
echo "  fire[0] declining 3 consecutive → kill"
echo "  fire[0] < 0.25 at any checkpoint → kill"
echo "  Hard cap: kill at 3M steps"
echo ""
echo "Monitor:"
echo "  INSTANCE_ID=$INSTANCE_ID PUB_IP=$PUB_IP"
echo "  bash $ART/watch_phase1.sh"
