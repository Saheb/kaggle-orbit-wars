#!/usr/bin/env bash
# Launch Phase 1 blitz — replicate 141208 HB-blitz on Phase 1 model.
#
# Resume: torch_step_6094848_20260529_160908.pt (rev5 6M peak = 38.7% HB, 54.3% Zach, 61.7% Suneet)
# Single blitz window: 2M steps, checkpoint every 500K
# No BC warmstart upload needed (IL removed from blitz script).
#
# Usage: bash launch_phase1_blitz.sh
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
INSTANCE_JSON=$(bash "$ROOT/gpu_run_artifacts/launch_gpu.sh")
INSTANCE_ID=$(echo "$INSTANCE_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['Instances'][0]['InstanceId'])")
echo "Instance ID: $INSTANCE_ID"

echo "Waiting for instance to reach running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
PUB_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "Public IP: $PUB_IP"

echo "Waiting 60s for SSH to become available..."
sleep 60

SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=30"

echo "=== Uploading repo ==="
rsync -az --exclude='.git' --exclude='gpu_run_artifacts/hellburner_spot/checkpoints' \
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
  tmux new-session -d -s training 'bash gpu_run_artifacts/hellburner_spot/run_remote_phase1_blitz.sh 2>&1 | tee /tmp/training_startup.log'
  sleep 3
  tmux list-sessions
  echo '--- last 5 lines of startup log ---'
  tail -5 /tmp/training_startup.log 2>/dev/null || true
"

echo ""
echo "=== Blitz launched ==="
echo "Instance: $INSTANCE_ID @ $PUB_IP"
echo "Resume: torch_step_6094848_20260529_160908.pt (rev5 6M peak — HB=38.7%, Zach=54.3%, Suneet=61.7%)"
echo "Config: pool-fraction=0.9, pool-external-fraction=1.0, lr=3e-5, no IL, 2M steps"
echo "Checkpoints: every 500K steps"
echo ""
echo "Monitor with:"
echo "  INSTANCE_ID=$INSTANCE_ID"
echo "  PUB_IP=$PUB_IP"
echo "  bash $ART/watch_phase1.sh"
