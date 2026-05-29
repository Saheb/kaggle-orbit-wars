#!/usr/bin/env bash
# ============================================================
# launch_gpu.sh — canonical GPU instance launcher for orbit-wars
# ALWAYS use this script to start training instances.
# It bakes in cost-safety defaults so no agent forgets them.
#
# Usage:
#   bash launch_gpu.sh [--spot] [--type g5.2xlarge] [--script run_remote_foo.sh]
#
# Defaults:
#   instance type : g5.2xlarge
#   on-demand     : yes (use --spot for spot)
#   shutdown-behavior : terminate  ← shuts down AND destroys instance (no idle billing)
#   key           : samosa-key
#   security group: from env or default
# ============================================================
set -euo pipefail

INSTANCE_TYPE="g5.2xlarge"
SPOT=false
RUN_SCRIPT=""
KEY_NAME="samosa-key"
AMI_ID="ami-0b9c99b766a895d68"     # PyTorch Deep Learning AMI (last used successfully)
SG_ID="${AWS_SG_ID:-sg-07f813c351bb5e011}"   # default VPC SG; override with AWS_SG_ID env var
SUBNET_ID="${AWS_SUBNET_ID:-}"     # optional; AWS picks AZ if empty

while [[ $# -gt 0 ]]; do
  case "$1" in
    --spot)        SPOT=true; shift ;;
    --type)        INSTANCE_TYPE="$2"; shift 2 ;;
    --script)      RUN_SCRIPT="$2"; shift 2 ;;
    --ami)         AMI_ID="$2"; shift 2 ;;
    --sg)          SG_ID="$2"; shift 2 ;;
    --key)         KEY_NAME="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Build user-data to run the training script on boot (optional)
USER_DATA=""
if [ -n "$RUN_SCRIPT" ]; then
  USER_DATA=$(base64 < "$RUN_SCRIPT")
fi

COMMON_ARGS=(
  --image-id "$AMI_ID"
  --instance-type "$INSTANCE_TYPE"
  --key-name "$KEY_NAME"
  --security-group-ids "$SG_ID"
  # CRITICAL: shutdown -h inside the instance terminates (not stops) it
  --instance-initiated-shutdown-behavior terminate
  --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=orbit-wars}]"
)

[ -n "$SUBNET_ID" ] && COMMON_ARGS+=(--subnet-id "$SUBNET_ID")
[ -n "$USER_DATA" ] && COMMON_ARGS+=(--user-data "$USER_DATA")

if $SPOT; then
  echo "Launching SPOT $INSTANCE_TYPE..."
  RESULT=$(aws ec2 run-instances "${COMMON_ARGS[@]}" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
    --count 1 \
    --query 'Instances[0].[InstanceId,State.Name]' \
    --output text)
else
  echo "Launching ON-DEMAND $INSTANCE_TYPE..."
  RESULT=$(aws ec2 run-instances "${COMMON_ARGS[@]}" \
    --count 1 \
    --query 'Instances[0].[InstanceId,State.Name]' \
    --output text)
fi

INSTANCE_ID=$(echo "$RESULT" | awk '{print $1}')
echo "Launched: $RESULT"
echo "Instance ID: $INSTANCE_ID"
echo ""
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

PUB_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "Ready: $INSTANCE_ID @ $PUB_IP"
echo "SSH:   ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUB_IP"
echo ""
echo "REMINDER: instance will TERMINATE (not stop) on shutdown — all local data lost."
echo "Sync checkpoints before shutdown or rely on the watcher script."
