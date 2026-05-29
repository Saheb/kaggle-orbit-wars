#!/usr/bin/env bash
# ============================================================
# launch_gpu_gcp.sh — canonical GCP GPU instance launcher
# Mirrors launch_gpu.sh but for Google Cloud (g2-standard-8, L4 GPU).
#
# Usage:
#   bash launch_gpu_gcp.sh [--zone us-central1-b] [--name orbit-wars-training]
#
# Prerequisites:
#   - gcloud auth: gcloud auth login && gcloud config set project orbit-wars-rl
#   - GPUS_ALL_REGIONS quota > 0 (request at console.cloud.google.com/iam-admin/quotas)
#   - G2_CPUS quota allocated (request same page, search G2_CPUS, region us-central1)
#
# Key differences vs AWS:
#   - Zone: us-central1-b has L4 capacity; us-central1-a is often OOS
#   - GPU: NVIDIA L4 (23GB) ≈ A10G, ~68% SPS vs AWS g5.2xlarge
#   - Cost: ~$1.13/hr vs AWS $1.21/hr, but slower → AWS better $/step
#   - Auth: uses OS Login (gcloud manages SSH keys); use gcloud compute ssh
#   - orbit_wars env: NOT in PyTorch base image — run setup/install_orbit_wars.sh
#   - No --terminate-on-done support without extra setup; manually terminate
#
# NEVER leave instance running when done — terminate with:
#   gcloud compute instances delete <name> --zone=<zone>
# ============================================================
set -euo pipefail

ZONE="us-central1-b"
INSTANCE_NAME="orbit-wars-training"
PROJECT="orbit-wars-rl"
MACHINE_TYPE="g2-standard-8"
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"
DISK_SIZE="200GB"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zone)   ZONE="$2"; shift 2 ;;
    --name)   INSTANCE_NAME="$2"; shift 2 ;;
    --type)   MACHINE_TYPE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Checking for existing GCP instances ==="
gcloud compute instances list --project="$PROJECT" \
  --filter="status=RUNNING OR status=STAGING" \
  --format="table(name,zone,machineType,networkInterfaces[0].accessConfigs[0].natIP,status)" 2>/dev/null || true

echo ""
echo "=== Launching $MACHINE_TYPE in $ZONE ==="
gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$DISK_SIZE" \
  --no-restart-on-failure \
  --maintenance-policy=TERMINATE \
  --tags=orbit-wars

GCP_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
  --zone="$ZONE" --project="$PROJECT" \
  --format="get(networkInterfaces[0].accessConfigs[0].natIP)")
echo "Instance: $INSTANCE_NAME @ $GCP_IP"

echo ""
echo "=== Waiting for SSH ==="
until gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT" \
  --command="echo ready" --ssh-flag="-o StrictHostKeyChecking=no -o ConnectTimeout=10" 2>/dev/null; do
  sleep 5
done

echo ""
echo "=== Adding instance to SSH config (enables rsync) ==="
gcloud compute config-ssh --project="$PROJECT" 2>/dev/null
SSH_ALIAS="$INSTANCE_NAME.$ZONE.$PROJECT"

echo ""
echo "=== Uploading code via rsync ==="
rsync -az \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='gpu_run_artifacts/hellburner_spot/checkpoints' \
  --exclude='gpu_run_artifacts/hellburner_spot/panels' \
  --exclude='gpu_run_artifacts/hellburner_spot/logs' \
  --exclude='gpu_run_artifacts/logs' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  "$ROOT/" "${SSH_ALIAS}:~/orbit_wars_rl/"
echo "Code uploaded"

echo ""
echo "=== Installing orbit_wars env ==="
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT" \
  --ssh-flag="-o StrictHostKeyChecking=no" \
  --command="cd ~/orbit_wars_rl && pip install -q kaggle-environments && bash setup/install_orbit_wars.sh"

echo ""
echo "=== Ready to train ==="
echo "Instance : $INSTANCE_NAME @ $GCP_IP"
echo "Zone     : $ZONE"
echo "SSH      : gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "rsync    : rsync -az ... ${SSH_ALIAS}:~/orbit_wars_rl/"
echo ""
echo "Upload seed checkpoint then start training:"
echo "  gcloud compute scp <checkpoint.pt> $INSTANCE_NAME:~/orbit_wars_rl/seed_checkpoints/phase1_resume.pt --zone=$ZONE"
echo "  gcloud compute scp <bc_warmstart.pt> $INSTANCE_NAME:~/orbit_wars_rl/seed_checkpoints/bc_phase1_warmstart.pt --zone=$ZONE"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- 'tmux new-session -d -s training \"bash /tmp/start_training.sh\"'"
echo ""
echo "When done, TERMINATE (not stop):"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
