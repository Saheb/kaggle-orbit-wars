#!/usr/bin/env bash
# ============================================================
# launch_gpu_gcp.sh — canonical GCP GPU instance launcher
# Mirrors launch_gpu.sh but for Google Cloud (g2-standard-8, L4 GPU).
#
# Usage:
#   bash launch_gpu_gcp.sh [--zone asia-south1-b] [--name orbit-wars-training] [--run rev55]
#   --run revX  -> syncs checkpoints/log to gpu_run_artifacts/revX/ (top-level, like Jarvis)
#   L4 is often STOCKOUT in us zones; default zone is asia-south1-b. Also try
#   asia-south1-c / europe-west4-a / us-west1-a — the create error names a free zone.
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

# Default zone = asia-south1-b: L4 (g2) capacity in us-central1-b/-c is frequently
# STOCKOUT. When this zone is also exhausted, DON'T just retry us zones — L4 is often
# available in NON-US regions: try --zone asia-south1-c, europe-west4-a/b/c,
# us-west1-a/b, us-east4-a. The create error message names a zone with capacity.
ZONE="asia-south1-b"
INSTANCE_NAME="orbit-wars-training"
PROJECT="orbit-wars-rl"
MACHINE_TYPE="g2-standard-8"
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"
DISK_SIZE="200GB"
RUN=""   # artifact/watcher subdir under gpu_run_artifacts/ (e.g. rev55). Defaults below.

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zone)    ZONE="$2"; shift 2 ;;
    --name)    INSTANCE_NAME="$2"; shift 2 ;;
    --type)    MACHINE_TYPE="$2"; shift 2 ;;
    --run)     RUN="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Per-run artifact dir (top-level revX, mirrors the Jarvis layout) instead of the
# legacy shared hellburner_spot/. Pass --run revX; defaults to a timestamped dir.
[[ -z "$RUN" ]] && RUN="gcp_$(date +%Y%m%d_%H%M%S)"

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
# Only send what the training instance needs: RL code, opponents, setup, docs.
# Explicitly exclude everything large or irrelevant — wrong excludes cost ~1.6GB per launch.

RSYNC_EXCLUDES=(
  --exclude='.git'
  --exclude='.venv'
  --exclude='orbit_wars_rl/.venv'
  --exclude='.claude'
  --exclude='**/__pycache__'
  --exclude='**/*.pyc'
  --exclude='**/*.pt'
  --exclude='archive'
  --exclude='leader-replays'
  --exclude='kernels'
  --exclude='orbit-wars-data'
  --exclude='**/checkpoints'
  --exclude='seed_checkpoints'
  --exclude='gpu_run_artifacts'
  --exclude='gpu_pool_artifacts'
  --exclude='**/gpu_pool_artifacts'
  --exclude='orbit_wars_rl/episode_data'
  --exclude='orbit_wars_rl/replays*'
  --exclude='orbit_wars_rl/episode_index'
  --exclude='pytorch_docs'
  --exclude='**/*.pkl'
  --exclude='**/*.npz'
  --exclude='submission_*.py'
  --exclude='main_submitted.py'
)

# ⚠️ Pre-flight size check — fail fast if excludes are wrong
MAX_MB=100
DRY_OUTPUT=$(rsync --dry-run --stats "${RSYNC_EXCLUDES[@]}" "$ROOT/" /tmp/dummy_dest/ 2>/dev/null || \
             rsync --dry-run --stats "${RSYNC_EXCLUDES[@]}" "$ROOT/" localhost:/tmp/dummy_dest/ 2>&1 || true)
TRANSFER_BYTES=$(rsync -azn --stats "${RSYNC_EXCLUDES[@]}" "$ROOT/" /tmp/rsync_size_check/ 2>/dev/null | \
  awk '/Total transferred file size/ {gsub(/,/,"",$(NF-1)); print $(NF-1)}')
if [ -z "$TRANSFER_BYTES" ]; then
  # Fallback: estimate from local du
  TRANSFER_BYTES=$(du -sb "$ROOT" --exclude='.git' --exclude='.venv' --exclude='archive' \
    --exclude='leader-replays' --exclude='checkpoints' --exclude='gpu_run_artifacts' 2>/dev/null | awk '{print $1}' || echo 0)
fi
TRANSFER_MB=$(( ${TRANSFER_BYTES:-0} / 1024 / 1024 ))
echo "Estimated transfer size: ~${TRANSFER_MB}MB"
if [ "$TRANSFER_MB" -gt "$MAX_MB" ]; then
  echo ""
  echo "❌ ERROR: rsync transfer size ${TRANSFER_MB}MB exceeds ${MAX_MB}MB limit."
  echo "   Large files not excluded — check for new .pkl, replay, or submission files."
  echo "   Top offenders:"
  du -sh "$ROOT"/* "$ROOT"/orbit_wars_rl/*.pkl "$ROOT"/orbit_wars_rl/episode_data \
    "$ROOT"/orbit_wars_rl/replays "$ROOT"/leader-replays 2>/dev/null | sort -rh | head -10
  echo ""
  echo "   Fix: add missing paths to RSYNC_EXCLUDES in this script, then re-run."
  exit 1
fi

rsync -az "${RSYNC_EXCLUDES[@]}" "$ROOT/" "${SSH_ALIAS}:~/orbit_wars_rl/"
echo "Code uploaded (~${TRANSFER_MB}MB)"

# Bundled eval opponents carry gitignored model assets, so the broad *.pt/*.npz excludes above
# intentionally skip them. Sync the declared assets explicitly and compare hashes; otherwise a
# wrapper can load without its weights and a remote panel can silently count no-op games as wins.
EVAL_ASSETS=(
  opponents/ender_bundle/checkpoint_2p.pt
  opponents/ender_bundle/checkpoint_4p.pt
  opponents/yijie_bundle/inference_2p/weights/weights_2p_u53000.npz
  opponents/yijie_bundle/inference_2p/weights/weights_2p_u55000.npz
)
for asset in "${EVAL_ASSETS[@]}"; do
  if [[ ! -f "$ROOT/$asset" ]]; then
    echo "❌ ERROR: required bundled opponent asset missing locally: $asset"
    exit 1
  fi
done
printf '%s\n' "${EVAL_ASSETS[@]}" | \
  rsync -azL --files-from=- "$ROOT/" "${SSH_ALIAS}:~/orbit_wars_rl/"
LOCAL_ASSET_HASHES=$(cd "$ROOT" && for asset in "${EVAL_ASSETS[@]}"; do
  printf '%s  %s\n' "$(shasum -a 256 "$asset" | awk '{print $1}')" "$asset"
done)
REMOTE_ASSET_HASHES=$(ssh -o StrictHostKeyChecking=no "${SSH_ALIAS}" \
  "cd ~/orbit_wars_rl && sha256sum ${EVAL_ASSETS[*]}")
if [[ "$LOCAL_ASSET_HASHES" != "$REMOTE_ASSET_HASHES" ]]; then
  echo "❌ ERROR: bundled opponent asset hash mismatch after sync"
  exit 1
fi
echo "✓ Bundled opponent assets synced and checksummed."

# Verify the sync actually landed — rsync can drop mid-transfer on flaky SSH
echo "Verifying sync..."
for key_file in orbit_wars_rl/train_torch.py orbit_wars_rl/torch_env.py orbit_wars_rl/eval.py; do
  if ! ssh -o StrictHostKeyChecking=no "${SSH_ALIAS}" "test -f ~/orbit_wars_rl/$key_file" 2>/dev/null; then
    echo "❌ SYNC FAILED: $key_file missing on remote. Re-running rsync..."
    rsync -az "${RSYNC_EXCLUDES[@]}" "$ROOT/" "${SSH_ALIAS}:~/orbit_wars_rl/"
    break
  fi
done
echo "✓ Sync verified."

# Clear stale .pyc cache so new code is always used, not old bytecode
ssh -o StrictHostKeyChecking=no "${SSH_ALIAS}" \
  "find ~/orbit_wars_rl -name '*.pyc' -delete 2>/dev/null; \
   find ~/orbit_wars_rl -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null; true"
echo "✓ .pyc cache cleared."

echo ""
echo "=== Installing orbit_wars env ==="
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT" \
  --ssh-flag="-o StrictHostKeyChecking=no" \
  --command="cd ~/orbit_wars_rl && pip install -q kaggle-environments wandb && bash setup/install_orbit_wars.sh"

echo ""
echo "=== W&B: push credential so runs log natively (drop --no-wandb to use) ==="
# Extract ONLY the api.wandb.ai block from the local ~/.netrc and append it to the instance's
# netrc — avoids copying the whole netrc (other secrets). Runs log ~60 metrics when wandb is
# installed + authed AND --no-wandb is NOT passed.
if grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
  awk '/machine api.wandb.ai/{p=1} p{print} p&&/password/{exit}' "$HOME/.netrc" | \
    ssh -o StrictHostKeyChecking=no "${SSH_ALIAS}" "cat >> ~/.netrc && chmod 600 ~/.netrc" \
    && echo "✓ W&B credential pushed" || echo "⚠ W&B credential push failed (runs fall back to text log)"
else
  echo "⚠ no api.wandb.ai in local ~/.netrc — run 'wandb login' locally first for native logging"
fi

echo ""
echo "=== Starting local checkpoint watcher ==="
ART="$ROOT/gpu_run_artifacts/$RUN"
mkdir -p "$ART/checkpoints" "$ART/logs"
WATCHER_LOG="$ART/logs/watcher_gcp_$(date +%Y%m%d_%H%M%S).log"
nohup bash -c "
  while true; do
    rsync -az '${SSH_ALIAS}':~/orbit_wars_rl/checkpoints/ '$ART/checkpoints/' 2>/dev/null
    rsync -az --include='train_gpu_phase1_*.log' --exclude='*' \
      '${SSH_ALIAS}':~/orbit_wars_rl/ '$ART/logs/' 2>/dev/null
    sleep 180
  done
" > "$WATCHER_LOG" 2>&1 &
echo "Watcher PID: $!  log: $WATCHER_LOG"

echo ""
echo "=== Ready to train ==="
echo "Instance : $INSTANCE_NAME @ $GCP_IP"
echo "Zone     : $ZONE"
echo "SSH      : gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "Logs     : $ART/logs/"
echo ""
echo "Upload seed checkpoint then start training:"
echo "  gcloud compute scp <checkpoint.pt> $INSTANCE_NAME:~/orbit_wars_rl/seed_checkpoints/phase1_resume.pt --zone=$ZONE"
echo "  gcloud compute scp <bc_warmstart.pt> $INSTANCE_NAME:~/orbit_wars_rl/seed_checkpoints/bc_phase1_warmstart.pt --zone=$ZONE"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- 'tmux new-session -d -s training \"bash /tmp/start_training.sh\"'"
echo ""
echo "When done, TERMINATE (not stop):"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
