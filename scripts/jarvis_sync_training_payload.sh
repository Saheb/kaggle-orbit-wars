#!/usr/bin/env bash
# Sync the standard Orbit Wars training payload to a running Jarvis container.
#
# Usage:
#   scripts/jarvis_sync_training_payload.sh <public_ip> <checkpoint.pt> [remote_ckpt_name] [extra_file ...]
#
# Example:
#   scripts/jarvis_sync_training_payload.sh 217.18.55.112 \
#     gpu_run_artifacts/revedge1/checkpoints/torch_step_4718592_revedge1_20260616_095906.pt \
#     revedge1_4718592.pt \
#     gpu_run_artifacts/peeler_c1/run_remote_peeler_c1_t1_jarvis.sh
#
# Shell footgun this avoids:
#   Do not run a scalar like $SSH root@$IP when SSH="ssh -i ...".
#   zsh does not word-split scalars, so it tries to execute a literal command
#   named "ssh -i ...". Use an array for ssh argv instead.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <public_ip> <checkpoint.pt> [remote_ckpt_name] [extra_file ...]" >&2
  exit 2
fi

ROOT="${ROOT:-/Users/saheb/home/kaggle-orbit-wars}"
KEY="${KEY:-$HOME/.ssh/jarvis-labs-key}"
IP="$1"
CHECKPOINT="$2"
REMOTE_CKPT_NAME="${3:-phase1_resume.pt}"
shift 2
if [[ $# -gt 0 ]]; then
  shift 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no)
RSYNC_RSH="ssh -i $KEY -o StrictHostKeyChecking=no"

"${SSH[@]}" "root@$IP" \
  "mkdir -p /home/orbit_wars_rl /home/opponents /home/setup \
     /home/orbit_wars_rl/seed_checkpoints /home/checkpoints \
     /home/orbit_wars_rl/panels && \
   ln -sfn /home/checkpoints /home/orbit_wars_rl/checkpoints"

rsync -az --delete \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pkl' \
  --exclude='episode_data' --exclude='replays' --exclude='replays_4p_heuristic' \
  --exclude='episode_index' --exclude='checkpoints' --exclude='panels' \
  --exclude='seed_checkpoints' \
  -e "$RSYNC_RSH" \
  "$ROOT/orbit_wars_rl/" "root@$IP:/home/orbit_wars_rl/"

rsync -az --delete --exclude='__pycache__' --exclude='*.pyc' \
  -e "$RSYNC_RSH" \
  "$ROOT/opponents/" "root@$IP:/home/opponents/"

rsync -az --delete \
  -e "$RSYNC_RSH" \
  "$ROOT/setup/" "root@$IP:/home/setup/"

rsync -azL \
  -e "$RSYNC_RSH" \
  "$CHECKPOINT" "root@$IP:/home/orbit_wars_rl/seed_checkpoints/$REMOTE_CKPT_NAME"

for extra in "$@"; do
  if [[ ! -f "$extra" ]]; then
    echo "extra file not found: $extra" >&2
    exit 1
  fi
  rel="${extra#$ROOT/}"
  if [[ "$rel" = /* ]]; then
    echo "extra file must be under repo root: $extra" >&2
    exit 1
  fi
  remote="/home/orbit_wars_rl/$rel"
  "${SSH[@]}" "root@$IP" "mkdir -p \"$(dirname "$remote")\""
  rsync -az -e "$RSYNC_RSH" "$extra" "root@$IP:$remote"
done

"${SSH[@]}" "root@$IP" \
  "ls -lh /home/orbit_wars_rl/train_torch.py \
          /home/opponents/candidate_peeler_t1.py 2>/dev/null || true; \
   ls -lh /home/orbit_wars_rl/seed_checkpoints/$REMOTE_CKPT_NAME"
