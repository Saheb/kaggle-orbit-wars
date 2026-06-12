# GCP Training Runbook

Covers everything needed to launch, monitor, and terminate a training run on GCP.
Mirrors the AWS workflow but using `gcloud` instead of `aws`.

---

## Prerequisites (one-time)

### Auth
```bash
gcloud auth login
gcloud config set project orbit-wars-rl
```

### Quotas required
Both must be non-zero in `us-central1`. Check at:
`console.cloud.google.com/iam-admin/quotas?project=orbit-wars-rl`

| Quota | Required | How to get it |
|-------|----------|---------------|
| `GPUS_ALL_REGIONS` (global) | ≥ 1 | Request increase; approved in minutes |
| `NVIDIA_L4_GPUS` (us-central1) | ≥ 1 | Already allocated (limit=1) |

> **Note:** `G2_CPUS` does NOT appear in the console — it is implicitly covered
> by `GPUS_ALL_REGIONS`. You don't need to request it separately.

---

## Instance details

| | Value |
|---|---|
| Machine type | `g2-standard-8` (1× NVIDIA L4, 8 vCPU, 32GB RAM) |
| GPU | NVIDIA L4, 23GB VRAM — equivalent to AWS A10G |
| SPS | ~600 (vs ~880 on AWS g5.2xlarge) — ~68% speed |
| Cost | ~$1.13/hr on-demand |
| Zone | **`us-central1-b`** — `us-central1-a` is frequently out of stock |
| Base image | `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` from `deeplearning-platform-release` |
| User | `saheb` (OS Login, not `ubuntu` like AWS) |

---

## Launch a training run

### Step 1 — Create instance and set up environment
```bash
cd /path/to/kaggle-orbit-wars
bash gpu_run_artifacts/launch_gpu_gcp.sh
# Optional overrides:
#   --zone us-central1-c   (if b is out of stock)
#   --name orbit-wars-run2
```

This script:
- Creates the instance
- Waits for SSH
- Runs `gcloud compute config-ssh` (adds instance to `~/.ssh/config` for rsync)
- rsyncs the repo (excludes `.git`, `.venv`, checkpoints, panels, logs)
- Installs `kaggle-environments` + copies `orbit_wars` env from `setup/orbit_wars_env/`

### Step 2 — Upload seed checkpoint
```bash
INSTANCE=orbit-wars-training   # or whatever --name you used
ZONE=us-central1-b

gcloud compute ssh $INSTANCE --zone=$ZONE -- "mkdir -p ~/orbit_wars_rl/seed_checkpoints"

# Upload the resume checkpoint
gcloud compute scp <local_checkpoint.pt> \
  $INSTANCE:~/orbit_wars_rl/seed_checkpoints/phase1_resume.pt --zone=$ZONE

# Upload BC warmstart (only needed if not already on instance)
gcloud compute scp seed_checkpoints/bc_phase1_warmstart.pt \
  $INSTANCE:~/orbit_wars_rl/seed_checkpoints/bc_phase1_warmstart.pt --zone=$ZONE
```

### Step 3 — Start training in tmux
```bash
# Upload the start script
cat > /tmp/start_training.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd $HOME/orbit_wars_rl
TS="$(date '+%Y%m%d_%H%M%S')"
LOG="train_gcp_phase1_${TS}.log"
echo "Starting: $TS  log: $LOG"
PYTHONUNBUFFERED=1 python3 orbit_wars_rl/train_torch.py \
  --resume seed_checkpoints/phase1_resume.pt \
  --total-steps 30000000 \
  --lr-schedule-steps 60000000 \
  --learning-rate 0.0003 \
  --num-envs 512 --rollout-steps 64 --num-minibatches 32 \
  --ppo-epochs 2 \
  --checkpoint-interval 1000000 \
  --pool-checkpoint-interval 500000 --pool-max-size 20 \
  --pool-mode mixed --pool-fraction 0.75 \
  --external-opponents candidate_hellburner.py \
  --pool-external-fraction 0.05 \
  --win-margin-coeff 0.5 \
  --action-decode target \
  --pool-pfsp-min-games 30 \
  --pool-mastered-threshold 0.99 \
  --pool-mastered-min-games 500000 \
  --il-lambda 0.01 \
  --il-decay-frac 1.0 \
  --il-ref seed_checkpoints/bc_phase1_warmstart.pt \
  --srcs-multi-penalty 0.001 \
  --srcs-multi-threshold 2.0 \
  --terminate-on-done \
  2>&1 | tee "$LOG"
EOF

gcloud compute scp /tmp/start_training.sh $INSTANCE:/tmp/start_training.sh --zone=$ZONE
gcloud compute ssh $INSTANCE --zone=$ZONE -- \
  "tmux new-session -d -s training 'bash /tmp/start_training.sh'"

# Verify it started — use log file directly, NOT tmux capture-pane.
# tmux capture-pane shows history buffer including old failed runs.
# The log file is the ground truth.
sleep 30
gcloud compute ssh $INSTANCE --zone=$ZONE -- \
  "ls -t ~/orbit_wars_rl/train_gcp_phase1_*.log 2>/dev/null | head -1 | xargs grep '^iter\|Resumed\|Error\|Traceback' 2>/dev/null | head -5"
```

---

## Monitor a running instance

### Check training log
```bash
INSTANCE=orbit-wars-training; ZONE=us-central1-b
gcloud compute ssh $INSTANCE --zone=$ZONE -- \
  "grep '^iter' ~/orbit_wars_rl/train_gcp_phase1_*.log | tail -3"
```

### Live watchers — use the CONTROLLER (sync + held-out eval)
Never hand-roll an ad-hoc rsync loop (those survive across runs → watch the *previous* run's folder, the
recurring stale-watcher bug; this literally happened with p2rev4). Use the `gcp` preset:
```bash
gcloud compute config-ssh                              # ensures the SSH alias exists
SSH_ALIAS="$INSTANCE.$ZONE.orbit-wars-rl"
bash gpu_run_artifacts/run_watchers.sh start <run> gcp "$SSH_ALIAS"   # target = the SSH alias
bash gpu_run_artifacts/run_watchers.sh status          # active run + live procs
bash gpu_run_artifacts/run_watchers.sh stop            # kill all
```
`start` tears down all existing watchers first; each self-terminates when `.active_run` changes. The `gcp`
preset resolves to `ssh <alias>` + `~/orbit_wars_rl/...` paths and syncs every 120s into
`gpu_run_artifacts/<run>/{logs,checkpoints}`, auto-running the held-out Ajay full-panel per checkpoint
(masks gate3/floor0/no-forward-only; override `REINFORCE_MASKS`). **Name the GCP training log
`train_gpu_phase1_<run>_*.log`** (the phase-2 convention p2rev4 used) so the sync glob matches. Launch
scripts should end with a `run_watchers.sh start` call.

### Sync checkpoints locally (one-off, e.g. final pull before terminate)
```bash
SSH_ALIAS="$INSTANCE.$ZONE.orbit-wars-rl"   # set by gcloud compute config-ssh
rsync -azL "${SSH_ALIAS}:~/orbit_wars_rl/checkpoints/" \
  gpu_run_artifacts/hellburner_spot/checkpoints/
```

### Watch training (live tail)
```bash
gcloud compute ssh $INSTANCE --zone=$ZONE -- \
  "tail -f ~/orbit_wars_rl/train_gcp_phase1_*.log"
```

---

## Kill training
```bash
gcloud compute ssh $INSTANCE --zone=$ZONE -- "tmux send-keys -t training C-c"
# Wait a few seconds, then verify
gcloud compute ssh $INSTANCE --zone=$ZONE -- "tmux capture-pane -t training -p | tail -5"
```

---

## Terminate instance (ALWAYS do this when done)

```bash
INSTANCE=orbit-wars-training; ZONE=us-central1-b

# 1. Pull all checkpoints first
SSH_ALIAS="$INSTANCE.$ZONE.orbit-wars-rl"
rsync -azL "${SSH_ALIAS}:~/orbit_wars_rl/checkpoints/" \
  gpu_run_artifacts/hellburner_spot/checkpoints/
echo "checkpoints: $(ls gpu_run_artifacts/hellburner_spot/checkpoints/*.pt | wc -l)"

# 2. Pull training log
rsync -azL --include='train_gcp_*.log' --exclude='*' \
  "${SSH_ALIAS}:~/orbit_wars_rl/" \
  gpu_run_artifacts/hellburner_spot/logs/

# 3. Terminate
gcloud compute instances delete $INSTANCE --zone=$ZONE --quiet
echo "Terminated."
```

> **Never leave the instance in a stopped state** — GCP still bills for the disk.
> Always delete, never stop.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ZONE_RESOURCE_POOL_EXHAUSTED` in us-central1-a | L4 stock-out | Use `--zone us-central1-b` |
| `Quota 'GPUS_ALL_REGIONS' exceeded` | Global GPU quota = 0 | Request increase at IAM & Admin → Quotas |
| `ModuleNotFoundError: orbit_wars` | orbit_wars not installed | `bash setup/install_orbit_wars.sh` |
| rsync fails / no files land | OS Login key not propagated | Use `gcloud compute config-ssh` first, then rsync to SSH alias |
| tmux session exits immediately | Training script crashes on startup | Run command directly (not in tmux) to see the error |
| `G2_CPUS` not in quota console | Normal — it's implicit under `GPUS_ALL_REGIONS` | Don't worry about it |
