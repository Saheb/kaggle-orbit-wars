# AWS GPU training runbook

End-to-end steps for running `train_torch.py` on an EC2 GPU instance. Captures
what worked for the 25M-step self-play run in `gpu_run_artifacts/` so we don't
relearn it next time.

## Instance spec

- **Type:** `g5.2xlarge` — NVIDIA A10G, 24 GB VRAM, 8 vCPU, 32 GB RAM, ~$1.21/hr on-demand.
- **AMI:** AWS Deep Learning AMI (Ubuntu). PyTorch + CUDA + NVIDIA drivers preinstalled — no manual CUDA setup.
- **Pricing model:** on-demand. Spot would be cheaper but interruptible; current
  training loop doesn't auto-resume from S3 checkpoints, so on-demand is the
  safer default.
- **Region:** whichever has g5 capacity for your account.

## One-time setup (per instance)

1. **Launch the instance** with:
   - DLAMI (Ubuntu) selected as the AMI.
   - Key pair: `samosa-key` (private key at `~/.ssh/samosa-key.pem`, mode 600).
   - Security group: inbound TCP 22 from your laptop's public IP only.
   - **Shutdown behavior: Terminate** (under "Advanced details" at launch, or via CLI flag below). This is what makes `--terminate-on-done` end billing instead of just stopping the box.
   - Root volume: 100 GB gp3 is plenty.

   CLI equivalent (substitute the AMI ID for your region):

   ```bash
   aws ec2 run-instances \
     --instance-type g5.2xlarge \
     --image-id ami-XXXXXXXX \
     --key-name samosa-key \
     --security-group-ids sg-XXXXXXXX \
     --instance-initiated-shutdown-behavior terminate \
     --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
     --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=orbit-wars-train}]'
   ```

2. **Note the instance ID and public IP** — `watch_gpu_run.sh` takes both as args.

## Push code + install deps

From the laptop, repo root:

```bash
PUB_IP=<instance-public-ip>
KEY=~/.ssh/samosa-key.pem

# Push the RL package (exclude bulky local artifacts)
rsync -az --progress \
  --exclude '.venv/' --exclude '__pycache__/' \
  --exclude 'checkpoints/' --exclude 'episode_data/' \
  --exclude 'replays/' --exclude 'replays_4p_heuristic/' \
  --exclude 'top_agent_replays/' --exclude '*.pkl' \
  -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  orbit_wars_rl/ ubuntu@${PUB_IP}:~/orbit_wars_rl/

# Also push main.py if eval-vs-heuristic is wanted
rsync -az -e "ssh -i $KEY" main.py ubuntu@${PUB_IP}:~/
```

On the instance:

```bash
ssh -i $KEY ubuntu@${PUB_IP}
# DLAMI ships a prebuilt PyTorch venv at /opt/pytorch — use it directly.
# Do NOT `python3 -m venv .venv` (ensurepip is absent on the system python).
source /opt/pytorch/bin/activate
pip install kaggle-environments wandb   # only the deps not in DLAMI
# Sanity: should print torch + a CUDA device
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Launch training

On the instance, inside a `tmux`/`screen` session so an SSH drop doesn't kill it:

```bash
tmux new -s train
cd ~/orbit_wars_rl && source .venv/bin/activate
PYTHONUNBUFFERED=1 python train_torch.py \
  --resume checkpoints/<latest>.pt \
  --total-steps 100_000_000 \
  --num-envs 512 --rollout-steps 64 --num-minibatches 32 \
  --ppo-epochs 2 --learning-rate 0.0001 \
  --eval-interval 5_000_000 --eval-heuristic ~/main.py \
  --early-stop-patience 999 \
  --terminate-on-done \
  2>&1 | tee train_gpu.log
# Detach: Ctrl-b d
```

Notes:
- `--terminate-on-done` runs `sudo shutdown -h +1` at the end. Combined with the
  instance's `terminate` shutdown behavior, this ends billing automatically.
- `--early-stop-patience 999` effectively disables eval-based early stop (the
  default of 3 killed the previous run at 25 % of budget — see ADR / project memory).
- Resume auto-skips LR warmup; pass `--with-warmup` to force it back on.

## Watch + sync from laptop

In a separate laptop terminal:

```bash
cd ~/home/kaggle-orbit-wars/orbit_wars_rl
./watch_gpu_run.sh <instance-id> <public-ip>
```

It polls every 3 min, rsyncs `checkpoints/` + `train_gpu.log` into
`~/home/kaggle-orbit-wars/gpu_run_artifacts/`, and runs
`aws ec2 terminate-instances` when it sees `Training complete` or `Early stop:` in the log.

## Teardown checklist (when run is done)

The auto-terminate path covers most of it, but verify:

```bash
aws ec2 describe-instances --instance-ids <id> \
  --query 'Reservations[].Instances[].State.Name' --output text
# Should print "terminated".
```

Manual fallback:
```bash
aws ec2 terminate-instances --instance-ids <id>
```

## Hyperparameter notes

- **Learning rate = 1e-4, not the config default 3e-4.** Pass `--learning-rate 0.0001`
  explicitly. At the default 3e-4 the policy's `clip_frac` creeps monotonically
  (0.10 → 0.30+) and the optimizer eventually loses the race against value-head
  sharpening — observed in earlier runs, matches the canonical transformer-RL
  failure mode. Cutting LR to 1e-4 keeps clip_frac flat. If you ever raise LR,
  watch `clip_frac` per iter — a sustained rise is the warning sign and demands
  rolling back LR (or capacity) **before** entropy collapses.
- **--num-minibatches 32 with --num-envs 512 on the A10G (24 GB).** P=2 symmetric
  rollouts double the per-rollout sample count vs the original code, so 16
  minibatches would OOM in the PPO backward at this env count. 32 keeps the
  per-minibatch size identical to pre-P=2 runs.
- **Pass `--ppo-epochs 2` explicitly.** The config default is `4`, which
  quadruples PPO compute per rollout (4 epochs × 32 minibatches = 128 updates
  vs 2 × 16 = 32 historically). On the A10G that swings SPS by ~2× — a 100 M
  run goes from ~23 h to ~42 h purely from the missing flag.
- **One change at a time.** When iterating on architecture or PPO config, ship
  one knob per cloud run. Stacked changes can't be attributed when training
  regresses — a working "stupider" config is worth more than an unstable
  "smarter" one. Add the new knob, verify clip_frac / entropy stay healthy,
  *then* compose with the next change.

## Gotchas hit in past runs

- **Security group IP rotation**: laptop's public IP changes on network switches
  → rsync silently failing was the old behavior. `watch_gpu_run.sh` now surfaces
  SSH/rsync errors so we notice (commit `213cbfd`).
- **Heuristic eval can crash training**: wrapped in try/except in `train_torch.py`
  so a broken `eval-heuristic` agent doesn't take down the whole run (commit `e63c2eb`).
- **CUDA OOM in attention — recurring.** A10G (24 GB) PyTorch backward keeps
  hitting OOM as the per-rollout sample count grows. Track record:
    - First cloud run: default `--num-minibatches 4` at `--num-envs 512` OOM'd in
      attention. Fix: bump to `16` (commit `006d207`, also moved rollout
      storage to CPU).
    - Today's run (post P=2 symmetric data): `--num-minibatches 16` OOM'd again
      because P=2 doubles the per-rollout sample count (32 K → 65 K). Fix:
      bump to `32` so per-minibatch size matches the pre-P=2 era.
  Rule of thumb: keep per-minibatch size at ≤ 2 K samples on A10G with this
  model + feature dims. If a future change doubles batch again, double
  `--num-minibatches` first, *then* run. As a last resort,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` can reduce fragmentation.
- **Early-stop-patience 3 + noisy self-play winrate** terminated at 25 M of 100 M
  steps. Use 999 unless you've changed the eval signal to something monotonic.
- **LR warmup re-runs on resume** — fixed; warmup auto-skipped when `--resume` is set.
