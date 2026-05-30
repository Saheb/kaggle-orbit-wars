# AWS GPU training runbook

End-to-end steps for running `train_torch.py` on an EC2 GPU instance. Captures
what worked for the 25M-step self-play run in `gpu_run_artifacts/` so we don't
relearn it next time.

## Instance spec

- **Type:** `g5.2xlarge` — NVIDIA A10G, 24 GB VRAM, 8 vCPU, 32 GB RAM.
  - On-demand: ~$1.21/hr. Spot (us-east-1): ~$0.56/hr.
- **AMI (us-east-1, current):** `ami-0b9c99b766a895d68` — Deep Learning OSS Nvidia
  Driver AMI GPU PyTorch 2.10 (Ubuntu 24.04). PyTorch + CUDA + NVIDIA drivers
  preinstalled. The DLAMI PyTorch venv lives at `/opt/pytorch`.
  - To refresh: `aws ec2 describe-images --owners amazon --filters
    "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu*"
    --query 'reverse(sort_by(Images,&CreationDate))[0].[ImageId,Name]'`
- **Security group:** `sg-07f813c351bb5e011` (default VPC SG). Keep ingress
  TCP 22 restricted to your current IP only — rotate when your IP changes:
  `aws ec2 authorize-security-group-ingress --group-id sg-07f813c351bb5e011
  --protocol tcp --port 22 --cidr $(curl -s https://checkip.amazonaws.com)/32`.
- **Key pair:** `samosa-key` (private key `~/.ssh/samosa-key.pem`, mode 600).
- **Pricing model:** **spot** by default. Checkpoints land every
  `--checkpoint-interval` env steps; `watch_gpu_run.sh` rsyncs them locally so
  an interruption costs at most one checkpoint interval of work. Use on-demand
  only if you've explicitly disabled checkpoint sync.
- **Region:** `us-east-1`.

## One-time setup (per instance)

1. **Launch the instance** with:
   - DLAMI (Ubuntu) selected as the AMI.
   - Key pair: `samosa-key` (private key at `~/.ssh/samosa-key.pem`, mode 600).
   - Security group: inbound TCP 22 from your laptop's public IP only.
   - **Shutdown behavior: Terminate** (under "Advanced details" at launch, or via CLI flag below). This is what makes `--terminate-on-done` end billing instead of just stopping the box.
   - Root volume: 100 GB gp3 is plenty.

   CLI (spot, us-east-1, current AMI):

   ```bash
   aws ec2 run-instances \
     --instance-type g5.2xlarge \
     --image-id ami-0b9c99b766a895d68 \
     --key-name samosa-key \
     --security-group-ids sg-07f813c351bb5e011 \
     --instance-initiated-shutdown-behavior terminate \
     --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
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
cd ~/orbit_wars_rl && source /opt/pytorch/bin/activate
LOG=train_gpu_$(date +%Y%m%d_%H%M%S).log
PYTHONUNBUFFERED=1 python train_torch.py \
  --resume checkpoints/<latest>.pt \
  --total-steps 100_000_000 \
  --num-envs 512 --rollout-steps 64 --num-minibatches 32 \
  --ppo-epochs 2 --learning-rate 0.0001 \
  --checkpoint-interval 5_000_000 \
  --pool-mode none \
  --terminate-on-done \
  2>&1 | tee $LOG
# Detach: Ctrl-b d
```

Notes:
- `--terminate-on-done` runs `sudo shutdown -h +1` at the end. Combined with the
  instance's `terminate` shutdown behavior, this ends billing automatically.
- **No in-training eval flag** — the prior `--eval-interval` + frozen-baseline
  probe gave false positives (see docs/bugs.md #2). Source of truth is local
  `eval.py` on rsynced checkpoints against raw Suneet/Zach/Rahul.
- Resume auto-skips LR warmup; pass `--with-warmup` to force it back on.
- Stop criterion: watch local eval at each checkpoint. Kill manually when two
  consecutive checkpoints (10M apart) plateau within ±2pp, or when win rate
  regresses >5pp from running max.

## Watch + sync from laptop

In a separate laptop terminal:

```bash
cd ~/home/kaggle-orbit-wars/orbit_wars_rl
./watch_gpu_run.sh <instance-id> <public-ip>
```

It polls every 3 min, rsyncs `checkpoints/` + `train_gpu.log` into
`../gpu_run_artifacts/` relative to the checkout containing
`watch_gpu_run.sh`, and runs
`aws ec2 terminate-instances` when it sees `Training complete` or `Early stop:` in the log.
Set `LOCAL_DIR=/absolute/path/to/artifacts` before invoking the watcher to
override the destination explicitly.

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
- **Early-stop on vs-frozen-initial probe was misleading and is removed.**
  Both `eval_vs_baseline` and the `--early-stop-patience` flag are gone — they
  reported steady "improvement" while replays showed policy collapse. See
  `docs/bugs.md` #2. Use local `eval.py` on synced checkpoints to decide stop.
- **LR warmup re-runs on resume** — fixed; warmup auto-skipped when `--resume` is set.
- **Spot reliability is poor for g5.2xlarge in us-east-1 (2026-05-24/25):** 4
  interrupts in one session, average run length under 2h before reclaim. For
  pool training (which doesn't tolerate frequent restarts cleanly), use
  on-demand. For pure self-play runs that save checkpoints every 5M, spot is
  still fine — accept losing up to a checkpoint of work.
- **Fraction-head decode is mode-aware (2026-05-25):** every checkpoint now
  carries `ship_bin_mode` in its `config` blob — either `"absolute"` (legacy
  32-bin SHIP_COUNTS lookup) or `"fraction"` (10-bin [0.1..1.0] × source).
  `train_torch.py` reads this BEFORE creating the env, so `VecTorchEnv` and
  `actions_from_policy` decode consistently. Loading a fraction-head .pt
  without `ship_bin_mode` set will silently default to "absolute" and give
  nonsense behaviour (this is what tonight's "maskbin0 95% Zach" results
  measured — see docs/bugs.md). Always confirm the line
  `Checkpoint declares ship_bin_mode=<mode>` appears in the log.

## New CLI flags (2026-05-25)

  --bc-coef            float; coefficient on auxiliary BC cross-entropy loss
                       during PPO. Requires --bc-samples. Typical 0.5–2.0.
                       Anchors policy to teacher's actions via supervised
                       gradient (alternative to il-lambda).
  --bc-samples         path to .pkl from extract_teacher_samples.py or the
                       cache from bc_frac.py.
  --il-lambda          float; KL-to-frozen-policy penalty coefficient.
                       Decays linearly to 0 over --il-decay-frac of training.
                       Note: KL-on-distributions can keep KL small while
                       argmax flips — not a strong-enough anchor in
                       practice. Prefer --bc-coef.
  --il-decay-frac      default 0.8
  --il-ref             optional separate frozen reference .pt; defaults to
                       the --resume checkpoint.
  --min-ship-bin       int; mask ship bins < this to -inf in model forward.
                       For fraction-head (10 bins), set to 1 to remove the
                       10%-of-source bin that PPO collapsed to in cold-start
                       (still not a complete fix — see docs/bugs.md).
  --heuristic-workers  int; size of multiproc worker pool for external
                       heuristic opponents in pool-mode=mixed. Default 0
                       = (cpu_count - 1). Gives ~2.5× SPS on g5.2xlarge
                       when external is Suneet.

## Pool tuning notes

- `--pool-fraction 0.1` keeps SPS at ~600 with Suneet but the asymmetric
  seat assignment in pool envs broke the seat symmetry the BC checkpoint had
  (Zach winrate dropped 92% → 65%, seat asymmetry +64pp). Suneet 0% didn't
  budge in 5M of pool.
- `--pool-fraction 0.3` → ~290 SPS, more Suneet gradient per minute. Not
  yet validated end-to-end (interrupted runs).
- Always panel-eval after pool training: pool envs assign the learner to
  one seat per env, so a single-seat eval will look great while panel
  exposes seat asymmetry. The 128-seed panel (`eval.py --panel`) is the
  required diagnostic, not optional.
