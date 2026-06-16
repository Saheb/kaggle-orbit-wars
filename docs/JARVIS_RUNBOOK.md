# Jarvis Labs Training Runbook

End-to-end steps for running Orbit Wars GPU experiments on Jarvis Labs with the
`jl` CLI. This mirrors the AWS/GCP runbooks but uses Jarvis instance IDs and
Jarvis lifecycle commands.

---

## Prerequisites

### CLI in the local venv

The CLI package is `jarvislabs`; it installs the `jl` executable. Use the repo's
documented local venv:

```bash
source /Users/saheb/home/.venv/bin/activate
uv pip install --python /Users/saheb/home/.venv/bin/python jarvislabs
jl --version
```

Verified locally: `jl 0.2.15`.

### Auth

The `.env` file currently contains `JARVIS_API_KEY`. The Jarvis CLI expects
`JL_API_KEY`, so bridge it when running commands:

```bash
source /Users/saheb/home/.venv/bin/activate
source .env
export JL_API_KEY="${JL_API_KEY:-$JARVIS_API_KEY}"

jl status
```

If you want the CLI token persisted in its config, run:

```bash
jl setup --token "$JL_API_KEY" --yes
```

### SSH key — MUST add to agent before using `jl exec` / `jl run`

The private key is at `~/.ssh/jarvis-labs-key`. The `jl` CLI uses plain `ssh`
with no `-i` flag, so it relies on the SSH agent. **Always run this before any
`jl exec` or `jl run` command or they will get `Permission denied (publickey)`:**

```bash
ssh-add ~/.ssh/jarvis-labs-key
```

To verify the key is loaded:
```bash
ssh-add -l | grep jarvis
```

Direct rsync/scp still works without the agent if you pass `-i ~/.ssh/jarvis-labs-key` explicitly.

To register the key with Jarvis (one-time):

```bash
chmod 600 ~/.ssh/jarvis-labs-key
ssh-keygen -y -f ~/.ssh/jarvis-labs-key > ~/.ssh/jarvis-labs-key.pub
jl ssh-key add ~/.ssh/jarvis-labs-key.pub --name orbit-wars-mac
jl ssh-key list
```

---

## Instance model

Jarvis supports two useful modes:

| Mode | Use it for | Notes |
|------|------------|-------|
| Container | Most Orbit Wars training | Default. Preconfigured with PyTorch/Jupyter/IDE. |
| VM | Bare SSH-only debugging | Pass `--vm` to `jl create` or `jl run`. |

Start with a container unless the managed image blocks something specific.

Useful resource checks:

```bash
jl gpus
jl resources
jl templates
```

---

## GPU choice and SPS benchmarks

Our model is small (404K params) and memory-bandwidth limited. Benchmarked SPS
for `num-envs=512 rollout-steps=64` Phase 1 training:

| GPU | ₹/hr | SPS (measured) | 30M step cost | Notes |
|-----|------|----------------|---------------|-------|
| L4  | ₹41  | **775**        | ~₹444         | Baseline; good for evals |
| A30 | ₹39  | **TBD**        | TBD           | Cheapest; 3× L4 memory BW — benchmark pending |
| A100 40GB | ₹84 | TBD   | TBD           | Best $/step candidate if ≥2× L4 SPS |
| A100 80GB | ₹141 | TBD  | TBD           | Diminishing returns vs 40GB for small model |
| H100 | ₹255 | TBD           | TBD           | Only if wall-clock time critical |

Update this table as benchmarks are run. Rule of thumb: A30 is the first thing
to try before paying more — it costs less than L4 and has ~3× the memory bandwidth.

### Measured findings — p2rev2 resume + heuristic-pool (2026-06-11)

**This workload is CPU-bound, NOT GPU-bound — the GPU sits at ~0%.** The model is tiny
(391K params) and self-play envs run free on GPU; the real cost is CPU-side simulation of the
external heuristic opponents (`--pool-external-fraction 0.25`), funnelled through a
**single-threaded main loop** (main proc pegs ~1 core). So **GPU choice barely affects SPS —
CPU core count and VRAM (env capacity) do.**

| GPU (spot ₹/hr) | cores | VRAM | best config | SPS | notes |
|---|---|---|---|---|---|
| H200 (₹189) | 28 | 143 GB | 512 envs / 6 workers | **~850** | GPU 0%; 512 is the sweet spot (1024 doesn't scale — single-thread cap) |
| A100 40 GB (₹75) | 16 | 40 GB | **256** envs / 4 workers | ~431 | **512 envs OOMs (~45 GB needed)** — must cap envs; 16 cores oversubscribe at load ~20 |
| A100 80 GB (`A100-SXM4-80GB`) | 28 | 80 GB | 512 envs / 12 workers | **~416** | no OOM; **must request `--gpu A100-80GB`** (see gotcha below) |

**Practical rules:**
- **VRAM:** 512 envs needs **~45 GB** → 40 GB cards OOM (the historical "A100 didn't work"). Use
  ≤256 envs on 40 GB, or an ≥80 GB card for 512.
- **Don't pay for flagship GPU on this workload** — it runs at 0% GPU. Optimise for **many CPU
  cores + ≥80 GB VRAM**. An A100 80 GB (₹84) likely matches the H200 (₹189) if cores are comparable.
- **Sweet spot is ~512 envs** (amortises the single-thread per-step overhead); past that doesn't
  scale because the main loop is the cap. Match `--heuristic-workers` to cores (~6 on 28-core, ~4 on 16-core).

For **eval only**: L4 is fine. 256-game panel takes ~12-13 min solo (vs ~40 min
on local Mac CPU). Two concurrent panels on same instance: ~20-25 min each.

---

## Launch a training instance

> **⚠️ GPU-NAME GOTCHA — how to actually get an 80 GB A100 (2026-06-16).** `jl create --gpu A100`
> gives the **40 GB PCIE** variant (`A100-PCIE-40GB`) → 512 envs OOMs, must cap to 256. The **80 GB**
> card is a SEPARATELY-NAMED gpu type: **`--gpu A100-80GB`** → provisions `A100-SXM4-80GB` (80 GB,
> 28 cores) → runs 512 envs. **`--storage N` is DISK GB, not VRAM** — it does NOT influence the card.
> So for 512-env training prefer `--gpu A100-80GB` (or H100/H200, also 80 GB+). The deploy/launch
> scripts auto-detect VRAM and cap NUM_ENVS to 256 below 70 GB, but requesting the right gpu type
> avoids the fallback. (Burned ~4 spot boxes rediscovering this — don't.)

```bash
source /Users/saheb/home/.venv/bin/activate
source .env
export JL_API_KEY="${JL_API_KEY:-$JARVIS_API_KEY}"
ssh-add ~/.ssh/jarvis-labs-key

jl create --gpu A100-80GB --storage 100 --name orbit-wars-train --spot --yes --json   # 80 GB / 28 cores / 512 envs
jl list
```

Record the integer machine ID. Prefer `--json` so you can extract `machine_id`
and `public_ip` programmatically.

---

## Remote directory layout (containers)

**Critical:** `jl upload <id> orbit_wars_rl /home/orbit_wars_rl` uploads the
*contents* of `orbit_wars_rl/` to `/home/orbit_wars_rl/`. So `train_torch.py`
lives at `/home/orbit_wars_rl/train_torch.py` — **not** nested deeper.

Confirmed layout after upload:
```
/home/
  orbit_wars_rl/          ← Python package files flat here (train_torch.py, eval.py, ...)
    checkpoints/           ← written here during training
    panels/                ← panel eval results
    seed_checkpoints/      ← resume checkpoints uploaded here
  opponents/               ← candidate_*.py files
  setup/                   ← install_orbit_wars.sh
  train_gpu_phase1_*.log   ← log lands in /home/ (NOT inside orbit_wars_rl/)
```

**Training must be launched from `/home`** so Python's module resolution works:
```bash
cd /home
python3 orbit_wars_rl/train_torch.py ...   # ✓ correct
# NOT: cd /home/orbit_wars_rl && python3 train_torch.py  (imports break)
```

**Always create this symlink at the top of every training script** — checkpoints save to `/home/checkpoints/` (cwd-relative) but watchers expect `/home/orbit_wars_rl/checkpoints/`:
```bash
mkdir -p /home/checkpoints
ln -sfn /home/checkpoints /home/orbit_wars_rl/checkpoints
```
Without this, the local watcher syncs from the wrong path and checkpoints/panels never appear locally.

**Eval** uses absolute path and also needs `/home` as cwd:
```bash
cd /home
python3 orbit_wars_rl/eval.py --checkpoint /home/orbit_wars_rl/checkpoints/... ...
```

Log files land at `/home/train_gpu_phase1_*.log` — sync from there.

---

## Push code and seed checkpoints

Use rsync directly (faster than `jl upload` for large trees, supports excludes):

```bash
KEY=~/.ssh/jarvis-labs-key
IP=<public_ip>
ROOT=/Users/saheb/home/kaggle-orbit-wars

# Upload package (exclude large local artifacts)
rsync -az \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pkl' \
  --exclude='episode_data' --exclude='replays' --exclude='replays_4p_heuristic' \
  --exclude='episode_index' --exclude='checkpoints' \
  -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$ROOT/orbit_wars_rl/" root@$IP:/home/orbit_wars_rl/

rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$ROOT/opponents/" root@$IP:/home/opponents/
rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$ROOT/setup/" root@$IP:/home/setup/

# Upload resume checkpoint
ssh -i $KEY -o StrictHostKeyChecking=no root@$IP \
  "mkdir -p /home/orbit_wars_rl/seed_checkpoints /home/orbit_wars_rl/checkpoints /home/orbit_wars_rl/panels"
scp -i $KEY -o StrictHostKeyChecking=no \
  <local_checkpoint.pt> root@$IP:/home/orbit_wars_rl/seed_checkpoints/phase1_resume.pt
```

Then install the Kaggle env:
```bash
ssh -i $KEY -o StrictHostKeyChecking=no root@$IP \
  "cd /home && pip install -q kaggle-environments && bash setup/install_orbit_wars.sh"
```

---

## Start training in tmux

```bash
ssh -i ~/.ssh/jarvis-labs-key -o StrictHostKeyChecking=no root@<IP>
```

On the remote:
```bash
tmux new-session -s orbit -x 220 -y 50
# window 0: training
tmux rename-window -t orbit:0 train
tmux send-keys -t orbit:train 'bash /home/orbit_wars_rl/gpu_run_artifacts/hellburner_spot/run_remote_phase1_rev28_jarvis.sh' Enter

# window 1: panel watcher (runs evals automatically on new checkpoints)
tmux new-window -t orbit -n panels
tmux send-keys -t orbit:panels 'bash /home/orbit_wars_rl/gpu_run_artifacts/hellburner_spot/run_panel_watcher_jarvis.sh' Enter
```

The training script must `cd /home` and call `python3 orbit_wars_rl/train_torch.py`.
No venv activation needed — PyTorch is in system Python on the pytorch template.

---

## Local watcher (sync checkpoints + logs + held-out eval)

Use the **controller** — never hand-roll an ad-hoc rsync loop (those survive across runs and end up
watching the *previous* run's folder, the recurring stale-watcher bug):

```bash
bash gpu_run_artifacts/run_watchers.sh start <run> jarvis <IP>   # e.g. start p2rev5 jarvis 217.18.55.11
bash gpu_run_artifacts/run_watchers.sh status                    # active run + live procs
bash gpu_run_artifacts/run_watchers.sh stop                      # kill all
```

`start` tears down ALL existing watchers first; each watcher self-terminates when `.active_run` changes
(so stale watchers can't accumulate). The `jarvis` preset resolves to `root@<IP>` + `/home/...` paths.
It syncs `train_gpu_phase1_<run>_*.log` + `torch_step_*/pool_step_*/torch_best_*` every 120s into
`gpu_run_artifacts/<run>/{logs,checkpoints}`, and auto-runs the held-out Ajay full-panel on each new
checkpoint (masks gate3/floor0/no-forward-only; override via `REINFORCE_MASKS`). Launch scripts should
end with a `run_watchers.sh start` call. (Still create the `/home/orbit_wars_rl/checkpoints` symlink in
the training script — see above — so checkpoints land where the watcher pulls from.)

---

## Monitor a run

```bash
KEY=~/.ssh/jarvis-labs-key
IP=<public_ip>

# Latest iter line
ssh -i $KEY -o StrictHostKeyChecking=no root@$IP \
  "grep '^iter' /home/train_gpu_phase1_*.log | tail -3"

# GPU utilisation
ssh -i $KEY -o StrictHostKeyChecking=no root@$IP \
  "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"

# Panel results as they land
ssh -i $KEY -o StrictHostKeyChecking=no root@$IP \
  "grep 'Overall:' /home/orbit_wars_rl/panels/*.log 2>/dev/null"

# tmux pane peek
ssh -i $KEY -o StrictHostKeyChecking=no root@$IP \
  "tmux capture-pane -t orbit:train -p | grep -v '^\$' | tail -10"
```

---

## Pause, resume, destroy

| Command | Billing impact | Data impact |
|---------|----------------|-------------|
| `jl pause <id>` | Stops compute billing | Keeps all home dir data |
| `jl resume <id> --gpu <type>` | Restarts billing | Keeps data; can swap GPU type |
| `jl destroy <id>` | Stops all billing | **Permanently deletes everything** |

> ⚠️ **`--terminate-on-done` does NOT free a Jarvis spot instance.** The training flag runs an OS
> `poweroff`, which Jarvis treats as the *instance still allocated* — `jl list` shows it **Running and
> still billing** (~₹189/hr on H200) even though SSH is dead and the log printed "powering off". You MUST
> run `jl destroy <id> --yes` explicitly after a run completes (confirmed 2026-06-11 on p2rev2/425211:
> log said "powering off" but the instance billed for hours until destroyed). `jl destroy` prompts `[y/N]`
> and defaults to **No** non-interactively — always pass `--yes`. Sync-verify checkpoints first (below),
> then destroy.

```bash
source /Users/saheb/home/.venv/bin/activate
source .env
export JL_API_KEY="${JL_API_KEY:-$JARVIS_API_KEY}"

jl pause 420340 --yes
jl resume 420340 --gpu A30 --yes --json   # can swap GPU on resume
jl destroy 420340 --yes
```

Always sync checkpoints locally before destroying:
```bash
rsync -az -e "ssh -i ~/.ssh/jarvis-labs-key -o StrictHostKeyChecking=no" \
  root@<IP>:/home/orbit_wars_rl/checkpoints/ \
  gpu_run_artifacts/jarvis/checkpoints/
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `jl: command not found` | venv not active | `source /Users/saheb/home/.venv/bin/activate` |
| Auth fails despite `.env` key | CLI expects `JL_API_KEY` not `JARVIS_API_KEY` | `export JL_API_KEY="${JL_API_KEY:-$JARVIS_API_KEY}"` |
| `jl exec` / `jl run` → `Permission denied (publickey)` | Key not in SSH agent | `ssh-add ~/.ssh/jarvis-labs-key` |
| `python3: can't open file '...train_torch.py'` | Wrong working dir | `cd /home` before running, not `cd /home/orbit_wars_rl` |
| `ModuleNotFoundError: orbit_wars` | Kaggle env not installed | `pip install -q kaggle-environments && bash setup/install_orbit_wars.sh` |
| Log file is empty / not found | Logs land at `/home/` not `/home/orbit_wars_rl/` | `ls /home/train_gpu_phase1_*.log` |
| Panel eval crashes with `No module named 'torch'` | Wrong local python | Use `source /Users/saheb/home/.venv/bin/activate && python3` for local evals |
| Instance still billing | Running or paused | `jl destroy <id>` after syncing |
