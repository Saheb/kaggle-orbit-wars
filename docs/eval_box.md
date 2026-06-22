# GCP CPU Eval Box — Fast Ajay Panels & Autonomous Live-Run Tracking

How to spin a high-vCPU GCP box that evals checkpoints vs Ajay **fast**, and run an
autonomous daemon that auto-evals live runs' new checkpoints into their CSVs.

Built 2026-06-22 to offload eval from the Mac (local panels were the bottleneck while
4 runs trained). Companion to the sharding code in `orbit_wars_rl/eval.py` +
`orbit_wars_rl/merge_panel_shards.py` and the drivers in `gpu_run_artifacts/eval_box/`.

---

## Why a box, and why sharding

**Evals are CPU-bound, not GPU-bound.** The cost is the `orbit_lite` planner opponents
(Ajay/Producer/deb), a single-threaded Python forward-sim (~16 s/game vs Ajay). Our model
is tiny and eval runs `CUDA_VISIBLE_DEVICES=""`. So:

- A **GPU box (L4) is the wrong tool** — GPU idle, few cores.
- OMP threads don't help — the planner is Python-serial.
- **A single-process 256-game panel vs Ajay is ~68 min** — *slower than the Mac*. The box
  only wins by **parallelism across cores**.

The fix is **panel sharding**: split the fixed 256-game panel into K deterministic shards
(by game index), run one process per core, and merge. A 256-game panel drops from ~68 min
to **~3–4 min** at K=30, with numbers **byte-identical** to a full panel.

---

## 1. Create the box (~10 min to ready)

```bash
# Account that owns the project:
gcloud config set account sahebcredit@gmail.com
# ⚠️ Use the project ID, NOT the display name. `orbit-wars-rl` is only the NAME;
#    compute.* calls fail with it. The real ID:
gcloud config set project orbit-wars-rl-499921

gcloud compute instances create orbit-wars-eval \
  --zone=asia-south1-a \
  --machine-type=n2-standard-32 \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB
gcloud compute config-ssh   # → alias orbit-wars-eval.asia-south1-a.orbit-wars-rl-499921
```

| | Value |
|---|---|
| Machine | `n2-standard-32` (32 vCPU, 128 GB) — the project's global `CPUS_ALL_REGIONS=32` ceiling |
| GPU | none (evals are CPU-bound) |
| Zone | `asia-south1-a` |
| Image | `ubuntu-2204-lts` (plain Ubuntu — avoids the DL-image driver hang on a GPU-less box) |
| Cost | ~$1.55/hr on-demand |

## 2. Environment (Python 3.11 — ke 1.29.1 needs ≥3.11; Ubuntu 22.04 ships 3.10)

```bash
HOST=orbit-wars-eval.asia-south1-a.orbit-wars-rl-499921
ssh $HOST '
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv venv ~/evenv --python 3.11
  uv pip install --python ~/evenv/bin/python torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python ~/evenv/bin/python kaggle-environments==1.29.1 numpy
'
```
Match `kaggle-environments==1.29.1` (the sim) so numbers stay comparable to history. torch/numpy
versions don't affect eval numerics.

## 3. Sync code + install the env

```bash
ssh $HOST 'mkdir -p ~/kaggle-orbit-wars/{orbit_wars_rl,opponents,setup}'
rsync -azL --exclude __pycache__ --exclude '*.pyc' orbit_wars_rl/ $HOST:~/kaggle-orbit-wars/orbit_wars_rl/
rsync -azL --exclude __pycache__ --exclude '*.pyc' opponents/   $HOST:~/kaggle-orbit-wars/opponents/   # incl orbit_lite
rsync -azL setup/ $HOST:~/kaggle-orbit-wars/setup/
# install the orbit_wars env into the venv (script uses python3 → put evenv first on PATH):
ssh $HOST 'cd ~/kaggle-orbit-wars && PATH=~/evenv/bin:$PATH bash setup/install_orbit_wars.sh'
```
Also rsync the drivers (gitignored, see §7) to `~/kaggle-orbit-wars/`.

> **Arch must match the checkpoints.** This worktree (`orbit-audit`) is the phase4 arch
> (pairwise-20); `main` is phase5 (pairwise-41 + NO_OP head). Sync the worktree whose code
> matches the checkpoints you're evaling.

---

## 4. Eval one checkpoint (sharded)

`orbit_wars_rl/eval.py` flags (added 2026-06-22): `--panel-shards K --panel-shard-idx i
--shard-out f.pkl`. Each shard runs the deterministic 1/K partition of the 256-game panel and
pickles its per-game records; `merge_panel_shards.py` replays all records through the identical
accumulation (`_accumulate_panel_records` / `add_conversion` are pure-additive → exact).

Driver does the whole thing:
```bash
ssh $HOST 'cd ~/kaggle-orbit-wars && \
  bash run_panel_sharded.sh \
    gpu_run_artifacts/<run>/checkpoints/<ckpt>.pt \
    opponents/candidate_ajay_1200.py <tag> 30'
# → eval_out/<tag>/report.txt  (full panel report; ~3-4 min)
```
- **K=30**, not 15: wall time = the slowest shard, and some games run the full 500-step
  timeout, so finer shards shrink the straggler tail.
- **Do NOT pass gate flags** — the reinforce gate auto-loads from the checkpoint metadata
  (`ckpt_cfg.get("reinforce_gate_min_planets")`). Passing them overrides train/eval parity.

## 5. Queue many checkpoints

```bash
# jobs file lines: <ckpt>|<opponent_py>|<tag>
ssh $HOST 'cd ~/kaggle-orbit-wars && nohup bash run_eval_queue.sh eval_out/jobs.txt 30 \
  >> eval_out/queue.log 2>&1 &'
```
Runs each as a full-core sharded panel **sequentially**, streaming a WR line to
`eval_out/summary.txt`. Total time is work-bound (~#games ÷ (#cores/16 s)): ~10 checkpoints
vs Ajay ≈ 35 min.

## 6. Pull results back to the run folders

`sync_eval_results.sh` (local) pulls each completed `report.txt` into
`gpu_run_artifacts/<run>/eval_logs/eval_<ckpt-stem>__ajay_1200.log` (provenance header +
cleaned report) **and** appends a row to `gpu_run_artifacts/<run>/eval_ajay_1200.csv`, using
the **same field extraction as `run_watchers.sh`** (dedup by checkpoint). Idempotent.

CSV columns: `utc_time,step,win_rate,seat0_wr,seat1_wr,outmassed_pct,open_capatk_WON,
mid_capatk_WON,peelrate_WON,planets100_WON,reinf_step_early,reinf_step_mid,reinf_dir_fwd,
games,checkpoint`.

```bash
bash gpu_run_artifacts/eval_box/sync_eval_results.sh   # add runs in its case statement
```

---

## 7. Autonomous daemon (hands-off live-run tracking)

`gpu_run_artifacts/eval_box/auto_eval_loop.sh` — runs locally (nohup). Every scan it takes the
**newest** local checkpoint of each tracked run not yet in its CSV, evals it on the box
(sharded), and appends the CSV row + log. **Strictly serial** (waits while a box eval/queue
runs) and **SSH-drop-robust** (eval launched detached on the box, then polls for `report.txt`).

```bash
# start (tracks runs listed in its RUNS=(...) line, e.g. mb4 + fs3):
nohup bash gpu_run_artifacts/eval_box/auto_eval_loop.sh >/dev/null 2>&1 &
tail -f gpu_run_artifacts/eval_box/auto_eval_loop.log      # monitor
pkill -f auto_eval_loop.sh                                 # stop
```
- Requires the run's **local `_sync` watcher** to be running (so new checkpoints land locally
  for the daemon to pick up).
- Edit `RUNS=("phase4fs_mb4:mb4" "phase4fs_fs3:fs3")` to add/drop runs (drop ones that
  collapse).

---

## Script locations

| Script | Where | Tracked? |
|---|---|---|
| `eval.py` (`--panel-shards/-shard-idx/--shard-out`) | `orbit_wars_rl/` | yes |
| `merge_panel_shards.py` | `orbit_wars_rl/` | yes |
| `run_panel_sharded.sh`, `run_eval_queue.sh`, `sync_eval_results.sh`, `auto_eval_loop.sh` | `gpu_run_artifacts/eval_box/` | **no** (gitignored) — rsync `.sh` to the box |

---

## Troubleshooting

- **`compute.* Required permission for projects/orbit-wars-rl`** → project set to the NAME.
  `gcloud config set project orbit-wars-rl-499921`.
- **SSH 255 storm / box unresponsive** → usually self-inflicted: killing/relaunching box
  queues over a flaky link leaves orphaned `run_panel_sharded.sh` and can spawn **overlapping
  queues** that overload the box until sshd stops accepting. Don't churn queues. To recover:
  `gcloud compute instances reset orbit-wars-eval --zone=asia-south1-a` (boot disk + completed
  reports survive the reboot; nohup processes die), then launch **one** queue.
- **Verify exactly one queue** (ps `args` self-matches your SSH command — use `grep -v grep`):
  ```bash
  ssh $HOST 'ps -eo pid,args | grep "run_eval_queue.sh eval" | grep -v grep'
  ```
- **A panel looks stuck** → progress prints only every 16 games; check shard completion:
  `ls eval_out/<tag>/shard_*.pkl | wc -l` (of K).

## Cost — delete when done

The box bills (~$1.55/hr) and the daemon keeps it alive indefinitely.
```bash
gcloud compute instances delete orbit-wars-eval --zone=asia-south1-a
```
