# Orbit Wars — Command Reference

Copy-paste ready. Every path and flag here exists at HEAD (refreshed in the 2026-10 cleanup —
the EC2 / BC-dataset / producer-audit / Kaggle-submission sections were removed; they are in git
history at tag `pre-cleanup-2026-10`).

---

## ⚠️ Before running anything locally — use the repo venv

```bash
source orbit_wars_rl/.venv/bin/activate      # or call orbit_wars_rl/.venv/bin/python directly
which python                                 # → .../kaggle-orbit-wars/orbit_wars_rl/.venv/bin/python
```

`python` alone resolves to the system Python 3.14 without the deps, and `nohup python ...` fails
with "No such file or directory". Activate first, or use the full path.

---

## 1. Launch and watch a training run (GCP L4)

Full runbook: `docs/GCP_RUNBOOK.md` (Jarvis: `docs/JARVIS_RUNBOOK.md`). Every launch has a
`gpu_run_artifacts/<run>/start_training.sh` recording the exact recipe and hypothesis — the
champion's is `gpu_run_artifacts/binarygates100m_l4/start_training.sh`.

```bash
bash gpu_run_artifacts/launch_gpu_gcp.sh --name <instance> --run <run> \
     --zone <zone> --project orbit-wars-rl-499921       # --project is REQUIRED
# verify the rsync landed before starting training:
ssh <alias> "ls ~/orbit_wars_rl/orbit_wars_rl/train_torch.py"
```

Watchers ONLY via the controller (sync + held-out eval):
```bash
# platform = gcp (target=config-ssh alias) | jarvis (target=IP) | custom (RSYNC_SSH/HOST/REMOTE_*_DIR env)
gcloud compute config-ssh --project=orbit-wars-rl-499921        # for the gcp alias
bash gpu_run_artifacts/run_watchers.sh start <run> gcp <alias>
bash gpu_run_artifacts/run_watchers.sh add-eval <run> opponents/candidate_yijie.py [from-latest]
bash gpu_run_artifacts/run_watchers.sh status
bash gpu_run_artifacts/run_watchers.sh stop
```
Held-out eval masks default to `REINFORCE_MASKS='--reinforce-gate-min-planets 2'`; override only
if a run trains a different gate. **When done: DELETE the GCP instance (not stop).**

Synced checkpoints and watcher output:
```bash
ls -lht gpu_run_artifacts/<run>/checkpoints/torch_step_*.pt | head
tail -20 gpu_run_artifacts/<run>/watcher_sync.log
cat gpu_run_artifacts/<run>/eval_ajay_1200.csv gpu_run_artifacts/<run>/eval_yijie.csv
```

---

## 2. ⭐ Behavioural probes — run these BEFORE proposing a lever

Cheap (minutes); on 2026-07-16 they killed three beliefs and a ~30h experiment (CLAUDE.md Key
Lessons 12–14).

```bash
# What does a top-10 agent actually SEND?  ships_sent / source_garrison histogram.
CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/ender_sizing.py --seeds 6
CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/ender_sizing.py --seeds 5 \
    --opponent opponents/candidate_ender.py          # strong-vs-strong control (do not skip)
CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/ender_sizing.py --seeds 6 \
    --agent-checkpoint <ckpt.pt> --opponent opponents/candidate_ender.py   # OURS, like-for-like

# Why do we lose captures?  Per-capture forensics. Refuses to run on allow_reinforce=False.
CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/peel_diagnosis.py --seeds 6 --checkpoint <ckpt.pt>

# Coordination / overkill (ground-truth redundant-on-arrival + same-turn multi-source)
CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/coord_overkill_probe.py --seeds 6 \
    --agent-checkpoint <ckpt.pt> --opponent opponents/candidate_ajay_1200.py

# How much of the action space do the legacy "full" gates delete?  (was: 80.2%)
python gpu_run_artifacts/ender_ref/probe_binary_gate_pressure.py
```

⚠️ A probe that plays one of our checkpoints must build it with `eval.load_eval_model(path, cfg)` —
never hand-copy model attributes. `build_agent_fn` reads the mask contract (commit gates, reinforce
gate, cooldown) off the model object; a forgotten attribute silently changes the policy being
measured. It happened twice (Key Lesson 14); `tests/test_load_eval_model.py` now enforces it.

---

## 3. Panel eval

**Primary metric: Yijie. Guard: Ajay. Reference: Ender.** Always write output to a file
(`PYTHONUNBUFFERED=1 … | tee`), never a bare pipe — a lost Ender panel is an hour of CPU.

```bash
# Full panel (256 games; ~40 min locally vs Ajay, 1h+ vs Ender)
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/eval.py \
  --checkpoint <path>.pt --opponent opponents/candidate_yijie.py --panel --target-decode \
  2>&1 | tee gpu_run_artifacts/<run>/eval_yijie_<step>.log

# Save the per-game records too, so later metrics can be recomputed without re-playing:
#   add  --panel-out gpu_run_artifacts/<run>/panel_yijie_<step>.pkl
python orbit_wars_rl/recompute_panel.py gpu_run_artifacts/<run>/panel_yijie_<step>.pkl

# Quick eval (16 games, ~2 min) — trend tracking only
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/eval.py \
  --checkpoint <path>.pt --opponent opponents/candidate_ajay_1200.py --games 16 --target-decode \
  2>&1 | tee gpu_run_artifacts/<run>/eval_ajay_quick_<step>.log
```

> ⚠️ `opponents/orbit_lite/` must be present for Ajay. Opponent paths are relative to the repo root.
> Discipline masks and commit gates auto-load from the checkpoint; `--reinforce-gate-min-planets` overrides.
> Checkpoints trained with a removed legacy mask (presres1 / stgpr1) are refused — eval them from
> tag `pre-cleanup-2026-10`. As opponents they run from their frozen tarballs.

---

## 4. Long local jobs: use `tmux`, not `nohup … &`

Detached shell jobs on this machine can disappear without a useful log; `tmux` keeps a real
terminal attached.

```bash
tmux new-session -d -s ender_panel \
  'cd /Users/saheb/home/kaggle-orbit-wars && \
   PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" orbit_wars_rl/.venv/bin/python orbit_wars_rl/eval.py \
     --checkpoint <ckpt.pt> --opponent opponents/candidate_ender.py --panel --target-decode \
     2>&1 | tee gpu_run_artifacts/<run>/eval_ender_<step>.log'
tmux attach -t ender_panel            # watch live; detach with C-b d
tmux kill-session -t ender_panel      # when finished
```

Check / kill local evals: `ps aux | grep eval.py | grep -v grep`.

---

## 5. Diagnose a silent eval failure

`eval.py` hard-exits with a traceback on any agent decode crash (kaggle would otherwise swallow it
as "no move" and report a fake 0%). If a log shows only startup noise (`INFO: OpenSpiel…`) and no
`panel progress:` lines:

| Symptom | Cause | Fix |
|---------|-------|-----|
| `nohup: python: No such file or directory` | wrong / no venv | `source orbit_wars_rl/.venv/bin/activate` |
| `FileNotFoundError` on an opponent | path not relative to repo root | `opponents/candidate_<name>.py` |
| `Do not trust this eval` (asset check) | a bundled opponent's assets missing | restore `opponents/<name>_bundle/` |
| `legacy discipline masks … removed` | presres1/stgpr1-era checkpoint | use tag `pre-cleanup-2026-10` |
| eval hangs with 0 progress >10 min | CPU contention | `ps aux \| grep eval.py`; ≤3 parallel evals on the Mac |
| CUDA OOM on a training box | training holds the GPU | prefix `CUDA_VISIBLE_DEVICES=""` |

---

## 6. Download game replays (Kaggle CLI 2.x)

Useful for studying winners' games. The agent index matches `info.TeamNames` order in the replay.

```bash
kaggle competitions episodes <SUBMISSION_ID>            # table (find ids: kaggle competitions submissions orbit-wars)
kaggle competitions episodes <SUBMISSION_ID> -v         # CSV (col 1 = episode id) — for scripting
kaggle competitions replay <EPISODE_ID> -p ./replays    # → episode-<id>-replay.json
kaggle competitions logs <EPISODE_ID> 0 -p ./logs       # YOUR agent's logs only
```

200 top-agent replays are already on disk in `archive/replays/top_agent_replays/` (untracked; the
bulk downloader `fetch_analyze_top_replays.py` is archived in `archive/cleanup_2026-07/rl_scripts/`).

---

## 7. Export a checkpoint as a standalone agent

Export inlines features / action_mask / timeline / binary_policy, so the file runs anywhere with
`kaggle_environments` + torch (e.g. as an opponent under `opponents/`). Mask contract and commit
gates are baked from the checkpoint. It then plays a 4-game sanity check vs Zach.

```bash
python orbit_wars_rl/export_agent.py --checkpoint <ckpt.pt> --output <agent>.py --target-decode
```

> Fixed 2026-10: export had been broken for every timeline/binary checkpoint (it emitted a file
> that didn't compile, then raised every turn). The exported champion now matches eval.py
> move-for-move. If a sanity check reports 0 wins, call `agent(obs)` directly to see the exception.

---

## 8. Run unit tests

```bash
orbit_wars_rl/.venv/bin/python -m pytest orbit_wars_rl/tests/ -q            # ~10 s
orbit_wars_rl/.venv/bin/python -m pytest orbit_wars_rl/tests/ -q --runslow  # + env-symmetry (~4 min)
```

---

## Key files — what does what

| File | Role |
|------|------|
| `orbit_wars_rl/train_torch.py` | Training entry point (GPU self-play PPO) |
| `orbit_wars_rl/eval.py` | Panel eval (source of truth); `load_eval_model` = THE checkpoint→model path |
| `orbit_wars_rl/torch_env.py` | Vectorised GPU env (physics, features, action decode) |
| `orbit_wars_rl/features.py` / `timeline.py` | Eval/export feature extraction; projected timeline channels |
| `orbit_wars_rl/action_mask.py` | Eval/export action decode + masks (inlined into exports) |
| `orbit_wars_rl/model.py` / `ppo.py` | Entity transformer / PPO learner (+ anchor KL) |
| `orbit_wars_rl/opponent_pool.py` | Self-play pool: PFSP + pinned RL champions + hard ramp |
| `orbit_wars_rl/export_agent.py` | Checkpoint → standalone agent .py |
| `orbit_wars_rl/eval_panel.py` / `recompute_panel.py` | 128-seed stratified panel / offline metric recompute |
| `orbit_wars_rl/ender_sizing.py`, `peel_diagnosis.py`, `coord_overkill_probe.py` | Behavioural probes (§2) |
| `gpu_run_artifacts/run_watchers.sh`, `launch_gpu_gcp.sh` | Live infra (tracked; the rest of gpu_run_artifacts is not) |
| `opponents/orbit_lite/` | Ajay dependency (must be present) |

---

## Common mistakes to avoid

| Mistake | Fix |
|---------|-----|
| A probe building its own model | `eval.load_eval_model(path, cfg)` — see §2 |
| Eval output piped to bare `tail` | `PYTHONUNBUFFERED=1 … 2>&1 \| tee gpu_run_artifacts/<run>/eval_*.log` |
| Forgetting `--panel` | Without it you get `--games N` single-seat games, not the 256-game panel |
| Ad-hoc per-run `*_watch.sh` | Only `run_watchers.sh start` (stale watchers watch the previous run) |
| Stopping (not deleting) a GCP instance | `gcloud compute instances delete …` — stopped disks still bill |
| Resuming from `_final.pt` | Resume from an interval checkpoint (it has Adam + the `pool_step` file) |
| Running many evals in parallel on the Mac | CPU contention → each 10× slower; ≤3 at a time |
| Eval on a training box OOMs | Prefix `CUDA_VISIBLE_DEVICES=""` |
