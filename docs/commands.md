# Orbit Wars — Command Reference

Sorted by frequency. Copy-paste ready. No guessing paths.

---

## ⚠️ Before running anything locally — activate the venv

```bash
# One-time per terminal session
source /Users/saheb/home/.venv/bin/activate

# Verify: should print /Users/saheb/home/.venv/bin/python
which python
```

`python` alone will not work on this Mac (Python 3.14 is system default, wrong version,
and `nohup python ...` will fail with "No such file or directory"). Always activate first
OR use the full path `/Users/saheb/home/.venv/bin/python`.

---

## 1. Monitor a running training run

```bash
# Tail the latest log on remote (replace IP)
ssh -i ~/.ssh/samosa-key.pem -o StrictHostKeyChecking=no ubuntu@<PUB_IP> \
  "ls -t ~/orbit_wars_rl/train_gpu_phase1_*.log 2>/dev/null | head -1 | xargs tail -5"

# Last 10 iter lines only (fire[0], srcs_multi, fleet_size)
ssh -i ~/.ssh/samosa-key.pem -o StrictHostKeyChecking=no ubuntu@<PUB_IP> \
  "ls -t ~/orbit_wars_rl/train_gpu_phase1_*.log 2>/dev/null | head -1 | xargs grep '^iter' | tail -10"

# Check tmux session is alive
ssh -i ~/.ssh/samosa-key.pem -o StrictHostKeyChecking=no ubuntu@<PUB_IP> "tmux list-sessions"

# GPU utilisation
ssh -i ~/.ssh/samosa-key.pem -o StrictHostKeyChecking=no ubuntu@<PUB_IP> \
  "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
```

> ℹ️ Run name varies by revision. If the above finds nothing, try:
> `ls -t ~/orbit_wars_rl/*.log 2>/dev/null | head -5` to find the active log.

---

## 2. Check locally synced checkpoints

```bash
ls -lht gpu_run_artifacts/hellburner_spot/checkpoints/*.pt | head -10

# Latest watcher heartbeat (look for rsync lines)
tail -20 gpu_run_artifacts/hellburner_spot/logs/watcher_rev7.log
```

---

## 3. Panel eval

### Full panel (256 games, ~40 min locally)

```bash
source /Users/saheb/home/.venv/bin/activate

# Ajay (primary metric — harder than Zach)
python orbit_wars_rl/eval.py \
  --checkpoint <path>.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --panel --target-decode \
  > /tmp/eval_ajay.log 2>&1 &

# Zach (secondary — saturating ~88-89%)
python orbit_wars_rl/eval.py \
  --checkpoint <path>.pt \
  --opponent opponents/candidate_zach_public.py \
  --panel --target-decode \
  > /tmp/eval_zach.log 2>&1 &

wait
grep "Overall" /tmp/eval_ajay.log /tmp/eval_zach.log
```

### Quick eval (16 games, ~2 min — for trend tracking)

```bash
python orbit_wars_rl/eval.py \
  --checkpoint <path>.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --games 16 --target-decode
```

> ⚠️ `orbit_lite` package must be at `opponents/orbit_lite/` for Ajay/Producer to work.
> ⚠️ Always `--target-decode` for Phase 1. Opponent paths relative to repo root.

### Full panel eval (legacy — 256 games, ~40 min/opponent on EC2; much slower on Mac CPU)

**Activate venv first** (see top section), then run from repo root:

```bash
# Run all three opponents — use & to run in parallel, save logs
source /Users/saheb/home/.venv/bin/activate

python orbit_wars_rl/eval.py \
  --checkpoint gpu_run_artifacts/hellburner_spot/checkpoints/<name>.pt \
  --opponent opponents/candidate_hellburner.py \
  --panel --target-decode \
  2>&1 | tee gpu_run_artifacts/hellburner_spot/panels/eval_hellburner.log &

python orbit_wars_rl/eval.py \
  --checkpoint gpu_run_artifacts/hellburner_spot/checkpoints/<name>.pt \
  --opponent opponents/candidate_zach_public.py \
  --panel --target-decode \
  2>&1 | tee gpu_run_artifacts/hellburner_spot/panels/eval_zach.log &

python orbit_wars_rl/eval.py \
  --checkpoint gpu_run_artifacts/hellburner_spot/checkpoints/<name>.pt \
  --opponent opponents/candidate_suneet_lb1200.py \
  --panel --target-decode \
  2>&1 | tee gpu_run_artifacts/hellburner_spot/panels/eval_suneet.log &

wait  # block until all three finish
```

**Check progress while running:**
```bash
tail -3 gpu_run_artifacts/hellburner_spot/panels/eval_hellburner.log
tail -3 gpu_run_artifacts/hellburner_spot/panels/eval_zach.log
tail -3 gpu_run_artifacts/hellburner_spot/panels/eval_suneet.log
```

**Read results when done** (look for `Overall:` line):
```bash
grep "Overall" gpu_run_artifacts/hellburner_spot/panels/eval_hellburner.log
grep "Overall" gpu_run_artifacts/hellburner_spot/panels/eval_zach.log
grep "Overall" gpu_run_artifacts/hellburner_spot/panels/eval_suneet.log
```

> ⚠️ Opponent paths are relative to **repo root**, not `orbit_wars_rl/`. Path is `opponents/candidate_*.py`.
> ⚠️ Always `--target-decode` for Phase 1 checkpoints (absolute 32-bin mode).
> ⚠️ Do NOT use `nohup python ...` — `python` alone fails. Either activate venv first or use full path.

---

## 4. EC2 instance management

```bash
# List running/stopped instances
aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=orbit-wars" "Name=instance-state-name,Values=running,stopped" \
  --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceType,PublicIpAddress]' \
  --output table

# Terminate (only after rsync — see section 5)
aws ec2 terminate-instances --instance-ids <id>

# Check state after terminate
aws ec2 describe-instances --instance-ids <id> \
  --query 'Reservations[0].Instances[0].State.Name' --output text
```

> ⚠️ Never terminate without pulling checkpoints first (section 5).
> ⚠️ Never leave in `stopped` state — EBS still bills. Always terminate.

---

## 5. Final rsync before terminating an instance

```bash
# SSH in first and confirm checkpoints exist
ssh -i ~/.ssh/samosa-key.pem ubuntu@<PUB_IP> "ls ~/orbit_wars_rl/checkpoints/*.pt | wc -l"

# Pull all checkpoints
rsync -az -e "ssh -i ~/.ssh/samosa-key.pem -o StrictHostKeyChecking=no" \
  ubuntu@<PUB_IP>:~/orbit_wars_rl/checkpoints/ \
  gpu_run_artifacts/hellburner_spot/checkpoints/

# Pull training log (using include/exclude to avoid pulling entire directory)
rsync -az -e "ssh -i ~/.ssh/samosa-key.pem -o StrictHostKeyChecking=no" \
  --include='train_gpu_*.log' --exclude='*' \
  ubuntu@<PUB_IP>:~/orbit_wars_rl/ \
  gpu_run_artifacts/hellburner_spot/logs/

# Confirm checkpoint count locally
ls gpu_run_artifacts/hellburner_spot/checkpoints/*.pt | wc -l

# Now safe to terminate
aws ec2 terminate-instances --instance-ids <id>
```

---

## 6. Launch a new training run

```bash
# Always use the launch script — never raw aws ec2 run-instances
bash gpu_run_artifacts/hellburner_spot/launch_phase1_rev7.sh

# The launch script handles: EC2 launch, rsync, W&B auth, seed checkpoint upload, tmux start
```

**After launch — start local watcher** (syncs checkpoints every 3 min):
```bash
nohup bash gpu_run_artifacts/hellburner_spot/watch_phase1.sh \
  > gpu_run_artifacts/hellburner_spot/logs/watcher_rev7.log 2>&1 &
echo "Watcher PID: $!"
```

> ⚠️ Edit `watch_phase1.sh` first — hardcode `INSTANCE_ID` and `PUB_IP` from the launch output.
> ⚠️ Use `nohup bash ...` not `nohup python ...`. The watcher is a shell script.

---

## 7. Kill/check local background processes

```bash
# Check what eval jobs are running (shows checkpoint name and opponent)
ps aux | grep eval.py | grep -v grep | awk '{print $2, $11, $14, $16}'

# Check what watchers are running
ps aux | grep "run_panel_eval_watcher\|watch_phase1" | grep -v grep | awk '{print $2, $11}'

# Kill all panel eval watchers (use when changing opponent paths or run version)
ps aux | grep "run_panel_eval_watcher" | grep -v grep | awk '{print $2}' | xargs kill

# Kill evals using wrong opponent paths (e.g., old candidate_*.py not opponents/)
ps aux | grep "eval.py" | grep -v grep | grep -v "opponents/" | awk '{print $2}' | xargs kill
```

### Long local jobs: use `tmux`, not plain `nohup ... &`

For long-running local Python jobs such as BC training, prefer a dedicated
`tmux` session. On this machine, detached shell jobs can disappear without
leaving a useful log, while `tmux` keeps a real terminal attached to the
process.

```bash
# Start a persistent tmux session for BC training
tmux new-session -d -s bc_isaiah_hober_5k \
  'cd /Users/saheb/home/kaggle-orbit-wars && \
   exec stdbuf -oL -eL orbit_wars_rl/.venv/bin/python -u orbit_wars_rl/bc.py \
     --samples /tmp/targeted_bc/isaiah_hober_pressure_merged.pkl \
     --steps 5000 \
     --save orbit_wars_rl/checkpoints/bc_isaiah_hober_pressure_5k.pt \
     2>&1 | tee /tmp/targeted_bc/bc_isaiah_hober_pressure_5k.log'

# Confirm session exists
tmux list-sessions | grep bc_isaiah_hober_5k

# Attach to the live job
tmux attach -t bc_isaiah_hober_5k

# Read the live log without attaching
tail -f /tmp/targeted_bc/bc_isaiah_hober_pressure_5k.log

# Stop the job from another shell
tmux send-keys -t bc_isaiah_hober_5k C-c

# Remove the finished session
tmux kill-session -t bc_isaiah_hober_5k
```

> ⚠️ For local BC/training, treat `tmux` as the default. Use plain background
> shell jobs only for small helpers/watchers.

---

## 8. Diagnose a silent eval failure

If eval logs show only startup noise (INFO: OpenSpiel...) and no `panel progress:` lines:

```bash
# Check the .failed files the panel watcher creates
ls gpu_run_artifacts/hellburner_spot/panels/*.failed | head -5
cat gpu_run_artifacts/hellburner_spot/panels/<name>.vs_hellburner.panel.txt.failed | grep -E "Error|Traceback|FileNotFoundError"
```

Common causes:
| Symptom | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: candidate_hellburner.py` | Old path, before `opponents/` move | Change to `opponents/candidate_hellburner.py` |
| `nohup: python: No such file or directory` | `python` not found, wrong venv | Activate venv: `source /Users/saheb/home/.venv/bin/activate` |
| eval hangs with 0 progress for >10 min | other CPU-heavy processes competing | Check `ps aux \| grep eval.py`; kill duplicates |
| W&B auth failure on remote | key not passed to tmux session | Re-run `wandb login --relogin <key>` on the remote |

---

## 9. Review a submission's loss replays

### One command: fetch loss replays and run the target audit

Use this as the default daily workflow after a submission has enough episodes.

```bash
orbit_wars_rl/.venv/bin/python orbit_wars_rl/review_submission_targets.py \
  --submission-id 53359633 \
  --checkpoint seed_checkpoints/rev31_31M_resume.pt \
  --player-name Saheb
```

Outputs go to:
```text
/tmp/sub53359633_review/
  episodes.json          # full Kaggle manifest for the submission
  selected_losses.json   # metadata for the loss slice actually audited
  replays/*.json         # downloaded replay payloads
  target_audit.json      # machine-readable audit
  target_audit.md        # readable report
```

Useful flags:
```bash
# only 2-player losses
orbit_wars_rl/.venv/bin/python orbit_wars_rl/review_submission_targets.py \
  --submission-id 53359633 \
  --checkpoint seed_checkpoints/rev31_31M_resume.pt \
  --player-name Saheb \
  --only-two-player

# cap review size and focus on the opening
orbit_wars_rl/.venv/bin/python orbit_wars_rl/review_submission_targets.py \
  --submission-id 53359633 \
  --checkpoint seed_checkpoints/rev31_31M_resume.pt \
  --player-name Saheb \
  --max-episodes 10 \
  --step-limit 40
```

What the wrapper does:
- fetches the richer Kaggle `EpisodeService/ListEpisodes` manifest
- selects losses for the requested submission
- downloads replay JSONs with retry/backoff
- runs the same target audit used for manual replay review

### Audit target selection with the actual checkpoint

Use this when the replay viewer makes a move look like an aiming miss or a bad
planet choice and you want to know what the submitted policy actually did.

```bash
source /Users/saheb/home/.venv/bin/activate

python orbit_wars_rl/audit_submission_targets.py \
  --checkpoint gpu_run_artifacts/h100_rev31/checkpoints/torch_step_10485760_rev31_20260603_153146.pt \
  --replay-dir /tmp/sub53336058_eps \
  --player-name Saheb \
  --output-json /tmp/rev31_target_audit.json \
  --output-md /tmp/rev31_target_audit.md
```

Read the saved summary:
```bash
sed -n '1,160p' /tmp/rev31_target_audit.md
```

Audit just a few episodes:
```bash
python orbit_wars_rl/audit_submission_targets.py \
  --checkpoint gpu_run_artifacts/h100_rev31/checkpoints/torch_step_10485760_rev31_20260603_153146.pt \
  --replay-dir /tmp/sub53336058_eps \
  --episode-id 78686024 \
  --episode-id 78680551 \
  --episode-id 78656975 \
  --player-name Saheb \
  --output-md /tmp/rev31_target_audit_focus.md
```

The audit reports, per launch:
- decoded chosen target
- nearest / cheapest / best-tempo alternatives
- weakest nearby / highest-production nearby alternatives
- open-space / decode-fail count
- angle error vs the decoded target intercept

This is the quick way to separate:
- actual aiming/intercept misses
- target-priority mistakes
- “looks wrong in viewer, but the model deliberately chose that farther planet”

### Compare tempo metrics across checkpoints on fixed Ajay seeds

Use this when you want the exact comparison we just did: generate fresh replays
for a fixed opponent/seed slice, audit them, and print checkpoint-to-checkpoint
tempo metrics in one command.

```bash
orbit_wars_rl/.venv/bin/python orbit_wars_rl/compare_tempo_checkpoints.py \
  --checkpoints \
    gpu_run_artifacts/jarvis_rev33/checkpoints/torch_step_1572864_rev33_20260604_144227.pt \
    gpu_run_artifacts/jarvis_rev33/checkpoints/torch_step_3145728_rev33_20260604_144227.pt \
    gpu_run_artifacts/jarvis_rev33/checkpoints/torch_step_4194304_rev33_20260604_144227.pt \
    gpu_run_artifacts/jarvis_rev33/checkpoints/torch_step_5242880_rev33_20260604_144227.pt \
  --opponent opponents/candidate_ajay_1200.py \
  --target-decode \
  --step-limit 40 \
  --output-dir /tmp/rev33_ajay_compare
```

Outputs:
- per-checkpoint replay JSONs under `/tmp/rev33_ajay_compare/<checkpoint-stem>/replays`
- per-checkpoint audit files:
  - `audit.json`
  - `audit.md`
- merged comparison summary:
  - `/tmp/rev33_ajay_compare/summary.json`
  - `/tmp/rev33_ajay_compare/summary.md`

The comparison summary includes:
- `farther_than_nearest_rate`
- `mean_eta_gap_vs_nearest`
- `mean_distance_gap_vs_nearest`
- `tempo_match_rate`
- `nearest_match_rate`
- `mean_first_capture_step`
- `invalid_raw_argmax`

### Build a conversion-focused auxiliary BC dataset

Use this when target ranking has improved but the opening still captures too
late because it spends too many ships before the first productive capture.

This extracts teacher openings that:
- face early pressure
- get the first extra planet early
- keep pre-capture ship spend restrained

It can also mix in the earlier target-relabel failure dataset so the resulting
auxiliary BC still carries the tempo-target correction.

```bash
orbit_wars_rl/.venv/bin/python orbit_wars_rl/build_conversion_bc.py \
  --teacher-replay-dir /tmp/orbit_episodes \
  --teacher-agent "Isaiah @ Tufa Labs" \
  --teacher-agent "Hober Malloc" \
  --teacher-agent "Ebi" \
  --teacher-steps-max 40 \
  --teacher-require-opponent-first-fire-by 12 \
  --teacher-max-first-capture-step 14 \
  --teacher-max-ships-before-first-capture 36 \
  --teacher-max-launches-before-first-capture 3 \
  --teacher-stop-at-first-capture \
  --failure-audit-json /tmp/sub53359633_target_audit.json \
  --failure-audit-json /tmp/ajay_panel_seat0_audit.json \
  --failure-relabel-mode tempo \
  --failure-min-eta-regret 4 \
  --failure-steps-max 40 \
  --samples-out /tmp/targeted_bc/conversion_mix_loose.pkl \
  --summary-out /tmp/targeted_bc/conversion_mix_loose_summary.json
```

**Calibration (full June-3 dataset, loose thresholds):**
- 316 teacher replays kept → 436 teacher samples
- 817 failure relabels
- Total: 1,253 samples → `seed_checkpoints/conversion_mix_loose.pkl`

**Strict thresholds** (first-capture ≤12, ships ≤28, launches ≤2) yield only ~9 samples — too tight.
**Loose thresholds** (first-capture ≤14, ships ≤36, launches ≤3) yield 1,253 — use these.

What it writes:
- standard BC sample pickle for `bc.py`
- JSON summary with:
  - kept replay count
  - teacher sample count
  - failure relabel count
  - filter-drop counts

### Low-level notes

Auth split still matters if you debug this manually:

| Endpoint | Auth method |
|----------|-------------|
| `EpisodeService/ListEpisodes` | Basic auth (`~/.kaggle/kaggle.json`) |
| `v1/competitions/episodes/{id}/replay` | Bearer token (`~/.kaggle/access_token`) |

The replay endpoint intermittently EOFs. Always use retry/backoff when
downloading many episodes.

---

## 10. Export a checkpoint as a submission agent

**⚠️ Phase 1 checkpoints require `--target-decode`. Forgetting it produces an agent that scores ~87 on LB (loses everything).**

```bash
source orbit_wars_rl/.venv/bin/activate

# Phase 1 checkpoints (32-bin absolute, action_decode=target) — ALWAYS use --target-decode
python3 orbit_wars_rl/export_agent.py \
  --checkpoint gpu_run_artifacts/hellburner_spot/checkpoints/<name>.pt \
  --output submission_agent.py \
  --target-decode

# Old architecture checkpoints (10-bin fraction) — no --target-decode
python3 orbit_wars_rl/export_agent.py \
  --checkpoint <old_arch_checkpoint>.pt \
  --output submission_agent.py
```

**Verify before submitting:**
```bash
source orbit_wars_rl/.venv/bin/activate
python3 -c "
import kaggle_environments as ke, re
code = open('submission_agent.py').read()
print('target_decode:', re.search(r'_TARGET_DECODE = (\w+)', code).group(1))
print('signature:', re.search(r'def agent\([^)]+\)', code).group(0))
g = {}; exec(code, g); agent = g['agent']
wins = sum(ke.make('orbit_wars').run([agent,'random'])[-1][0]['reward'] > 0 for _ in range(5))
print(f'vs random: {wins}/5')
"
```

**Then submit:**
```bash
kaggle competitions submit orbit-wars -f submission_agent.py -m "description"
```

**Known bugs fixed (2026-05-31):**
1. `def agent(obs)` → `def agent(obs, cfg=None)` — was crashing silently on every LB step
2. Missing `--target-decode` → used angle decode, scored 87 on LB vs 894 expected

---

## 10. Run unit tests

```bash
source /Users/saheb/home/.venv/bin/activate
python -m pytest orbit_wars_rl/tests/ -x -q
```

---

## Key files — what does what

| File | Role |
|------|------|
| `orbit_wars_rl/train_torch.py` | Main training entry point |
| `orbit_wars_rl/eval.py` | Full 256-game panel eval (source of truth) |
| `orbit_wars_rl/quick_eval.py` | Sanity check only — **never use for decisions** |
| `orbit_wars_rl/torch_env.py` | Vectorised GPU env, ship bin decode, SHIP_COUNTS |
| `orbit_wars_rl/features.py` | Feature extraction (Phase 1: planet=20, fleet=13, global=11, pairwise=12) |
| `orbit_wars_rl/model.py` | Entity transformer |
| `orbit_wars_rl/ppo.py` | PPO update + IL regularisation |
| `orbit_wars_rl/opponent_pool.py` | Self-play pool + PFSP sampling |
| `orbit_wars_rl/export_agent.py` | Export checkpoint → submission .py |
| `orbit_wars_rl/bc.py` | BC loss (lazily imported by ppo.py when --il-lambda set) |
| `orbit_wars_rl/build_conversion_bc.py` | Build conversion-focused BC dataset (fast-capture teacher + failure relabels) |
| `orbit_wars_rl/compare_tempo_checkpoints.py` | Compare first-capture/conversion metrics across checkpoints on fixed Ajay seeds |
| `orbit_wars_rl/step_firep.py` | Compare FireP at steps 0-3 across multiple checkpoints (opening aggression) |
| `opponents/orbit_lite/` | Ajay/Producer dependency — intercept aiming, fleet routing (must be present) |
| `orbit_wars_rl/env.py` | Old kaggle_environments wrapper (used by validate_training + tests) |
| `orbit_wars_rl/eval_panel.py` | 128-seed stratified panel used inside eval.py |
| `gpu_run_artifacts/launch_gpu.sh` | **Only** way to launch EC2 — bakes in terminate-on-shutdown |
| `gpu_run_artifacts/hellburner_spot/` | All active training scripts, logs, checkpoints |
| `seed_checkpoints/phase1_resume.pt` | Current Phase 1 resume point (rev5 6M peak, HB=38.7%) |
| `seed_checkpoints/bc_phase1_warmstart.pt` | BC warmstart (IL reference, not used in Rev7+) |
| `opponents/candidate_hellburner.py` | Primary HB opponent |
| `opponents/candidate_zach_public.py` | Primary Zach opponent |
| `opponents/candidate_suneet_lb1200.py` | Primary Suneet opponent |

---

## Common mistakes to avoid

| Mistake | Fix |
|---------|-----|
| RL-agent external opponent (141208) without `--heuristic-workers 2` | Auto-default uses `cpu_count-1=7` workers; each loads a neural net → all 8 CPUs saturated, GPU drops to 0%, training deadlocks. **Always add `--heuristic-workers 2` for RL-agent opponents.** If already hung: `gcloud compute instances reset` (SSH unresponsive). |
| `python train_torch.py` without activating venv | `source /Users/saheb/home/.venv/bin/activate` first, or use full path |
| `nohup python ...` on Mac | Fails silently — activate venv first so `python` resolves |
| `--opponent candidate_hellburner.py` | Path is `opponents/candidate_hellburner.py` |
| `--external-opponents candidate_hellburner.py` | Path is `opponents/candidate_hellburner.py` |
| Forgetting `--panel` in eval | Without it you get a single game, not 256 |
| Forgetting `--target-decode` in eval | Phase 1 checkpoints need this; wrong mode = garbage results |
| Trusting `quick_eval` results | Always run full panel before making decisions |
| Launching with `aws ec2 run-instances` raw | Use `launch_gpu.sh` — raw launch defaults to stop-not-terminate |
| Leaving instance in stopped state | Stopped still bills for EBS; terminate after pulling everything |
| Panel watcher running with old opponent paths | Kill watcher (`ps aux \| grep run_panel_eval_watcher ... xargs kill`), fix paths, restart |
| Running many evals in parallel on Mac | CPU contention → each eval takes 10× longer; limit to 3 at most |
