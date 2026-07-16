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

## 2b. ⭐ Behavioural probes — run these BEFORE proposing a lever

Cheap (minutes), and on 2026-07-16 they killed three beliefs and a ~30h experiment. See
CLAUDE.md Key Lessons 12–14 and docs/training.md.

```bash
# What does a top-10 agent actually SEND?  ships_sent / source_garrison histogram.
# Verdict: Ender all-ins 97.3% vs Ajay, 97.7% vs ITSELF => a learned middle is worth <=3%.
CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/ender_sizing.py --seeds 6
CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/ender_sizing.py --seeds 5 \
    --opponent opponents/candidate_ender.py          # strong-vs-strong control (do not skip)
CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/ender_sizing.py --seeds 6 \
    --agent-checkpoint <ckpt.pt> --opponent opponents/candidate_ender.py   # OURS, like-for-like

# Why do we lose captures?  Per-capture forensics; splits "took what we can't hold" from
# "held fine, out-produced later". Refuses to run on allow_reinforce=False.
CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/peel_diagnosis.py --seeds 6

# How much of the action space do the hardcoded gates delete?  (was: 80.2%)
python gpu_run_artifacts/ender_ref/probe_binary_gate_pressure.py
```

⚠️ **Any probe that builds its own agent must set the model attributes `evaluate_checkpoint` sets**
(`allow_reinforce`, `reinforce_gate_min_planets`, `reinforce_garrison_floor`,
`reverse_edge_cooldown`, `sufficient_commit_factor`). `build_agent_fn` reads them **off the model
object** (eval.py:391/:1560), NOT from its kwargs — forgetting them silently disables
reinforcement and you measure your own config. See CLAUDE.md Key Lesson 14.

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

### Cross-checkpoint eval (cycling / forgetting detector)

Evals a checkpoint vs HELD-OUT fast heuristics **and our own best past selves**
(exported to `.py`). A drop vs a fixed heuristic = absolute regression; a drop vs
a *past self* = forgetting/cycling — the failure self-play win-rate is blind to.
Run every ~1M steps during training.

```bash
bash gpu_run_artifacts/cross_eval/run_cross_eval.sh <checkpoint.pt> [games=48]
```

Opponent set (all held-out, none in the training pool): zach, 1166-peak,
1043-simple (floor), and exported `self_rev38` / `self_rev53b`.

Re-export a past self as an opponent (needs same feature arch = pairwise-15; rev38+):
```bash
python orbit_wars_rl/export_agent.py --checkpoint <best>.pt \
  --output opponents/ourbest/<name>.py --target-decode
```

### Pool diversity — preseed cross-run best selves

Fold our best *cross-run* checkpoints into the training pool as fast GPU `self`
members (more strategic diversity than dense same-run snapshots → less cycling).
Symlinked under `gpu_run_artifacts/preseed_pool/` (rev38, rev53b — pairwise-15).
Add to a training launch (keep it a *separate* delta — don't bundle with a
method test like VDN):
```bash
  --preseed-pool ../preseed_pool   # appended as 'self' members; step parsed from filename
```
> ⚠️ Older checkpoints (rev31/rev32b = pairwise-12) won't load under the current
> pairwise-15 model — they need the old config to export/preseed (follow-up).

### Producer-style ranking audit on replay losses

```bash
source /Users/saheb/home/.venv/bin/activate

python orbit_wars_rl/producer_ranking.py \
  --checkpoint gpu_run_artifacts/jarvis/checkpoints/<checkpoint>.pt \
  --replay-dir /tmp/ajay_seed_replays \
  --player-slot 0 \
  --step-limit 40 \
  --output-json /tmp/producer_ranking.json \
  --output-md /tmp/producer_ranking.md
```

This compares, per launch:
- our decoded target
- our target-head top target
- producer-style shortlist best target (`H=18` in 2P)

Use it to check whether failures come from:
- target ranking
- source/shortlist mismatch
- commitment / no valid producer candidate

### Producer whole-action ranking audit

```bash
source /Users/saheb/home/.venv/bin/activate

python orbit_wars_rl/producer_action_ranking.py \
  --replay-dir /tmp/ajay_seed_replays \
  --player-slot 0 \
  --step-limit 40 \
  --output-json /tmp/producer_action_ranking.json \
  --output-md /tmp/producer_action_ranking.md
```

This compares replay launches against Producer-best **whole actions**:
- best `(source, target, ships)` candidate in the state
- whether replay source matches producer-best source
- whether replay target matches producer-best target
- rank/score of the replay move under the same action scorer

### Producer-target BC dataset for target-head supervision

```bash
source /Users/saheb/home/.venv/bin/activate

python -m orbit_wars_rl.scripts.build_producer_target_bc \
  --replay-dir /tmp/sub53359633_eps \
  --player-name Saheb \
  --step-limit 40 \
  --mismatch-repeat 4 \
  --samples-out /tmp/producer_target_bc.pkl \
  --summary-out /tmp/producer_target_bc_summary.json
```

Then fine-tune only the target modules:

```bash
source /Users/saheb/home/.venv/bin/activate

python orbit_wars_rl/bc.py \
  --samples /tmp/producer_target_bc.pkl \
  --init-checkpoint seed_checkpoints/rev31_31M_resume.pt \
  --trainable-param tgt_q \
  --trainable-param tgt_k \
  --trainable-param target_scorer \
  --steps 300 \
  --save orbit_wars_rl/checkpoints/bc_producer_target_smoke.pt
```

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

**After launch — start watchers via the CONTROLLER** (sync + held-out eval; never ad-hoc per-run scripts — they go stale and watch the *previous* run's folder):
```bash
# platform = jarvis (target=IP) | gcp (target=config-ssh alias) | custom (set RSYNC_SSH/HOST/REMOTE_*_DIR env)
bash gpu_run_artifacts/run_watchers.sh start <run> jarvis <IP>      # e.g. start p2rev5 jarvis 217.18.55.11
bash gpu_run_artifacts/run_watchers.sh status                       # active run + live procs
bash gpu_run_artifacts/run_watchers.sh stop                         # kill all
```

> `start` tears down ALL existing watchers first, and each watcher self-terminates when `.active_run`
> changes — so stale prior-run watchers can't accumulate. Held-out eval defaults to Ajay full-panel with
> masks gate3/floor0/no-forward-only (override via `REINFORCE_MASKS` if a run trains different masks).
> Launch scripts should end with a `run_watchers.sh start` call so this is never forgotten. Old ad-hoc
> `watch_phase1.sh`/`*_watch.sh`/`sync_watcher.sh` are DEPRECATED.

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

### Kaggle CLI 2.1.2+ — episodes / replays / logs (native subcommands)

The simplest way to pull a submission's games — no API tokens or custom scripts needed
(`kaggle --version` ≥ 2.0.2). The agent index matches `info.TeamNames` order in the replay.

```bash
# List episodes for a submission (find SUBMISSION_ID via `kaggle competitions submissions orbit-wars`)
kaggle competitions episodes <SUBMISSION_ID>            # table
kaggle competitions episodes <SUBMISSION_ID> -v         # CSV (col 1 = episode id) — for scripting

# Download a replay JSON (for analysis or the viewer)
kaggle competitions replay <EPISODE_ID>                 # -> episode-<id>-replay.json in cwd
kaggle competitions replay <EPISODE_ID> -p ./replays    # into a dir

# Download YOUR agent's logs (index 0..3; you can only fetch your own seat)
kaggle competitions logs <EPISODE_ID> 0 -p ./logs
```

**Quick 2p-vs-4p win-rate from a submission's replays** (the FFA-baseline check, 2026-06-08):
```bash
mkdir -p /tmp/sub_replays
kaggle competitions episodes 53802378 -v 2>/dev/null | tail -n +2 | cut -d',' -f1 \
  | while read ep; do kaggle competitions replay "$ep" -p /tmp/sub_replays >/dev/null 2>&1; done
orbit_wars_rl/.venv/bin/python - <<'PY'
import json, glob
from collections import Counter
pc=Counter(); n2=w2=n4=0; w4=0
for f in glob.glob('/tmp/sub_replays/episode-*-replay.json'):
    d=json.load(open(f)); names=d['info']['TeamNames']
    if 'Saheb' not in names: continue
    i=names.index('Saheb'); rew=d['rewards']; win=rew[i]==max(rew) and rew.count(max(rew))==1
    if len(names)==2: n2+=1; w2+=win
    elif len(names)==4:
        n4+=1; w4+=win
        sc=[0.0]*4                              # final placement from last-step planet ships
        for p in d['steps'][-1][0]['observation']['planets']:
            o=int(p[1]);  sc[o]+= p[5] if 0<=o<4 else 0
        pc[1+sum(s>sc[i] for s in sc)]+=1
print(f"2p {w2}/{n2} ({100*w2/max(n2,1):.0f}%, par 50%) | 4p {w4}/{n4} ({100*w4/max(n4,1):.0f}%, par 25%) | 4p placement {dict(pc)}")
PY
```
> Win = our reward is the unique max (orbit_wars rewards are ±1). Mind the baselines: 2p par = 50%, 4p par = 25%.
> ⚠️ The `4p placement` above is **naive (final-score)** and is an ARTIFACT — FFA ends with one survivor, so all eliminated players tie at score 0 and look "2nd." For real 4p placement, rank by **elimination order** (last step each player still owned a planet/fleet), not the final-score snapshot. See `project_ffa_not_the_gap`.

### One command: fetch loss replays and run the target audit

Use this as the default daily workflow after a submission has enough episodes.

```bash
orbit_wars_rl/.venv/bin/python -m orbit_wars_rl.scripts.review_submission_targets \
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
orbit_wars_rl/.venv/bin/python -m orbit_wars_rl.scripts.review_submission_targets \
  --submission-id 53359633 \
  --checkpoint seed_checkpoints/rev31_31M_resume.pt \
  --player-name Saheb \
  --only-two-player

# cap review size and focus on the opening
orbit_wars_rl/.venv/bin/python -m orbit_wars_rl.scripts.review_submission_targets \
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
orbit_wars_rl/.venv/bin/python -m orbit_wars_rl.scripts.compare_tempo_checkpoints \
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
orbit_wars_rl/.venv/bin/python -m orbit_wars_rl.scripts.build_conversion_bc \
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
| `orbit_wars_rl/scripts/build_producer_target_bc.py` | Build producer-labeled target-only BC dataset from replay launches |
| `orbit_wars_rl/scripts/build_conversion_bc.py` | Build conversion-focused BC dataset (fast-capture teacher + failure relabels) |
| `orbit_wars_rl/scripts/compare_tempo_checkpoints.py` | Compare first-capture/conversion metrics across checkpoints on fixed Ajay seeds |
| `orbit_wars_rl/scripts/eval_joint_opening.py` | Opening-only live prototype: joint action scorer picks early moves, base policy handles the rest |
| `orbit_wars_rl/scripts/step_firep.py` | Compare FireP at steps 0-3 across multiple checkpoints (opening aggression) |
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
| Eval OOM on training instance | Training occupies GPU; eval auto-detects CUDA and OOMs. Always prefix: `CUDA_VISIBLE_DEVICES="" python3 orbit_wars_rl/eval.py ...` on any instance running training |

---

## 11. Download top-player / winner replays (for BC datasets)

`fetch_analyze_top_replays.py` pulls a **score-sorted slice** of Kaggle's daily Orbit Wars
episode datasets (`kaggle/orbit-wars-episodes-YYYY-MM-DD`), not the full ~20GB/day. These are
the top-of-leaderboard games used to build the replay-action BC corpus (`ar_stage0/replays/top2`).
Needs `~/.kaggle/kaggle.json`. Today's dataset publishes ~00:10 UTC next day; unpublished dates
return 403 and are skipped, so `--last-days` is always safe to run.

```bash
# Download the last 14 days' top-100 two-player games into a worktree dir (no analysis):
/Users/saheb/home/.venv/bin/python -m orbit_wars_rl.scripts.fetch_analyze_top_replays \
  --last-days 14 --n-per-day 100 --agent-count 2 \
  --out-dir gpu_run_artifacts/ar_stage1/replays

# Specific dates instead of recent N:
/Users/saheb/home/.venv/bin/python -m orbit_wars_rl.scripts.fetch_analyze_top_replays \
  --dates 2026-06-17 2026-06-18 --n-per-day 100 --agent-count 2 --out-dir <dir>

# Add --analyze to also run the winner behavioural characterisation; --no-download to
# re-analyze an existing dir; --player / --exclude to focus on or skip a named player.
```

Then rebuild the BC dataset over the combined replay dirs (see `docs/replay-action-bc.md`):
build with the worktree's `build_replay_action_bc.py` so samples are emitted at the phase4
**20-dim** feature width (`PAIRWISE_FEATURE_DIM=20`), not main's 41-dim.
