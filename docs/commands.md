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

## 3. Full panel eval (256 games, ~40 min/opponent on EC2; much slower on Mac CPU)

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

## 9. Download our LB episode replays for analysis

### Step 1 — list our submission's episodes
```python
# Run from repo root with venv activated
import requests, json

with open('/Users/saheb/.kaggle/access_token') as f:
    token = f.read().strip()
with open('/Users/saheb/.kaggle/kaggle.json') as f:
    creds = json.load(f)

# Our best submission ID (check: kaggle competitions submissions --competition orbit-wars --csv)
SUB_ID = 53076736

url = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
resp = requests.post(url,
    auth=(creds['username'], creds['key']),
    json={"submissionId": SUB_ID},
    headers={"Content-Type": "application/json", "Accept": "application/json"})

data = resp.json()
print(f"Found {len(data['episodes'])} episodes")
json.dump(data, open('/tmp/our_episodes.json', 'w'))
```

### Step 2 — analyse win/loss patterns
```python
import json, collections

with open('/tmp/our_episodes.json') as f:
    episodes = json.load(f)['episodes']
OUR_SUB = 53076736

wins, loss_data = 0, []
for ep in episodes:
    our = next((a for a in ep['agents'] if a['submissionId'] == OUR_SUB), None)
    if not our: continue
    if our['reward'] > 0:
        wins += 1
    else:
        for a in ep['agents']:
            if a['submissionId'] != OUR_SUB and a['reward'] > 0:
                loss_data.append((a['initialScore'], ep['id'], our['initialScore']))

print(f"W={wins} L={len(loss_data)} WR={wins/(wins+len(loss_data))*100:.0f}%")
# Losses to weaker opponents (most actionable):
weak = [(s,ep,our) for s,ep,our in loss_data if s < our]
print(f"Losses to weaker opponents: {len(weak)}/{len(loss_data)}")
```

### Step 3 — download specific replay JSONs
```python
import requests, os, time

with open('/Users/saheb/.kaggle/access_token') as f:
    token = f.read().strip()
# ⚠️ MUST use Bearer token — basic auth returns 401 for replay endpoint

os.makedirs('/tmp/our_losses', exist_ok=True)
ep_ids = [78025831, 78083575, ...]  # from step 2

for ep_id in ep_ids:
    r = requests.get(
        f"https://www.kaggle.com/api/v1/competitions/episodes/{ep_id}/replay",
        headers={"Authorization": f"Bearer {token}"}, timeout=30)
    if r.status_code == 200:
        open(f"/tmp/our_losses/{ep_id}.json", "wb").write(r.content)
    time.sleep(0.3)  # avoid 429
```

> **Auth note:** The `~/.kaggle/access_token` file is a Bearer token (separate from `kaggle.json` which has API key for basic auth). Both are needed for different endpoints.
>
> | Endpoint | Auth method |
> |----------|-------------|
> | `EpisodeService/ListEpisodes` | Basic auth (username + API key) |
> | `v1/competitions/episodes/{id}/replay` | Bearer token (`access_token`) |

### Step 4 — analyse behaviour in replays
```python
import json, glob, statistics

def analyze_replay(path):
    ep = json.load(open(path))
    rewards = ep['rewards']
    winner_slot = rewards.index(max(rewards))
    results = {}
    for slot, name in enumerate(a['Name'] for a in ep['info']['Agents']):
        fire_steps = total = multi = 0; ships = []
        for step in ep['steps'][1:]:
            if slot >= len(step): continue
            acts = step[slot].get('action', [])
            total += 1
            if acts:
                fire_steps += 1
                ships.extend(a[2] for a in acts if len(a) >= 3)
                if len(acts) > 1: multi += 1
        results[slot] = dict(name=name, won=(slot==winner_slot),
            fire_rate=fire_steps/max(total,1),
            avg_ship=statistics.mean(ships) if ships else 0,
            multi_rate=multi/max(fire_steps,1), n_steps=len(ep['steps']))
    return results
```

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
