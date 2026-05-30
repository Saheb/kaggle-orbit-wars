# Orbit Wars — Agent Operating Instructions

Rules every Claude Code agent must follow in this project. These are not suggestions.

---

## Repo Map

```
orbit_wars_rl/          ← ALL active RL code lives here
  train_torch.py        ← PRIMARY: GPU self-play training (always use this)
  eval.py               ← PRIMARY: full 256-game panel eval
  quick_eval.py         ← sanity check only (never use for decisions)
  export_agent.py       ← export checkpoint → submission agent
  model.py              ← entity transformer model
  torch_env.py          ← vectorised GPU environment (ship bins, action decode)
  features.py           ← feature extraction (Phase 1 feature bundle)
  ppo.py                ← PPO learner + IL regularisation
  opponent_pool.py      ← self-play pool + PFSP
  config.py             ← ModelConfig / PPOConfig
  action_mask.py        ← action masking for eval
  bc.py / bc_frac.py    ← behaviour cloning (used to create warmstart)
  tests/                ← unit tests

candidate_hellburner.py    ← PRIMARY eval opponent (HB)
candidate_zach_public.py   ← PRIMARY eval opponent (Zach)
candidate_suneet_lb1200.py ← PRIMARY eval opponent (Suneet)

seed_checkpoints/          ← resume points uploaded to training instances
  phase1_resume.pt         ← current Phase 1 resume checkpoint
  bc_phase1_warmstart.pt   ← BC warmstart (IL reference)

setup/                     ← install orbit_wars kaggle env (run once per instance)
docs/                      ← logs, runbooks, checklists
gpu_run_artifacts/         ← training scripts, watchers, synced checkpoints (gitignored)
archive/                   ← dead code, old logs, old eval scripts (ignore unless archaeology)
```

---

## EC2 / GPU Instances

**NEVER launch an EC2 instance directly with `aws ec2 run-instances` raw.**
Always use `gpu_run_artifacts/launch_gpu.sh` which bakes in cost-safety defaults.

**Why:** Raw `run-instances` defaults `--instance-initiated-shutdown-behavior stop`.
When training finishes and calls `sudo shutdown -h`, the instance *stops* (billing continues for EBS).
`launch_gpu.sh` sets `--instance-initiated-shutdown-behavior terminate` so the instance is destroyed on shutdown.

**Termination rule:** When a training run finishes:
1. SSH into the instance first and confirm all checkpoints are present on disk.
2. Run a final manual rsync before terminating: `rsync -az ubuntu@<ip>:~/orbit_wars_rl/checkpoints/ gpu_run_artifacts/hellburner_spot/checkpoints/`
3. Also pull the full training log: `rsync -az ubuntu@<ip>:~/orbit_wars_rl/train_gpu_*.log gpu_run_artifacts/hellburner_spot/logs/`
4. Only then terminate: `aws ec2 terminate-instances --instance-ids <id>`
5. Never leave an instance in *stopped* state — stopped still bills for EBS storage.

**If instance is already stopped (not terminated):** It still has all data on EBS. SSH in, pull everything, then terminate. Do NOT terminate without pulling first.

**On-demand vs spot:** Always launch **on-demand** (no `--spot` flag). Spot instances can be reclaimed by AWS at any time mid-training, losing all unsaved checkpoints. The watcher syncs every 3 min so at most 3 min of work is lost on-demand if the instance fails; a spot interruption can happen at any point with 2 min notice and no guarantee checkpoints have been synced. On-demand cost for a full training run (~12h on g5.2xlarge) is ~$15 — acceptable. Never pass `--spot` to `launch_gpu.sh` or `launch_phase1.sh`.

**Verify status before starting new runs:**
```bash
aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=orbit-wars" "Name=instance-state-name,Values=running,stopped" \
  --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceType,PublicIpAddress]' \
  --output table
```

---

## Training Runs

- All training scripts live in `gpu_run_artifacts/hellburner_spot/`
- Watcher scripts sync checkpoints + logs every 3 min; always run one when training
- Panel evals (`run_panel_eval_watcher.sh`) auto-run 256-game full panels on each new checkpoint
- **Never trust `quick_eval` for decisions — full panel only**
- One change per run; record hypothesis before launching

### Key flags
| Flag | Purpose |
|------|---------|
| `--terminate-on-done` | Shuts down instance OS when training ends (triggers terminate if launched correctly) |
| `--pool-external-fraction` | Fraction of pool samples guaranteed to go to external opponents (bypasses PFSP) |
| `--pool-pfsp-min-games` | Use wr=0.5 for PFSP weight until N games played (prevents death-spiral) |
| `--instance-initiated-shutdown-behavior terminate` | Set at launch time via `launch_gpu.sh` |

---

## Eval

Full panel command:
```bash
python orbit_wars_rl/eval.py \
  --checkpoint <path> \
  --opponent <candidate_foo.py> \
  --panel \
  --target-decode
```
256 games (128 seeds × 2 seats). Takes ~40 min per opponent on local CPU.

Opponent paths are relative to repo root (not `orbit_wars_rl/`):
- `candidate_hellburner.py`
- `candidate_zach_public.py`
- `candidate_suneet_lb1200.py`

---

## Current Baselines (full panel, 256 games)

| Checkpoint | Hellburner | Zach | Suneet |
|---|---|---|---|
| torch_step_1015808_20260526_141208 | **55.5%** | 74.2% | 80.1% |
| torch_step_1015808_20260526_123203 | 44.5% | 75.4% | 75.8% |
| torch_step_1015808_20260526_174758 (blitz 1M) | 42.6% | 76.6% | 75.0% |

Target: >75% on all three opponents simultaneously.
Best Hellburner checkpoint ever: `torch_step_1015808_20260526_141208.pt` (55.5%).

---

## Key Lessons

1. **Heavy Hellburner pressure (external_fraction=1.0) causes regression** — policy overfits to opponent-specific patterns, loses generalization across panel archetypes. The best Hellburner result (55.5%) came from a run that was mostly self-play.
2. **PFSP death-spiral**: small N of games → noisy wr → near-zero weight → never sampled again. Fix: `--pool-pfsp-min-games 30`.
3. **Pool summary truncation**: `summary(max_rows=8)` drops external opponents once 8+ self-checkpoints accumulate. Fixed in `opponent_pool.py` — externals always shown.
4. **Cosine LR decay to 0**: with `--total-steps 3M`, LR hits 0 at the end. Last ~500k steps have negligible gradient. Use constant LR or longer schedule for longer runs.
