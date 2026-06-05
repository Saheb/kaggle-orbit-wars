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
  model.py              ← entity transformer model (encode_state() + forward())
  torch_env.py          ← vectorised GPU env (SSDR, ship bins, action decode)
  features.py           ← feature extraction (Phase 1 feature bundle)
  ppo.py                ← PPO learner + IL regularisation
  opponent_pool.py      ← self-play pool + PFSP
  config.py             ← ModelConfig / PPOConfig
  action_mask.py        ← action masking for eval
  bc.py / bc_frac.py    ← behaviour cloning (used to create warmstart)
  compare_tempo_checkpoints.py  ← conversion metrics vs Ajay across checkpoints
  build_conversion_bc.py        ← build conversion-focused BC dataset
  step_firep.py                 ← compare FireP at steps 0-3 across checkpoints
  tests/                ← unit tests

opponents/                 ← eval + training opponents
  candidate_hellburner.py
  candidate_zach_public.py   ← primary Zach opponent
  candidate_ajay_1200.py     ← ⭐ PRIMARY eval metric (1200 LB, harder than Zach)
  candidate_producer_1200.py
  candidate_suneet_lb1200.py
  orbit_lite/                ← dependency for Ajay/Producer (intercept aiming etc.)

seed_checkpoints/          ← resume points uploaded to training instances
  rev32b_6M_resume.pt      ← Rev32b 6M (best Zach: 88.7%, 20/21 loss seeds)
  rev35c_1M_resume.pt      ← Rev35c 1M (best Ajay: 3.1%)
  bc_isaiah_hober_pressure_5k.pt  ← BC warmstart (Isaiah+Hober openings)

setup/                     ← install orbit_wars kaggle env (run once per instance)
docs/                      ← runbooks and logs
  commands.md              ← ⭐ copy-paste command reference (start here)
  training.md              ← current training state, full run history, key config
  submissions.md           ← full submission log with Kaggle IDs and checkpoint paths
  GCP_RUNBOOK.md           ← GCP L4 launch, monitoring, terminate
  JARVIS_RUNBOOK.md        ← Jarvis H100 spot instances
gpu_run_artifacts/         ← training scripts, watchers, synced checkpoints (gitignored)
archive/                   ← dead code, old logs (ignore unless archaeology)
```

---

## GPU Instances

**Primary:** GCP L4 (`g2-standard-8`, ~$1.13/hr) — use `bash gpu_run_artifacts/launch_gpu_gcp.sh`
**Secondary:** Jarvis H100 spot (₹112/hr, ~4250 SPS) — use `jl` CLI, see JARVIS_RUNBOOK.md

**Hard rules:**
- GCP: always DELETE (not stop) instances after training — `gcloud compute instances delete`
- Jarvis: always DESTROY (not pause) spot instances — data may not persist on preemption
- **After `launch_gpu_gcp.sh`**: verify sync with `ssh ... "ls ~/orbit_wars_rl/orbit_wars_rl/train_torch.py"` before starting training — rsync can drop mid-transfer
- **Eval on training instances**: always prefix `CUDA_VISIBLE_DEVICES="" python3 orbit_wars_rl/eval.py` — training occupies GPU, eval OOMs otherwise
- Rsync checkpoints: use `-L` flag to follow symlinks (`rsync -azL`)
- One change per run; record hypothesis before launching

### Key training flags
| Flag | Purpose |
|------|---------|
| `--ssdr-frac 0.3` | SSDR: 30% of self-play resets grant opponent 1-2 extra planets |
| `--ssdr-max-steps 2` | Max extra planets granted to opponent in SSDR |
| `--min-ship-bin 4` | Ban bins 0-3 (1-4 ships) to prevent degenerate 1-ship probing |
| `--first-strike-steps 50` | Double capture reward for t<50 (was the LB record fix) |
| `--pool-pfsp-min-games 30` | Prevents PFSP death-spiral |
| `--pool-external-fraction` | Fraction of pool samples to external opponents |

---

## Eval

**Primary metric: Ajay** (`opponents/candidate_ajay_1200.py`) — harder than Zach, requires `orbit_lite/`
**Secondary: Zach** — saturating at ~88-89%, use only as sanity check

```bash
# Full panel (256 games, ~40 min locally)
CUDA_VISIBLE_DEVICES="" python3 orbit_wars_rl/eval.py \
  --checkpoint <path> \
  --opponent opponents/candidate_ajay_1200.py \
  --panel --target-decode

# Quick eval (16 games, ~2 min) — for trend tracking only
CUDA_VISIBLE_DEVICES="" python3 orbit_wars_rl/eval.py \
  --checkpoint <path> \
  --opponent opponents/candidate_ajay_1200.py \
  --games 16 --target-decode
```

Opponent paths relative to repo root. Always `--target-decode` for Phase 1.

---

## Current Baselines (full panel, 256 games vs Ajay)

| Checkpoint | Zach | Ajay | LB | Notes |
|---|---|---|---|---|
| Rev31 10M | 84.8% | — | **918.8** | LB record |
| Rev32b 6M | **88.7%** | 0.8% | pending | Best Zach ever |
| Rev35c 1M | — | **3.1%** | — | Best Ajay ever |

Target: Top 10 LB ≈ 1153 (gap ~234 points from 918.8).

---

## Key Lessons

1. **Zach panel is saturated** (~88-89%) — use Ajay as the primary signal. Ajay uses `orbit_lite` for targeting; our agent also uses orbital intercept — the gap is conversion timing, not routing.
2. **SSDR** (asymmetric planet starts) improves Ajay from 0.8% → 3.1% but improvement is transient — self-play Nash reforms after ~2M steps regardless of pool mask gating.
3. **ship0 collapse** — agent learns to send 1-ship probes when behind. Fix: `--min-ship-bin 4`. Does NOT fix the underlying SSDR regression.
4. **First Strike** (`--first-strike-steps 50 --first-strike-mult 2.0`) fixed opening paralysis and scored 918.8 LB. It's a reward shaping band-aid, not a structural fix.
5. **BC aux at bc-coef=0.05** — too small to move the needle but can disrupt conversion (Rev34: us_first_cap 14→136 after 1M).
6. **Pool mask gating** — SSDR should only apply to self-play envs, not pool envs. Call `env.set_ssdr_mask(mask)` each rollout. Slows regression but doesn't stop it.
7. **PFSP death-spiral**: small N games → noisy wr → never sampled. Fix: `--pool-pfsp-min-games 30`.
8. **LR=0.000025** is the mature-run rate (halved twice). Use LR=0.0001 for fresh BC warmstarts.
9. **BC warmstart from partial/diagnostic checkpoints** causes clip_frac=0 (frozen policy). Always use a strong PPO checkpoint as `--resume`, BC aux via `--bc-samples`.
10. **Export**: always use `--target-decode` for Phase 1 checkpoints. Run 10/10 vs random before submitting.
