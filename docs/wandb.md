# W&B Tracking — How To

W&B is **on by default**. Every training run logs ~60 metrics organized by prefix.
This doc covers setup, the panels that matter, and the decisions you make from them.

---

## Setup (one-time per box)

```bash
pip install wandb
wandb login          # paste API key from https://wandb.ai/authorize
```

Verify:
```bash
python -c "import wandb; print(wandb.__version__)"
```

If wandb is not installed, training runs fine without it (silent skip + warning).
To explicitly disable for a run: `--no-wandb`.

---

## What's logged

Metrics are grouped by prefix — W&B auto-organizes them into sections. You only
pin the panels you care about. Here are the ones that matter, by purpose:

### PBRS run — the 5 panels to pin

| Panel | Metric | What it tells you | Decision |
|-------|--------|-------------------|----------|
| Staging Φ | `staging/phi` | Is the agent staging inflight toward neutrals? | Should rise if PBRS is biting. Flat = PBRS inert. |
| Fire fraction | `policy/fire_fraction` | Mean fraction of owned planets firing per step | Entropy spike pushes this up. If flat at 1-2M → clip bottleneck. |
| Neutral targeting | `policy/target_share_neutral` | Are fires going to neutrals (PBRS-directed) or enemies/own? | Should climb with PBRS. If fire_frac up but this flat → entropy noise, not PBRS. |
| Critic health | `ppo/explained_variance` | Is V(s) tracking returns? | Should stay >0.8 after warmup. Dropping = critic broken. |
| Game reward | `train/reward_p0` | Zero-sum episode reward (p0 vs p1, should mirror) | Sanity check only — real signal is Ajay panel eval on checkpoints. |

### General health — pin if investigating

| Prefix | Key metrics | When to look |
|--------|-------------|--------------|
| `ppo/` | clip_frac, approx_kl, value_loss | PPO divergence (clip>0.3, kl>0.05) |
| `entropy/` | fire, ship, target | Collapse (any →0) or spray (fire too high) |
| `policy/` | fire_0, ship_bin0_rate, reinforce_rate | Degenerate behavior (slot0>0.8, bin0>0.5) |
| `value/` | mean, std, adv_std | Critic scale drift |
| `dm/` | gap, cross, ratio, nearmiss | Force-concentration diagnostic (decisive-mass floor) |
| `il/` | kl, coef | IL anchor (zero when anchor off) |
| `train/` | sps, lr | Throughput + LR schedule |

---

## The entropy-spike manual-cut workflow

The PBRS run launches with `--entropy-coef-fire 0.05` (5x normal). You cut it
to `0.02` manually when PBRS is clearly directing fires. Here's how to decide:

1. **Watch `policy/fire_fraction`** (wandb). Entropy alone will push it up
   from baseline (~0.10) toward ~0.20-0.30. This is expected — NOT sufficient
   to cut yet.

2. **Run the spare-fire diagnostic on the latest checkpoint:**
   ```bash
   /Users/saheb/home/.venv/bin/python orbit_wars_rl/value_spare_diagnostic.py \
     --checkpoint <latest.pt> \
     --opponent opponents/candidate_ajay_1200.py \
     --seeds 12
   ```
   Look at the `fired-spare rate` line.

3. **Decision:**
   - Spare-fire > 8% AND fire_fraction climbing → **cut entropy to 0.02**
     (stop + resume with `--entropy-coef-fire 0.02`). PBRS is directing fires
     to the right place; entropy has done its job.
   - Fire_fraction up but spare-fire still ~4% → **keep 0.05**. Entropy is
     causing noise, PBRS hasn't taken hold yet. Investigate (check `staging/phi`
     is rising, check Φ isn't blocked by floor being too conservative).
   - Fire_fraction flat at 1-2M → **clip bottleneck confirmed**. The policy
     can't move through the low-probability trap. Consider higher entropy
     (0.08) or investigate the fire-head logits directly.

4. **Spray tripwire:** if fire_fraction balloons past ~0.4 without
   `policy/target_share_neutral` climbing, kills are not improving → spray
   pathology. Cut entropy immediately and investigate the PBRS gate.

---

## Comparing runs

W&B groups runs by project (`orbit-wars` by default, override with
`--wandb-project`). To compare two runs:

1. Open the project page.
2. Select both runs in the left sidebar.
3. Any panel now overlays both lines.

Useful comparisons:
- **PBRS vs baseline:** overlay `staging/phi`, `policy/fire_fraction`,
  `policy/target_share_neutral`. PBRS run should separate on all three.
- **Entropy spike effect:** overlay `entropy/fire` and `policy/fire_fraction`
  across the cut point (resume creates a new W&B run; compare side by side).

---

## Notes

- W&B run name defaults to the training timestamp. Override with `--run-name`.
- The run config (all CLI flags including `staging_shaping_coef`,
  `staging_topk`, `entropy_coef_fire`) is logged at init — visible in the
  W&B UI under the run's Config panel.
- Resume creates a new W&B run (not a continuation). Use `resume="allow"`
  in the init call if you want to resume a W&B run by name.
- If the box loses internet, W&B caches locally and syncs when reconnected.
  Training is unaffected.
