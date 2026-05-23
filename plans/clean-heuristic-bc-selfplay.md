# Plan: Clean essential heuristic → BC → BC+self-play PPO

## Context

Recent evidence forced a strategic reset:

- **Pure self-play (15 M+ steps in the on-demand cloud run)**: 0 measurable progress vs raw Suneet/Zach. Replay inspection of yesterday's `torch_step_10027008.pt` checkpoint showed the policy has collapsed to a degenerate behavior — **median fleet size = 1 ship**, 1.1 moves/active-turn vs Suneet's 3.6, only 5 planets captured per 5 games vs Suneet's 40. This is the classic self-play cold-start failure: in mirror self-play, "do nothing" is locally safe, so the policy never learns to send meaningful fleets or expand.
- **The pool/PFSP experiment we built** stacked 7 deltas in one branch (P=2 symmetric, skip-warmup, opponent pool, PFSP, externals, persistence, preseed) — violating "one delta at a time." Its `material 0 → 99` partial signal vs Zach is the only positive data point, but un-attributable.
- **Action-space discretization** (16 log-spaced ship bins) materially weakens any heuristic that emits exact ship counts. Real, but secondary to the cold-start collapse.
- **Competitor's hard-earned lesson**: keep the reward simple, change one thing at a time, ship the "stupid baseline" first.

The intended outcome is the **original project plan we abandoned**: build a clean, essential heuristic agent → BC warmstart → PPO on top — with the new learnings from this session baked in (kill stacked deltas, monitor `clip_frac`, validate non-degenerate behavior at each stage).

## Phase 0 — Decide what goes in the heuristic *(reading only, no code yet)*

Before writing any heuristic code, pull and read **pilkwang/orbit-wars-structured-baseline** from Kaggle and compare it against:

- Old `main.py` at commit `b865a26` (Roman A1–A8 strategies, already surveyed)
- `candidate_zach_public.py` (576 lines, intercept solver, opening focus)
- `candidate_suneet_lb1200.py` (3239 lines, forward-search heavyweight — for reference only, NOT to clone)

```bash
kaggle kernels pull pilkwang/orbit-wars-structured-baseline -p /tmp/pilkwang
```

**Deliverable**: a short list of 5–10 essential primitives the new heuristic will encode, agreed with the user before coding. Anchor points (from the existing-agent survey) likely include: capture-cost calculation, defense reserve, target value ranking, orbital interception, sun avoidance, and in-flight commitment tracking. Pilkwang's structure may shift the cut.

## Phase A — Write the essential heuristic

**File**: `orbit_wars_rl/teacher.py` — separate from `main.py` (which stays as the Kaggle submission).

**Interface** (matches what `bc.py` and `eval.py` already expect — no new infrastructure needed):

```python
def agent(obs: dict) -> list[list[int | float]]:
    # returns list of [from_planet_id, angle_rad, ships]
```

**Constraint**: keep to the primitive list from Phase 0. No phase-aware tactics, no gang-up multipliers, no 140 hand-tuned constants until the simple version is proven inadequate.

**Validation**:
- 32 games vs random opponent → expect >80% win rate
- 32 games vs `candidate_zach_public.py` → expect >25% (proves competence, not necessarily wins)
- Code stays under ~500 lines and uses ≤ ~15 named constants

## Phase B — BC warmstart from the new heuristic

Use the existing `bc.py` pipeline as-is (interface already takes a `.py` path to a heuristic with `agent(obs)`):

```bash
cd orbit_wars_rl
python bc.py --agent teacher.py --num-games 200 --steps 5000 \
  --save checkpoints/bc_teacher_v1.pt
```

Past artifacts (`bc_warmstart.pt`, `bc_warmstart_v2.pt` from 20–21 May) and the `bc_adaptive_120g_15000.log` show this path works and converges to ~1.2 loss. We will reuse the same machinery; only the teacher source changes.

**Validation** before moving to Phase C:
- BC train+val loss converges similarly to the prior log
- BC checkpoint wins >70% vs random
- **Replay-inspector check on BC checkpoint vs Suneet**: median ship count > 5 and the agent fires from >1 planet per game. This is the explicit cold-start check — if the BC checkpoint still emits 1-ship fleets, BC failed and we debug before any PPO.

## Phase C — PPO from BC checkpoint, pure self-play

**Per user's directive: one delta at a time.** BC + pure self-play first. No pool, no externals.

Use the *pre-pool* version of `train_torch.py` (the P=2 + skip-warmup fixes are keepers; the pool machinery isn't engaged with `--pool-mode none`, which is the default).

```bash
python train_torch.py \
  --resume checkpoints/bc_teacher_v1.pt \
  --total-steps 50_000_000 \
  --num-envs 512 --rollout-steps 64 --num-minibatches 32 \
  --ppo-epochs 2 --learning-rate 0.0001 \
  --checkpoint-interval 5_000_000 \
  --eval-interval 5_000_000 \
  --pool-mode none
```

**Validation during the run**:
- `clip_frac` stays below 0.30 (hard alert at 0.40 → cut LR by 2× and restart from last checkpoint)
- `H_fire` does not collapse to <0.05 (lagging warning)
- `r_p0` and `r_p1` mirror near zero (P=2 symmetry held)

**Post-run validation (local eval, kaggle env)**:
- Win rate vs raw Suneet, Zach, Rahul — at the 5 M / 25 M / 50 M-step checkpoints
- Median ship count per move (from the replay inspector) — must stay above the BC-checkpoint baseline; if it regresses toward 1, the policy is collapsing back to "do nothing"

## Phase D — Decision point (no code yet, just a fork)

After Phase C completes (or stalls), we decide the *next single delta* based on data — explicitly not predetermined. Likely candidates, to be ranked from evidence: opponent pool (re-engaging the work already in `pool-training` branch), action-space refinement (more ship bins or continuous head), or PBRS reward shaping.

## Critical files

- **New**: `orbit_wars_rl/teacher.py` — the essential heuristic agent
- **Reused as-is**: `orbit_wars_rl/bc.py` (collection + training), `orbit_wars_rl/eval.py` (validation), `orbit_wars_rl/replay_inspect.py` (degenerate-behavior check)
- **Reused with `--pool-mode none`**: `orbit_wars_rl/train_torch.py` on the `pool-training` branch — the P=2 and skip-warmup fixes apply; pool stays dormant
- **Reference only, do not modify**: `main.py` (Kaggle submission), `candidate_*.py` files

## Background state to clean up before starting

- The on-demand baseline cloud run (`i-0ebc2d020a22781d3`) is at ~34 M post-resume steps, no measurable progress vs heuristics. Let it run to next eval for data, then terminate.
- The spot pool run (`i-0c079633c768d719a`) at ~3 M steps. Let the in-flight local eval (pool@3M vs Suneet/Zach/Rahul) finish for data, then terminate.
- The `pool-training` branch's commit (`851a85e`) stays — we don't lose the work, just won't engage it in Phase C.

## Verification (end-to-end, after Phase C run completes)

1. **Heuristic sanity**: `python eval.py --checkpoint <bc_ckpt> --opponent /path/to/teacher.py --games 32 --opponent random` — should win > 80%
2. **BC quality**: `python replay_inspect.py --checkpoint checkpoints/bc_teacher_v1.pt --opponent .../candidate_suneet_lb1200.py --seed 1 --seed 2 --seed 3` — median ship count must be > 5
3. **PPO from BC progress**: re-run replay_inspect on the 50 M PPO checkpoint, same opponents. Compare median ship count, fleet diversity, planets-captured-per-game vs the BC-only baseline. Run local eval against Suneet/Zach/Rahul/random.
4. **Health log inspection** during training: grep `clip_frac` and `H_fire` from `train_gpu.log`; both should stay within healthy bands described in Phase C.
