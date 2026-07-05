# Ajay distillation → fast neural pool opponent (DAgger)

**Problem:** Ajay/Producer are the most strategically *distinct* opponents we have
(selective production-horizon targeting), exactly the diversity our cross-play
matrix showed the heuristic ladder lacks. But orbit_lite is single-env CPU planning
— it **cannot** be GPU-batched (`batched_planner.py` confirms), so it's too slow for
the training pool (rev43/44 CPU deadlock).

**Fix:** distill Ajay into our entity-transformer (a `self`-type pool member → fast
GPU inference, fully compatible). The clone need not match Ajay's *strength* — it
needs to carry Ajay's *style* (selective targeting), the diversity we're missing.

## Success criteria (validate with `study_opponents.py`, ≥32 games)
1. **Style match:** clone's avg game length ≈ Ajay's (long/positional), not the
   short/decisive heuristic profile.
2. **Strategic match:** our current best agent's win-rate vs `ajay_clone` ≈ its
   win-rate vs real Ajay (within ~10pp). I.e. it poses Ajay's *kind* of problem.
3. **Pool value (the real test):** adding `ajay_clone` to the pool improves
   held-out **real-Ajay** win-rate over a matched run without it.

## Why DAgger, not pure replay-BC
Pure BC on Ajay's own games fails on the states *our* agent forces (covariate
shift — the clone only knows states Ajay visits). DAgger fixes it by labeling the
states the **clone/our-agent actually induce**. Ajay is ideal for this: unlike a
top-LB player we can only see in replays, we can **query Ajay on any state, unlimited**.

## Pipeline

### Stage 0 — seed dataset (round-0, from Ajay-vs-our-agent replays)
Generate replays of `our_agent vs Ajay` (eval.py / compare_tempo already do this),
then extract whole-action BC samples:
```bash
python orbit_wars_rl/build_producer_action_bc.py \
  --replay-dir /tmp/ajay_vs_us_replays --player-name "Ajay @ ..." \
  --samples-out /tmp/dagger/ajay_round0.pkl
```
> ⚠️ Decision: `build_producer_action_bc` labels the *producer-best* action. For
> faithful imitation prefer Ajay's **actual recorded action**. Ajay≈producer so
> they nearly coincide, but the DAgger query (Stage 2) uses Ajay's *real* action —
> keep that the source of truth; round-0 producer-labels are just a warm start.

### Stage 1 — train clone v0
Distill into our arch, all policy heads (fire+ship+target) + backbone. Init from a
strong checkpoint for a good prior:
```bash
python orbit_wars_rl/bc.py --samples /tmp/dagger/ajay_round0.pkl \
  --init-checkpoint gpu_run_artifacts/rev53b/checkpoints/torch_step_10485760_rev53b_*.pt \
  --steps 5000 --save /tmp/dagger/ajay_clone_v0.pt
```

### Stage 2 — DAgger rounds (NEW tool: `dagger_collect.py`)
For K = 0,1,2:
1. Roll out `our_agent vs ajay_clone_vK` for M games (~64), recording **every obs
   the clone faced** (states the clone induces vs our agent).
2. For each recorded obs, query **real Ajay**: `act = ajay_agent(obs)`.
3. Encode `(obs, act)` → BC samples (reuse the action→per-slot encoding in
   `build_producer_action_bc` / `features.py`; the *real* Ajay action, not producer-best).
4. Aggregate with prior rounds → `ajay_round{K+1}.pkl`.
5. Retrain: `bc.py --samples <all rounds> --init-checkpoint ajay_clone_vK.pt
   --save ajay_clone_v{K+1}.pt`.

`dagger_collect.py` (~120 lines) = eval.py's game loop (to roll out + record obs)
+ `candidate_ajay_1200.agent(obs)` (to label) + `build_producer_action_bc`'s
encoder (to write samples). It is the only new code.

### Stage 3 — validate
```bash
python gpu_run_artifacts/cross_eval/study_opponents.py 32 \
  "ajay_clone,rev53b,rev38,h1166,hellburner"   # + real Ajay once (slow, one-off)
```
Check criteria 1–2. Export the clone as a `.py` opponent for the cross-eval panel.

### Stage 4 — deploy + measure value
Add to the training pool as a `self` member:
```bash
  --preseed-pool ../dagger_pool   # dir containing ajay_clone.pt
```
Run a matched A/B (pool + clone vs pool − clone); criterion 3 = held-out real-Ajay
win-rate goes up. THIS is the test of whether distillation actually buys diversity.

## Risks / decisions
- **Clone strength < Ajay** (BC has no planning). For pool diversity, *style* > strength,
  but if too weak, PFSP deprioritizes it → init from a strong checkpoint + enough DAgger
  rounds. If still weak, optionally fine-tune the clone with a little PPO vs the ladder.
- **Action mapping** (Ajay `(source,target,ships)` → our per-slot `(fire,ship,target)`):
  already solved in `build_producer_action_bc.py` — reuse, don't reinvent.
- **Cost:** offline Ajay queries are slow but parallelize freely (no training-loop
  deadlock); BC is cheap. One-time per clone.
- **Generalizes:** the same pipeline distills Producer, or any slow planner, or
  (with replay-only round-0) a top-LB player.
