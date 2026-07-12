# Top-100 Writeup Lessons — One Change at a Time

Study log for `writeups/top100/`. Goal: understand why we fell short, then apply **single,
measurable changes** in order — not copy whole agents. One lever per run; record hypothesis
before launching; verdict after.

Source writeups: SimJeg "N < 10th" (~top 5), Ender (top 10), kiyotah "GPU Poor PPO Rich"
(~top 40), Isaiah "Scaling RL to the Stars" (top), Rank ~55, novice #99, rule-based ~top 75.

Context that hangs over every verdict below: **the training-scale gap**. Our best checkpoints
are 5–10M env steps; the top-100 range was ~400M–15B (kiyotah 412M for $25, Ender 1B,
SimJeg 10B, Isaiah 15B). Enabled by env rewrites (Rust/JAX/C) at 15K–40K SPS vs our
~350–4250. Any lever we test at 5M steps may be under-trained, not wrong.

---

## 1. Projected-future timeline features — ✅ CONFIRMED (tl100m, 2026-07-12)

**Verdict:** 100M from-scratch pure self-play with the timeline (sparse ±1, no shaping,
noop-KL 0.3): **Ajay full panel 0% → 74.6% (best 77.7% @96.5M)** vs 57.4% pre-timeline best
(stgpr1, spray-inflated) — and clean (launch_rate 0.092, discipline learned, no collapse).
Full run record + stage-2 continuation in docs/training.md "tl100m".

**Status:** studied 2026-07-04. Projection core built + verified 2026-07-09
(`orbit_wars_rl/timeline.py`): `project_timeline(state, K=24)` → per-planet (owner, garrison)
over 24 future steps assuming no new launches. Vectorized: resolve fleets→(target,ETA)
(`resolve_target_eta`, mirrors `_resolve_targets_at`), scatter to (N,K,P,players) arrivals,
K-step recurrence (production + engine combat/flip mirroring `torch_env_fn.physics_core`).
**Parity vs stepping the functional env 24× no-action: 100.0% owner agreement, 0.01 garrison
MAE** (tests/test_timeline_projection.py).

**Wiring DONE 2026-07-10** — planet token dim **20 → 116** (+96 = 4 ch × 24 steps:
mine/enemy/neutral one-hot + log-garrison, `timeline_features()`):
- training: `torch_env.get_features` (projects over the FULL fleet set, not the 128 view);
- eval/export: `features.extract_features(timeline=...)` calls the SAME timeline.py on a
  batch of 1 (no numpy mirror); eval/export infer the flag from `planet_proj` width, so
  presres1/stgpr1 (20-wide) stay evaluable/exportable;
- submission: export_agent inlines timeline.py (made import-free for this);
- train/eval parity suite: 0 error incl. timeline channels; export smoke both eras ✅.
Breaks resume of ALL pre-timeline checkpoints (guard added) — next run is from scratch,
which is the plan anyway (pure self-play + anchoring, docs/training.md).
GPU SPS gate + from-scratch run both DONE — see verdict above (tl100m ran
`--compile-features`, ~537 SPS on L4 at 116-dim).

**What they did (5 of 7 writeups; the most universal ingredient):**
- **SimJeg:** per body, 10 features × 20 future steps — literally steps the env forward with
  no new actions. Combat resolved, ownership flips visible, `ships_for_capture` per step.
  This timeline is nearly his entire feature space; encoded by a 1D-CNN into the planet token.
- **kiyotah:** 23 steps × 6 channels (owner one-hot + self/enemy/neutral ships). Quote: "The
  explicit timeline was one of the biggest improvements I found… a planet becoming enemy-owned
  in three steps is not the same as one changing in twenty."
- **Ender:** 24 bins net incoming (inter-fleet combat pre-resolved) + 24 bins projected
  owner/garrison under ceasefire. PLUS: target MLP sees the projection **assuming the candidate
  fleet launches** (counterfactual) → fixed launching too-early/too-late; let the value head
  "respond immediately and confidently to good and bad launches in training".

**Our gap (features.py):**
- ch12/13 planet inbound pressure = untimed scalar sums.
- Pairwise ch10 `ships_at_arrival` = ships + prod×eta — **ignores all fleets in flight**
  (bug-grade gap independent of this lever).
- ch20 `enemy_mass_soon` = one 6-step window; ch21 `threat_imminence` = min-ETA only, no mass.
- Projected **ownership** appears nowhere. Friendly/enemy arrival interleaving not representable.
- Our ROI/deflation stack (ch12/13/16/17/19 + deflations) hand-computes conclusions the
  winners let the model draw from a raw resolved timeline — and each patch carried bugs.

**Why it maps to our failure:** conversion timing was our diagnosed gap verbatim (Ender's
too-early/too-late). Losses were mid-game eliminations, gap opening steps 25–75 — exactly where
contested planets flip and untimed aggregates mislead. Also a credit-assignment fix: bad
launches become visible to the critic the step they happen (shaped-reward-like benefit, no
reward shaping).

**Scoped change:** per-planet projected timeline, K=24 steps × 4 ch (owner one-hot +
log-garrison), kiyotah's form (NOT Ender's counterfactual yet — that's a separate later lever).
Small conv/MLP encoder into the planet token.

**Implementation notes:** fleet trajectories are fixed at launch and we already resolve each
fleet to its single target at 98.4% (`torch_env._fleet_target_idx`) → projection = bin resolved
fleets into (envs, planets, 24) arrival tensor by ETA, then a 24-step vectorized recurrence
(add production if owned, apply arrivals with engine combat/flip rules). NOT "run the env 24
extra times".

**Verification before GPU:** (1) projection parity vs actually stepping the env 24 steps
no-action; (2) scalar mirror in features.py + train/eval parity test; (3) SPS benchmark.

**Read on the run:** hold-loss / out-massed% (conditional on reinforce held high), planets@50,
underkill/conversion lines, Ajay panel.

---

## Backlog (studied at overview level only; deep-dive one at a time)

### 2. Action heads & decoding structure — studied 2026-07-04 (deep-dive done, change not yet scoped)

**Head designs across the field:**
| Agent | Fire/no-op | Target | Ship size | Cross-planet coordination |
|---|---|---|---|---|
| SimJeg (top5) | Bernoulli per body FIRST | attention over bodies | none (all-in) | none (parallel) |
| Isaiah (top) | Bernoulli per source FIRST | Q·K attention | logistic mixture, conditioned on target via V(target) injection | none (parallel) |
| kiyotah (~40) | fused: no-op = 45th option in target softmax | pair-MLP | none (all-in) | none — his #1 unsolved problem |
| #99 | fused (target-or-no-op) | stage 1 | stage 2 conditioned on target + same-turn proposal context | PARTIAL (proposal context) |
| Ender (top10) | global launch/halt per micro-step | conditioned on origin+fraction, abort option | joint origin×fraction | FULL AR: fleet added to obs, ≤16 micro-steps |
| rank ~55 | STOP token | AR source→target→amount | AR | FULL AR (AlphaStar-style) |

**Key patterns:** (1) ship size always conditioned on chosen target (nobody sizes blind);
(2) no-op as first-class competitor in the target softmax (rank-55's policy once "held" by
aiming at unreachable planets behind the sun — restraint needs a legal expression);
(3) Ender's AR insight: "the next decision after committing a launch = as if the fleet were
already in play" → later launches see earlier ones; (4) #99's proposal-context = cheap partial
coordination without AR throughput cost.

**Where ours differs (train_torch.py:88-107, ppo.py:217):**
- We are target-FIRST: sample target, then fire|target and ship|target (per-(slot,target)
  pairwise scorers). Top agents are fire-first or fused.
- Target log-prob enters the PPO joint EVEN WHEN fire=0 → at winner launch rates (~0.04),
  ~96% of target-head gradient comes from steps where the choice had no effect. Fire-first /
  fused designs don't have this. Candidate quiet drag on target learning.
- Across slots: fully parallel, zero coordination signal — the shape of our
  force-concentration wall (N planets deciding peel/commit independently).

**Counterweight:** Isaiah reached the top with fully parallel sampling at 200M params / 15B
steps — parallel isn't disqualifying, scale can substitute. At our scale, Ender/#99 got
coordination from decoding structure instead.

**Candidate levers (pick ONE when this item comes up):** (a) fire-first refactor + stop
crediting target log-prob when fire=0; (b) #99-style proposal context into the ship head;
(c) full AR micro-steps (biggest change, throughput cost).

### 3. Action-space simplification + KL-to-prior regularization
SimJeg reached ~top 5 with TWO actions per body (no-op vs all-in, ETA<20 targets). kiyotah
same. Ender's fix for the many-small-fleets disease (our spray: launch_rate ~0.38 vs winner
~0.04): replace entropy bonus with **KL penalty toward a prior** (halt_prob 0.9, fraction
ratio 1:1:1:1:10 favoring full send) — "huge immediate improvement". Principled version of
what our min-ship-bin band-aid groped at.

### 4. Throughput / scale (the env rewrite question)
All top agents rewrote the env (Rust/JAX/C+CUDA) → 15K–40K SPS → 400M–15B steps. Decide:
invest in a rewrite vs accept that our levers get tested under-trained. kiyotah: "run fewer
experiments for longer; learning curves change substantially later."

### 5. Sparse rewards, no shaping band-aids
kiyotah: pure −1/0/+1 ("tried production shaping, did not see results"). Ender: 0/1/2
positive-only to avoid loss-delaying stalling; SimJeg: 0.5 for timeout-wins (anti-stall).
Nobody used first-strike-style multipliers or SSDR-style curricula. Consistent claim: fix the
signal path (features/value/action space), not the reward.

### 6. Trustworthy local eval / model selection
Rank-55: local round-robin ladder incl. all strong public bots + own exploiters; promote only
on beating ALL stronger priors. kiyotah's cautionary tale: his gate picked a 668M ckpt when
412M was better on LB. Ender/Isaiah: gate promotion at 70–80% WR vs reference checkpoint;
league of past checkpoints doubles as an Elo progress meter.

### 7. Misc notable single facts
- SimJeg: from-scratch RL **beat** his IL-warmstarted models within 5 days once throughput
  existed (questions our BC-warmstart track).
- Isaiah: action masking (e.g. no-sun-shots) made his model WORSE — hypothesis: unmasked
  forced better internal physics modeling. Re-masked only for late fine-tune + test time.
  (Questions our gate/mask stack.)
- Ender: exploiters (train a fresh model purely to beat the main one, then fold back into
  league) broke the passive 4p equilibrium; +15–17pp first-place rate. [rank-55 writeup]
- Novice #99: value-head EV was the turning point (EV −0.9 → +0.92 via 128-step rollouts +
  economy-delta shaping); "RL debugging is systems debugging."
