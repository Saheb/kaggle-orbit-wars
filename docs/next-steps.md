# Next steps / idea backlog

Living doc — efforts and ideas, roughly prioritized. Discipline: **one delta per cloud run.**
Status tags: 🔵 in-flight · 🟢 ready-to-build · 🟡 idea · ✅ done · ⏸ parked.
Current focus: **Phase 2 — reinforcement** (docs/phase2.md). VDN/per-slot concluded (doesn't work; back to standard arch).

---

## ⚠️ LIVE INSTANCE — p2rev4 (2026-06-11)

- **Jarvis A100-40GB SPOT, IP 217.18.55.74, IN2, name `orbit-wars-p2rev4`, RUNNING + BILLING (~₹179/hr).**
  Running **p2rev4** in tmux `training`. The ONE delta vs p2rev3: **`--reinforce-garrison-floor 10 → 0`**
  (unblock reinforcement — veto probe showed the floor blocked 62% of wanted reinforces). Resume of the
  p2rev3 4M checkpoint (`seed_checkpoints/p2rev4_resume.pt`). Forced throughput-only deviation: **256 envs
  / 4 workers** (was 512/6 — this box is 40GB, 512 OOMs >40GB). Script: `gpu_run_artifacts/p2rev4/run_remote_p2rev4.sh`.
  iter 1 healthy: EV 0.706, clip 0.141, no OOM. Launched ~18:01 UTC.
- **⚠️ BALANCE ≈ ₹524 at launch → only ~3hr runtime (~5M steps).** Won't reach 10M without a top-up; spot-out
  loses ≤500k (watcher syncs every 60s). Enough for the decisive `lost-cap` read (floor effect shows by 1-2M).
- **WATCH:** EV recovery + clip<0.25 (resume warmup), then the PAYOFF metrics — **`lost-cap` should FALL /
  `median-hold` should RISE** (reinforcement now unblocked → can hold) and **`reinf@13+` should climb**.
  Caveat: floor=0 lets hard-commit reinforces strip a source to 0 → if `lost-cap` WORSENS, that's over-drain
  → add a small floor back (2-3). Selection PURE: held-out win-rate.
- Watchers (local): sync `P2REV4_IP=217.18.55.74 watch_p2rev4_sync.sh` (real-dir `/home/checkpoints/` fix),
  deb panel/ckpt, zach panel/2M, cross-eval/3M → `gpu_run_artifacts/p2rev4/`.
- SSH: `ssh -i ~/.ssh/jarvis-labs-key root@217.18.55.74`. Tail: `grep '^iter' /home/train_gpu_phase1_p2rev4_*.log | tail`.
- **TERMINATE when done:** `JL_API_KEY=$JARVIS_API_KEY jl destroy <id> --yes` (find id via `jl list`; was 425420-class).
  ⚠️ SPOT — destroy, never pause.

---

## ⭐ 2026-06-11 session — findings + next levers (prioritized)

**Biggest finding — the gap vs strong planners is the MID-GAME (steps 50→100) hold, NOT the opening.**
⚠️ **CORRECTED 2026-06-11 by the conversion metrics** (was "the gap is the opening"). p2rev2 @4.7M
conversion eval (masks-on) vs Zach 64g (77% WR) and Deb 32g (0% WR): **`planets@16/32/50/100` is the
clean win/loss discriminator** — Zach **2/4/7/10** (monotone, end 17) vs Deb **2/4/6/3** (peaks @50,
collapses to 3 by @100, end 0.1, dead ~step 122). **The opening is IDENTICAL win/loss (2/4 at steps
16/32) — we do NOT lose the early exchange; the divergence is the mid-game.** Metric verdicts:
`redundant-launch` is at the elite floor in BOTH (0.12/0.17; ref ~0.15) → **the over-fire /
friendly-coverage roi-deflation p2rev2 is built on is a non-bottleneck, no headroom** (not hurting,
but misdirected for the Deb gap). `cap/atk-launch` fine (0.59/0.44). `churn` degenerate on elimination
(end→0). Mechanism: we over-garrison mid-game (garr_frac@50 0.63–0.75, ships/pp 34–46 vs Isaiah 0.54/22);
Deb's planner peels the parked planets, Zach can't. **Lever the data points to = `--pool-seed-rl`** (a
strong RL self in the loop that punishes the over-garrison → forces holding), not deflation or First-Strike.
Tooling: `churn`+`redundant-launch` added to `eval.py game_conversion()`; `conversion_from_replays.py`
for the top-2 baseline; defs/confounds in `docs/metrics.md`. Won-game HTML: `/tmp/p2rev1_WIN_*_seed{1948,9013}.html`.
p2rev1 9.8M panels (masks-on): **Ajay 0.4% · debatreya 0.8% · Zach ~62%** — weak vs strong planners
(opening), strong vs Zach. (Ajay/debatreya NOT LB-predictive — yardstick, not objective. Best-ever
Ajay refs: rev53b 10.9%, rev54 1M 5.5%, rev38 3.1%.)

- 🟢 **`--pool-seed-rl` BUILT + TESTED — use next run.** Pins fixed RL champions into the pool via the
  GPU "self" path (fast, **sim-gap-immune** — unlike the heuristics). Never FIFO-evicted, survives
  save/load. `opponent_pool.py` (`add_pinned_rl` + `pinned` flag), `tests/test_pool_pinned.py`. Next run:
  `--pool-seed-rl gpu_run_artifacts/rev38/checkpoints/torch_step_5242880_rev38_20260605_181635.pt,gpu_run_artifacts/preseed_pool/torch_step_10485760_rev53b.pt`
  (rev38 = first-strike aggressor → opening pressure; rev53b = selective). **Keep rev31/rev32b
  HELD-OUT** (cross-eval forgetting detector). p2rev1 loses to all four (cross-eval 7.7M) → real
  pressure. Compat verified (pairwise15; rev38's stale `angle_head` keys auto-filtered).
- 🟡 **Opening-strength shaping** — if the pinned aggressors aren't enough, revisit First-Strike
  (`--first-strike-steps/--first-strike-mult`) for the from-scratch reward to win the early exchange
  more often. The 9.8M analysis says this is THE lever vs strong planners.
- 🟢 **RELAX `--reinforce-garrison-floor` (TOP lever — evidence-backed, cheapest).** Veto probe
  (`orbit_wars_rl/garrison_floor_probe.py`, 4M ckpt vs deb, 2026-06-11): the policy WANTS to reinforce 26%
  of fire-decisions but only **13% are allowed — the garrison-floor blocks 62%** (82% of forward-legal
  reinforces); at floor=0 allowed jumps to 55%. The floor=10 directly FIGHTS forward-staging (drain-rear vs
  keep-10) and the policy commits hard so nearly any reinforce trips it. **Next single-delta run: resume
  p2rev3 4M with garrison_floor dropped to 0 (or 2-3), else identical; watch `lost-cap`/`median-hold`.**
  Caveat: no floor → hard-commit can strip a source to 0 → policy must re-learn commitment. **Secondary
  blocker = `--reinforce-forward-only`** (blocks pulling ships BACK to defend a peeled rear planet; 23%→42%
  of intents once the floor is gone) — relax next if floor-relax alone isn't enough. **Sequencing: relax
  masks BEFORE/WITH deb-as-pool** — a peeler in the pool is useless if 87% of intended reinforces are masked out.
- 🟢 **THREAT HEAD (retention lever — strongest principled candidate).** Per owned planet, predict
  P(lost within K steps), self-supervised from the trajectory; feeds reinforce target selection
  (reinforce the planet you predict you'll lose). Full spec: `docs/phase2.md` "Deferred track". **Decision
  gate = is the target head undertrained?** Evidence as of p2rev3 ~2.5M: **`H_tgt` is RISING 1.07→1.70**
  (the documented "where-head going uniform" canary) AND aggregate reinf is high (0.5+) while empire-binned
  `reinf@13+` is LOW (0.25) with poor retention ⇒ **reinforce is mis-TARGETED, not under-rate** — exactly
  the threat head's domain. Build it ONLY as a self-supervised PREDICTION head (NOT target reward = rev49
  carpet-bomb graveyard; NOT KL-to-heuristic-targets = rev54 crater). Build landmine: per-planet labels hit
  the VDN planet-id-reorder corruption (scatter in planet-id space, not slot space). Highest-value/lowest-risk
  per phase2.md; do this if the p2rev3 retention metric (`lost-cap`) stays bad through 5M.
- 🟡 **Deb (or Ajay/producer) as the ONLY external pool opponent — "learn by losing to the exploit."**
  Fast hammers don't peel mid-game planets the way a planner does → never punish our retention weakness; a
  planner would. MECHANICALLY FEASIBLE: route via the `--external-opponents X --heuristic-workers N`
  SUBPROCESS path (NOT in-process `--planner-externals`, which stalls) — rev56 ran debatreya_1300 as sole
  external stably (256 envs / 4 workers / ext-frac 0.15, SPS ~400-660). Tradeoffs: **(1) deb is our HELD-OUT
  eval — training on it forfeits the clean generalization signal; swap in Ajay/producer as the new held-out.
  (2) sim-gap — deb-in-torch_env plays weaker than in Kaggle, so transfer to LB is not guaranteed (still a
  stronger/differently-exploiting opponent than hammers). (3) throughput retune** vs the 512/6 fast-hammer
  config. Queue as a single-delta experiment after p2rev3 concludes. (rev56 mechanism: memory
  `feedback_orbit_lite_is_slow`.)
- ⏸ **Train/eval sim gap — documented + PARKED** (`docs/train-eval.md`, memory
  `project_train_eval_sim_gap`). Pool wr is torch_env fiction (cross-eval: all 3 hammers **2–4% in
  kaggle** vs training ema 0.40–0.48). Found+fixed the **144-bin angle quantization** of opponents
  (continuous-angle override via `torch_env.step(angle_overrides=…)`, `tests/test_ship_bin_decode.py`)
  BUT a hammer-vs-hammer A/B = 52.6% (noise) → it's a **minor** contributor. Real gap likely our agent
  **overfitting torch_env physics** + win/timeout resolution. Decisive probe (parked): same checkpoint
  vs a hammer in BOTH engines, then diff one shared-seed trajectory. Aim fix kept (correct + free).
- 🟡 **clip_frac creep on p2rev1** — 0.069 (8.0M) → 0.158 (9.4M), first KL/EV wobble @9.4M (the
  entropy-0.05 / LR-1e-4 drift). Standing authority: **halve LR at 0.25**. Wait 2–3 checkpoints
  (9.2/9.8/10.3M panels) before concluding; conversion dipped slightly (cap/atk 0.56→0.51), likely the
  same drift.
- ✅ **Eval hygiene fixed:** debatreya watcher now masks-on + newest-first (`sort -rV`) + `MIN_STEP`
  watermark (backlog cleared at 9.2M); conversion eval prints a **reinf-by-empire-size ramp** vs the
  top-player ramp (aggregate `reinf_share` is opponent/empire-confounded — read the ramp; `eval.py` +
  `docs/metrics.md`); `cross_eval` now includes the pool hammers (`pool_lb1152/1138/1084`) + masks.
- ⏸ **orbit_lite candidate opponents parked** — locutus + producerlite (Kaggle kernels) ≈ 11 ms/call
  (vs hammer 0.7, ~debatreya 9.3) → too slow for the pool, eval-only. Rule: imports `orbit_lite` ⇒
  slow, conclude from imports (memory `feedback_orbit_lite_is_slow`). At `/tmp/{locutus,producerlite}_x/`.

---

## 0. In-flight — Phase 2 reinforcement, Tier-1 (docs/phase2.md)

**rev58/58b resume probes both flooded — cost knob dead, root cause re-diagnosed.** rev58 (cost 0)
and rev58b (cost 0.001) both *drifted* from a healthy gated start into a flood (~330–400k: reinf
0.69–0.75, `p90` 357–408, `Vμ`→negative). Drift-from-healthy ⇒ a property of the reward **objective**
(recurs from scratch), and the pump is **`defense_coef` itself**: in a symmetric self-play mirror,
hold-everything-via-reinforce dominates risky attacking, so the term meant to *incentivise*
reinforcement *is* the flood. Full write-up: `docs/phase2.md` top Update.

**Tier-1 design (locked 2026-06-10) — "outcome-tied attribution":**
- 🟢 **Forward-staging mask (BUILT + unit-tested):** own reinforce target legal only if closer to the
  nearest enemy than the source (`--reinforce-forward-only`) → rear hoard impossible by construction.
  `torch_env.py` + `tests/test_reinforce_mask.py`.
- 🟢 **Drop `defense_coef`** — reinforcement gets no shaping reward; credited purely via terminal +
  early_capture through GAE. Drop `reinforce_cost` (dead). Keep gate(3), garrison-floor(10),
  expansion 0.03, speed 0.3, early_capture anneal, fire-entropy 0.005.
- 🟢 **Small aggressive pool (rev53b-proven)** — held-out LB archetypes so hoarding *loses games*
  (the asymmetry that makes reinforcement instrumentally valuable; a pure mirror gives flood OR
  passivity). (c)-attribution removes the bad incentive; the pool supplies the attack pressure — both needed.
- 🟢 **p2rev1 READY** (fresh Phase-2 numbering, not the rev5x lineage): snowball-BC warmstart
  (`bc_snowball_pairwise15.pt`, aggressive winners, 53% reinforce coverage) + forward mask + drop
  defense + pool (lb1152 hammer + debatreya_1300 @0.25). Target-head diagnostics (`H_tgt`, target
  own/neutral/enemy share) added. Script: `gpu_run_artifacts/p2rev1/`. Awaiting launch.

Selection stays PURE: win-rate/Elo vs the held-out ladder decides; `reinforce_rate`/game-length are
diagnostics. Tier-2 (full causal fleet attribution) held in reserve if Tier-1 underperforms.

## 1. Phase 2 fallbacks / follow-ups (conditional on rev58b)

- 🟢 **From-scratch (BC warmstart) run** — only if we conclude the *base* matters after all (unlikely per the drift
  finding). BC seed options analyzed: Isaiah (controlled) vs Jake/aggressive-cohort (snowball, full 0→0.61 reinforce
  ramp). Snowball selector to pull aggressive replays without player names: high `avg_score` + `size_bytes`<3.5MB.
- ✅ **Eval/export gate parity (DONE 2026-06-11, first Phase-2 submit sub 53574885).** Correction: `action_mask.py`
  `actions_from_target_policy` ALREADY had the full reinforce-discipline parity (gate/forward-only/garrison-floor, both
  logit-mask + post-argmax reject). The real gap was `export_agent.py`: it baked only `allow_reinforce` and let the three
  discipline params fall to their off-defaults (gate0/forward-off/floor0) → exported agent would reinforce <3 planets /
  backward / drain source = self-sabotage. The three params are NOT stored in the checkpoint (only `allow_reinforce` is),
  so export now takes them as CLI flags defaulting to the locked Tier-1 values (gate3/forward/floor10) and bakes them into
  the template. Also fixed a pre-existing export crash: the template didn't `import os`, surfaced by features.py's
  module-level `os.environ` (friendly-deflation) read. Verify after any reinforce export: grep `_REINFORCE_GATE_MIN`/etc.
- 🟡 **Reinforce-aware BC warmstart from top-player replays** (180 games in `/tmp/fresh_validate`, 89 snowball in
  `/tmp/snowball`; analyzer `orbit_wars_rl/fetch_analyze_top_replays.py`, timing-corrected).

## 2. Diversity / anti-cycling (still the structural lever, independent of Phase 2)

- 🟢 **Pool curation** — fold cross-run best selves as fast `self` members; keep self-pool small (8–12 diverse) so PFSP
  gets ≥30 games/opponent. `--preseed-pool`.
- 🟢 **Distill Ajay → fast neural clone (DAgger)** — adds the selective-targeting style we lack; orbit_lite can't be
  GPU-batched. Spec: `docs/ajay_distillation_spec.md`; tool to build: `dagger_collect.py`.
- 🟡 **Neural exploiters / mini-league** — principled anti-cycling; all-neural avoids the slow-opponent problem. Endgame.
- ✅ **Cross-checkpoint eval panel** (`gpu_run_artifacts/cross_eval/run_cross_eval.sh`) — use every ~1M during training.

## 3. Hyperparameter / dynamics probes

- 🟡 **gamma 0.999** (not 1.0) — longer horizon propagates the win signal earlier → less shaping dependence.
- 🟡 **rollout 32 + ppo_epochs 1** — top-LB suggestion: short rollout = local credit; ppo_epochs 1 = more on-policy.

## 4. Measurement / instrumentation

- 🟢 **Always-on held-out ladder during training** — eval vs a fixed diverse set every ~1M; self-play WR + `Vμ`/EV are
  blind to absolute regression. Selection is PURE: win-rate decides, not shaped reward / `Vμ` / Ajay panel alone.

## 5. Parked

- ⏸ **FFA / 4p** — above par on wins, no validated lever, no faithful local eval. Don't spend GPU. (`project_ffa_not_the_gap`)
- ⏸ **VDN / per-slot credit** — concluded: per-slot ship-credit → undercommitment / collapse. Back to joint-credit standard arch.

## 6. Ops / tech debt

- 🟢 **Bake auto-terminate-on-completion into launch scripts** (GCP `--terminate-on-done` not wired; manual delete required).
- 🟡 **Run scripts + seed_checkpoints are rsync-EXCLUDED** by `launch_gpu_gcp.sh` → must scp them after launch (caught us on rev58).
- 🟡 **Pool-opponent `state_dict` reload** fires ~123×/iter — rollout-throughput inefficiency; optimize only if SPS bottlenecks.
