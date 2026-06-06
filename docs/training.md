# Orbit Wars — Training State & History

---

## Phase 1 Run History — Quick Reference

| Run | Delta from prev | Expected | What happened | Killed because | Best ckpt LB |
|---|---|---|---|---|---|
| **Rev5** | Added srcs-multi-penalty | Guard carpet-bomb collapse | 6M peak: HB=38.7%, Zach=54.3%, Suneet=61.7%. Passive drift after 6M. | fire[0] 0.35→0.25 by 12M | — |
| **Rev6** | Cosine decay on srcs penalty | Reduce over-penalisation | Carpet-bomb returned immediately | HB=0%, collapsed | — |
| **HB blitz** | 90% HB, lr=3e-5, no IL | Replicate 141208 blitz | Destroyed everything in 500K steps | HB=0%, Zach=4.3% | — |
| **Rev7** | Removed IL anchor | IL amplifies passivity; remove it | 1M peak HB=40.2%, then passive drift | fire[0] 0.32→0.25 by 4M | 750 LB |
| **Rev8** | `--shaping-coef 0.05` | Reward material gain per step → more firing | HB dropped to 30% at 1M. fire[0] hit 0.22 by 1.5M | Passive + panel regression | — |
| **Rev9** | `--entropy-coef-fire 0.05` | Higher H_fire suppresses passive locking | H_fire rose (✓) but HB=25%, Zach=42% — entropy = random firing, not strategic | Panel regression on all metrics | — |
| **Rev10d** | `rollout-steps 512, envs=64` | Better credit assignment for sparse reward | HB=28.1%, Zach=46.9%, Suneet=53.8%. avgfleet oscillated 85-92 post-1M | Panel below rev7 on all metrics | 772 LB (1M) |
| **Rev11** | Pure self-play, 10M target, pool=40, kill floor 0.20 | Passive phases may recover at scale; HB was wrong opponent | clip_frac=0.199↓ (lowest ever). 2M best (fire=0.34, fleet=77). 4M/5M/6M declined. vs 141208: 2M=17%, 4M=8%, 5M=6%. vs Zach: 2M=41%, 5M=36%. Killed at 6M. | **Best: 2M** (`torch_step_2031616_20260531_065423.pt`) — submit tomorrow | — |
| **Rev12** | Top-team stabilizer recipe as a unit (+1/-1, rollout 512, minibatch 16, grad-clip 10, entropy 0.02/0.03/0.02, ~const LR) | Drift needs the recipe *combination*, not duration; dense ckpts capture peak | **WORKED.** Peaked at 6M (Zach 69.9%, vs-rev11-2M 73.4%) — best Phase-1 ever. Declined after (9M/11M lose to 6M head-to-head). Stopped ~15M. | Submitted 6M to LB 2026-06-01; stopped (clean monotonic decline past peak) | **6M = best Phase-1** |
| **Rev13** | Chain: resume rev12-6M, same recipe, 2M steps | 123203→141208 pattern: chaining the peak pushes higher | 1M: Zach 68.4%, vs-rev12-6M 47.7% — no improvement over rev12-6M. Chaining didn't help. | Lost to rev12-6M across both checkpoints | — |
| **Rev14** | `--expansion-coef 0.01` (production-lead shaping) | Break opening-expansion passivity; dense signal for planet capture | 1M: Zach 71.9%, vs-rev12-6M 41.8%. 2M: Zach 69.9%, vs-6M 47.3%. Peak same as rev12. | Same ceiling; expansion-coef alone didn't break equilibrium | — |
| **Rev15** | Resume rev12-6M, expansion-coef=0.01, win-margin=0.5 | Combine expansion incentive with decisive-win bonus | **Best Phase-1 ever.** 2M: Zach 69.9%, vs-6M 52.3%. 3M: Zach 71.5%. 4M declined. | Passive drift after 3M; 4M loses to 2M | **796 LB (2M)**, 784 (4M) |
| **Rev16** | `--defense-coef 0.02` (planet-loss penalty) | Break late-game collapse: penalise losing planets | Lost to rev15-2M at 1M/2M/3M (panels incomplete). Teaching holding, not aggression. | Consistently below rev15-2M | — |
| **Rev17** | `--handicap-frac 0.3` (asymmetric starts) | Force exposure to losing positions; learn to fight from behind | 1M: Zach 68.4%, vs-rev15-2M 52.7%. 2M: Zach 60.5%, vs-rev15-2M 43.8%. Declined. | Below rev15-2M by 2M | — |
| **Rev18** | `--external-opponents 141208 --pool-external-fraction 0.25` | Break passive Nash by forcing play against RL aggressor | 1M-3M: Zach ~68-70%, vs-rev15-2M ~45-50%. 4M: Zach 71.5%, vs-rev15-2M 40.6% — peaked early, declined. | Peak same ceiling as rev15; fire_rate didn't change | — |
| **Rev19** | `rollout=64, envs=512` (replicate 141208 training conditions) | rollout=64 forces aggressive opening gradient — 141208's key structural difference | Only reached 1M (AWS capacity issues); insufficient data | AWS us-east-1 quota exhausted mid-run | — |
| **Rev20** | rollout=64, heuristic BC warmstart | Test if rollout=64 + clean warmstart breaks passive equilibrium | fire_rate=0.15 at 1M, 2M — indistinguishable from passive baseline. vs-rev15-2M: 0% at 1M, 2% at 2M. | Wrong BC: heuristic fires indiscriminately (fire_rate 0.50, no game-sense) | — |
| **Rev20b** | rollout=64, **141208-teacher BC** warmstart, entropy=0.02 | Correct BC prior: selective aggression from RL teacher | 1M: Zach 14.5%. 2M: Zach 19.5%, vs-rev15-2M 4.7%, vs-141208 1.6%. BC prior erodes by 2M. | Self-play erases BC prior; still far below rev15-2M | — |
| **Rev21** | rollout=64, heuristic BC, entropy=0.05 | Higher entropy + wrong BC | Same failure as Rev20 — 0.8% vs rev15-2M at 1M | Wrong BC + high entropy = random spray | — |
| **Rev21b** | rollout=64, **141208-teacher BC**, entropy=0.05 | Higher entropy to resist passive collapse | 2M: Zach 16.8%, vs-rev15-2M 5.1% — slightly worse than Rev20b | Entropy=0.05 worse than 0.02 with 141208-BC | — |
| **Rev22** | `--early-capture-coef 0.0025` (cumulative holding, 100-step decay), resume rev15-2M | Terminal reward blind to early expansion timing | fire[0] dropped 0.32→0.19-0.21 at 1M. Holding reward wrong design: symmetric self-play cancels advantages, wrong 100-step window | fire[0] below baseline | — |
| **Rev23** | `--gae-lambda 0.99` + delta-capture(0.07/400) + resume rev15-2M | Isolate horizon fix; rev15-2M critic stale under λ=0.95, expected instability | Critic shock confirmed: avgfleet ballooned 26→100+, fire[0] degraded to 0.23-0.25 at 3M. Still running on GCP | Still running (GCP) | — |
| **Rev24** | Same horizon fix + **rank1 BC warmstart** (fresh critic) | Fresh critic eliminates shock; delta-capture should give early-expansion gradient | fire[0] rose to 0.42-0.52 — most aggressive Phase 1 ever. BUT carpet-bomb collapse: srcs_multi hit 6.84 at 5.7M. 5M Zach=28.9% (best BC-warmed run, but degrading). **Root cause:** delta-capture rewards ALL simultaneous captures, incentivising carpet-bomb. | srcs_multi=6.84, carpet-bomb | — |
| **Rev25** | Same as rev24 but delta-capture **capped at 1 capture/step** (prevents carpet-bomb incentive) | Remove degenerate multi-capture reward; preserve single-probe gradient | p90fleet still 200-370 throughout. fire[0]=0.19-0.25. No qualitative improvement over rev24 post-fix. Games still passive. | Still passive; no improvement over rev24 at matching steps | — |
| **Rev26** | Same as rev25 + **speed_coef=0.5** (time-to-victory velocity bonus: `reward_win += ((500-T)/500)*0.5`) | Pressure agent to close games fast, reduce timeout games | **New failure: ship-bin-0 collapse.** 51-66% of fires argmax to bin 0 (1 ship) from 500K steps onwards. 1-ship fleets can't capture neutrals → attacks useless. p90fleet reached 179 at 2M (best ever!) but via artificial mechanism (ships drained, not games decisive). avgfleet=76 at 2M. fire[0]=0.20-0.28. 29 ship-bin-0 collapse warnings. No external opponents in pool (pure self-play Nash). | ship-bin-0 collapse + not scoring real wins | — |
| **Rev27** | Drop speed_coef (0→0), `--entropy-coef-ships 0.02→0.05`; rank1 BC warmstart | Fix ship-bin-0 collapse from rev26; keep delta-capture+λ=0.99 architecture | Same failure: iters 1-7 r_p0=+0.000 (no terminals), delta-capture fires ~zero times. BC prior gives brief fire[0]=0.22 at iter 9 then erodes to 0.12-0.14 by iter 13. Identical pattern to all previous runs. | Same passive Nash; 0.07 coeff too small (~65x below terminal signal) | — |
| **Rev28** | `--early-capture-coef 0.07→0.30` (4x); keep entropy-ships=0.05, no speed-coef; rank1 BC warmstart | 0.30 coeff makes step-0 capture = +0.30 vs +1.0 win — large enough to dominate sparse-game early-training gradient | **Breakthrough.** Slow start (0.8% at 1M, 8.2% at 3M) then takeoff after 6M: Zach 39.5% at 7M → 43.4% at 16M → 48.4% at 26M → **58.2% at 32M** (H100 10M checkpoint). Wins are decisive (85-110 step eliminations), not timeout luck. rewNZ=0.50+ (dense rewards). meanshipbin climbing to 18-19. Still running on H100 (rollout=64, 2048 envs) targeting 100M. Key insight: run needed 6M+ steps to develop game sense — same slow-burn pattern as 62M run that scored 894 LB. | **Best Phase 1 result ever. Still climbing.** | TBD |
| **Rev29** | Same as Rev28 but **rollout=512, num-envs=256** (same total transitions=262K). Resume from Rev28 10M checkpoint. | Test if full-game credit assignment fixes value function horizon blindspot | Declined: 1M=57.8% → 4M=52.0% → plateau ~52%. Pool mismatch (40 strong opponents from rollout=64 run), lower SPS (1,750), 256 envs hurt diversity. Not a clean test of rollout=512 alone. Killed at 13M. | Pool mismatch + low SPS confounded the test | — |
| **Rev30** | **Symmetric capture reward** (`planet_delta = clamp(-1,1)` — losses penalised = no planet-tennis arbitrage) + **exponential decay + 0.10 floor** (no hard cliff at 400) + **expansion_coef 0.01→0.03**. Resume from Rev28 27M (77.3% peak). | Fix late-game gradient desert: symmetric reward eliminates farming exploit so capture signal can run to step 500; floor keeps gradient alive on rotating boards (seed6462 went 473-step timeout → 81-step decisive win). **Halved LR to 0.00005 at 12M** when clip_frac hit 0.28. | **New all-time Phase 1 record.** Peak at 11M: **84.8% vs Zach**. Submitted. loss-seed isolation test: 15/21 at peak. seed6462 seat0 fixed (473→81 steps). Remaining failures: `low_prod__mostly_static` boards and seat1 complacency (agent goes ahead at step 100, stops firing, Zach overtakes). | Peak at 11M then drifts 82-83%. Submitted 11M. | **TBD (submitted)** |
| **Rev30b** | Resume Rev30 from 17M checkpoint. **Halve LR again: 0.00005→0.000025**. Spot H100 Jarvis (₹112/hr). | Test if second LR halve stabilises past 84.8% peak | **85.2% vs Zach at 4M**. Clip_frac stable 0.23. 7M checkpoints. LB episode analysis revealed opening paralysis: FireP=0.001 at step 0 — policy confident in no-fire. Zach panel doesn't catch this (Zach also waits). | New panel record, but LB gap vs rev28 | TBD |
| **Rev31** | `--first-strike-steps 50 --first-strike-mult 2.0`: capture reward doubled for t<50. Resume from Rev30b 4M. LR=0.000025. | Overcome opening paralysis: step-2 FireP 0.003→0.772, making policy fire 2 steps earlier. Value function must unlearn "waiting is safe" for opening steps. | **Breakthrough: 918.8 LB** (new all-time record). 84.8% Zach at 10M (balanced seats: 85%/84%). 16/21 loss seeds. GCP continuation peaked at 31M (84.4% Zach, 17/21 loss seeds). | Peaked at 31M | **918.8** |
| **Rev32** | rollout=128, BC warmstart (`bc_isaiah_hober_pressure_5k.pt`), LR=0.0001. Jarvis H100 spot. | Test rollout=128 as middle ground between 64 and 512. | Spot instance preempted at 16M, checkpoints not synced (rsync symlink bug). Run lost. | Instance preempted + sync bug | — |
| **Rev32b** | First Strike 4×t<20 (stronger/shorter window). Resume Rev31 31M. GCP L4, rollout=64. | Push step-0 firing further — 2×t<50 fixed step 1-2, try 4×t<20 for step 0. | **New Zach record: 88.7% at 6M. 20/21 loss seeds.** Ajay panel = 0.8% (same as Rev31 — intercept aiming gap structural). | Peak at 6M | **pending LB** |
| **Rev33** | Resume Rev31 31M + `--bc-samples tempo_mix_small.pkl --bc-coef 0.05`. Jarvis H100 spot. | BC auxiliary nudge from tempo dataset (2817 samples). | Ran to 10M. Ajay quick-16: 1/16 (6.25%) — marginal improvement. Killed, checkpoints saved locally. | Peak unclear, marginal Ajay gain | — |
| **Rev34** | Resume Rev33 7M + `--bc-samples conversion_mix_loose.pkl --bc-coef 0.05`. GCP L4. | Conversion-focused BC (1253 samples). | Killed at 2M — conversion regressed badly (us_first_cap 14→136). BC disrupted existing policy. Paused, resume tomorrow. | BC disrupted conversion | — |
| **Rev35** | SSDR v1 (random play warmup). First Strike removed. Resume Rev32b 6M. | Break symmetric-start passive Nash. | Carpet-bomb collapse at 1M (p90fleet=2754, srcs_multi=5.8). Random 60% fire warmup taught "fleet traffic = fire more". Killed. | Fleet explosion | — |
| **Rev35b** | SSDR v2 (asymmetric planet assignment, frac=0.3/max=2). entropy-ships 0.08. | No random play — opponent gets 1-2 extra planets at reset. Clean board, no fleet chaos. | Peaked 2.0% Ajay @ 5M. ship0 collapse at 7M. entropy fix delayed but didn't prevent. | ship0 collapse | — |
| **Rev35c** | SSDR v2 + `--min-ship-bin 4`. Resume Rev35 5M. | Ban 1-4 ship bins to prevent degenerate 1-ship probing. | **Peaked 3.1% Ajay @ 1M** (new best). Regressed 2.0% by 4M — pool contamination. ship0=0 throughout (mask trivially enforced). | Pool contamination | — |
| **Rev35d** | SSDR v2 + min-ship-bin=4 + **pool mask gating** (pool envs get symmetric starts). Resume Rev35c 1M. | Fix pool contamination: old checkpoints poisoned by asymmetric boards they weren't trained on. | **3.1% @ 1M, 2.3% @ 2M** — still regressing, just slower. Mask gating helped but self-play Nash reforms regardless. SSDR gives 1M burst then fades. | Self-play Nash reformation | — |
| **Rev38** | New pairwise features (roi_20, roi_50, enemy_contest, pairwise=15). Zero-padded warmstart from `rev32b_6M_pairwise15.pt`. 256 envs, rollout=128. GCP L4. | Test if 3 new pairwise features + rollout=128 improve Ajay win rate. | Peaked **2.7% Ajay @ 5M** (full panel 7/256), **89.1% Zach @ 5M**. Collapsed to 0% Ajay by 6M, stayed 0% through 20M. New feature weights stayed near-zero throughout (norm 0.035–0.09 vs orig 0.8–1.4). Zero-padding = dead gradient signal from iter 1. Same passive Nash pattern as Rev37. | New features never activated (zero-padded start) | **1006.8** (first 1000+, new record) |
| **Rev39** | BC v2 warmstart (`rev32b_6M_ajay_bc5k_v2.pt` — BC-trained on Ajay self-play with pairwise=15, new feature norms 0.17–0.24). BC aux loss: `ajay_bc_1k_v2.pkl` (78,493 samples), bc_coef=0.01→0.02→0.03. First Strike linear decay (mult=2.0, steps=200). 512 envs, rollout=128. Jarvis H100 spot. | New features activated from start (BC gave them real weights). BC aux maintains Ajay signal during PPO. Continuous FS decay instead of cliff. | Peaked **1.6% Ajay @ 8M** (4/256). Deteriorated to ~0.9% @ 12M — passive Nash reasserting. Ran to 15M (timestamp rev39_20260606_055742). New features confirmed active (roi_20=0.193, roi_50=0.215, enemy_contest=0.224 @ 4.7M). Metrics improved (fire[0] 0.17→0.19, srcs_multi 0.82→1.01) but didn't translate to wins. | Passive Nash @ ~8M despite active features | — |

**What we know:**
- **Rev31 First Strike**: opening paralysis fixed, 918.8 LB record. Was a band-aid for symmetric-start problem.
- **Rev32b**: First Strike 4×t<20 → Zach 88.7%, 20/21 loss seeds — new records.
- **Ajay's strategy**: greedy production-horizon model (only fire if ΔProduction×H > ships_cost). Uses `orbit_lite` for targeting. Our `--target-decode` also uses orbital intercept — gap is NOT routing, it's conversion timing.
- **SSDR verdict**: Asymmetric planet starts DO help (0.8% → 3.1% vs Ajay). But improvement is transient — peaks at ~1M then self-play Nash reforms. Min-ship-bin=4 prevents ship0 collapse. Pool mask gating slows regression but doesn't stop it. The SSDR gradient signal is real but not sustained.
- **Best Ajay result**: Rev35c 1M = **3.1% (8/256)** — `gpu_run_artifacts/gcp_rev35c/checkpoints/torch_step_1048576_rev35c_20260605_052334.pt`
- **Rev34 lesson**: BC auxiliary disrupts existing conversion even at bc_coef=0.05.
- **Zero-padding new features = dead signal**: Rev38 started with roi_20/roi_50/enemy_contest zeroed out. Norms never exceeded 0.09 across 20M steps (vs 0.8–1.4 for orig features). BC warmstart needed to give new features real gradient signal from iter 1.
- **Passive Nash cliff at ~6M**: Rev38 (and Rev37) both collapsed to 0% Ajay at exactly 6M regardless of new features or rollout=128. Self-play Nash is a structural attractor — BC aux during PPO is the proposed fix.
- **BC aux + FS decay delays but doesn't stop Nash**: Rev39 peaked at 1.6% @ 8M (vs Rev38's 2.7% @ 5M). BC warmstart successfully activated new features, but passive Nash still reformed by ~10-12M. The delay (~8M vs ~6M cliff) is modest. Likely need stronger reward signal (higher FS mult/steps) to sustain aggression.
- **Continuous First Strike**: linear decay from `first_strike_mult` at t=0 to 1.0 at t=`first_strike_steps`. Replaced binary cliff in torch_env.py. Better gradient at boundary.
- **pairwise=15 feature layout**: `pair_kv.weight` is [192, 111]. Cols 96–107 = orig 12 pairwise, cols 108–110 = new 3 (roi_20, roi_50, enemy_contest).
- Zach panel saturating ~88–89%. Ajay full panel is primary signal.
- Eval on training instances: always `CUDA_VISIBLE_DEVICES=""` to avoid GPU OOM.
- launch_gpu_gcp.sh: now verifies rsync landed + clears .pyc cache after sync.
- Export requires `--target-decode` for Phase 1.

**LB scores:** Rev38 5M = **1006.8** ← record (first 1000+) | Rev32b 6M = 874.3 | Rev31 10M = 918.8 | Rev30 11M = 866.3 | Rev28 27M = 843.9
**Target:** Top 10 needs ~1153. Gap = ~234 points.

---

## Current State (2026-06-06)

**Active runs:** None. Rev39 ended, Jarvis instance 422162 destroyed.

**In progress:**
- Rev38 checkpoint sweep: running full 256-game Ajay panels on 4M and 6M to find best checkpoint (known: 5M=2.7%, 10M=2.3%)
- Next training run planned: Rev40 with `--first-strike-mult 4.0 --first-strike-steps 400` (user suggestion: stronger/wider FS window)

**Best checkpoints locally:**
- Rev32b 6M: `gpu_run_artifacts/gcp_rev32b/checkpoints/torch_step_6815744_rev32b_20260604_112217.pt` — Zach 88.7%, Ajay 0.8%, LB 874.3
- Rev35c 1M: `gpu_run_artifacts/gcp_rev35c/checkpoints/torch_step_1048576_rev35c_20260605_052334.pt` — Ajay **3.1%** (best Ajay result)
- Rev38 5M: `gpu_run_artifacts/rev38/checkpoints/torch_step_5242880_rev38_20260605_181635.pt` — Ajay 2.7% (7/256), Zach 89.1%, submitted (pending LB)

**Rev38 Ajay panel results:**
| Checkpoint | Ajay wins | Ajay % |
|---|---|---|
| 4M | in progress | — |
| 5M | 7/256 | **2.7%** |
| 6M | queued | — |
| 10M | 6/256 | 2.3% |

**Key diagnostic tools:**
- `orbit_wars_rl/diagnose_opening.py` — FireP per step on episode JSON
- `orbit_wars_rl/test_seed6462.py` — 21-seed isolation test
- `orbit_wars_rl/step_firep.py` — compare FireP at steps 0-3 across checkpoints
- `opponents/candidate_ajay_1200.py` + `opponents/orbit_lite/` — Ajay now works as panel opponent
- `docs/submissions.md` — full submission log with Kaggle IDs and checkpoint paths

**Ajay gap root cause (identified 2026-06-04):**
- `orbit_lite.intercept_aim` computes where planet will BE when fleet arrives → fleets land in ~4 steps
- Our agent fires at current planet position → fleets take ~12 steps (3× slower)
- At step 5: Ajay has 2 planets, we have 1 → snowballs from there
- Fix needed: intercept-aware action decode in `eval.py` / `export_agent.py`

**Next priorities:**
1. Complete Rev38 checkpoint sweep (4M, 6M panels) — submit best
2. Rev40: `--first-strike-mult 4.0 --first-strike-steps 400`, resume from best Rev39 checkpoint (15M: `gpu_run_artifacts/jarvis_rev39/checkpoints/torch_step_15728640_rev39_20260606_055742.pt`)

**Rev30b** completed: peak 85.2% vs Zach at 4M (LR=0.000025). Best checkpoint: `torch_step_4194304_rev30b_20260603_143644.pt`
- Resume: Rev28 22M checkpoint (=28M overall from scratch)
- Total target: 100M steps. Currently ~32M overall.
- Best panel: **58.2% vs Zach at 32M** and still climbing
- Panel watcher: Zach + HB only (Suneet removed), auto-runs on each 1M checkpoint

**Rev29** — rollout=512, 256 envs, H100 (machine 420547, `217.18.55.95`)
- Resume: Rev28 10M checkpoint (=32M overall)
- Hypothesis: full-game credit assignment without 8-hop bootstrapping chain
- Early panels: 1M=57.8% (+11pp vs rev28 at same step), 4M=52% (oscillating)
- V_loss stable (no critic shock), ship0≈0, meanshipbin=19-21
- clip_frac=0.213 still drifting down (healthy)
- **Tomorrow (2026-06-01):** Submit rev11 2M as slot 1. Then consider rev12 direction.

**Rev12 (decided 2026-05-31): stabilized scaled self-play, 50M target.**
- **Hypothesis:** passivity drift is not fixed by duration alone (rev11 proved that — pure self-play 10M still peaked at 2M). Our single-delta tests (rev8 shaping, rev9 fire-entropy, rev10 rollouts) each isolated ONE top-team change and failed because the others fought back. Reproduce the top-10 team's stabilizer recipe **as a unit** (documented exception to one-delta-at-a-time).
- **Resume:** `torch_step_2031616_20260531_065423.pt` (rev11 2M best, fire=0.34).
- **Recipe deltas from rev11:** pure `+1/-1` (`--win-margin-coeff 0.0`); rollout 512 / num-envs 64 / **minibatch 16** (mb8 OOM'd at first PPO update on L4 23GB — mb16=2048/batch is the largest that fits; run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, GPU sits ~22/23GiB); `--max-grad-norm 10.0` (was hardcoded 0.5); entropy fire 0.02 / angle 0.03 / ships 0.02 (~2×, NOT team's 50%-of-max — rev9 showed our env punishes random firing); `--lr-schedule-steps 200000000` (~constant LR); `--total-steps 50000000`.
- **Running on GCP** (`orbit-wars-training`, us-central1-b, L4) as of 2026-05-31. Launched 10:12 UTC after an mb8 attempt at 10:04 OOM'd.
- **Code change:** exposed `--max-grad-norm`, `--entropy-coef-angle`, `--entropy-coef-ships` as CLI flags in `train_torch.py` (fields already existed in `config.py`).
- **Scripts:** `gpu_run_artifacts/hellburner_spot/run_remote_phase1_rev12.sh` + `launch_phase1_rev12.sh`.
- **Monitoring (NOT fire[0] kills):** dense 1M checkpoints + Zach/Suneet panel capture the peak wherever it lands; do NOT kill on fire[0] decline or HB regression. Watch `clip_frac` — if it creeps toward 0.30 (team canary), HALVE lr (intervention, not kill). First 5 iters: check `nvidia-smi` mem (bump minibatch to 16 if OOM). Hard stop 50M (budget ~$50).
- **Success:** any rev12 checkpoint exports (`--target-decode`) to LB > 894.

**LB scores for correctly-exported Phase 1 agents (2026-05-31):**

| Submission | Steps | LB Score | Notes |
|---|---|---|---|
| 141208 (old arch) | ~63M eff | **894.0** | Best ever, target to beat |
| rev10d 1M (Phase 1) | 1M | 772.4 | Better than rev7 1M on LB |
| rev7 1M (Phase 1) | 1M | 750.3 | Panel best, but LB < rev10d |
| rev10d 2M (Phase 1) | 2M | 600.0 | Passive drift hurt 2M ckpt |

Key insight: **rev10d 1M (772) > rev7 1M (750) on LB despite worse panel** — panel and LB don't correlate perfectly. More firing = worse vs HB (panel) but better on LB.

**Export bugs fixed (2026-05-31) — all previous Phase 1 submissions were broken:**
1. `def agent(obs)` → `def agent(obs, cfg=None)` — was crashing every step silently (score 87)
2. Missing `--target-decode` — was using angle decode (wrong for Phase 1)
Always use: `python3 orbit_wars_rl/export_agent.py --checkpoint <ckpt> --output <out> --target-decode`

---

---

## LB Episode Analysis (2026-05-31)

### How to fetch our episodes

```python
# Step 1: list our submission's episodes
import requests, json

with open('/Users/saheb/.kaggle/access_token') as f:
    token = f.read().strip()
with open('/Users/saheb/.kaggle/kaggle.json') as f:
    creds = json.load(f)

# EpisodeService API (not the standard v1 API)
url = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
resp = requests.post(url,
    auth=(creds['username'], creds['key']),
    json={"submissionId": 53076736},  # our best submission
    headers={"Content-Type": "application/json"})
episodes = resp.json()['episodes']
# Save: json.dump({'episodes': episodes}, open('/tmp/our_episodes.json','w'))

# Step 2: download individual replay JSON
# IMPORTANT: standard v1 API requires Bearer token (not basic auth)
ep_id = 78025831
r = requests.get(
    f"https://www.kaggle.com/api/v1/competitions/episodes/{ep_id}/replay",
    headers={"Authorization": f"Bearer {token}"}   # ~/.kaggle/access_token
)
open(f"/tmp/{ep_id}.json", "wb").write(r.content)
```

> ⚠️ The replay endpoint **requires Bearer token** (`~/.kaggle/access_token`), not basic auth.
> Basic auth returns 401. The `access_token` file is separate from `kaggle.json`.

### Key findings from 159 LB episodes (submission 53076736, score 894.4)

**Win/loss summary:**
- 63 wins (40%) / 96 losses (60%)
- We are **not** matched against top-10 agents — highest opponent score that beat us was ~1036
- **38 out of 95 losses (40%) are to weaker opponents** (their score < ours at game start)

**Loss breakdown by opponent score:**
| Bracket | Losses | % of losses |
|---------|--------|-------------|
| < 800 | 6 | 6% |
| 800–900 | 38 | 40% |
| 900–1000 | 46 | 48% |
| 1000–1100 | 5 | 5% |
| > 1100 | 0 | 0% |

**Root cause: bimodal fire behavior (same checkpoint, two modes)**

From replay analysis of 10 losses + 5 wins:

| Metric | Our WINS | Our LOSSES | Winner in losses |
|--------|----------|------------|-----------------|
| fire_rate | **0.468** | **0.223** | 0.528 |
| multi_rate | 0.502 | 0.316 | 0.633 |
| avg_ship_bin | 31.0 | 34.3 | 53.3 |
| n_steps | 232 | 281 | 281 |

**Same agent, same checkpoint — we fire 2× more often in games we win.**
In losses we play passively (22% fire rate); the winner fires on 53% of steps.
Longer game length in losses (281 vs 232 steps) confirms we're being snowballed.

**Implication:** The policy has a passive mode it locks into for specific game states.
The training self-play equilibrium creates states where holding is locally optimal.
Training `fire[0]~0.32` is an average that masks this bimodal behavior.

### Rev9 fix: entropy-coef-fire

`--entropy-coef-fire` (default 0.01) controls the entropy bonus on the fire head specifically.
Increasing it prevents the policy from becoming deterministically non-firing.

Rev9 uses `--entropy-coef-fire 0.05`. H_fire rose from ~0.09 (rev7/rev8) to 0.143 by iter 3.

---

## Leaderboard Reality Check (2026-05-31)

| Rank | Agent | Score | Notes |
|---|---|---|---|
| 1 | Isaiah @ Tufa Labs | 1751.4 | Target for top 10 |
| 2 | typeIIIfairy | 1716.7 | |
| 3 | Vadasz | 1614.9 | |
| 7 | **Zachary Ruhe** | 1596.0 | ← Our `candidate_zach_public.py` (old version) |
| 17 | kovi | 1412.1 | Beats even rank1; large ship commitment |
| 23 | Shun_PI | 1354.1 | Was rank1 when replays were collected |
| 100 | ttmn | 1153.0 | Score needed for top 100 |
| **669** | **Saheb (us)** | **894.4** | Current position |
| 962 | **Suneet Saini** | 827.4 | ← Our `candidate_suneet_lb1200.py` — **below us** |
| N/A | Hellburner | — | Not on LB — local test bot only |

**LB game type split (rev7 1M submission, 159 episodes):**
- 1v1: 76 games — **50% WR** (we're competitive in head-to-head)
- FFA: 83 games — **30% WR** (badly losing in 4-player games)

**The FFA problem:** Our agent is trained in 2-player mode only. 52% of LB games are FFA, where it plays a game it was never trained for. This is the primary LB score drag. Top-10 team trained *separate* 2p and 4p policies.

**Critical implication**: Our panel opponents don't represent actual LB threats.
- Suneet (rank 962) is below us — optimising vs Suneet adds no signal.
- HB not on LB at all.
- Zach (rank 7) is real but `candidate_zach_public.py` is an old version.
- Top threats (Isaiah, typeIIIfairy, Vadasz, 213tubo, bowwowforeach, 3Comets) untested.

**What kovi does differently** (beats rank1 Shun_PI — 8 wins from 116 games):
- avg_ship_bin=40.6 vs Shun_PI 27.5 → commits ~50% more ships per attack
- multi_src_rate=0.46 vs 0.43 → fires from more sources simultaneously
- fire_rate=0.47 vs 0.43 → slightly more aggressive

**Panel opponent priority for next update:** find/build proxies for Isaiah, typeIIIfairy, Vadasz.

---

## Current State (original — 2026-05-30)

**Active run:** None — Rev7 killed at 4.6M steps (2026-05-30 ~17:40)

**Next:** Rev8 — decide single delta based on Rev7 1M results below

**Rev7 1M panel results** (`torch_step_1015808_20260530_104829`, 256 games, 2026-05-30):

| Opponent | Score | vs Phase 1 6M peak |
|----------|-------|--------------------|
| Hellburner | 40.2% | +1.5pp vs 38.7% |
| Zach | 50.8% | -3.5pp vs 54.3% |
| Suneet | 59.8% | -1.9pp vs 61.7% |

HB up slightly vs 6M peak; Zach/Suneet slightly below — all expected at 1M steps.
Trajectory looks healthy (Suneet ~60% at 1M is strong). Rev7 6M checkpoint will be the real comparison.

**Rev7 post-mortem:**
- Instance `i-0a5129fba17cf3de6` terminated; all checkpoints + log pulled locally
- Ran to 4.6M steps — blew past 3M hard cap due to lack of automated kill
- `fire[0]` declined across all 4 checkpoints: 0.34 → 0.32 → 0.29 → 0.25 (same passivity pattern as rev5)
- `avgfleet` rose steadily: 72 → 76 → 88 → 92 (passivity signal)
- `srcs_multi` stayed clean: 1.0–1.7 throughout (penalty working, not the problem)
- IL removal alone was not enough to prevent passivity drift — just slowed it slightly
- **Best checkpoint expected: 1M** (`torch_step_1015808_20260530_104829`) — consistent with 141208 pattern

**Rev7 fire[0] progression:**

| Checkpoint | Steps | fire[0] | avgfleet |
|---|---|---|---|
| 1M | 1,015,808 | ~0.32 | ~74 |
| 2M | 2,031,616 | ~0.29 | ~82 |
| 3M | 3,047,424 | ~0.27 | ~86 |
| 4M | 4,063,232 | ~0.25 | ~90 |

**Ops fix:** panel eval watcher had 3 bugs fixed (2026-05-30):
1. Opponent paths broken after moving to `opponents/` — fixed
2. `EVAL_EVERY_N_CHECKPOINTS=3` skipped 1M checkpoints — changed to 1
3. N/A — python path was valid

**Kill signals (standard, for all runs):**
1. `fire[0]` declining 3 consecutive 1M checkpoints → kill immediately
2. `fire[0] < 0.25` at any checkpoint → kill immediately
3. `srcs_multi > 4.0` or `fire[0] > 0.55` → kill (collapse)
4. Hard cap: 3M steps regardless — **enforce this strictly**

---

## Architecture

### Phase 1 (current)
- **Ship bin mode:** absolute, 32 bins
  - `SHIP_COUNTS = [1,2,3,4,5,6,7,8,9,10,12,14,16,19,22,26,30,35,42,50,60,72,86,102,122,145,173,206,245,290,350,420]`
  - Bin 0 = 1 ship — 0-masking built in by design
- **Features:** planet=20, fleet=13, global=11, pairwise=12, max_owned=16
- **Warmstart:** `bc_phase1_warmstart.pt` (BC on top-200 Kaggle replays)
- **Action decode:** `target` mode

### Old architecture (141208 chain — best ever)
- **Ship bin mode:** fraction, 10 bins (0.1–1.0 × source fleet)
- **Features:** planet=18, fleet=9, global=10, pairwise=10
- `min_ship_bin=1` masked the 0-ship bin (a no-op fire)
- No IL anchor in any blitz run
- ⚠️ Not interchangeable with Phase 1 checkpoints

---

## Training Flag Glossary

### PPO / Optimiser

| Flag | Default | What it does |
|---|---|---|
| `--learning-rate` | 3e-4 | PPO learning rate (Adam). |
| `--lr-schedule-steps` | — | Total steps over which LR cosine-decays to 0. Set to 60M so LR barely moves in short runs. |
| `--num-envs` | 512 | Parallel game environments. More = faster SPS but more GPU memory. |
| `--rollout-steps` | 64 | Steps collected per environment before each PPO update. Longer = better credit assignment for sparse terminal reward, but uses more GPU memory. |
| `--num-minibatches` | 32 | How many minibatches to split each rollout into for PPO gradient updates. Fewer = larger batches = more memory. Keep `total_buffer / num_minibatches ≈ 2000` to avoid OOM on L4. |
| `--ppo-epochs` | 2 | How many times to iterate over each rollout during the PPO update. |
| `--total-steps` | 30M | Training runs until this many env steps. Watcher hard-kills at 3M (10M for rev11). |

### Self-play pool

| Flag | Default | What it does |
|---|---|---|
| `--pool-mode` | mixed | `mixed` = sample both recent snapshots and older ones. `latest` = only recent. |
| `--pool-fraction` | 0.75 | Fraction of games played against the pool (vs current self). |
| `--pool-max-size` | 20 | Max checkpoints kept in the opponent pool. Bigger = more diverse opponents. |
| `--pool-checkpoint-interval` | 500K | How often to snapshot current model into the pool (env steps). |
| `--pool-external-fraction` | 0 | Fraction of pool games guaranteed to go to external opponents (bypasses PFSP). |
| `--pool-pfsp-min-games` | 30 | Use uniform sampling until an opponent has played this many games (prevents death-spiral from noisy early win rates). |
| `--pool-mastered-threshold` | 0.99 | Win rate above which an opponent is considered "mastered" and removed from pool. |
| `--external-opponents` | — | Path(s) to external rule-based agent files. We used `opponents/candidate_hellburner.py`. |

### Reward shaping

| Flag | Default | What it does |
|---|---|---|
| `--win-margin-coeff` | 0.5 | Scales the reward by win margin (more ships remaining = bigger reward). Encourages decisive wins. |
| `--shaping-coef` | 0 | Per-step reward = `coef × (my_material_delta)`. Tried in rev8 — hurt performance because passive resource collection also gives material gain. |

### Entropy / exploration

| Flag | Default | What it does |
|---|---|---|
| `--entropy-coef-fire` | 0.01 | Entropy bonus specifically on the fire head. Higher = more random firing decisions. Tried in rev9 — raised H_fire metric but made play less strategic, hurting panel/LB. |

### Collapse guards

| Flag | Default | What it does |
|---|---|---|
| `--srcs-multi-penalty` | 0 | Per-step penalty when the agent fires from too many sources simultaneously (carpet-bomb). Set to 0.001. |
| `--srcs-multi-threshold` | 2.0 | Sources above this number get penalised. |

### IL (behaviour cloning anchor) — NOT used in rev7+

| Flag | What it does |
|---|---|
| `--il-lambda` | KL penalty toward the BC warmstart on every PPO update. Removed in rev7 — it amplifies passivity drift. |
| `--il-ref` | Reference checkpoint for IL. |
| `--il-decay-frac` | Fraction of training over which IL weight cosine-decays to 0. |

### Action decode

| Flag | What it does |
|---|---|
| `--action-decode target` | Phase 1 mode: model outputs a target planet logit; argmax at inference. Required for Phase 1 checkpoints. **Always pass `--target-decode` when exporting.** |

---

### Key training metrics (in log lines)

> ⚠️ **Thresholds are empirical** — derived from watching our own runs, not from studying top-ranked agents' training internals. They are proxy signals, not ground truth. LB score is ground truth.

### Data-driven targets from rank1 replays (Shun_PI, 116 1v1 games)

| Metric | Rank1 WINS | Rank1 LOSSES | kovi (beats rank1) | Our rev7 wins | Our rev7 losses |
|---|---|---|---|---|---|
| fire_rate | **0.320** | 0.366 | 0.310 | 0.468 | 0.223 |
| multi_rate | 0.325 | 0.364 | 0.326 | 0.502 | 0.316 |
| avg_ship/attack | 31.7 | 17.6 | **45.7** | ~31 | ~34 |
| n_steps | **145** (decisive) | 167 | 167 | 232 | 281 |

**Key insights:**
1. **Lower fire_rate = winning** — rank1 fires *less* in wins (0.32) than losses (0.37). Selective, decisive attacks beat reactive spray.
2. **Ship commitment is the differentiator** — kovi beats rank1 not by firing more often but by sending **45.7 avg ships per attack** vs rank1's 31.7. Big forces at the right time.
3. **Our fire_rate analysis may be misleading** — our LB losses show fire_rate=0.223, but this may be *reactive* (responding to being attacked) rather than passive. Rank1 also has lower fire_rate in wins.
4. **Our avg_ship (~31) matches rank1** — the problem may not be ship commitment size. It may be tactical timing.
5. **Shorter games = winning** — rank1 wins in 145 steps, loses in 167. We win in 232, lose in 281. We should be closing games faster.

**Revised focus:** Rather than maximising fire_rate, target *decisive* attacks — fewer but larger (meanshipbin ≥ 18), and learn to close games before step 200.

| Metric | Healthy range | Warning | Why it matters |
|---|---|---|---|
| `fire[0]` | 0.28–0.45 | < 0.25, or declining 3 consecutive checkpoints | Fraction of steps the agent fires from planet-slot 0 (its first owned planet). Proxy for aggression — if it stops firing from there, it's hoarding ships. Kill floor 0.25 was set empirically: below this, panel scores reliably crash. |
| `avgfleet` | 60–85 | > 95 and trending up | Average ships on planets. **Rising = passive.** Ships accumulate when you *hold* rather than *attack*. An aggressive agent keeps fleets lean by sending them constantly. If avgfleet climbs, the agent has learned to build up fleets rather than spend them. |
| `srcs_multi` | 1.0–2.5 | > 4.0 | Average number of source planets firing simultaneously. A value > 4 signals the **carpet-bomb collapse**: agent fires from all planets at once, empties defences, rarely wins. This is a degenerate training artifact. Normal multi-source attacks use 2-3 sources. Threshold 4 was empirically tuned. |
| `clip_frac` | 0.10–0.22 | > 0.30, creeping upward | PPO's measure of how often the policy update was clipped (policy changed too much). Creeping upward = gradients are too large = value function estimates go stale = training destabilises. From standard PPO/IMPALA/AlphaStar literature: sustained > 0.30 is a red flag. Our rev11 shows clip_frac drifting *down* to 0.205 which is unusually good. |
| `H_fire` | 0.08–0.15 | < 0.05 (deterministic) | Entropy of the fire-head distribution. Low = policy is near-deterministically choosing fire or no-fire in each state. Too low = rigid passive policy that can't adapt. **Note:** higher is NOT automatically better — rev9 showed that boosting H_fire with `--entropy-coef-fire` made firing random rather than strategic, hurting LB. |
| `meanshipbin` | 15–18 | > 19 trending up | Mean ship-count bin when firing (Phase 1 has 32 bins; bin 16 ≈ mid-range fleet fraction). Rising alongside avgfleet = passive (building bigger fleets before attacking). Rising with stable avgfleet = sending larger forces per attack (may be OK). |
| `SPS` | 650–730 | < 500 | Environment steps per second. Drops if: rollouts are longer (rev10d hit 499 with envs=64), pool workers are slow, or GPU is doing other work. **⚠️ If SPS drops to ~0 and GPU shows 0% utilisation with CPU at 100%: the heuristic worker pool is deadlocked — see note below.** |

---

## Best Checkpoints Ever

| Checkpoint | Arch | Steps | HB | Zach | Suneet |
|---|---|---|---|---|---|
| `torch_step_1015808_20260526_141208` | Old | ~63M effective | **55.5%** | 74.2% | 80.1% |
| `torch_step_1015808_20260526_123203` | Old | ~62M effective | 44.5% | 75.4% | 75.8% |
| `torch_step_1015808_20260526_174758` | Old | ~63M + 1M blitz | 42.6% | 76.6% | 75.0% |
| `torch_step_6094848_20260529_160908` | Phase 1 | 21.7M | **38.7%** | 54.3% | 61.7% |

**Target:** >75% on all three simultaneously.

---

## Run History

### Old Architecture

#### Foundation (62M self-play, no IL, no penalty)
- Pure self-play, no external opponents, no IL
- Result: `fire[0]` dropped from 0.25 → 0.09 over 62M steps
- **All checkpoints scored poorly on full panel eval** — too passive to attack vs real opponents
- ⚠️ **The foundation checkpoints themselves were bad agents — panel score, not fire[0], is truth**
- But built deep game-sense that the blitz chain exploited

#### 123203 chain → 141208 blitz (best ever)
- `torch_best_123203` (44.5% HB) was produced by an unknown blitz on the foundation
- **141208 run:** resumed from 123203, +1M steps, no IL, no penalty
  - 1M = **55.5% HB** ← peak, best ever
  - 2M = 47.3%, 3M = 46.1%, 4M = 45.3%, 5M = 44.1%
  - **Peaked exactly at 1M, declined every checkpoint after**
- Key: foundation had ~63M of self-play game-sense; blitz injected HB aggression before passivity reset

#### ⚠️ What we don't fully understand about the 1M blitz mechanism
The 45M and 62M foundation checkpoints all scored poorly on the panel. The 1M blitz on 123203
produced a large jump (+11pp HB to 55.5%). But the *same* blitz on 141208 immediately regressed
(-13pp HB to 42.6%). Open questions:
1. **Why did the 1M blitz work on 123203 but not 141208?**
   - 123203 was itself the product of a prior blitz on the foundation — was it at an optimal
     "tipping point" in weight-space that made the HB gradient land well?
   - 141208 had already absorbed the HB signal; a second blitz overfit?
2. **Is foundation depth actually the key variable?**
   - The story is "foundation gives game-sense, blitz injects aggression" — but we haven't
     proven the foundation checkpoints had useful game-sense. They scored 0% vs HB.
   - Alternative: 123203 was good *despite* being on a bad foundation, not *because* of it.
3. **Fire[0] is a proxy; panel eval is ground truth.**
   - We've been killing runs based on fire[0] decline, but the 62M foundation had terrible
     fire[0] AND terrible panel scores. A run could have healthy fire[0] but still be a bad agent.
   - Implication: **run panel eval on every 1M checkpoint**, don't just trust fire[0].

#### Post-141208 blitz (174758)
- +1M more blitz on 141208 best
- HB=42.6%, Zach=76.6%, Suneet=75.0% — HB regressed, Zach/Suneet held

---

### Phase 1

#### Rev1–Rev3 (BC warmstart, il-decay-frac=0.5)
- Resume: `bc_phase1_warmstart.pt`
- `--il-decay-frac=0.5` — IL decayed to 0 halfway through, too short
- Peak: Zach=12.9%, HB=0.8%
- IL anchor kept policy too conservative; not enough self-play to build game-sense

#### Rev4 (resume rev3 3M peak, +6M)
- Same config as rev3
- `srcs_multi` collapse at iter 200+ — carpet-bomb failure mode
- Peak same as rev3, no improvement

#### Rev5 (resume rev4 3M, +6M, srcs-multi-penalty added)
- Added `--srcs-multi-penalty 0.001 --srcs-multi-threshold 2.0` to guard collapse
- **6M peak: HB=38.7%, Zach=54.3%, Suneet=61.7%** ← Phase 1 best
- Regression after 6M: fire[0] 0.35→0.25 by 12M, avgfleet rising
- `srcs_multi` stayed 1.2–1.8 throughout (penalty worked, not the cause of passivity)
- IL anchor (`--il-lambda=0.01`) + self-play equilibrium together drove passivity

#### Rev6 blitz (resume rev5 6M, 3M, cosine penalty decay)
- Tried cosine decay on the srcs-multi penalty to 0
- HB=0%, Zach=19.9% — penalty decayed too aggressively, carpet-bomb returned

#### HB blitz attempt (2026-05-30, pool-fraction=0.9, pool-external-fraction=1.0)
- Tried to replicate 141208 blitz on Phase 1: 90% HB exposure, lr=3e-5, no IL
- **Destroyed everything in 500K steps: HB=0%, Zach=4.3%**
- Root cause: Phase 1 at 21.7M effective steps lacks the deep foundation (~63M) the 141208 blitz relied on
- Heavy HB overwrites fragile generalisation — blitz only works as final polish on fully-developed model

#### Rev7 (killed 2026-05-30, resume rev5 6M, no IL)
- **Single delta from rev5: IL anchor removed entirely**
- Rationale: 141208 chain had no IL; IL pulls toward conservative BC → amplifies self-play passivity drift
- Config: pool-fraction=0.75, pool-external-fraction=0.05 (5% HB, 75% self-play), lr=3e-4 cosine over 60M schedule
- `fire[0]=0.57` at iter 1 (optimizer transient), settled ~0.32 at 1.5M, declined to 0.25 by 4M
- srcs_multi stayed 1.0–1.7 throughout — penalty working, not the problem
- **Verdict:** IL removal alone insufficient; passivity still set in, just slightly slower than rev5
- **Best checkpoint: 1M** (`torch_step_1015808_20260530_104829.pt`) — HB=40.2%, Zach=50.8%, Suneet=59.8%
- Ran to 4.6M before kill (3M hard cap was not enforced automatically)

#### Rev8 (killed 2026-05-31, resume rev7 1M, shaping-coef)
- **Single delta from rev7: `--shaping-coef 0.05`**
- Rationale: per-step material-delta reward to incentivise attacking over passive fleet-building
- Panel at 80 games: **HB=30%** (vs rev7 1M baseline 40.2%) — clear regression
- fire[0] hit 0.22 by iter 45 (1.5M steps), killed early
- H_fire stayed ~0.09 throughout — entropy unchanged, passive mode unchecked
- **Verdict: failed.** Shaping rewards material gain, which can be achieved passively (collect more planets); did not incentivise firing specifically
- **Best checkpoint: 1M** (`torch_step_1015808_20260530_184448.pt`) — HB~30%, worse than rev7

#### Rev9 (killed 2026-05-31, resume rev7 1M, entropy-coef-fire)
- **Single delta from rev7: `--entropy-coef-fire 0.05` (5× default 0.01)**
- Rationale: LB episode analysis showed bimodal behavior — same checkpoint fires at 0.47 rate in wins vs 0.22 in losses
- H_fire rose from ~0.09 to 0.121–0.153 at 1M (entropy working mechanically)
- Panel at 1M (112/192 games): **HB=25%, Zach=42%** — worse than rev7 AND rev8
- **Verdict: failed.** Entropy coef raises H_fire but does not improve strategic play.
  - Higher entropy = more random firing, not more strategic firing
  - HB and Zach are rule-based opponents that punish inefficient/random attacks
  - The bimodal fire behavior is game-state-driven (not entropy-driven); entropy can't fix it
  - The passive mode in losses reflects strategic holding in specific positions, not entropy collapse
- Pattern: rev7→rev8→rev9 on HB = 40.2%→30%→25% — each reward/entropy tweak regresses
- **Best checkpoint: 1M** (`torch_step_1015808_20260530_193824.pt`) — HB~25%, worst so far
- Lesson: stop tuning reward signals and entropy. Try training setup changes instead.

#### Rev10 (killed 2026-05-31, resume rev7 1M, longer rollouts)
- **Attempted deltas:** rollout_steps=512 (OOM × 3 attempts), final working config: `num_envs=64, rollout_steps=512, num_minibatches=32`
- Note: 512×512 and 256×512 both OOM'd on L4 23GB; 64×512 = same 65K buffer size as rev7, worked
- SPS=499 (vs rev7's 680 — fewer envs = slower)
- **Panel results at 1M:** HB=28.1%, Zach=46.9%, Suneet=53.8% — all below rev7 1M
- **Panel results at 2M:** fire[0]=0.32, avgfleet=88, passive drift 1.5-2M
- **LB scores:** rev10d 1M = **772.4**, rev10d 2M = **600.0**
- **Key finding from local replay analysis:**
  - Rev10d fires MORE aggressively than rev7 in losses (fire_rate 0.479 vs 0.387)
  - vs HB: we WIN by playing PASSIVELY (fire_rate=0.238 in wins, 0.479 in losses) — HB is a rush aggressor
  - More aggression = worse vs HB panel, better on LB
  - **HB is the WRONG panel metric** — it rewards passivity which is anti-LB strategy
- **Verdict:** Longer rollouts did not clearly help. Panel misleading due to HB. LB: rev10d 1M (772) > rev7 1M (750) — slightly better.
- Export bugs fixed during this run — all prior Phase 1 LB submissions were broken (scored ~87)

#### Rev11 (active 2026-05-31, resume rev7 1M, pure self-play 10M)
- **Key deltas from rev7:**
  1. **No external opponents** — pure self-play only (HB was wrong opponent, rewarding passivity)
  2. **10M step target** — never ran past 3M; top-10 team ran 600M
  3. **pool-max-size=40** (doubled) — more diverse self-play partners
  4. **fire[0] kill threshold 0.20** (vs 0.25) — allow passive phases rather than killing them
- Resume: `torch_step_1015808_20260530_104829.pt` (rev7 1M, 32-bin absolute)
- SPS=674, clip_frac=0.213 (still drifting down — healthier than all previous runs)
- **1M checkpoint:** fire[0]=0.32, avgfleet=78.1 — comparable to rev7 1M
- **Post-1M behaviour:** avgfleet oscillating 78-85 (NOT monotonically climbing like rev7-rev10)
- clip_frac=0.213 is lowest steady-state of any run — policy settling more stably
- **Submission plan:** not submitting today (4/5 slots used). Submit 3M or 5M checkpoint tomorrow if fire[0] holds.
- **Success criterion:** LB score > 894 (beat 141208 old arch)

---

## External Reference: Top-10 Team Approach (discussion/697725)

Team "Light" + Claude Opus (top-10, was #1 briefly). Posted 2026-05-31. Read the full post.

### Their setup vs ours

| Item | Theirs | Ours (rev9) |
|---|---|---|
| Environment | **JAX rewrite** | Vectorised PyTorch GPU |
| SPS | **~10,000** (basic) → ~2,000 (complex) | ~685 |
| Total steps | **600M** (3 days, RTX 5090) | ~22M effective |
| Model size | ~600K params | 404K params |
| Architecture | Entity transformer | Entity transformer |
| Self-play | **Pure** — no external opponents | Mixed + HB 5% |
| Reward | **`+1/-1` only** (2p mode) | +1/-1 + win-margin coeff |
| Rollout steps | **512** | 64 |
| Num minibatches | **1** | 32 |
| Grad clip | **99** | 0.5 (standard) |
| GPU cost | ~$150 (5090, 3 days) | ~$50 so far |

### Key quotes

> *"About 100M samples with pure self-play should beat all public agents by 90%"*

> *"clip_frac starts creeping up monotonically (0.10 → 0.30+) before entropy_fire collapses or KL spikes. When you see that creep, cut lr or revert capacity. Don't wait for the blow-up."*

> *"+1/-1 is enough for 2p mode."*

> *"Forget sample efficiency. You are doing RL. [Fast environment] is non-negotiable."*

> *"We don't have scale. So, put as many inductive biases as possible."*

> *"Add one architecture delta at a time. Always."* (they violated this, broke training, had to recover)

> *"entropy is like 50% of max"* — far higher than our H_fire ~0.11

### What this means for us

1. **SPS is the primary bottleneck.** 685 vs 10,000 = 14× slower. 100M steps takes us ~41 hours; takes them ~3 hours. A JAX rewrite is the path to their training scale, but it's a large project.

2. **`rollout_steps=512` worth trying** (rev10 candidate). Longer rollouts = better credit assignment for sparse terminal reward. Directly comparable experiment, cheap delta.

3. **`num_minibatches=1`** — full-rollout batch updates. More conservative gradient steps. Worth pairing with longer rollouts.

4. **Pure self-play** — they reached top-10 without any external opponent. Our HB external may be adding overfitting risk not signal.

5. **Entropy 50% of max** — they run with far more entropy than us. Consistent with our rev9 hypothesis (entropy suppresses passive collapse). Their target is much higher than our H_fire~0.12.

6. **Current training chain is structurally sound** — same entity transformer, same PPO + self-play, same "one delta at a time" rule. The gap is scale (600M vs 22M) and environment speed.

### What NOT to copy

- JAX rewrite: too large a change to validate mid-competition
- Their exact features: not shared publicly
- `num_minibatches=1` alone without testing: changes gradient dynamics significantly

---

## ⚠️ HB Is The Wrong Benchmark (discovered 2026-05-31)

Replay analysis of local games (10 games each, rev7 and rev10d vs HB) revealed:

| | Rev7 WINS vs HB | Rev7 LOSSES vs HB | Rev10d WINS vs HB | Rev10d LOSSES vs HB |
|---|---|---|---|---|
| fire_rate | **0.185** | 0.387 | **0.238** | 0.479 |
| n_steps | **500 (timeout)** | 243 | **500 (timeout)** | 186 |

**We beat HB by playing PASSIVELY.** HB is a rush aggressor. We win by holding ships and outlasting it on timeout. We lose when we attack too much (HB captures our empty planets).

This is the **opposite** of LB strategy. LB analysis showed:
- LB wins: fire_rate=0.468, shorter games (232 steps)
- LB losses: fire_rate=0.223, longer games (281 steps)

**Consequence:** Every experiment that increased fire rate (shaping-coef, entropy-coef) improved LB strategy but hurt HB panel score. This explains why rev8/rev9 showed lower HB scores despite training metrics looking healthier. We were optimizing against the wrong opponent.

**Rules going forward:**
1. **Do NOT use HB panel score for kill/keep decisions.** HB rewards passivity which is the opposite of LB strategy.
2. **Use Zach and Suneet panel scores** — they are better proxies (both are on LB, play more strategically).
3. **LB submission score is ground truth.** Panel is for directional guidance only.
4. **Rev10d interpretation:** HB 25.9% (we fire more = worse vs HB), Zach 44.9%, Suneet 65.6%. Net vs rev7: Zach -6pp, Suneet +6pp. Roughly a wash. LB score will decide.

---

## Why Passivity Sets In

Self-play equilibrium rewards holding over firing — `fire[0]` declines monotonically in long runs. Observed in every unconstrained run:
- 62M foundation (no IL, no penalty): `fire[0]` 0.25 → 0.09
- Rev5 (with IL): `fire[0]` 0.35 → 0.25 by 12M

**IL makes it worse:** `--il-lambda` penalises KL divergence from the BC warmstart each update. BC is methodical/conservative (it imitates top human players who build large fleets before attacking). IL + self-play passivity compound each other.

**srcs-multi-penalty is NOT the cause** — in rev5, `srcs_multi` stayed 1.2–1.8 the entire run, well below the 2.0 threshold. Passivity set in anyway.

---

## Short-Run Chain Strategy (current)

Replicate the 123203→141208 structure on Phase 1:

1. Run 2–3M steps, kill before passivity takes hold
2. Resume from best checkpoint → next run
3. Each run is a single delta from the previous
4. Build foundation depth through chaining, not one long run
5. HB blitz only as final polish once foundation is solid (>60M equivalent)

**Chain so far:**
```
bc_phase1_warmstart → rev1/2/3 → rev4 → rev5 (6M peak) → rev7 (1M peak=40.2% HB)
  → rev8 (failed, shaping-coef) → rev9 (active, entropy-coef-fire=0.05)
```

**Rev10 candidate (if rev9 1M looks healthy):** `--rollout-steps 512 --num-minibatches 4`
- Rationale: top-10 team uses rollout_steps=512 vs our 64; longer rollouts improve credit assignment for sparse terminal reward
- Single delta — keep everything else from rev9
- num_minibatches=4 (not 1 — their extreme is untested; 4 keeps batch size sane)

---

## Key Config (Rev7 / current standard)

```bash
python orbit_wars_rl/train_torch.py \
  --resume seed_checkpoints/phase1_resume.pt \
  --total-steps 30000000 \
  --lr-schedule-steps 60000000 \   # cosine over 60M so LR doesn't hit 0 at 3M
  --learning-rate 0.0003 \
  --num-envs 512 --rollout-steps 64 --num-minibatches 32 \
  --ppo-epochs 2 \
  --checkpoint-interval 1000000 \
  --pool-checkpoint-interval 500000 --pool-max-size 20 \
  --pool-mode mixed --pool-fraction 0.75 \
  --external-opponents opponents/candidate_hellburner.py \
  --pool-external-fraction 0.05 \
  --win-margin-coeff 0.5 \
  --action-decode target \
  --pool-pfsp-min-games 30 \
  --pool-mastered-threshold 0.99 \
  --pool-mastered-min-games 500000 \
  --srcs-multi-penalty 0.001 \
  --srcs-multi-threshold 2.0 \
  --terminate-on-done
```

**Flags intentionally absent vs rev5:**
- No `--il-lambda` (removed in rev7 — key delta)
- No `--il-ref` / `--il-decay-frac`

---

## Metrics to Watch

| Metric | Healthy range | Warning | Kill |
|--------|--------------|---------|------|
| `fire[0]` | 0.30–0.55 | Declining 2 consecutive | Declining 3 consecutive or <0.25 |

| `srcs_multi` | 1.0–2.5 | 2.5–4.0 | >4.0 (carpet-bomb collapse) |
| `avgfleet` | 50–90 | >100 and rising | — |
| `clip_frac` | 0.10–0.25 | >0.30 sustained | — |
| `H_fire` (entropy) | 0.05–0.15 | <0.03 (deterministic) | — |

---

## ⚠️ External Opponent Gotcha: `--heuristic-workers` (rev18, 2026-06-01)

**Symptom:** Training hangs after a few iters. GPU shows **0% utilisation** despite process being alive. All CPU cores at 100%. `ps aux` shows 7+ Python worker processes each consuming ~82% CPU.

**Cause:** The heuristic worker pool defaults to `os.cpu_count() - 1` workers (7 on a g2-standard-8). This is fine for lightweight rule-based opponents (Hellburner, Zach) which run fast Python logic. But **141208 and other RL-agent opponents run a full neural network on CPU** — each worker loads the model and runs inference. With 7 workers all inferring simultaneously on an 8-core machine, all CPU is consumed and the main training process (which also needs CPU to coordinate the GPU) deadlocks waiting for workers that can never finish.

**Fix:** Always pass `--heuristic-workers 2` (or 3) when using RL-agent opponents as externals:
```bash
--external-opponents opponents/archive/main_rl_141208.py \
--pool-external-fraction 0.25 \
--heuristic-workers 2        # ← essential for RL-agent opponents
```

**Rule of thumb:**
- Lightweight heuristic (Hellburner, Zach, landgrab): auto workers fine, or `--heuristic-workers 4`
- RL-agent opponent (141208, any exported .py with embedded weights): `--heuristic-workers 2`

**Recovery:** If already hung, `gcloud compute instances reset` (SSH will be unresponsive since all CPUs are saturated).
