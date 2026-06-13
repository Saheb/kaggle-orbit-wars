# Orbit Wars — Training State & History

> 📋 **Ideas & efforts backlog:** [`docs/next-steps.md`](next-steps.md) — prioritized next experiments
> (shaping anneal, diversity/exploiters, Ajay distillation, hyperparameter probes, VDN follow-ups).

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
| **Rev38** | New pairwise features (roi_20, roi_50, enemy_contest, pairwise=15). Zero-padded warmstart from `rev32b_6M_pairwise15.pt`. 256 envs, rollout=128. GCP L4. | Test if 3 new pairwise features + rollout=128 improve Ajay win rate. | 5M: Ajay **2.7%** (7/256), Zach 89.1%. 6M: Ajay **3.1%** (8/256) — best panel. Collapsed to 0% Ajay by 10M. New feature weights stayed near-zero throughout. | New features never activated (zero-padded start) | **950.5** (record) |
| **Rev39** | BC v2 warmstart (`rev32b_6M_ajay_bc5k_v2.pt` — BC-trained on Ajay self-play with pairwise=15, new feature norms 0.17–0.24). BC aux loss: `ajay_bc_1k_v2.pkl` (78,493 samples), bc_coef=0.01→0.02→0.03. First Strike linear decay (mult=2.0, steps=200). 512 envs, rollout=128. Jarvis H100 spot. | New features activated from start (BC gave them real weights). BC aux maintains Ajay signal during PPO. Continuous FS decay instead of cliff. | Peaked **1.6% Ajay @ 8M** (4/256). Deteriorated to ~0.9% @ 12M — passive Nash reasserting. Ran to 15M (timestamp rev39_20260606_055742). New features confirmed active (roi_20=0.193, roi_50=0.215, enemy_contest=0.224 @ 4.7M). Metrics improved (fire[0] 0.17→0.19, srcs_multi 0.82→1.01) but didn't translate to wins. | Passive Nash @ ~8M despite active features | — |
| **Rev40** | Resume Rev39 15M (`rev32b_6M_ajay_bc5k_v2.pt` warmstart lineage). rollout=256, 64 envs. Hypothesis: wider rollout improves credit assignment. | Break passive Nash with longer rollout horizon. | **Carpet-bomb collapse.** srcs_multi 5–7 from start (penalty=0.001 too weak at 64 envs × rollout=256). fire[0] wildly oscillating. Full Ajay panel at 3M: **0/160 wins (0%)** — confirmed useless. Died/killed at 4.4M. | srcs_multi penalty=0.001 too weak for wide rollout config; carpet-bombing from early on | — |
| **Rev41** | penalty=0.01 (10×), threshold=2.0. BC warmstart (ajay_bc5k_v2). 64 envs, rollout=128. GCP L4. | Suppress carpet-bombing with stronger penalty. | srcs_multi suppressed to 0.45 — OVER-penalised. fire[0]=0.14 (passive collapse, not carpet-bomb). Threshold=2.0 penalises legitimate Ajay-style play (Ajay mean=1.9 sources). Killed at 1.3M. | Penalty over-suppressed firing; threshold too low | — |
| **Rev42** | penalty=0.01, threshold=5.0 (loosen threshold). BC warmstart. 64 envs. | Threshold=2.0 was penalising Ajay-style play; raise to 5.0 to allow normal multi-source. | Same passive collapse. srcs_multi=0.6–0.7 (fire=0 satisfies penalty at any threshold). Killed at 1M. | Penalty with floor=0 → fire=0 is Nash minimum regardless of threshold | — |
| **Rev43** | + Ajay external opponent (fraction=0.1), 64 envs, 7 heuristic workers. | Ajay signal during training maintains aggressive prior. | **Hung at iter 12** — all 8 vCPUs saturated by 7 heuristic workers × Ajay's orbit_lite inference. GPU at 0%. CPU deadlock. Killed. | Ajay is CPU-intensive; 7 workers × orbit_lite = all CPUs gone | — |
| **Rev44** | 32 envs, Ajay external 0.25 fraction, 2 workers (fix CPU hang). | Fewer envs + fewer workers to avoid CPU saturation. | Same passive collapse within 50 iters. srcs_multi<0.5. Ajay external doesn't prevent fire=0 Nash when both players benefit from not firing. Killed. | BC warmstart provides no aggression floor — PPO finds fire=0 beats passive self-play | — |
| **Rev45** | Rev38 config (256 envs, rollout=128) + threshold=3.5, penalty=0.02. Jarvis H100, instance 422375. | Calibrate penalty: threshold=3.5 (above Ajay mean+1σ), stronger penalty=0.02. | **fire=0 collapse by iter 84.** srcs_multi=0.07, fire[0]≈0.02. V_loss exploded (critic shock). Both agents converged to zero-fire together — fire=0 is global Nash minimum for any symmetric penalty with floor=0. Instance 422375 destroyed. | Penalty design is fundamentally broken — any penalty with floor=0 makes fire=0 the Nash minimum in symmetric self-play | — |
| **Rev46** | Rev38 EXACT config. penalty=0.001, threshold=2.0 (same as Rev38). 256 envs, rollout=128. Resume rev32b_6M_pairwise15.pt. Jarvis H100, instance 422380. | Rev38 reached 3.1% organically without tuning. Reproduce exactly to confirm baseline. Accept 6M peak; submit that checkpoint. | **3.1% Ajay @ ~6M** (matched Rev38 baseline). srcs_multi collapsed to 5+ after 8M, 0% by 10M. Confirmed Rev38 6M is reproducible. | Nash collapse after 8M | — |
| **Rev47** | fleet_activity_coef=0.002 (binary: fire=any → +0.002) + srcs_multi penalty=0.005/threshold=3.5. Feature warmup on new cols. Resume rev32b_6M_pairwise15_warmed.pt. Jarvis H100. | Binary activity reward breaks fire=0 Nash; penalty caps sources at natural Ajay range. | **Token-fire Nash.** fire[0]=0.11, avgfleet=160. Agent fires 1 source to collect activity reward, hoards everywhere else. 0/16 Ajay at 3M. Nash = fire exactly 1 source (minimum cost, full reward). | Binary reward created new degenerate Nash | — |
| **Rev48** | Proportional activity reward (each source up to threshold adds activity_coef, not binary) + aux_roi_coef=0.02. Jarvis H100, instance 422419. | Proportional reward: Nash = fire up to threshold, not 1. ROI aux loss anchors roi_20/roi_50/enemy_contest features. | srcs_multi stable 2–4 throughout 22M steps (proportional reward worked structurally!). BUT **8M Ajay = 2.3%** (6/256, below Rev46's 3.1%). Root cause: early_capture spike was count-based — prod=1 and prod=5 captures gave identical reward. No gradient to prefer high-ROI targets. ROI features stayed inert. Preempted at 22.8M. | Targeting gradient missing: count-based capture ≠ ROI-based targeting | — |
| **Rev49** | Production-weighted capture spike: ec_rewards = production_delta (not planet_count_delta). early_capture_coef 0.30→0.10 (avg_prod=2.65 keeps magnitude ~0.27). No aux_roi_coef (caused L4 OOM — fixed by conditionally allocating `pairwise_features` storage). Resume rev32b_6M_pairwise15_warmed.pt. GCP L4. Ran full 30M steps. | prod=5 capture gives 5× reward vs prod=1 → gradient pressure for ROI targeting. roi_20/roi_50 now optimise same objective as reward. | **Worse than baseline — 0/16 Ajay at 5M, 6M, 8M (0%, vs Rev38/46's 3.1%).** Production-weighted reward caused carpet-bombing, not selective targeting: rewarding total production_delta means firing at ALL planets simultaneously maximizes reward. By 30M: clip_frac=0.878, KL=16-19, srcs_multi=5-6, fire[0]=0.57-0.63 — agent fires from majority of sources every step. Opposite of intended ROI-selective behavior. | Reward design flaw: production_delta rewards quantity not selectivity | — |
| **Rev51** | Resume rev38 lineage, clean self-play. (Diagnostic run for the target-head bug.) | Establish clean self-play baseline with new metrics. | **KL stuck ~2 regardless of LR** — LR-independent, so not an optimiser problem. Diagnosed as the **target-head pairwise-storage bug**: rollout storage gated `pairwise_features` on `aux_roi_coef>0` (Rev49 OOM fix), so when aux was off the update forward got zeroed pairwise → uniform target head → KL blowup. | Pairwise-storage bug (fixed) | — |
| **Rev52** | Clean self-play, entropy 0.05, target-head bug fixed. | Confirm entropy=0.05 provides collapse-resistance (not "null"). | Ran to **7M with no carpet-bomb / no fire=0 collapse** — entropy 0.05 is the stabiliser, keep it. Baseline for the selectivity experiment. | Baseline established | — |
| **Rev53b** | **Heuristic-pool selectivity** + fixed intercept aimer + early_capture=0 + entropy 0.05. Pool mix 25% current-self / 60% snapshots / 15% heuristics (1166 ladder). Resume rev38 lineage. GCP L4. | Let the agent learn selective targeting from the 1166 heuristic ladder instead of reward-engineering it. | **Cracked the long-standing Ajay ~3% ceiling.** Held-out 1166: 50→**~62%**. Ajay: 3.9→**9.0% @ 9.4M eff**, **10.9% @ 13.6M eff** (best Ajay panel ever, vs prior 3.1% record). Wins are decisive eliminations, not timeout luck. clip stable 0.23–0.28, KL <0.025, EV ~0.9. | Still running (this session) | **TBD** |
| **Rev54 (v1)** | Resume **rev38 5M** + 3 heuristic-hammer externals (lb1084/1138/1152) @0.25 + **early_capture training-wide cosine anneal** (0.3→0, frac 0.67) + metric cleanup. LR 1e-4, entropy 0.05. | Diversity from the heuristic ladder + remove the shaping liability *over training* (dense→sparse anneal: shaping bootstraps competence but creates a degenerate Nash if kept). | Ran to 1M, then **mistakenly killed on a false collapse alarm** (Vμ −0.02 + avgfleet + srcs_multi — all non-signals, see `docs/metrics.md`). Held-out eval proved the 1M **HEALTHY: 5.5% Ajay** (2× rev38's 2.7%) + **4.7% vs debatreya_1300 planner** — **best Ajay checkpoint to date.** Submitted 1M to LB (sub 53527873, 2026-06-10). | False-positive collapse alarm (non-signals) | **1M submitted (pending)** |
| **Rev54 v2** | Resume the **Rev54 1M** + DROP the 3 heuristic hammers, ADD **debatreya_1300 @0.15 only** (1300-LB forward-sim planner, Producer/Ajay archetype) + entropy 0.05 + early_capture anneal kept. LR 1e-4. GCP L4, ran 15M. | The 1300 planner is the diversity archetype we lack and can't be out-sprayed → real pressure *without* the over-fire reward loop that beatable aggressive bots create. Target LB > 993.9. | **clip_frac sat ~0.3+ for most of the run** (entropy 0.05 + LR 1e-4). **Late checkpoints regressed to ~0% Ajay / 0.4% 1300**; early <0.25-clip checkpoints stayed healthy. **Lesson: pair higher entropy with lower LR** (entropy 0.05 pushes clip ~0.30). | clip>0.3 → late-run policy degradation | — |
| **Rev55** | Resume **Rev54 1M** + `--allow-reinforce` (own planets become legal targets — **the reinforcement lever**). Keep v2 diversity config (1300 @0.15, early_capture anneal) but **LR 2.5e-5** and **entropy 0.05→0.03**. GCP L4. | Top-player EDA: Vadasz **reinforces ~57% of launches** (sends to own planets); our agents do **0%** (own planets were masked out of targets). Structural lever to break the ceiling — physics already supports it. | **Over-fire collapse in <400K steps** — hard t=0 unmask is too violent. `Vμ` +1.17→−0.07, `Rμ` +1.07→−0.28, `fire_frac` 0.40→0.77, `avgfleet` 52→138, `p90` 104→346, clip 0.04→0.29. Cause: the target head's `is_mine`-input weight was never trained (own-targets masked all of rev54's lineage), so at t=0 own planets are scored on generic attractiveness → policy sprays into them (safe-fire). Killed; instance deleted. | Hard-unmask shock; cutting entropy 0.05→0.03 removed collapse-resistance | — |
| **Rev57** | Resume **rev38 5M** (the 967.6 LB-record base, NOT rev54 1M) + `--allow-reinforce` + curriculum (bias −8→0 frac 0.3) + **reinforcement discipline: `--reinforce-garrison-floor 10` (#1 mask) + `--reinforce-cost 0.001` (#2 per-ship transit cost)**. Clean self-play (no external). entropy 0.05, LR 5e-5. GCP L4. | rev54 1M scored **848.6** LB (119pts UNDER the rev38 record) despite 5.5% Ajay → the whole reinforce lineage was built on an underwater base; Ajay panel 4×-confirmed not LB-predictive. Test disciplined reinforcement on our actual best. #1 (training-time veto, no Nash risk) kills the "drain-a-planet-then-lose-it" regression; #2 (per-ship cost) prices the rev56 flood — a 50-ship stage costs 0.05, a 2000-ship flood costs 2.0. Watch reinforce_rate → 0.4-0.6 (Vadasz 0.57). | Clean start (iter 1 EV 0.797/clip 0.081). **clip crept 0.25→0.285 by 1.97M** (rev54-v2 creep: entropy 0.05 + LR 5e-5). → intervention rev57b. | superseded by rev57b |
| **Rev57b** | **Clip intervention**: resume **rev57's 1.5M checkpoint** at **HALVED LR 2.5e-5** + added the **`reinforce_rate` training metric** (diag `reinf`, CKPT_METRICS, W&B — measured over the current policy's launches, train_mask-filtered). Same discipline/curriculum. GCP L4. | clip>0.25 is the standing-authority intervention line; halving LR is the documented fix for the entropy-0.05 creep. Curriculum re-runs from −8 on the more-trained 1.5M weights (gentler re-intro). | Launched 2026-06-10. iter 1: LR 2.5e-5, clip 0.085, EV 0.783, Vμ +0.44, `reinf 0.00` (bias re-suppressed, will climb). | TBD |
| **Rev56** | Resume **Rev54 1M** + `--allow-reinforce` + **curriculum** `--reinforce-bias-init -8 --reinforce-anneal-frac 0.3` (own-target logit bias −8→0 over 4.5M; enemy/neutral untouched) + entropy **0.05** + LR **5e-5**. GCP L4. | The curriculum stops the rev55 t=0 shock by phasing reinforcement in gradually; RL learns the reinforce value from reward as own-targets surface. | **Curriculum worked (no t=0 collapse) but the flood reasserts.** Vμ dipped then recovered to **+0.53 @1.25M**, then **declined to +0.15 @2M** as **p90 ballooned 107→368** (past rev55) and clip crept to 0.287. Eval @2M (bias=0): **reinforce_rate 0.73–0.80** (vs Vadasz 0.57; lever WORKS) but **flooding** (~30× launch volume, own→own) → Ajay WR 2.08%/material 270 (below rev54 baseline 4.17%/760, above BC-seed 0/0). **Root cause: reinforcement is near-costless** (friendly arrival can't lose ships → material conserved), so with any fire incentive the policy floods. Curriculum necessary but **not sufficient** — it times availability, adds no cost. Killed at 2M; instance deleted. | Costless-reinforce flood (no opportunity cost / garrison guard) | — |
| **Rev58** | Resume **rev38 5M** + empire-gate(3) + `defense_coef 0.03` + fire-entropy 0.005, **`reinforce_cost 0`** (the phase2 locked "masks + defense, no cost" design). GCP L4. | Gate + defense + low fire-entropy should self-cap reinforcement without a cost tax. | Started healthy (reinf 0.21 @32k, Vμ +0.63) then **drifted into a flood** by ~400k (reinf 0.75, p90 408, Vμ→0). Drift-from-healthy ⇒ a reward-**objective** attractor (not a mature-base artifact). | `defense_coef` rewards holding → hold-via-reinforce dominates attack in a mirror | — |
| **Rev58b** | Rev58 + **`reinforce_cost 0.001`** (only delta — the §3 back-pocket lever). GCP L4. | The small cost paired with gate+low-entropy should cap the flood. | **Flooded earlier**, ~330k (reinf 0.69, p90 357, avgfleet 175, fire_frac 0.83, Vμ −1.25), clip 0.27. **Cost knob dead** (both 0 and 0.001 flood). Killed @524k; instance deleted. → pivot to Tier-1 (forward-staging mask + drop `defense_coef` + aggressive pool); see `docs/phase2.md`. | Cost did not bind; `defense_coef` is the flood pump | — |

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
- **srcs_multi penalty design flaw**: Any penalty with floor=0 makes fire=0 the self-play Nash minimum — both agents converge to zero-fire together. Confirmed across 5 runs (Rev41–45) with thresholds 2.0, 5.0, 3.5 and penalties 0.01, 0.02. Do NOT use srcs_multi penalty to suppress carpet-bombing. The penalty (0.001) in Rev38/Rev46 is too small to cause this — it's essentially decorative.
- **BC warmstart provides no aggression floor**: BC trained on Ajay behavior (mean 1.9 sources). PPO immediately finds fire=0 beats passive self-play. Collapsed within 30–84 iters across Rev42–44 regardless of threshold/penalty tuning. BC warmstart cannot hold the line.
- **Ajay CPU bottleneck**: orbit_lite makes Ajay inference CPU-intensive. With ≥7 heuristic workers, all vCPUs saturate and GPU drops to 0%. Always `--heuristic-workers 2` when using Ajay as external opponent. Fix in Rev44, but still couldn't prevent passive Nash.
- **Rev46 = Rev38 exact**: Confirmed 3.1% Ajay at ~6M. Carpet-bombing at 8M+ is the Nash — don't fight it with penalties.
- **Targeting root cause (Rev48 diagnosis)**: early_capture_coef spike was count-based: `planet_delta = (owned - prev_owned).clamp(-1, 1)`. prod=1 and prod=5 captures gave identical reward. ROI features (roi_20/roi_50) stayed inert because PPO never needed them. Fix: `ec_rewards = production - prev_owned_prod` (production delta instead of count delta). Scaling: 0.30→0.10 coef since avg_prod=2.65 keeps spike magnitude ~0.27.
- **pairwise_features OOM on L4 22GB**: `pairwise_features` tensor in rollout storage (shape: rollout_T × N × P × MAX_OWNED × max_planets × pairwise_feature_dim) was unconditionally allocated even when aux_roi_coef=0. Caused OOM during backward pass (18GB baseline + 1.5GB allocation = OOM). Fix: conditional allocation — only allocate when `args.aux_roi_coef > 0` (train_torch.py line 604).
- Zach panel saturating ~88–89%. Ajay full panel is primary signal.
- Eval on training instances: always `CUDA_VISIBLE_DEVICES=""` to avoid GPU OOM.
- launch_gpu_gcp.sh: now verifies rsync landed + clears .pyc cache after sync.
- **GCP instance naming**: use `--name orbit-wars-training` (default). After training ends, delete immediately — instance left running costs ~$1.13/hr even with no jobs. Rev38 instance ran idle 19+ hrs before discovered.
- Export requires `--target-decode` for Phase 1.

**LB scores:** Rev38 5M + fixed aimer = **967.6** ← record (drifted from 978.3) | Rev38 5M = 950.5 | Rev53b 13.6M (Ajay 10.9%) = **953.2** | Rev38 6M = submitted (Ajay 3.1%) | Rev32b 6M = 872.4 | Rev31 10M = 918.8 | Rev30 11M = 866.3 | Rev28 27M = 843.9 | Rev54 1M (Ajay 5.5%) = **848.6** (53527873) — 119pts UNDER rev38 record despite 2× Ajay panel; 4th confirmation Ajay panel ≠ LB-predictive
**LB record stays rev38 5M + fixed aimer = 967.6.** Target: Top 10 needs ~1153. Gap = ~186 points.

---

## Current State (2026-06-13)

### ⭐ PHASE 3 STARTED — Stage A (teacher-KL anti-cycling anchor); closed-loop fidelity is SPLIT
**Comet FEATURES done** (path-aware, parity probe CLEAN — see the comet section below). **Closed-loop fidelity check
(frozen p2rev5 4M vs the pins, threshold-decode):** the new `frozen_vs_pins_torch.py` duel (torch_env+comets) vs the
full kaggle 256-panel — **rev38 torch 44.2% ≈ kaggle 44.5% ✅ (faithful)** but **rev53b torch 10.4% vs kaggle 32.8%
❌ (3× gap)**. So the comet PHYSICS fix is solid (rev38 confirms) but a **real opponent-specific obs/closed-loop gap
remains for rev53b-style play** (beyond the min_owned_dist cap bug). The old 48g seat-0 anchors (27/37.5%) were noise.
**Implication for Phase 3 pools: pin rev38 (faithful, beatable); treat rev53b with suspicion as a pin** (torch makes
it look ~unbeatable at 10% when it's really 32.8% → win-starvation trap). Localize the rev53b gap later via
`frozen_vs_pins_torch.py` + `sim_gap_probe.py`.
**Phase 3 = ratcheted teacher-KL + league (`docs/phase3.md`)** — the structural anti-cycling fix (PPO clip is a
*relative* trust region blind to slow drift; teacher-KL is the missing *absolute* anchor; ratchet = re-anchor to
rising held-out-best → climb without ceiling). Machinery exists (`ppo.py` `frozen_il_model`/`_il_kl_penalty`,
`--il-ref`/`--il-lambda`); only the ratchet is a new build. **Stage A LAUNCHED 2026-06-13 (GCP L4, `p3stageA`):**
resume p2rev5 4M + ONE delta = a CONSTANT self-anchored teacher-KL (`--il-lambda 0.05 --il-ref <4M> --il-decay-frac
100`), else identical to p2rev5 (deb pool, gate3/floor0, early_capture off). Held-out = kaggle Ajay (kaggle-native →
faithful selection signal, unaffected by the rev53b torch gap). **WATCH:** held-out WR HOLDS past the ~4-5M band
instead of peak-then-fall (un-anchored p2rev5: peak 5.9%@4M → ~4.5% band). Canaries: clip_frac must NOT→0 (frozen),
il_kl moderate, entropy stable; tune il-lambda by these. Origin: [[reference_toad_brigade_rl_recipe]].

### ⭐⭐ THE BIG ONE — the train/eval sim gap = MISSING COMETS in torch_env (FOUND + FIXED 2026-06-13)

This is the root blocker behind the whole "panel not LB-predictive / pool wr is fiction / planets@50=6 / hoarding"
cluster. **torch_env never simulated comets** (`# comets not implemented in Phase 3a`); the real kaggle env spawns
4-comet groups at steps [50,150,250,350,450] — **collidable, capturable (carry ships+production), moving**. So
every torch_env game from step 50 on was a DIFFERENT, simpler game than kaggle — exactly the mid/late window where
we hoard-to-win and where planets@50→100 collapses for real. We were overfitting a comet-free world.
- **Found** via a new trajectory-diff harness (`orbit_wars_rl/sim_gap_probe.py`): drive both engines from one seed
  (shared generate_planets+RNG → identical board), replay a real game's EXACT actions in torch_env, diff every
  step. Byte-identical ~50–77 steps, then one fleet diverged — it hit a comet kaggle had and torch didn't. Proven
  independently in kaggle's own engine (fleets fly INTO comets and capture them). This also EXONERATED orbital
  motion, fleet movement, spawn, combat, win-resolution (all byte-match given identical actions).
- **Fixed:** comets implemented in `torch_env.py` (reuse kaggle generate_comet_paths; slots 44–47 so the policy
  observes them; lazy per-spawn compute; **vectorized 7.5× byte-identical** via `_comet_paths_fast`). Now
  byte-faithful for whole games incl comets. Tests: `tests/test_comet_fidelity.py`. Full write-up: `docs/train-eval.md`.
- **Board distribution ruled out (2026-06-13):** self-play boards = real LB boards (same generate_planets,
  symmetric — no SSDR/handicap in p2rev9); the eval PANEL's board features (count/production/orbiting/big-planet)
  also match the natural distribution. **WR A/B confirmed it** (`board_dist_ab.py`, p2rev5 4M vs Ajay, both seats,
  128 seeds each → 256 games each, run in the REAL kaggle env): PANEL **5.9%** (15/256) vs RANDOM **5.9%** (15/256)
  — IDENTICAL at convergence (the seat split just mirror-flips: panel s0 4.7/s1 7.0, random s0 7.0/s1 4.7). Random
  peaked ~9.8% mid-run (@112g) then regressed to the same mean — the earlier "RANDOM ~8%" note read it before it
  converged. NOT the 10×+ a board artifact would produce. So boards are NOT the gap — physics (comets) was. The low
  WR vs strong opponents is genuine skill, not a board/seat artifact. (Log: `/tmp/board_ab.log`.)

**Active run:** NONE. **p2rev9 (comet-faithful relaunch) KILLED @1.28M 2026-06-13** after confirming the fix works
live — GCP box `orbit-wars-p2rev9` DELETED (no billing). It was a validation run (resume p2rev5 4M + the POOL pivot
on the FIXED engine): iter 1 ran past the step-50 comet spawn cleanly, SPS ~485, comets active in the full loop,
and the **in-training pin WR moved DOWN with comets** (rev38 ema 0.60→0.53, rev53b 0.70→0.60) — the fix
recalibrating training difficulty toward kaggle reality (not the full drop to cross-eval 27–37% yet; resumed policy
mid-adapt + sampled-vs-decode). Harvested 524k+1M ckpts → `gpu_run_artifacts/p2rev9/`; old comet-free artifacts in
`p2rev9_cometfree_old/`. (p2rev7+p2rev8 also KILLED 2026-06-13.) **The fix itself is committed (branch
`comet-sim-fidelity-fix`), tested, and documented — the next run is the FEATURE phase below, ideally from-scratch.**

**NEXT (deliberate phase):** comet FEATURES — ✅ **DONE 2026-06-13** (path-aware, train/eval/export parity).
Step 1 (static parity probe, `orbit_wars_rl/feature_parity_comet_probe.py`) found the obs gap was BIGGER than
`is_comet`: comet slots also broke the orbital-prediction features (planet feat 9/10/11 + pairwise
sin/cos/dist/eta/roi) because comets carry stale `init_planets=(0,0)` but `is_orbiting=True` → `get_features`
predicted a circular orbit from (0,0) while `extract_features` fell back to the comet's CURRENT position (true on the
REAL kaggle obs too — not a reconstruction artifact). Step 2 (the fix): for comet slots OVERLOAD the orbital
channels (NO model-dim change → existing ckpts still load, frozen-vs-pins stays runnable) — feat 7 `is_comet`=1,
feat 10/11 = comet PATH position +5 steps (`paths[path_index+5]`), feat 9 = normalized steps-to-departure
(`(len(path)-path_index)/40`); pairwise treats comet targets as non-orbiting; `to_legacy_obs` now surfaces
`comets`/`comet_planet_ids` (+ fixed a pre-existing id-0 collision: comet slots polluted `initial_planets` with id 0,
corrupting real planet 0's orbit feats). Identical comet branch in `torch_env.get_features` +
`features.extract_features`/`compute_pairwise_features` (kaggle obs exposes `observation.comets` = paths+path_index,
so eval/export parity is automatic via features.py). Parity probe now CLEAN on all comet feats; comet fidelity
(physics) tests still pass. Pre-existing comet-independent `min_owned_dist` BOARD_SIZE-cap parity bug flagged
separately (task_ab4d5b3a). **THEN:** step 3 = frozen-vs-pins WR duel (now runnable), then a from-scratch run that
learns comets from step 0. The pool-pivot logic is now well-founded on a faithful sim.

**Why the pivot (the win-starvation finding):** 4 reward/mask levers all left `planets@50` pinned at 6 — the
opening ceiling is STRUCTURAL because we can't learn to beat a peeler we never beat (deb ~5%, all-loss = no
win-contrast; rare sim-wins are easy-board fiction). **Cross-eval 2026-06-13 confirmed the pins are the fix:
p2rev5 4M beats rev38 27% / rev53b 37.5%** (vs deb ~5%) — matched difficulty, a REAL transferable win-gradient.
rev38 (aggressor) punishes under-expansion at a beatable difficulty → the terminal win-contrast early_capture
lacked. See [[feedback_win_starvation]]. **WATCH:** `planets@50` climbs toward 9 + WR climbs from the 27-37%
base. If the pool moves it, early_capture/defense return as clean *win-anchored* follow-ups.

**⭐ WHY BOTH KILLED — the win-starvation finding (2026-06-13, user insight, the session's real conclusion):**
We can't learn to beat a peeler we almost never beat. In-training sim WR vs deb is ~0.25 BUT that's
sim-inflated (held-out Ajay ~5%, p2rev7 *declining* 5.9→3.1) AND the sim-wins are the wrong games (archetype
breakdown: wins on `low_prod__static` snowball boards, **0% on contested high_prod/mixed** where holding-under-peel
matters). So 75–95% of deb games are terminal −1 with no learnable win-contrast, and the rare wins teach
"snowball the easy board," not the skill. Consequences, now proven:
- **p2rev8 (early_capture 0.2): FAILED.** `planets@50` DEAD FLAT at **6** across all 9 held-out ckpts (524k→4.7M;
  winner 9), `open<50 cap/atk` flat ~0.37 (winner 0.51), WR flat ~4.5%. early_capture's within-game expansion
  reward has no terminal win that *requires* 9 planets to anchor it → no breakthrough. Same non-response as the
  p2rev6 mask.
- **p2rev7 (defense_coef 0.02): hold metric moved, but it's a self-play MIRROR artifact, not transferable.**
  `peel WON` declined 0.64→0.61→0.57→**0.52** (toward winner 0.43) — first lineage result to move it, flood
  guard GREEN (reinf ramp winner-like, garr_frac BELOW Isaiah) — BUT **WR DECLINED 5.9→3.1** and `planets@50`
  slipped 6→5 while `mid cap/atk` rose to 0.71. defense_coef + perpetual-deb-loss = the policy gets good at
  *converting/holding a SMALL empire* but expands less (the conservatism the design warned of). The peel gain is
  self-play copies symmetrically out-holding *each other*, which is exactly why it does NOT lift WR vs deb.
- **`planets@50 = 6` is now invariant across FOUR levers** (deb-pool p2rev5, sufficient-commit mask p2rev6,
  defense_coef p2rev7, early_capture p2rev8) → the opening-expansion ceiling is **STRUCTURAL** (a property of the
  pool/board/self-play setup), not reward-tunable. **Stop shaping past it with reward knobs; fix the SIGNAL.**

**THE PIVOT (next run): pool-seed-RL** — pin strong-but-BEATABLE, sim-immune RL selves (rev38 aggressor +
rev53b) so winning is achievable at matched difficulty AND demands the right play; keep ~30-40% self-play
(auto-curriculum); **demote deb to a small peel-flavor fraction or drop it** (cranking the unbeatable opponent
starves the win-gradient). Built+tested (`opponent_pool.py add_pinned_rl`). See `docs/next-steps.md` pool levers.

**Phase-2 lineage p2rev3 → p2rev9** (resume-chained; the gap = beating PLANNER-class peelers deb/ajay,
mechanism = capture-then-lose / mid-game hold; selection PURE on held-out, never self-play wr):
- **p2rev3** — the deb-era resume base. **0.5M = best-vs-deb (3.9%)** before the self-play Nash erased holding;
  the documented resume base for p2rev4/p2rev5. (config detail: `docs/phase2.md`.)
- **p2rev4** (GCP L4, resume p2rev3 4M, **garrison_floor 10→0**): unblock reinforcement (veto probe: floor=10
  blocked 62% of wanted reinforces). **Verdict: quantity ≠ the fix** — floor0 raised reinforce (mid 0.11→0.19)
  but peel did NOT drop (0.61→0.64), churn WORSE (16→23) ⇒ we reinforce *incorrectly* (wrong target/timing),
  not too little. Box deleted (synced through 1.5M). **floor=0 KEPT as the validated default.**
- **p2rev5** (Jarvis A100, resume p2rev3 0.5M, **deb peeler external @0.25**, floor0/no-forward): learn
  holding-under-peeling natively. Ran ~9.44M (box destroyed; held-out Ajay panels backfilled through 9.44M).
  **Verdict: deb-in-pool moved reinforce-SHAPE metrics (direction fwd 58%, less hoard) but NOT the two OUTCOME
  levers** — `open<50 cap/atk WON` FLAT ~0.38 (winner 0.51) and peel ~0.6 (winner 0.43). **Read held-out PANEL
  logs, not self-play diag** (the rising-H_tgt / reinf>100 "threat head" signal was a mirror artifact → threat
  head PARKED). **4M = the held-out WR peak (5.9% Ajay)** → the resume base for BOTH p2rev6 + p2rev7.
- **p2rev6** (Jarvis A100, resume p2rev5 4M, **`--sufficient-commit-factor 1.0`**): veto fragment launches
  (ships ≤ target defense) → force opening concentration. **CONCLUDED at 7.8M, box destroyed 2026-06-12.
  Verdict: FAILED — `open<50 cap/atk WON` FLAT ~0.34 over all 9 held-out ckpts** (≤ the p2rev5 base ~0.40),
  `planets@50` stuck at 6 (winner 9). A veto removes the bad behavior (fragments) but doesn't supply the good
  one (concentration) — agent just fired *less* ([[feedback_veto_mask_removes_not_teaches]]). None beat the
  p2rev5 4M base → that stays the resume point. clip resolved benign (peaked 0.27, receded). 14 ckpts harvested.
- **p2rev7** (GCP L4, resume p2rev5 4M, **`--defense-coef 0.02`**, sufficient-commit OFF): **KILLED @4M 2026-06-13.**
  Verdict above (win-starvation section): `peel WON` moved 0.64→0.52 but it's a self-play mirror artifact —
  WR DECLINED 5.9→3.1, `planets@50` 6→5 (conservatism: holds a small empire better, expands less). flood guard
  was green. Harvested 7 ckpts → 3.67M.
- **p2rev8** (Jarvis A100-80GB spot, resume **p2rev7 1M** + **`--early-capture-coef 0.2`** always-on, defense_coef
  0.02 carried fwd): the QUANTITY/opening lever (clean A/B off p2rev7 1M — p2rev7 = hold only, p2rev8 = + opening).
  **KILLED @5.77M 2026-06-13. Verdict: FAILED** — `planets@50` DEAD FLAT at 6 over all 9 ckpts, opening conversion
  flat ~0.37 (winner 0.51), WR flat ~4.5%. early_capture 0.2 did not break the opening ceiling (no terminal win
  needs 9 planets to anchor it — the win-starvation finding). Harvested 11 ckpts → 5.77M. (ran ~800 SPS, 28-core
  A100; fixed a watcher symlink-sync bug mid-run — [[feedback_watcher_symlink_ckpt_sync]].)
- **p2rev9** (GCP L4, resume p2rev5 4M, **POOL PIVOT**): the first NON-reward delta — pin rev38+rev53b (winnable
  RL champions, cross-eval 27/37% vs deb ~5%) + deb 0.25→0.10 + 35% self; reward reverted to p2rev5 clean
  baseline (early_capture/defense OFF). **LIVE** (full status in Current State above + `docs/next-steps.md`). 500k
  panel = baseline (planets@50 6, WR 4.7% — too early); watch the 1.5M–2.5M trend for planets@50 climbing.

**Decision rules confirmed this session:** (1) judge a delta by whether its OWN target metric trends toward
goal across MANY held-out checkpoints — flat over many ckpts = it isn't working → end + harvest (don't burn GPU
to 10M). (2) **improve-then-degrade** (held-out peaks ~500k–1M then drifts) = self-play Nash reforming around a
misaligned objective; durable gains need changing the ATTRACTOR (always-on structural masks + asymmetric pool
pressure that makes the bad behavior LOSE), not shaping the gradient — a transient gain is itself the signal
the delta is a nudge, not a structural fix. **Queued next levers** (`docs/next-steps.md`): pool-seed-RL + deb
(anti-drift), K-nearest reinforce-target mask, relaxed sufficient-commit 0.6.

---

## Prior State (2026-06-11)

**Active run:** none — both Phase-2 runs COMPLETE; Jarvis H200 spot 425211 **destroyed** (billing stopped).
All checkpoints synced + the 10.03M final verified-loadable locally (`gpu_run_artifacts/p2rev2/checkpoints/`).
⚠️ `--terminate-on-done` only OS-poweroffs a Jarvis spot — it keeps **billing** until `jl destroy --yes`
(now warned in JARVIS_RUNBOOK).

**Phase 2 progress: p2rev1 → 10M done · p2rev2 → 10.03M done.**
- **p2rev1 (from-scratch, 10M):** snowball-BC warmstart + forward-staging mask + drop `defense_coef` +
  3 fast heuristic hammers pool (lb1152/lb1138/lb1084 @0.25). Recovered general strength (strong vs Zach)
  but weak vs strong planners (~0.8% debatreya @9.8M). Diagnosed a concrete **opening over-fire** (re-firing
  at the same near-neutral until the in-flight fleet lands).
- **p2rev2 (resume p2rev1@10.3M + ONE delta: friendly-coverage roi-deflation, 10.03M):** deflate `roi_20/roi_50`
  by own ships already inbound so a planet we're already capturing stops reading attractive. Run healthy
  throughout (EV 0.57→0.91, V_loss→0.08, no degeneracy, reinf ~0.54 no flood; iter-clip peaked 0.242, never
  crossed 0.25). **⭐ CHAMPION = 8.91M (`torch_step_8912896`), NOT the 10.03M `_final` — export/submit 8.91M.**
  Deb panel WR PEAKED at 8.9M (3.1%, 8/256) then the last ~1M REGRESSED to 0.4%: eval `garr_frac@50` rebounded
  0.54→0.63 (re-hoard) as the **self-play hoarding Nash reformed** (same transient shape as SSDR). WR tracks
  `garr_frac@50` inversely run-wide (deploy→win; garr@50 = leading indicator). The regression = the policy
  CONVERGED into the hoarding Nash (NOT LR — it barely moved under the 200M schedule; KL stayed low). The
  per-ckpt clip fell 0.27→0.155 because the policy settled — into hoarding. Deploy was a transient excursion.
  **Lesson: deploy is not a stable attractor under 75%-self/25%-weak-heuristic — → `--pool-seed-rl` for p2rev3.**

**Key findings this session (diagnosis sharpened — full detail in `project_phase2_reinforcement` + docs/phase2.md):**
- **`planets@N` is THE win/loss discriminator.** vs Zach (win) it climbs 2/4/7/10; vs debatreya (loss) it
  peaks then collapses 2/4/6/3. **Opening is identical win OR lose (2/4) — the gap is the MID-GAME (steps
  50→100) hold, NOT the opening** (corrects the earlier "we lose the opening" framing).
- **debatreya WR is climbing with training: 1.2% (4.19M, 3 wins) → 3.1% (8.9M, 8 wins),** evenly across
  seats. The 3 wins @4.19M are **real contested comebacks** (Deb led 53–58% material, we held and won —
  not lucky snowballs). Mechanism: **deploy-not-hoard** — `garr_frac@50` fell from 0.63–0.66 → **0.54
  (Isaiah-level)** by 8.9M; the mid-game over-garrison we pinned as the collapse cause is normalizing.
- **roi-deflation verdict = INCONCLUSIVE** (not "not working" — a controlled A/B used a `redundant`
  metric broader than the fix's actual condition; being re-aligned to `cap_cost_at_arrival`). Opening
  over-fire was modest to begin with (p2rev1 native ~0.17 vs ~0.12 elite floor).
- **Next lever (p2rev3) = `--pool-seed-rl`** (pin rev38 aggressor + rev53b as sim-gap-immune RL selves) to
  punish the mid-game over-garrison and force holding — better-aimed at the `planets@N` collapse than more deflation.
- **New eval tooling:** `eval.py game_conversion` now emits `churn` (length-confounded — read `churn/100st`)
  + `redundant-launch<50` (opening-windowed); `conversion_from_replays.py` (top-2 baseline, eval==replay);
  `analyze_deb_wins_p2rev2.py` (win profiler + banner replays); `ORBIT_NO_FRIENDLY_DEFLATION` eval toggle.
  Always full 256-game `--panel` for WR. Defs/confounds in docs/metrics.md.

---

## Prior State (2026-06-10)

**Active run:** none. The reinforce-lever resume chain ran out: rev57/57b → **rev58 → rev58b, both
flooded** (~330–400k; see run table + `docs/phase2.md` top Update). Cost knob dead; root cause
re-diagnosed as `defense_coef` (the flood pump in a symmetric mirror). **Phase 2 pivoted to the
Tier-1 outcome-tied design** — forward-staging mask (built + unit-tested) + drop `defense_coef` +
small aggressive pool (rev53b-proven) + from-scratch. **p2rev1 ready** (fresh Phase-2 numbering): snowball-BC
warmstart + forward mask + drop defense + lb1152/debatreya pool @0.25; target-head diagnostics added; `gpu_run_artifacts/p2rev1/`
run script. GCP instance DELETED. Memory: `project_reinforcement_lever`, `project_phase2_reinforcement`.

**What landed this session:**
- **Rev54 1M is our best Ajay checkpoint to date (5.5% Ajay / 4.7% debatreya_1300)** — exported (`--target-decode` + fixed aimer) and **submitted to LB** (sub 53527873, pending). Came from Rev54 v1 (rev38 5M + 3 heuristic externals @0.25 + early_capture training-wide anneal); the run was *mistakenly* killed at 1M on a false collapse alarm (Vμ/avgfleet/srcs_multi — all non-signals), but held-out eval proved the 1M healthy.
- **Rev54 v2 negative result:** resuming the 1M with the 1300 planner @0.15 at LR 1e-4 + entropy 0.05 let **clip_frac sit ~0.3+** for most of 15M; late checkpoints regressed to ~0% Ajay. **Standing lesson: entropy 0.05 pushes clip toward 0.30 — pair it with a lower LR.** Rev55 applies this (LR 2.5e-5, entropy 0.03).

**LB reality check (scores drift as games play out):** the rev38 5M + aimer record now reads **967.6** (was logged 978.3/993.9) and rev53b reads **953.2** (was 933.0). Judge Rev54 1M against these current numbers when it scores.

### Reinforcement lever — rev55 collapse + BC-seed dead end → curriculum (2026-06-10)

Pursuing reinforcement (`--allow-reinforce`: own planets legal targets; top players reinforce ~57%, we do 0%). Two approaches failed before landing on the curriculum:

1. **Hard unmask (rev55)** — over-fire collapse in <400K steps (see run table). The target head's `is_mine`-input weight was never trained (own-targets masked all of rev54's lineage), so unmasking at t=0 lets the policy spray into own planets.
2. **BC-seed the target head — looked great, cratered play.** `ajay_bc_1k_v2.pkl` is pairwise-15 with **55,448 own-target labels** (directly usable). Target-head-only BC on rev54 1M set own-target top1 **6.3%→53.8%** and passed every BC/label gate — but held-out WR (reinforce off) **collapsed: Ajay 4.17%→0%, avg material 760→0** (eliminated every game). The shared `target_scorer` got dragged toward Ajay's enemy/neutral targeting (label-agreement *rose*), which **destroyed rev54's aggressive winning play.** ⇒ **We win by targeting differently from Ajay; copying its targets craters us. Label-accuracy is a misleading proxy** (Zach/srcs_multi trap again). The held-out WR gate caught it pre-GPU.
3. **Curriculum (chosen)** — annealed negative bias on own-target logits *only* (`bias·is_mine`, → 0 over `--reinforce-anneal-frac`). Leaves rev54's trained enemy/neutral target head **untouched** (the thing BC broke); RL learns the reinforce value from reward as own-targets phase in. No t=0 shock, no broken attack play. Resume rev54 1M, entropy 0.05, LR 5e-5. Memory: `project_reinforcement_lever`.

---

## Current State (2026-06-08)

**Active run:** Rev53b on GCP L4 (`orbit-wars-training`, us-central1-b), at ~11M steps (~13.6M eff). Local watcher syncing checkpoints+logs every 60s; held-out evaluator auto-runs 1166 (every 1M) + Ajay frontier panels on each checkpoint as they land.

This session was a metric/evaluation overhaul plus two free bugfixes, and it **cracked the long-standing Ajay ~3% ceiling (now 10.9%)**. Highlights:

### Two free bugfixes (no experiment needed)
1. **Intercept aimer fix** — `_target_intercept_angle` in both `action_mask.py` and `torch_env.py` was a crude approximation (73% hit rate on slawekbiel's aiming benchmark). Rewrote to a proper lead/intercept fixed-point solve (surface-gap subtraction, continuous ETA, 8 iters, current-position orbit): **94.9%** on the benchmark, **+14.5pp vs the 1166 heuristic**. The two implementations verified identical (4.6e-7 rad). Re-exporting Rev38 5M with the fixed aimer alone moved LB **950.5 → 978.3** (sub 53451535).
2. **Target-head pairwise-storage bug** — the Rev49 OOM fix gated `pairwise_features` rollout storage on `aux_roi_coef>0`. With aux off, the update-forward saw zeroed pairwise → uniform target head → **KL stuck ~2, LR-independent** (the rev51 symptom). Fixed by gating storage on `pairwise_feature_dim>0`. This unblocked all subsequent training.

### Evaluation philosophy changes
- **Zach is retired as a decision metric** (saturated ~88–89%). Primary held-out signal is now the **1166 heuristic ladder** (`opponents/orbit-wars-heuristic-bots/08_v13_3_R8_full_stack_lb1166_PEAK_HEURISTIC.py`) + the **Ajay** full panel. Both run as automatic daemons on each checkpoint (sync 60s + panel ~40min Ajay → they appear with lag, not manual triggers).
- **Ajay-in-training is unusable** as a pool opponent: orbit_lite inference saturates all vCPUs (rev43/44 CPU deadlock). Use the heuristic ladder in the pool instead and keep Ajay strictly for held-out eval.
- **srcs_multi is misleading** — it's empire-size-confounded (counts sources firing when ≥2 owned) and outlier-game dominated, so optimising it (rev5–rev48) never moved real wins. Replaced with **fire_fraction** (sources fired / owned, on firing steps), **owned_planets**, and **fire_rate**, computed in `ppo.py` and emitted at checkpoint time via the `CKPT_METRICS` line in `train_torch.py` (so metrics align exactly with the checkpoint, not the diag cadence). *Note: the current rev53b binary predates this code, so its fire_fr/owned columns are blank in `track.py`; they populate from the next launch.*
- **entropy = 0.05 is the stabiliser, not "null."** It provides collapse-resistance — rev52 ran to 7M with no carpet-bomb and no fire=0 collapse. Keep it; do not zero it.

### Win-mechanism finding (how/why we win, not just the aggregate)
Profiling 32 both-seats games (rotating vs static boards) and full-game phase analysis:
- **WINS = aggression → decisive elimination of Ajay** (~150–280 steps, fire 23–37%). Not timeout luck.
- **LOSSES = passivity** (fire ~18%) → the agent gets eliminated, or drifts passive after going ahead.
- **Score oscillation across checkpoints = passivity drift on rotating (hard) boards** specifically — static boards stay won.
The takeaway: the lever is *sustained aggression / not going passive once ahead*, and the honest signals for that are fire_rate + fire_fraction, not srcs_multi.

### Rev53b result (the experiment that worked)
Heuristic-pool selectivity (15% 1166-ladder in pool) + fixed aimer + early_capture=0 + entropy 0.05:

| eff (M) | 1166 % | Ajay % |
|---|---|---|
| 4.7 | 62.5 | — |
| 5.2 | 64.1 | 6.6 |
| 9.4 | — | **9.0** |
| 10.0 | 62.9 | — |
| 13.1 | 64.1 | 7.8 |
| 13.6 | — | **10.9** ← best Ajay ever |

vs the prior all-time Ajay record of 3.1% (Rev35c/Rev38). clip stable 0.23–0.28, KL <0.025 (mostly <0.012), EV ~0.9 — healthy throughout.

**Standing intervention authority for the live run:** if `clip_frac` crosses **0.25** (hard threshold — >0.28 actively degrades the policy, confirmed Rev54 v2: it sat at ~0.3+ for most of 15M and the late checkpoints regressed to ~0% vs Ajay / 0.4% vs 1300 while early <0.25-clip checkpoints stayed healthy), or KL>0.05, or H_fire<0.07, **halve LR** (intervention, not kill). Don't wait for 0.32. Note: raising entropy (0.02→0.05) pushes clip_frac up (~0.20→~0.30) — pair higher entropy with lower LR.

---

## Planet-Centric Stage 1 — per-slot PPO (2026-06-08)

**Stage 1 = a loss change, not a model change.** `ppo.py` `compute_loss` switched the policy
surrogate from the **joint** log-prob ratio (one ratio over fire+ship+target summed across all
owned planets) to a **per-slot (MAPPO) factorisation** — each owned planet's decision gets its
own clipped surrogate against the shared global advantage. Network, heads, value head, GAE, action
space, features are **byte-for-byte unchanged**, so rev53b checkpoints resume perfectly. Unit-tested
(`orbit_wars_rl/tests/test_per_slot_ppo.py`).

**Clean A/B (the experiment RESULTS.md prescribed):** resume rev53b 10M + the heuristic-ladder
externals (28-member pool: 3 bots lb1084/1138/1152, `external_fraction` self-restored from the pool
checkpoint), LR 5e-5 — **one delta** from the rev53b joint continuation. Jarvis H100 spot, run
`stage1_ext`. Evals run **locally** (panels must never co-locate with training — see below).

**Mechanistic win, no outcome win.** clip_frac decoupled from empire size exactly as designed
(joint ~0.20–0.28 → per-slot ~0.006–0.02). But the full per-slot loss is a **modest, compounding
regression** on both held-out opponents:

| eff | Ajay (perslot) | Ajay (rev53b joint) | 1166 (perslot) | 1166 (rev53b joint) |
|---|---|---|---|---|
| +1M | 6.6% | 9.4% | 59.4% | ~62.5% |
| +2M | 7.4% | 7.0% | 59.4% | ~62.9% |
| +3M | 5.9% | 8.2% | ~47% | ~64% |

**Root cause (training diags + replay audit) — it's ship-size undercommitment, NOT carpet-bomb.**
Per-slot **ship-size** credit reinforces each planet's fleet-size choice independently against the
global advantage, so cheap small launches riding along in winning envs get over-reinforced →
**ships spread thin across undersized launches** → on *contested* targets no single launch is big
enough to capture → first capture fails. Diag signature: `ship0` 0.00→0.27, `meanshipbin` 20→12,
`srcs_multi` **dropped** 5–7→2–4 (fewer sources, smaller fleets — opposite of carpet-bomb),
Vμ/rewμ drift negative. Replay smoking gun (`compare_tempo_checkpoints.py`, seed 11, the contested
board): per-slot +3M **never captures**, commits **40 ships** vs the joint policy's 53–59; meanwhile
`invalid_raw_argmax` *dropped* (valid targets, just too few ships). Fire/target per-slot is fine.

**Fix — hybrid loss (implemented + unit-tested in this worktree).** Fire/target keep per-slot
credit; **ship-size reverts to JOINT** (one clipped ratio per env, `ship_log_probs.sum(over slots)`
= the original credit that concentrates fleets). Added `clip_frac_ship` (joint ship trust-region
canary) alongside `clip_frac` (now the per-slot fire/target rate). New test
`test_ship_credit_is_joint_not_per_slot` proves the split (joint ship ratio clips on a delta the
per-slot fire/target ratio passes). Run `stage1_shipjoint` launched on H100 spot to A/B vs full
per-slot + rev53b joint.

**Ops notes (this session):**
- **Never co-locate a panel with a live train** — both are CPU-bound (12 heuristic workers + Ajay's
  orbit_lite) → load 34 on 24 cores, panel ~7 hr/panel, training SPS 3100→700. Eval **locally** or
  on the box **after** training stops. (Memory: `feedback_eval_not_on_training_box`.)
- **Jarvis L4 is 22 GB** (vs GCP L4's 24 GB) → this config OOMs on it even with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Use H100 spot (IN2, ₹112.59/hr, 24 cores) or
  A100 (40 GB) to keep `--num-minibatches 16`; L4 needs `--num-minibatches 32`.
- **jl create needs balance ≥ the on-demand rate** even for spot (H100 ⇒ ≥₹255), else "Insufficient
  balance".

**Next:** if hybrid recovers conversion → **Stage 2 = VDN per-slot value head** (a real model change:
per-slot value outputs replacing the scalar; needs a brief critic re-warm). Memory:
`project_stage1_per_slot_ship`.

---

## Stage 1 results, the collapse diagnostic, control, + Stage 2 VDN (2026-06-09)

All runs: resume rev53b 10M + heuristic-ladder externals + LR 5e-5, one delta apart. Evals **local**
(panels never co-located with training — CPU contention starves both). Ajay/1166 full panels (256g):

| eff | joint-warm | per-slot | hybrid (ship-joint) | **control** (joint+fresh-critic) | VDN (running) |
|---|---|---|---|---|---|
| Ajay +1M | 9.4 | 6.6 | 8.2 | 8.2 | (backfilling) |
| Ajay +2M | 7.0 | 7.4 | 5.9 | 9.4 | — |
| Ajay +3M | 8.2 | 5.9 | **4.3** | **9.0** | — |
| 1166 +3M | ~64 | 45.3 | **38.3** | **60.5** | — |

**1. The per-slot direction underperforms — hybrid *collapses* by +3M.** The hybrid (ship-joint fix)
had the best +1M (8.2/64.8) but then collapsed hardest (Ajay 8.2→5.9→4.3, 1166 65→55→38). Both per-slot
variants end well below joint by +3M.

**2. The collapse signature is `Vμ`/`rewμ` drifting NEGATIVE — found in the training metrics, not eval.**
NOT instability (EV ~0.9, KL <0.02, clip ~0.005 throughout). It's a smooth policy drift: per-slot/hybrid
**fire *more*** (`srcs_multi` 1.7→4.85, `fire_frac` 0.36→0.56) but **`Vμ` goes negative** (+0.62→−0.35),
i.e. *over-fire ineffectively* and lose more vs the pool. Joint stays `Vμ` positive throughout. Root cause:
per-slot **decouples** the planets' fire decisions, each optimising against the *global* advantage with no
signal of whether *its own* fire is valuable → coordination/selectivity failure. **`Vμ` is now a
first-class collapse canary** (watch it stay positive).

**3. CONTROL (joint loss + re-initialised scalar critic) — resume is trustworthy.** VDN needs a fresh
(cold) per-planet critic on resume; the worry was that the *cold critic itself* (not the method) breaks
things. Control isolates exactly that: known-stable joint loss + `--reinit-critic`, one variable. **It
stayed rock-stable (Ajay 8.2→9.4→9.0)** where per-slot/hybrid collapsed. So the cold critic is benign,
resume is fine, and the collapse was the **method** (factorisation), not the setup. (At real scale the
critic re-warms EV→0.9 in <300K steps with KL never spiking — no value-warmup needed.) Added
`--reinit-critic` to `train_torch.py`.

**4. Stage 2 VDN — built, tested, launched.** Per-planet value head (`V_total=Σ_p V_p`, regressed to the
global return) + per-planet advantages for fire/target (ship stays joint on global). Gated behind
`--vdn-value`. Per-planet GAE in **planet-id space** (scatter via `owned_indices`) — required because
owned slots reorder by planet array index when ownership changes, so a naive per-slot value TD is corrupt
on capture/loss steps. Tests: `test_vdn_gae.py` (single-planet≡standard GAE, Σ_k A_k=A_total, ownership
gating), CPU smoke + resume smoke. Checkpoint-compat: backbone+policy+scalar-head load; per-planet head
fresh. **Early read (~1M):** `Vμ` POSITIVE (+0.69, vs hybrid's declining) — per-planet credit *is*
suppressing wasteful fires; BUT behaviour went very selective (`srcs_multi` 1.7→0.34, `avgfleet` 37→97) —
watch for over-suppression into **passivity** (the opposite failure). Eval panels are the arbiter.
Hypothesis test: `Vμ` stays positive AND eval stops collapsing (matches/beats joint-warm) ⇒ Stage 2 works.

**5. Instrumentation added this session** (`gpu_run_artifacts/cross_eval/`):
- **Cross-checkpoint eval panel** (`run_cross_eval.sh`) — held-out heuristics + our exported past selves
  (rev38/rev53b) → a **cycling/forgetting detector** (self-play WR is blind to it). Run every ~1M.
- **Diversity study** (`study_opponents.py`, ≥32 games) — finding: our opponent set is a **transitive
  strength ladder, not diverse** (no non-transitivity); **Hellburner is surpassed** (rev38/rev53b beat it
  100% — the old "never beat HB" was weak early agents); `candidate_zach_public.py` is a **weak old Zach
  proxy**. Implication: fixed-heuristic diversity is limited → need **exploiters** (non-transitive) and a
  distilled Ajay (style).
- **Ajay distillation spec** (`docs/ajay_distillation_spec.md`) — orbit_lite can't be GPU-batched, so
  distill Ajay→fast neural clone (DAgger) for the pool. One new tool needed (`dagger_collect.py`).

**Metric discipline — `srcs_multi` REMOVED from the code (2026-06-09).** It's empire-size-confounded:
it counts sources firing conditioned on owning ≥2 planets, so it rises naturally with empire size and
indicates *empire-building, nothing more* — never a clean firing-intensity signal (and averaged over
all steps incl. no-fire, so it conflates fire *rate* with fire *breadth*). Optimising it (rev5–48) never
moved wins. **Use instead: `fire_fraction`** (sources fired ÷ owned, on firing steps — the true
carpet-bomb signal), **`owned_planets`** (expansion), **`fire_rate`**, **`avgfleet`** (hoarding), and
**`Vμ`/`rewμ`** (the unconfounded outcome canary). The `--srcs-multi-penalty` shaping lever still exists
(default off; "do not use" — floor=0 → fire=0 Nash) but is a separate, deprecated knob.

**Ops:** eval.py now tolerates VDN `value_pp_*` keys (eval ignores the value head); model.py prints the
VDN fresh-critic notice once (was spamming ~123×/iter via pool-opponent loads); always wire
**auto-destroy on training-completion** (not an intermediate checkpoint) so checkpoints aren't thrown away.

---

## Current State (2026-06-07)

> **Superseded by the 2026-06-08 section above.** Rev50 was not the path taken — the session instead fixed the aimer + target-head bug and ran the rev53b heuristic-pool selectivity experiment, which cracked the Ajay ceiling (3.1% → 10.9%). Kept below for history.

**Active runs:** None. Rev49 ran to completion (30M steps) on GCP L4 (`orbit-wars-training`, 136.111.191.182) — **failed, worse than baseline** (0/16 Ajay at 5M/6M/8M vs Rev38/46's 3.1%). Instance still running — needs deletion decision pending next move.

**Rev49 verdict — production-weighted capture reward backfired:**
Rewarding `production_delta` (not count) was meant to create a gradient toward high-ROI single-target captures. Instead it rewards firing at *everything simultaneously* — total production delta is maximized by carpet-bombing every reachable planet, not by selective targeting. By 30M steps: `clip_frac=0.878`, `KL=16-19`, `srcs_multi=5-6`, `fire[0]=0.57-0.63`. This is a reward-shaping design flaw, not a training instability — it converged to a clean (bad) Nash.

**Lesson for next attempt:** A per-capture reward that scales with target value will always be dominated by "capture more targets" unless it's normalized per-action or capped. Need either (a) a reward that's relative/comparative across available targets (e.g. bonus only for capturing the *highest-ROI available* target), or (b) Rev50's approach — direct behavioral signal from playing against Ajay (which already exhibits selective targeting), rather than reshaping the reward function further.

**Recommended next step: Rev50** — Ajay as external pool opponent (20% of envs), already scripted at `gpu_run_artifacts/hellburner_spot/run_remote_phase1_rev50_gcp.sh`. This sidesteps reward-engineering entirely — the model sees Ajay's actual selective-targeting behavior and gets gradient from losing to it.

Delete instance when done with diagnostics:
```bash
gcloud compute instances delete orbit-wars-training --zone=us-central1-b --quiet
```

**Best checkpoints locally:**
- Rev32b 6M: `gpu_run_artifacts/gcp_rev32b/checkpoints/torch_step_6815744_rev32b_20260604_112217.pt` — Zach 88.7%, Ajay 0.8%, LB 874.3
- Rev35c 1M: `gpu_run_artifacts/gcp_rev35c/checkpoints/torch_step_1048576_rev35c_20260605_052334.pt` — Ajay **3.1%** (best SSDR result)
- Rev38 5M: `gpu_run_artifacts/rev38/checkpoints/torch_step_5242880_rev38_20260605_181635.pt` — Ajay 2.7% (7/256), Zach 89.1%, LB **950.5** (record)
- Rev38 6M: `gpu_run_artifacts/rev38/checkpoints/torch_step_6291456_rev38_20260605_181635.pt` — Ajay **3.1%** (8/256) — best panel
- Rev48 checkpoints: `gpu_run_artifacts/rev48/checkpoints/` — 43 checkpoints (0.5M–22.8M)

**Rev38 Ajay panel results (baseline to beat):**
| Checkpoint | Ajay wins | Ajay % |
|---|---|---|
| 5M | 7/256 | **2.7%** |
| 6M | 8/256 | **3.1%** ← best |
| 10M | 6/256 | 2.3% |

**Next priorities:**
1. Rev49 first Ajay quick-panel at 5M — beat 3.1% to confirm production-weighted capture fix
2. If ≤ 3.1%: launch Rev50 (Ajay external pool) — accept slow SPS, direct Ajay gradient
3. If > 3.1%: run full 256-game panel, consider submit

**Key diagnostic tools:**
- `orbit_wars_rl/count_ajay_srcs.py` — Ajay source distribution analysis (mean=1.9 when firing, 48.5% steps no fire)
- `orbit_wars_rl/diagnose_opening.py` — FireP per step on episode JSON
- `orbit_wars_rl/step_firep.py` — compare FireP at steps 0-3 across checkpoints
- `opponents/candidate_ajay_1200.py` + `opponents/orbit_lite/` — Ajay panel opponent
- `docs/submissions.md` — full submission log with Kaggle IDs and checkpoint paths

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

## Leaderboard Reality Check (2026-05-31; our row updated 2026-06-08)

Other agents' ranks/scores are a 2026-05-31 snapshot. **Our current position: submission 53451535 = 982.1** (Rev38 5M + fixed intercept aimer — new all-time record, up from the 894.4 below). Rank not re-measured.

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

**LB game-type split (current submission 53451535, 70 episodes, measured 2026-06-08):**

| Mode | Our WR | Chance baseline | Read |
|---|---|---|---|
| 2p | **40.5%** (15/37) | 50% | **Below par** — we shed rating in 2p |
| 4p FFA | **42.4%** (14/33) | 25% | **Well above par** — we gain rating in 4p |

4p **true** placement (by elimination order — see ⚠️ below): **1st 42% (14/33) / 2nd 24% (8) / 3rd 15% (5) / 4th 18% (6)**, mean 2.09 (random 2.5). Above average, but with a real early-elimination tail: in **18 of 19** 4p losses we were knocked out *before the game ended* (often by step ~50–130 of a 100–250 step game) — i.e. over-exposed and eliminated, not narrowly out-scored. Controlled local eval (Rev38 5M, identical Zach opponent): Zach 2p 90.6%, Zach 4p 59.4% (par 25%) — 4p mechanics are healthy vs a weak bot.

> ⚠️ **Placement artifact (caught 2026-06-08):** a naive final-score placement reads "1st 42% / 2nd 58%, never 3rd/4th." That is WRONG — FFA games end with a single survivor, so all eliminated players sit at score 0 and tie for "2nd." Always rank by **elimination order**, not final-score snapshot.

**Rating mechanics (resolved + confirmed by official rules + a live 4p replay header):** the LB rating is **win-based TrueSkill (μ/σ).** The env interpreter (`orbit_wars.py:703-715`) awards `reward=+1` only to the max-score agent and `-1` to everyone else, so all non-winners are a **tied 2nd** — confirmed by a real 4p replay where all three losers were labeled `[2nd]` despite different final material (148/128/129) and elimination orders. **Placement beyond 1st (3rd/4th, by material OR survival) is invisible to the rating.** Per the competition's Evaluation page: *"The score by which your bot wins or loses an Episode does not affect the skill rating updates"*; update magnitude scales with deviation from the expected result (prior μ) and each submission's σ. Consequences: (1) the 4th-place-18% tail costs **nothing directly** — only winning pays; (2) for a **top-rated agent (we're 982)**, every FFA we don't win is a slight **net drag** — non-winners "draw" with each other and a draw pulls μ toward the (usually lower) field mean, so *not finishing last is worth nothing — we must WIN*; (3) **losing to a low-rated winner costs more** than losing to a strong one (bigger surprise), so handing the game to an opportunist via over-commit is doubly bad.

**But the early deaths are FFA-specific over-exposure, and they cost WINS (which do count).** Trajectory analysis of the 3 earliest deaths (ep 79067299/79093662/79064471): we expand well — *led the game* in one — then commit **65–94% of our army forward**, get hit by **two enemies at once** while a **third snowballs off the wreckage** (winner ends at 600–700 material vs our 0). This is the classic FFA mistake the 2p policy never learned: it plays winning 2p tempo (commit everything when ahead — correct vs ONE enemy) and gets ganged with three players on the board. Several of these were *winnable* games we threw. Converting even a third of the 18 over-commit deaths into survival-and-wins lifts the **win rate** on **52% of all LB games** → real, rating-relevant upside.

**Reserve-cap probe — NEGATIVE (2026-06-08).** Tested the over-exposure hypothesis directly via an inference-time reserve cap (`action_mask.actions_from_target_policy(reserve_frac=...)`, default 0.0 = no change): force the agent to keep a home garrison in 4p and see if survival/win rate recovers.
- vs Ajay (24 games, 4p): `reserve_frac=0.0` → 20.8% win, survival 0.75; `reserve_frac=0.5` → **0% win, survival 0.75 (unchanged)**.
- vs Zach (32 games): 0.0 → 59.4%; 0.5 → 43.8% (cap hurts a weak-opponent matchup, as expected).

**Conclusion:** keeping ships home did NOT reduce deaths (survival identical) — the causal over-exposure story is **not supported.** The replay correlation (we over-commit AND die) is likely **reverse causation**: a policy fires its army out *because* it's losing/threatened, so high in-flight fraction is a symptom, not the cause. **Confound:** vs Ajay×3 we survive to ~0.75 of the game — we are NOT dying early here (LB ganging deaths were at 0.37–0.65). Ajay×3 doesn't reproduce the LB gang-up; no local bot does. **Meta-finding: we cannot reproduce the LB FFA failure mode locally, so local 4p eval (survival 0.99 vs Zach, 0.75 vs Ajay) is a poor proxy for LB 4p.**

**Net — FFA back on the shelf (earned, not asserted):** 4p is above par on wins (the only metric that counts), there is **no validated lever** to improve it, and **no faithful local eval** to develop one. 4p mixed co-training remains *possible* (env is 4p-ready; only `train_torch.py P=2` is 2p-locked) but is a speculative bet against dynamics we can't measure — not justified now. The **2p-vs-strong-opponents conversion/targeting gap** (below par, locally measurable) is the clearest headroom. Memory: `project_ffa_not_the_gap`.

> 🗄️ **Superseded (kept for history):** earlier reading from the old 894-pt submission 53076736 (rev7-era, 159 episodes) was *1v1 50% WR / FFA 30% WR*, interpreted as "FFA is the primary LB drag, agent trained 2p-only." That was true for that weaker agent but is wrong for the current entity-transformer policy — do not act on it.

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

#### Rev12–Rev57 (2026-06-01 → 06-10) — see memory + git log, not duplicated here
LB record **rev38 5M (993.9 / 967.6)** with First Strike. rev55–57 = reinforcement attempts (own-planet
targeting), all FLOODED. Full reinforce-lever history: memory `project_reinforcement_lever`; checkpoints in
`seed_checkpoints/` + `gpu_run_artifacts/rev*/`. VDN/per-slot credit (rev5x) concluded: doesn't work, reverted.

#### Phase 2 — reinforcement via empire-size gate (2026-06-10) — docs/phase2.md
Reinforcement (sending ships to own planets) is the #1 LB skill-gap. Replay analysis of the top tier (Isaiah #1,
Jake #2, timing-corrected — action at steps[t] ↔ obs at steps[t-1]): reinforce ≈0 below ~3 planets, ramps with
empire size (Isaiah plateau ~0.34, snowball cohort to ~0.61); forward-staging ~68%; full-garrison commits; no
production targeting. → reward model where reinforcement has **NO reward term**: an empire-size GATE
(`--reinforce-gate-min-planets 3`, NEW mask in torch_env) + `defense_coef 0.03` (outcome-tied holding = the
instrumental incentive) + `speed_coef 0.3` + fire-entropy **0.005** (binary fire = flood pressure once reinforce
is a costless outlet). Anneal only `early_capture`; keep expansion/defense/speed on. Selection is PURE
(win-rate/Elo decides; reinforce_rate/game-length are diagnostics).

- **Rev58** (rev38 5M + gate + defense + speed + fire-entropy 0.005, NO cost): started HEALTHY (reinf 0.21 @32k,
  p90 99, Vμ +0.63) but **DRIFTED into a flood** (reinf 0.75, p90 408, Vμ→0) by ~400k. Killed. Drifted-in (not a
  t=0 shock) ⇒ a **reward-landscape attractor** (safe-hoarding-via-reinforce: attacking a defended enemy is risky,
  reinforcing is free+safe), NOT a mature-rev38 artifact ⇒ from-scratch would flood too. The dropped
  `reinforce_cost` was load-bearing; fire-entropy (already 0.005) was not the driver.
- **Rev58b** (IN-FLIGHT): rev58 + **`--reinforce-cost 0.001`** reinstated (now paired with the gate + low entropy
  that rev57 lacked when 0.001 alone flooded). Tests whether the small cost caps reinforce_rate at ~0.3–0.5
  through the ~400k window where rev58 flooded. Decision tree + live-instance handoff: **docs/next-steps.md**.
  Run scripts: `gpu_run_artifacts/rev58{,b}/run_remote_rev58*.sh`.

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
