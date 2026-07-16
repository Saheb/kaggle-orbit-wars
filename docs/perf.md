# Training-Loop Performance

Where the SPS goes, why, and which levers actually move it. The headline finding
settles a standing question: **is the loop env-bound or PPO-bound, and would a JAX
rewrite get us to the top-team ~10k SPS?**

## ⭐ Standard training config (going forward, 2026-07-09)

Add these to every real training launch — ~2× SPS over the old fp32/4-epoch default, at a
modest sample-efficiency cost from the epoch change (the other three are numerically free):

```
--precision bf16 --compile --gpu-storage --ppo-epochs 2
```

`--ppo-epochs 2` is now the `config.py` default (Jake Will's hyperparameter). `bf16` /
`--compile` / `--gpu-storage` are CUDA-only, opt-in flags — launch scripts must pass them.
Baseline: fp32/4-epoch ≈ 1,300 SPS → this stack ≈ 3,000+ SPS on an H200 at 0.5M.

## Production GPU throughput — intent/binary runs (2026-07-13/14)

These are long-run, steady-state `train_torch.py` numbers with the production feature bundle,
self-play pool fraction 0.5, rollout length 64, 32 minibatches, PPO epochs 2, bf16, compiled
model/features, and GPU-resident rollout storage. They are the numbers to use for capacity
planning; the earlier microbenchmarks below isolate individual levers.

| host / GPU | action mode | envs | batch/update | steady SPS | representative phase times |
|---|---|---:|---:|---:|---|
| GCP L4 24 GB | intent | 384 | 49,152 | **767-768** | `gf` 2.3s, `pool` 8.1s, `estep` 8.6s, `upd` 14.7s, wall 35.1s |
| Jarvis A100 80 GB | intent | 1,280 | 163,840 | **2,591-2,601** | `gf` 1.4s, `pool` 2.3s, `estep` 15.0s, `upd` 12.5s, wall 32.1s |
| Jarvis RTX PRO 6000 96 GB | binary | 1,280 | 163,840 | **4,082-4,101** | `gf` 1.1s, `pool` 2.6s, `estep` 8.3-8.9s, `upd` 8.0-8.2s, wall 20.6-21.1s |

The RTX production run sustained **4,082 SPS through 50.79M steps**, 1.57× the A100 run and
5.3× the L4 run in logged env-steps/sec. The L4 comparison is not geometry-matched: VRAM
limited it to 384 envs, so it paid much more pool/update overhead per env-step. The A100 and RTX
runs do match at 1,280 envs, but they are still not a pure GPU A/B: the RTX run used the binary
NOOP/COMMIT action space, which removes ship-head sampling/PPO loss and suppresses the tiny-fleet
population that makes `env.step` expensive. Consistent with that mixed hardware/workload change,
RTX improved both major buckets (`env.step` 15.0→8.3-8.9s and PPO update 12.5→8.0-8.2s).

**Operational choice:** for this 0.53M-parameter production model, RTX PRO 6000 at 1,280 envs
is the best measured single-GPU throughput. Do not generalise the 1.57× ratio to arbitrary
models or action spaces; the identical-checkpoint A/B below attributes only 5-7% to skipping
binary mode's unused ship branch.

### Binary ship-head bypass A/B (2026-07-15)

An identical-checkpoint A/B on a Jarvis A100 80 GB spot instance isolated the cost of computing
the unused ship branch in binary mode. Both arms resumed the 25.3952M binary checkpoint and used
1,280 envs × 64 steps, 32 minibatches, PPO epochs 2, bf16, compiled model/features, GPU storage,
and the same seed. The control retained the old behavior (compute ship prior/residual/logits, then
discard them); the treatment bypassed `ship_head`, `ship_q`, `ship_k`, and `ship_scorer`.

| binary implementation | timed steady iteration | steady SPS | seven-iteration post-first aggregate |
|---|---:|---:|---:|
| old: compute unused ship branch | 27.2s | **3,012** | **2,941 SPS** |
| new: bypass unused ship branch | 25.4s | **3,225** | **3,076 SPS** |

That is **+7.1%** on the directly timed steady iteration and **+4.6%** over the broader
post-first window. The range reflects residual shape compilation and normal iteration variance;
use **roughly +5-7%** as the supported effect size. This is worthwhile and policy-equivalent,
but it cannot explain the RTX run's 1.57× advantage over the earlier A100 intent run. Raw logs:
`gpu_run_artifacts/envperf/binary_ship_bypass_20260715/` (instance 446533 was destroyed).

### Rejected: nearest-16 sparse target scoring (2026-07-15)

The next identical-checkpoint A/B tested scoring only the 16 shortest-ETA legal targets per
source, while leaving source enrichment dense. This was deliberately opt-in because it restricts
the action set. The same A100 80 GB configuration above was used for eight iterations per arm.

| target scorer | timed steady iteration | steady SPS | seven-iteration post-first aggregate |
|---|---:|---:|---:|
| dense | 24.2s | **3,385** | 3,127 SPS |
| nearest 16 | 24.6s | **3,330** | 3,166 SPS |

The two views bracket the result from **-1.6% to +1.3%**: no supported throughput gain. The
smaller MLP input is offset by `topk`, gather, and scatter work, while the dense pairwise source
enrichment and environment remain unchanged. The experiment was removed rather than carrying a
behavior-changing path with no measured benefit. Do not retry this exact boundary; meaningful
sparsity would have to start before source enrichment or avoid materialising dense pairwise
features. Raw logs: `gpu_run_artifacts/envperf/sparse_target_k16_20260715/` (Jarvis run IDs
`r_44dbd649` and `r_82e7768a`; spot instance 446549 was destroyed).

### Rejected: cached-target-only fleet collisions (2026-07-15)

Fleet trajectories are fixed at launch, but the engine's first physical collision is not a
perfectly fixed target: orbiting planets and comets can make a cached resolver disagree with the
dense swept-collision result. Under the 25.3952M binary policy with the normal four-tick refresh,
the cached target agreed on **1,530/1,561 true hits (98.0%)**. We therefore tested the approximation
as opt-in only: gather one cached planet per fleet and collision-test `(env,fleet)` instead of the
dense `(env,fleet,planet)` cube.

| collision path | iteration 5 wall / `env.step` | iteration 10 wall / `env.step` | 11-iteration post-first SPS |
|---|---:|---:|---:|
| dense swept collision | 24.5s / 10.3s | **27.0s / 12.8s** | **3,057** |
| cached target only | 24.5s / 10.4s | 27.6s / 13.5s | 3,048 |

There was **no speedup**: the full post-first window regressed 0.3%, and the late/high-fleet
iteration regressed 2.2% overall (5.5% in `env.step`). Gather overhead and the rest of the env
dominate this already-vectorized cube. The path was removed because it was both slower and
physics-approximate. Raw logs: `gpu_run_artifacts/envperf/cached_target_collision_20260715/`
(Jarvis run IDs `r_242e6da1` and `r_5df9cfa9`; spot instance 446561 was destroyed).

**TL;DR — the loop is model/PPO-compute-bound, and more so as the model grows. A JAX
rewrite will not reach 10k SPS.** The top teams hit 10k because their bottleneck was a
cheap CPU env that XLA fusion crushed; ours is a heavy model (the mandatory
pairwise-attention scorer) run through a heavy PPO config. The env — the thing a JAX
rewrite would replace — is only 5–21% of the loop. The real SPS levers are bf16, fewer
PPO epochs, a lighter pairwise head, or more/bigger GPUs — none of them a framework swap.

---

## The 2026-07-09 profile (H200)

Six runs on a single H200 spot box. Config held fixed at **512 envs × 64 rollout,
`--num-minibatches 32`, `ppo_epochs 4`** (= 128 gradient steps per rollout), pure
self-play (`--pool-fraction 0`), comets on. The only variables are model size (525K
blessed vs ~8M) and the two profiling flags.

| Run | Model | Metrics / sync | SPS | `upd` | wall/rollout |
|-----|-------|----------------|----:|------:|-------------:|
| A | 525K | full | **1,318** | 14.5s | 23.3s |
| B | 525K | full + `--profile-sync` | 1,296 | 14.8s | 23.4s |
| E | 525K | `--lean-metrics` | 1,333 | 14.0s | ~23s |
| C | 8M | full | **320** | 84.2s | 102s |
| D | 8M | full + `--profile-sync` | 321 | 84.9s | 102s |
| F | 8M | `--lean-metrics` | 326 | 84.0s | 100s |

Logs: `gpu_run_artifacts/jax_profile/prof_*.log`. Instance 441688 was destroyed after
the run.

### Synced phase breakdown (per rollout of 32,768 env-steps)

`--profile-sync` puts a `cuda.synchronize` at each phase boundary so every bucket owns
the GPU work it launched (without it, async kernels land at whichever bucket syncs next).

| Phase | 525K (B) | 8M (D) | scales with model? |
|-------|---------:|-------:|--------------------|
| `get_features` | 0.8s | 0.8s | no — flat |
| `env.step` | 4.3s | 4.3s | no — flat |
| batch build (flatten + H2D) | 1.2s | 1.2s | no — flat |
| sample + D2H store | 0.8s | 3.4s | ~flat (noisy, D2H copy) |
| rollout model forward | 1.3s | **7.0s** | yes — 5.4× |
| **PPO update total** | **14.8s** | **84.9s** | yes — 5.7× |
| &nbsp;&nbsp;— update forward (+metrics) | 6.0s | 29.1s | yes |
| &nbsp;&nbsp;— update **backward** | 8.5s | **55.5s** | yes — 6.5× |
| &nbsp;&nbsp;— optimizer step | 0.3s | 0.3s | no — flat |

---

## What this means

**Model/PPO-compute-bound, not env-bound.** Everything env/data-side
(`get_features` + `env.step` + build) is **~6s and completely flat** across a 15× model
scale. Everything that grew is model matmul. At 8M, **PPO backward alone (55.5s) is 54%
of the whole loop** — larger than every other phase combined. A 15× parameter increase
tanked SPS 4.1× (1,318 → 320); if the loop were env-bound this would barely move.

**A JAX rewrite won't reach 10k SPS:**
- **Env-only rewrite** touches 5–21% of the loop. Pointless.
- **Whole-loop rewrite** — the dominant cost is `fwd`/`bwd` over big matmuls, which call
  the *same* cuBLAS/cuDNN kernels under JAX as under PyTorch. XLA's edge is fusing small
  ops and eliminating launch/host-sync overhead — but **synced ≈ unsynced (1,318 vs
  1,296)**, so the loop is already effectively synchronous and there is almost no
  overhead to reclaim. The SPS ceiling here is set by model FLOPs, not framework.

**The per-minibatch metric syncs cost ~0.** The metrics block in `ppo.py` fires ~40
`.item()` GPU→CPU syncs per minibatch. `--lean-metrics` computes them once per rollout
instead — and SPS barely moves (525K 1,318 → 1,333; 8M 320 → 326). The syncs read
results of compute already on the backward critical path, so they create no idle bubble.
**Do not strip diagnostics for SPS.** (Console diag prints every 5 iters — also
negligible. Eval does not run at these step counts.)

**Capacity is not free for us.** Jake Will's "20M trains at ~the same SPS as 7.5M" holds
only when you are CPU/env-bound. We are the opposite, so a bigger model costs
proportionally more throughput. The pairwise scorer activation `(B × 16 × 48 × D)` grows
with `D` and is what OOMs at `--num-minibatches 4` — widening the model pushes hardest on
the tightest memory. (Fix: keep `--num-minibatches` at 16–32 for the heavy-pairwise
config; the default 4 OOMs a 140 GB card at 512 envs.)

---

## Precision A/B — MEASURED (2026-07-09, H200, 0.5M production model)

`--precision {fp32,tf32,bf16}`. Same config as the profile (512 envs × 64 rollout, mb=32),
our real 0.5M model. **This is a free training-time speedup with no rewrite and no
measured harm to PPO health.**

| precision | SPS | vs fp32 | EV | KL | clip_frac |
|-----------|----:|--------:|---:|---:|----------:|
| fp32 | 1,296 | — | 0.941 | 0.0090 | 0.065 |
| **tf32** | **1,492** | **+15%** | 0.933 | 0.0102 | 0.069 |
| **bf16** | **1,593** | **+23%** | 0.938 | 0.0085 | 0.072 |

EV/KL/clip converge to the same place across all three (within run-to-run noise) — bf16
does **not** destabilize PPO. Mechanism (synced breakdown, bf16 vs fp32): PPO backward
8.5s → 5.8s (−32%), update forward 6.0s → 5.0s, total update 14.8s → 11.0s; env-side flat.
**bf16 implementation:** autocast the model forward (rollout + update), then upcast logits
/ value to fp32 so the PPO ratio/log-prob math stays fp32 (the bf16 matmuls + their backward
are unaffected — the upcast is just a cast node). No GradScaler (bf16 keeps fp32's exponent
range). tf32 is even cheaper (one backend flag, fp32 API unchanged) for +15%.

**Recommendation: turn on bf16 for real training runs.** +23% SPS at this scale, more on
bigger models (backward is the phase bf16 cuts most, and backward's share grows with model
size — so an 8M/Jake-class model should see a larger bf16 win than 0.5M's +23%).

## Lever ladder — MEASURED (2026-07-09, H200, 0.5M, cumulative)

Steady SPS read from the `timing | ... wall` line at iters 10–20 (warmup-immune — the
cumulative Final SPS understates `--compile` because of its first-iter warmup). Each row
adds one lever to the row above. fp32 baseline = 1,296 SPS.

| config | steady SPS | ×fp32 | EV | clip | free? |
|--------|-----------:|------:|---:|-----:|-------|
| bf16 | ~1,650 | 1.27× | 0.966 | 0.078 | ✅ numerically identical |
| + `--compile` | ~2,000 | 1.5× | 0.967 | 0.079 | ✅ identical |
| + `--gpu-storage` | ~2,500 | 1.9× | 0.966 | 0.078 | ✅ identical |
| + `--ppo-epochs 2` | ~3,470 | 2.7× | 0.953 | 0.051 | ⚠ sample-eff cost |
| + `--ppo-epochs 1` | ~4,030 | 3.1× | 0.892 | 0.033 | ⚠ larger cost |

**bf16 + compile + gpu-storage ≈ 1.9× over fp32 (2,500 SPS), all numerically free** — EV /
KL / clip are identical to the bf16 baseline, because none of the three changes the math
(compile fuses ops; gpu-storage just relocates buffers). **This is the recommended stack for
real runs.** `--gpu-storage` also removed the noisy D2H `samp_store` (0.8–3.6s → 0.4s) and
the entire H2D `build` phase (1.2s → ~0) by keeping rollout buffers GPU-resident (GPU→GPU
instead of PCIe round-trips), which is why its `wall` is much tighter (covered 100%).

**Fewer PPO epochs is NOT free.** 4→2→1 pushes SPS to ~3,470 / ~4,030, but EV falls
(0.966 → 0.953 → 0.892) and clip_frac collapses (0.078 → 0.033) — the policy barely moves
per rollout, i.e. you extract less learning per env-step. Raw SPS ≠ learning-per-hour here;
these need a convergence check before adopting, not just the throughput number.

**We are now at the structural ceiling (~4k SPS), and `env.step` is the new wall.** At
`--ppo-epochs 1 --gpu-storage`, the update is down to ~1.5s and `env.step` (~4–5s) is the
dominant remaining phase. Past ~4k SPS on a single GPU, the env itself must get faster
(compiled/Rust/JAX env — the one place a rewrite would finally pay) or go multi-GPU (DDP).

## torch.compile on the ENV — MEASURED DEAD END (2026-07-09)

`--compile-env` (torch.compile on `env.step`, fullgraph=False): steady wall **9.4s → 9.4s
(+0%)**, `estep` unchanged. Health identical (semantics-preserving). Verdict: **not worth it.**

Why: `env.step` is riddled with `scatter_add_` / `scatter_reduce_` (6+ sites — combat per-owner
sum, inflight-per-target, ETA, inbound threat, reinforce debit) plus Python-serial comet-spawn
and per-env reset. scatter is a data-dependent op inductor can't fuse across, so the compiled
graph fragments into tiny regions with nothing worth fusing. Two prerequisites had to be added
just to make it *trace* without hard-erroring: `@torch._dynamo.disable` on `_lazy_comets`,
`_check_done`, `_auto_reset` (the numpy/`.tolist()`/`set()` helpers). Model.compile helped
(matmuls fuse); env.compile does not (scatters don't).

**Implication:** a real env speedup needs the scatters restructured as dense masked-sums over
the existing (N,F,P) collision tensor (fusion-friendly but uncertain payoff), OR the compiled-
language rewrite the winners actually did (Rust: Isaiah, "Complete novice to Top 100"; JAX: the
~10k-SPS team) — where combat accumulation is a native loop / XLA segment-sum, not a graph
break. **Tip:** validate compile-*ability* locally on CPU (dynamo tracing is hardware-
independent) before spending GPU time on the speedup measurement.

## Functional env rewrite — MEASURED 15.8× physics speedup (2026-07-09, H200)

`torch_env_fn.py` is a pure-functional / immutable twin of `torch_env` (v1: no comets, no
actions yet, shaping off). Built in stages, each parity-checked against `torch_env` as the
oracle: physics_core (bit-parity over 40 steps with combat), reset_masked (numpy-free),
both `torch.compile(fullgraph=True)` with **ZERO graph breaks** — the thing the imperative
oracle env fundamentally couldn't do (mutation + numpy + boolean-masked scatter). Key
rewrite: combat uses a **maskless dense scatter** (non-combat fleets add 0 ships) instead
of the oracle's `flat_idx[combat_mask]` boolean index, so there's no data-dependent shape.

**Physics-only microbench (512 envs, 128 fleets, H200):**

| variant | env-steps/s | vs oracle |
|---------|------------:|----------:|
| oracle eager (`env.step(None)`) | 184,000 | 1.0× |
| functional eager | 273,000 | **1.48×** (maskless rewrite alone) |
| functional **compiled** (fullgraph) | **2,900,000** | **15.8×** |

**Full step now ported too (actions + physics), and it's even better:**

| benchmark (512 envs, H200) | oracle | functional compiled | speedup |
|---|---:|---:|---:|
| physics only (`step(None)`) | 184,000 | 2,900,000 | 15.8× |
| **full step (actions+physics)** | **49,800** | **877,000** | **17.6×** |

The action port (`apply_actions_core`: owned-slot topk, 8-iter intercept aimer, **maskless
scratch-slot fleet scatter** replacing the oracle's boolean-masked advanced index) is bit-parity
with the oracle over 30 steps of random actions, and `step_full_core` compiles fullgraph with
zero breaks. Note the oracle's full step is only 49.8k SPS (vs 184k physics-only) — action
application is *expensive* in eager (topk + masked scatter + aimer = many kernel launches); the
functional version fuses all of it.

**Caveats (do NOT read 17.6× as end-to-end training SPS):** this is env.step throughput only.
Still missing for a real end-to-end number: (a) `get_features` port (separate ~0.7s/rollout,
still eager), (b) train_torch integration (get_features + model must read the functional state),
(c) reinforce discipline / diagnostics / comets (deferred — additive, throughput-neutral).
Even functional-EAGER (1.48×) is free. And the JAX port is now near-mechanical: everything is
pure tensor-in/out functions (NamedTuple wrapper kept outside the compiled region since dynamo
2.11 chokes on NamedTuple inputs — exactly the shape JAX jit wants).

**MEASURED end-to-end rollout (bench_rollout.py — real 0.5M model compiled+bf16 + real
get_features, swapping ONLY env.step): oracle 18,100 → functional 28,000 SPS = 1.54×.**
This CORRECTS an earlier over-optimistic projection (~6,500 SPS): that leaned on the profile's
`estep` ≈ 4.6s, but that was measured UNSYNCED so get_features/model async work spilled into
the estep bucket and inflated it. The honest split: env.step is only ~36% of the *rollout*, so
porting it alone buys ~1.54× rollout (less on the full loop with the PPO update). **get_features
(still fully eager) + model forward are now the co-dominant wall** — the ~28k functional-rollout
ceiling is set by them, not by the (now-negligible) 877k functional step.

**2026-07-10 update — timeline features land in get_features.** The projected-timeline
channels (planet dim 20→116, `timeline.py`) add a scatter + K=24-step recurrence per
get_features call ≈ **62% of eager get_features on CPU** (~3× the phase; still small vs the
PPO update). `torch.compile(env.get_features)` still works with it — the K-loop unrolls, the
single `scatter_add` traces fine, and the timeline channels are **bit-identical** compiled vs
eager (the pre-existing pairwise channels show ~2e-6 reassociation noise on CPU inductor).
Re-gate SPS on GPU at the next launch; `--compile-features` is now more valuable, not less.

**RESOLVED — get_features needs NO functional rewrite, it torch.compiles as a DROP-IN.** Scan
found zero hostile patterns (no scatter/numpy/mask-index); it's already pure read→compute→return.
Compiled it is **bit-identical to eager, 1 graph break** (the `owned_count.tolist()`; disable the
guarded `_obs_*` diagnostics for training). Measured 3-way rollout (512 envs, H200, real 0.5M
model compiled+bf16):

| rollout config | SPS | vs baseline |
|---|---:|---:|
| eager get_features + oracle step | 17,900 | 1.00× |
| **compiled get_features** + oracle step | 23,800 | **1.33×** |
| **compiled get_features + functional step** | **41,600** | **2.33×** |

So the real end-to-end **rollout** win is **2.33×** — get_features compile (+33%, a drop-in) *then*
the functional step becomes visible (+75% more, no longer hidden behind eager get_features).
**Caveat:** this is rollout only; the PPO update is env-independent and dilutes the full-training-SPS
gain below 2.33× (update is ~half the loop). **Actionable now:** `torch.compile(env.get_features)`
+ disable `_obs` diagnostics is a cheap, bit-identical drop-in for train_torch — no rewrite.

**MEASURED FULL-ITERATION training SPS (rollout + real PPO update, bench_full_iter.py, 512 envs,
H200, 0.5M model, epochs2/mb16): baseline 4,605 → optimized 5,394 = 1.17×.** This is THE honest
end-to-end number, and it re-confirms the opening verdict: **the loop is PPO-update-bound.** The
update is ~75% of the iteration and is env-independent, so even a 17.6× env.step + 2.33× rollout
collapses to **1.17× at the full training loop.** And it gets WORSE with model size — a bigger
model makes the update dominate more, so the env work buys even less. **Bottom line on the whole
functional-env effort:** a beautiful engineering result (17.6× env.step, bit-parity, fullgraph,
JAX-ready) that delivers only ~1.17× end-to-end because the env was never the real bottleneck —
the PPO update (model fwd/bwd) is, exactly as the first profile said. The *cheap* piece
(`torch.compile(get_features)`, part of the 1.17×) is worth shipping; the full functional-env
rewrite's payoff is mostly future-facing (JAX portability; or if the update is ever made cheaper).
To actually move full-training SPS, attack the UPDATE (bf16 ✓, fewer epochs, lighter pairwise
head) — not the env.

## Other levers (not yet measured)

1. **Lighter pairwise target head** — it dominates the update forward and the activation
   memory.
3. **More / bigger GPUs** — throughput scales with hardware, not framework (how Isaiah #1
   actually scaled: DDP across up to 32 B200s, "throughput scaled linearly").
4. **Surrender / early-truncation** — two top writeups (Jake Will, Isaiah) cut compute on
   already-decided games; Jake's adaptive surrender cut 60–70% of turns. A sample-density
   lever, not a raw-SPS one, but it multiplies useful throughput. We don't have it.

**Winner-cited throughput levers** (from writeups): fast *compiled* env (Isaiah used Rust,
not JAX; another team used JAX for ~10k SPS), pinned/preallocated memory to minimize
blocking CPU↔GPU transfers, multi-GPU DDP, pure self-play, surrender. **No writeup cites
bf16/mixed-precision** — but bf16 in a JAX/TPU stack is often the silent default, so it may
be implicit. Our A/B confirms it's a real, safe win here regardless.

Reference point: **H200 525K baseline ≈ 1,300 SPS fp32 / ≈ 1,600 bf16.** The stale
"~4,250 SPS on H100" in `JARVIS_RUNBOOK.md` predates the mandatory pairwise head (it was a
391K pre-pairwise model) — the pairwise scorer is the drop.

---

## Reproduce

Instrumentation lives in `train_torch.py` / `ppo.py`:

- `--profile-sync` — `cuda.synchronize` at phase boundaries for true per-phase
  attribution (implies `--log-timing`; profiling only, adds overhead).
- `--lean-metrics` — compute the per-minibatch metrics block once per rollout instead of
  every update; isolates the logging-sync tax (gradients unchanged; disables KL
  early-stop, so throughput probes only).
- `--entity-dim` / `--num-layers` — model-capacity overrides (default 96/3 ≈ 525K params;
  `--entity-dim 352 --num-layers 4` ≈ 8.0M).
- `--precision {fp32,tf32,bf16}` — matmul precision (CUDA only; default fp32). `bf16`
  autocasts the model fwd/bwd and upcasts logits to fp32 for the ratio math. **Use `bf16`
  for real runs** (+23% SPS on 0.5M, PPO health unchanged).

```bash
# 525K synced phase breakdown
python3 orbit_wars_rl/train_torch.py --num-envs 512 --rollout-steps 64 \
  --num-minibatches 32 --pool-fraction 0 --no-wandb --device cuda \
  --profile-sync --total-steps 327680 --run-name prof_b

# 8M (capacity → SPS) — note --num-minibatches 32 (mb=4 OOMs)
python3 orbit_wars_rl/train_torch.py --num-envs 512 --rollout-steps 64 \
  --num-minibatches 32 --pool-fraction 0 --no-wandb --device cuda \
  --profile-sync --entity-dim 352 --num-layers 4 --total-steps 327680 --run-name prof_d
```

The `timing |` console line (every 5th iter) reports each bucket plus `covered %`
(fraction of wall-clock attributed). `ext_*` ⊂ `pool` and `upd_fwd/bwd/opt` ⊂ `upd`.
