# The Train/Eval Gap — why pool win-rates don't translate

## ✅ RESOLVED (2026-06-13) — the gap was MISSING COMETS in torch_env

The dominant cause of the torch_env↔kaggle fidelity gap was found and fixed: **`torch_env` did not
simulate comets at all** (`# comets not implemented in Phase 3a`). The real kaggle env spawns 4-comet
symmetric groups at steps **[50, 150, 250, 350, 450]** — they are **collidable** (fleets die hitting
them), **capturable** (they carry ships + production 1), and **moving** (elliptical paths). So **every
torch_env game from step 50 on was a different, simpler game than kaggle** — exactly the mid/late window
where our policy "wins" by hoarding and where `planets@50→100` collapses in reality. The policy overfit a
comet-free world; in kaggle, comets disrupt the hoard and add contested material → tactics don't transfer.

**How it was found (decisive, not by elimination):** a new trajectory-diff harness
(`orbit_wars_rl/sim_gap_probe.py`) drives BOTH engines from the same seed (they share `generate_planets`
+ RNG order → identical boards) and replays a real kaggle game's **exact action stream** in torch_env,
diffing state every step. Result: byte-identical for ~50–77 steps, then a single fleet diverged — it had
hit a **comet** in kaggle that didn't exist in torch (confirmed: a comet sat 1.5–2.1 units from the
diverging fleet). **This also exonerated everything else** — orbital motion, fleet movement, spawn,
combat resolution, and win-resolution all match byte-for-byte given identical actions.

**The fix:** comets implemented in `torch_env.py` (reuse kaggle's `generate_comet_paths` for byte-exact
ellipse math + RNG order; reserved planet slots 44–47 so the policy *observes* them; lazy per-spawn
compute for throughput). After the fix the harness is **faithful for entire games incl. comets** across
all seeds. Regression test: `tests/test_comet_fidelity.py`. **Expected consequence (validate on the next
run): in-training difficulty should now match cross-eval** (the rev38/rev53b pins were easy in torch_env
60–70% but 27–37% in kaggle — that split should shrink), and the hoarding/`planets@50` pathologies should
become trainable because the agent now faces the real comet-ful mid-game. **Perf:** comet-path generation
is vectorized (`_comet_paths_fast`, numpy — **7.5× faster, byte-identical to kaggle**, verified on 2000
seed/spawn combos; `np.cos/np.sin` match `math` bit-for-bit and segments ≪ comet_speed so the resample
searchsorted == kaggle's sequential append) **plus** lazy per-spawn compute (only spawns a game reaches).
Net: the one-time synchronized spawn-50 spike is ~3s and steady-state per-step comet cost is negligible
(~13ms when one env crosses a spawn). Byte-identity guarded by `tests/test_comet_fidelity.py`.

Everything below predates the fix (kept for the investigation record). The "two larger suspects" it
couldn't isolate (our overfitting / win-timeout) were **both downstream of the missing-comet game
divergence**, not independent causes.

---

**TL;DR (2026-06-11):** the in-training opponent-pool win-rate is measured **inside `torch_env`**
(our vectorised GPU reimplementation of the kaggle env), with **sampled** actions, as an **EMA/
cumulative** stat. It is a **PFSP sampling signal, not a performance metric.** For aiming-heavy
heuristic opponents it is wildly optimistic — `torch_env` weakens them. **Don't read pool `wr`/
`ema_wr` as "how good are we vs X." Use a clean `kaggle_environments` eval (cross-eval) instead.**

This file exists so we don't re-run this investigation. If you see a pool `wr` that looks great but
held-out eval disagrees, the answer is below — it's the sim gap.

---

## The phenomenon

The training log prints pool members like:

```
external_heuristic  14_main_k_v2_lb1152_LAST_HEURISTIC  wr=0.37(n=2070) ema_wr=0.48 ...
external_heuristic  11_v14_1n_lb1138_doom_evac_mega_hammer wr=0.34(n=3840) ema_wr=0.40 ...
external_heuristic  03_v12_7m_lb1084_4p_relative_gap_hammer wr=0.46(n=2003) ema_wr=0.48 ...
```

These read as "we win ~34–48% vs the lb1152/1138/1084 hammers." We do **not**. A clean eval in the
real kaggle env says otherwise.

## The decisive measurement (p2rev1 8.25M vs lb1152, masks-on)

Same checkpoint, same reinforce-discipline masks, 32 games each:

| environment | decode | wr vs lb1152 |
|---|---|---|
| `torch_env` (training pool) | sampled | **ema 0.48** / cumulative 0.37 |
| `kaggle_environments` (eval) | threshold-0.5 | **6.25%** (2/32) |
| `kaggle_environments` (eval) | sampled | **0.00%** (0/32) |

The ~**40–48 point** gap is **not**:
- **decode** — within the kaggle env, sampled (0%) is *worse* than threshold (6%); matching
  training's sampled decode does **not** close the gap (still ~0%). (NB: this is the opposite of
  the phase-1 zach finding where threshold>sample; don't assume either direction — measure.)
- **board distribution** — the eval was **non-panel** (default random boards ≈ training board gen),
  not the harder stratified `--panel` archetypes.
- **cumulative-vs-EMA** — even the responsive `ema_wr` (0.48, ~100-game window) is ~8× the kaggle
  number. The stale cumulative `wr` only makes it worse.

What's left, isolated by elimination: **the simulator.** Same policy, same sampled decode, only the
env differs → torch_env 0.48 vs kaggle 0.00. **It's a `torch_env` ↔ `kaggle_environments` fidelity
gap.**

## A real discrepancy found + fixed — but it's a MINOR contributor, not the gap (2026-06-11)

⚠️ **Honest status:** the 144-bin angle quantization below is a genuine torch_env↔kaggle discrepancy
and is now fixed, but a controlled A/B shows it explains only a sliver of the gap. **Hammer-vs-hammer
with the override on one side only (same agent, same boards, asymmetric only in aiming): continuous
52.6% vs quantized — 10 wins to 9, N=19, i.e. noise.** A ±2.5° handicap is small relative to the
hammer's strategic strength, so it cannot account for a 48-point (0.48→0) pool-vs-eval gap. The fix is
kept (it's correct, free, and makes opponents marginally truer in-sim) but the **dominant cause of the
gap is still open** — see "What still doesn't add up" below.

The hammers' aimers are **not** buggy — lb1152 runs a correct 6-iteration lead/intercept solver
(`AIM_MAX_ITERS=6`, orbital extrapolation via `predict_planet_position`, comet paths), fed the right
`angular_velocity` by `to_legacy_obs`. The physics constants match too (sun 10, rotation-limit 50,
fleet speed `1+5·(log(ships)/log1000)^1.5` cap 6.0).

**The precision was thrown away on the way INTO torch_env.** `_heuristic_moves_to_action_tensor`
quantized the hammer's continuous angle to a 144-bin grid:
`ab = int(ang / ANGLE_BIN_WIDTH)`, `NUM_ANGLE_BINS = 144` → `ANGLE_BIN_WIDTH ≈ 2.5°`, then the env
re-expanded to the bin **center**. The real kaggle env takes the continuous angle directly. So the
hammer ate a **±1.25–2.5° aiming error inside torch_env that it never has in the real env** — at a
40-unit flight that's ~0.9–1.7 units of lateral miss, on the order of a planet radius, enough to
sweep past orbital targets and trip the hammer's own K9F path-clear refusals. Aggregated over a
game → missed captures in-sim → weak in training, strong in eval → pool ema 0.48 vs kaggle 0%.

**Why it was asymmetric (and missed):** our agent uses `--target-decode`, which computes a
**continuous** intercept angle via the fixed aimer; the hammers were left on the **old 144-bin
angle path**. The continuous-aimer upgrade reached our agent but never reached the external-opponent
conversion, so torch_env handicapped opponents' aim relative to ours — exactly the "weak in
training, strong in eval" population.

**The fix (continuous-angle override):** `_heuristic_moves_to_action_tensor` now also returns the raw
continuous angle; it's threaded through `env.step(..., angle_overrides={seat: (N,MAX_OWNED) float})`
(NaN = no override) and applied in `_apply_actions` after the target-decode block, bypassing the bin
quantization for external rows only. The 4-col int action tensor is unchanged (the scatter at the
pool call-site still works); the angle rides a parallel float channel. Tests:
`tests/test_ship_bin_decode.py::test_continuous_angle_override_bypasses_bin_quantization` (+ NaN
fallback). Training-only — eval already runs opponents in the real continuous-angle kaggle env.

## What still doesn't add up (the gap is mostly elsewhere)

The A/B says the hammer is ~as strong quantized as continuous, so removing the quantization will
barely move our-agent-vs-hammer pool wr. So the 48-point gap is mostly NOT "hammer weak in torch_env."
That leaves two larger suspects, untested:

1. **Our agent overfits torch_env physics.** Our policy was *trained inside* torch_env and may exploit
   torch_env-specific quirks (motion, collision, combat resolution) that don't transfer to the real
   engine. In torch_env it beats the hammer ~half; in the real env its torch_env-tuned tactics don't
   transfer *and* the hammer is at full strength → we lose. This is a sim2real gap on OUR side, which
   the opponent-aim fix does nothing for.
2. **Win/timeout resolution.** The agent hoards (high `garr_frac`), games run long; if torch_env's
   `_check_done` (max score, ±1) scores a 500-step timeout differently than the kaggle engine, some
   torch_env "wins" aren't wins in the real env.

**Decisive next probe:** run the SAME fixed our-agent checkpoint vs the hammer in *both* torch_env and
the kaggle env and compare our win-rate. If torch_env≈0.48 but kaggle≈0 for the identical matchup,
and hammer-vs-hammer aim is ~neutral, the gap is (1) and/or (2) — not opponent aiming. Then diff a
single shared trajectory (same seed) between the two engines to localize the physics/resolution
divergence.

## How the pool wr is actually computed (so you know what it is, not what you wish it were)

- `opponent_pool.py`: `win_rate = wins/(wins+losses+draws)` **cumulative over the whole run** (lags
  current ability — includes games from when the agent was weak). `ema_win_rate` is an EMA,
  `alpha=0.01` (~100-game window), draws count as 0 — the better "current" estimate, still torch_env.
- `train_torch.py:701` (`compute_pool_actions`, `external_heuristic`): the opponent move is computed
  from `to_legacy_obs(env, ...)` by a persistent CPU worker pool, converted back via
  `_heuristic_moves_to_action_tensor` (shared with eval). The **move conversion** is shared; the
  **environment the opponent observes and acts in is torch_env**, not the kaggle env.
- The pool wr exists to drive **PFSP opponent sampling** (sample harder opponents more). That is all
  it is for. It was never an eval-comparable performance number.

## What to do about it

1. **Never treat pool `wr`/`ema_wr` as a performance/decision metric.** It's a torch_env + sampled +
   EMA sampling signal. Champion/checkpoint selection runs on **held-out eval only** (this is also
   the rev15 / `phase2.md §6` lesson).
2. **Use cross-eval for the comparable number.** `gpu_run_artifacts/cross_eval/run_cross_eval.sh`
   now evals the **same pool hammers** (`pool_lb1152/1138/1084`) in the **kaggle env**, alongside the
   held-out anchors and our past selves. Run it every ~1M steps. A large `pool ema` vs `cross-eval
   wr` split = this sim gap, re-confirmed live.
3. **Masks-on, always.** Reinforce checkpoints must eval with the training masks
   (`--reinforce-gate-min-planets 3 --reinforce-forward-only --reinforce-garrison-floor 10`) or they
   self-sabotage (see `metrics.md` decode note). The cross-eval and the debatreya watcher both pass
   them now; `allow_reinforce` is read from the checkpoint so the flags are no-ops for phase-1 ckpts.

## Verifying the fix

- **Mechanism:** unit test proves the continuous angle is applied (not re-quantized).
- **Outcome (the real confirmation):** re-measure **pool-wr vs cross-eval** on the next training run.
  If the fix worked, the hammers get meaningfully harder in training (pool wr drops from ~0.48 toward
  the real ~0–6%) and the gap closes. Bonus: the agent is now training against **true-strength**
  opponents, which is the pressure that should fix the persistent mid-game **hoarding** weakness
  (`garr_frac@50 ~0.80` vs top-player 0.54) — a weakness that self-play couldn't fix precisely
  because the aggressive opponents that punish hoarding were crippled in-sim.
- **Quick local A/B:** hammer-vs-hammer with the override on one side only (same agent, same boards,
  asymmetric only in aiming) — `/tmp/hammer_winrate_ab.py`. (Note: a hammer-vs-*passive* expansion
  test is the WRONG regime — at short range a 2.5° error rarely misses; the bug bites on long-range/
  orbital/contested shots over full games.)

## Remaining follow-up (only if a gap persists after the fix)

**Win/timeout resolution parity.** Run a hammer vs a fixed bot to a 500-step timeout in both envs and
compare who's declared the winner — torch_env's `_check_done` (max score, ±1) vs the kaggle engine.

## Pointers

- Pool wr / EMA / PFSP: `orbit_wars_rl/opponent_pool.py` (`record_result`, `win_rate`, `sample`)
- Training opponent execution: `orbit_wars_rl/train_torch.py:701` (`compute_pool_actions`),
  `to_legacy_obs` in `torch_env.py`
- Eval (kaggle env): `orbit_wars_rl/eval.py`; cross-eval: `gpu_run_artifacts/cross_eval/run_cross_eval.sh`
- Related: `docs/metrics.md` (trust order, decode note, conversion), `docs/phase2.md §6` (selection on
  outcomes only), the aimer fix in `docs/training.md` (2026-06-08).
