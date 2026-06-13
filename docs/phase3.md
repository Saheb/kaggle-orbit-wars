# Phase 3 — Structural anti-cycling: ratcheted teacher-KL + league

**Purpose:** fix the *structural* self-play failure that no reward knob has touched — **drift / cycling /
Nash-reform** — by anchoring self-play to a **ratcheted teacher-KL** alongside the **opponent league/pool**, so
improvement is *monotonic* instead of *cyclic*. Runs on the now-faithful (comet) sim, from-scratch, with the new
feature set (comets done + game-phase). This is NOT another reward-shaping delta; it changes the *training dynamics*.

Origin: the Toad Brigade / Isaiah Pressman (Lux AI 2021 winner) writeup — teacher-KL "stabilize[s] behavior and
prevent[s] strategic cycles, both of which plague a pure self-play setup" — mapped onto our exact documented failure.
See `writeups/Toad Brigade's Approach…md`, memory `reference_toad_brigade_rl_recipe`, `docs/next-steps.md`
FROM-SCRATCH FEATURE SET. Lineage context: `docs/phase2.md` (reinforcement) + `docs/training.md` Current State.

---

## 1. The failure this targets (recap)

Our signature: held-out WR (vs Ajay/deb) **peaks ~500k–1M then drifts down**, self-play WR stays ~50%,
`planets@50=6` invariant across **four** reward/mask levers (p2rev5 deb-pool, p2rev6 commit-mask, p2rev7
defense_coef, p2rev8 early_capture). Every shaping term we anneal gets **Nash-eaten**. This is **cumulative drift +
forgetting + drift-to-degenerate-equilibrium** — NOT a reward bug.

**Why reward knobs can't fix it (the crux):** PPO's clip is a *relative* trust region — "don't move too *fast* this
update." It is **blind to slow cumulative drift** over millions of steps (which is why `clip_frac` looked benign while
the policy rotted — [[feedback_clipfrac_lowkl_benign]]). The teacher-KL is the missing **absolute** anchor: "don't
move too *far* from this good point." Cycling/Nash-reform is a slow-far-from-good failure → exactly what an absolute
anchor damps and the clip cannot see.

---

## 2. The two complementary levers (different channels — do NOT conflate)

| | Pool / league (pinned-RL + externals) | Teacher-KL (this) |
|---|---|---|
| What the frozen ckpt does | **plays against** you in games | **scores your action distribution** at visited states |
| Channel | the **reward** (win/lose) | an **auxiliary loss** term |
| Message | "you must **beat** this" (pressure) | "**behave** like this" (anchor) |
| Failure it fixes | can't beat diverse strong strategies | drift / forgetting / cycling |

They are **orthogonal and run together**: the pool supplies the *climb*, the teacher keeps the climb *monotonic*.
Isaiah used both (self-play opponents + frozen teacher-KL). This is the league-self-play / iterated-best-response
architecture (AlphaStar, OpenAI Five) in its minimal single-anchor form.

---

## 3. Teacher-KL mechanics — and why it does NOT cap you at the teacher

Loss: `L = L_PPO + β · KL(π_current(·|s) ‖ π_teacher(·|s))`, averaged over **visited** states. It is a **soft**
constraint. At equilibrium the forces balance:

```
∂(win-rate)/∂θ  =  β · ∂(KL-to-teacher)/∂θ
```

so the policy **deviates from the teacher wherever win-rate is to be gained**, bounded by β. You *can* exceed the
teacher. With a **fixed** teacher the anti-drift strength and the ceiling-limiting are the **same knob (β)** — stronger
anchor ⇒ better damping but lower ceiling. The **RATCHET** resolves that tension:

```
anchor T0 → improve past it → re-snapshot teacher = new held-out-best (T1)
→ anchor T1 → improve past it → re-snapshot (T2) → …
```

Each anchor stops drift *back*; refreshing raises the ceiling. **Fixed teacher = soft-capped; ratcheted teacher =
no fixed ceiling.** In a from-scratch run the ratchet naturally starts at BC/weak and climbs with you — you never
anchor far *above* yourself (which would be distillation), only to your own rising best.

**The real limit is therefore the SELECTION SIGNAL, not the teacher.** The ratchet only ratchets *up* if "new best"
is truly better → it is **only as good as held-out WR's reliability**. A noisy/biased signal can ratchet *down*. This
ties Phase 3 directly to the closed-loop fidelity question (§9).

**Safety — anchor to your OWN strong self, never a heuristic.** KL toward a heuristic's targets craters us
(rev54: dragged the shared target-scorer to enemy/neutral targeting, destroyed our aggressive play). Own-self anchors
pull toward *our* winning behavior → none of that. This is the line that separates Phase 3 from the dead bc-coef-0.05
imitation aux: **same machinery, opposite intent** — stabilizer (don't drift from our good self), not imitation
(copy someone else).

---

## 4. What ALREADY exists (don't rebuild)

- `ppo.py`: `frozen_il_model` + `_il_kl_penalty()` — KL(π_current ‖ π_frozen) on **fire/ship/target** heads, masked
  to valid slots, frozen model run on the **same rollout states**, added to the loss as `il_coef · il_kl`. This is
  exactly an on-policy teacher-KL at visited states. ✅
- `config.py`: `il_lambda` (peak coef) — comment already reads "Anchors the policy to teacher competence — prevents
  drift to degenerate … policy can eventually exceed the teacher." `kl_target` early-stop. ✅
- `train_torch.py`: `--il-lambda`, `--il-ref <ckpt>` (defaults the reference to the `--resume` checkpoint), and a
  **linear decay `il_lambda → 0`** schedule. ✅
- Separate: `bc_coef`/`--bc-samples` = cross-entropy on a teacher's *argmax actions* from a fixed sample set (a
  different anchor; penalizes argmax-flip directly). NOT what we want here — we want the live KL-to-model.

**Gaps vs Phase 3:**
1. **Usage reframed:** use a STRONG own teacher + a *meaningful* β (a stabilizer, not the 0.05 nudge we concluded is
   too weak). Past `il_lambda` use was warmstart-from-partial; here it is the main event.
2. **The RATCHET is not implemented.** The current schedule **decays the anchor to 0** ("let it exceed the teacher")
   — that *releases* the anchor over time, the opposite of sustained anti-drift. We need **re-anchor to a rising
   best**, with β roughly *constant* (not decaying to 0).

---

## 5. The plan — staged (one mechanism proven at a time)

### Stage A — prove the anti-drift mechanism CHEAPLY (resume, single delta)
Isolate the teacher-KL on the *exact* drift we've documented, with NO from-scratch / new-feature confound.
- **Resume p2rev5 4M** (or the current strong base) + **`--il-lambda <β>` with `--il-ref` = that SAME checkpoint**
  (self-anchor at the resume point) + the SAME pool. ONE delta = the anchor. **Disable the decay-to-0** (or make it a
  very slow floor) so the anchor persists.
- **Watch:** does held-out WR (Ajay/deb) **drift LESS than the un-anchored p2rev lineage did** (which peaked ~1M then
  fell)? That is the whole hypothesis, as a *trend over many checkpoints*, not a point.
- **Tune β by the canaries** (§8): too weak → still drifts; too strong → `clip_frac → 0` / entropy collapse (frozen
  policy, our lesson #9). Start small, increase until drift is damped without freezing. (β is a STABILITY knob, not a
  reward knob — different, lower-risk failure profile than the shaping graveyard.)
- **Decision:** drift damped on a *fixed* self-anchor ⇒ mechanism works ⇒ proceed to Stage B. Flat-no-effect over many
  ckpts ⇒ β too weak or wrong heads; frozen ⇒ β too strong. (Our standard "judge by the target metric's trend over
  many ckpts" rule.)

### Stage B — the RATCHET + from-scratch + new features (the real Phase-3 run)
- **From-scratch** (BC warmstart) so the feature set (below) is learned natively + the policy acquires comets/phase
  from step 0.
- **Teacher = own best-held-out-so-far, refreshed (ratchet).** Naturally starts at BC (style insurance while weak) and
  climbs to strong RL selves. β roughly constant (no decay-to-0).
- **Ratchet controller (the one real build):** every ~1–2M steps / N checkpoints, the watcher computes held-out WR;
  if a new best, **refresh `--il-ref` to that checkpoint** and continue.
  - **v1 (manual/cheap):** the held-out eval watcher already ranks checkpoints; pick the new best and relaunch
    resume with the new `--il-ref` (a clean checkpoint boundary). Low build cost, proves the ratchet.
  - **v2 (in-process):** reload `frozen_il_model` mid-run from the new best (no relaunch). Build only if v1 works.
- **Pool:** the pool-seed-RL + deb league (the existing queued lever) — pinned strong-but-beatable RL selves +
  the peeler. Both levers together (pressure + anchor).
- **Watch:** held-out WR **climbs and HOLDS past 2M** (does not peak-then-fall); the ratchet's anchor rises over time.

### Stage C — full league (optional, later)
Multiple teachers / past selves (a real league) for true intransitivity, if the single ratcheted anchor plateaus.
Heavier; only if Stage B's single anchor is insufficient.

---

## 6. The from-scratch feature set (bundled into Stage B — model-dim change)

All input-dim changes MUST land together in the one from-scratch run. Source: `docs/next-steps.md` FROM-SCRATCH
FEATURE SET (authoritative; summarized here).
- **✅ Comet features (done):** is_comet + path-aware position/expiry, train/eval/export parity, regression test
  `feature_parity_comet_probe.py` CLEAN.
- **🟢 Game-phase features (to build):** (1) game-phase one-hot (early/mid/late, or `planets@16/32/50/100`-aligned
  buckets); (2) comet-cycle phase = normalized steps-to-next-spawn. One-hot channels fit our Linear-projected globals
  (no new embedding layer). Parity-safe (computable from `step`/`angular_velocity`/constant spawn steps).
- **⭐ Coupled hypothesis:** game-phase as an OBSERVATION may let us **retire the time-scheduled shaping**
  (`early_capture` exp-decay, `first_strike` t<50) — test whether the agent self-schedules the opening aggression we've
  been bribing it into. Clean falsifiable sub-experiment inside Stage B.

---

## 7. Monitoring / canaries / decision rules

| Signal | Healthy | Action |
|---|---|---|
| **held-out WR trend** (Ajay/deb, many ckpts) | climbs then HOLDS (no peak-then-fall) | THE decider; PURE — never select on self-play WR / shaped reward / Vμ |
| `clip_frac` | < 0.25, not → 0 | → 0 = anchor too strong (frozen policy) → lower β |
| entropy | stable, not collapsing | collapsing = over-anchored → lower β |
| KL-to-teacher (`il_kl`) | moderate, non-zero | ~0 = anchor inert (β too weak / teacher == current); huge = pulling hard |
| ratchet anchor strength (held-out WR of current `il-ref`) | non-decreasing over refreshes | decreasing = ratcheting DOWN → selection signal is lying (§9) |

Decision rule (standard): judge the delta by **its own target metric (held-out drift) trending over MANY
checkpoints** — flat/peak-then-fall over many ckpts = it isn't working → diagnose β/teacher, don't burn GPU to 10M.

---

## 8. Risks / open questions

- **Teacher choice (Stage A fixed anchor):** the resume base (self-anchor) for the cheap test; for Stage B the ratchet
  removes the choice (always = own best). For a fixed strong anchor, prefer **LB-validated** strength + the behaviors
  self-play erases (aggressive opening / holding), not panel-only.
- **β tuning:** one more knob — but a STABILITY knob (canary-tunable via clip_frac/entropy), not a reward knob (no
  fire=0 / carpet-bomb Nash-trap risk). Lower-risk profile than the shaping graveyard.
- **Ratchet-DOWN risk:** noisy/biased held-out WR can re-anchor to a worse point → ties to §9. Require a *margin* on
  "new best" before refreshing (don't ratchet on noise).
- **KL on the shared target-scorer:** the rev54 crater. Mitigated *because the teacher is our OWN strong self*, not a
  heuristic — the pull is toward our winning targeting, not someone else's. (Still watch target behavior.)
- **Throughput:** the frozen teacher does one extra forward on rollout states per update (already implemented). Cost
  known/modest; measure SPS at launch.
- **Interaction with the pool:** both are "frozen-ckpt-flavored" but act on different channels; they should compose.
  Guard against double-constraining (if both pin hard, the policy may stall) — read held-out WR.

---

## 9. Sequencing / gate

1. **GATE — closed-loop fidelity (in flight):** the frozen-vs-pins kaggle full-panel anchors (running) decide whether
   torch_env held-out WR is a TRUSTWORTHY selection signal. The ratchet *is* a selection-signal amplifier — if a
   residual closed-loop gap exists (torch_env disagrees with kaggle, e.g. the rev53b 10% vs 37.5% discrepancy), the
   ratchet would amplify a *biased* signal. **Resolve/understand the closed-loop verdict before relying on the
   ratchet.** (Stage A, which uses held-out *trend* not absolute level, is more robust to this and can proceed first.)
2. **Build game-phase features** (`docs/next-steps.md`) — needed for Stage B's from-scratch run.
3. **Stage A** (cheap resume + self-anchor) → read held-out drift → tune β.
4. **Stage B** (ratchet + from-scratch + features) → read held-out HOLD-past-2M.
5. **Stage C** (league) only if needed.

One delta per cloud run still holds *within* a stage (Stage A = the anchor; Stage B isolates the ratchet given A
proved the anchor). Selection stays PURE: held-out WR / Elo decides, never self-play WR / shaped reward / Vμ.
