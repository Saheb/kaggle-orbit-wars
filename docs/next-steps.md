# Next steps / idea backlog

Living doc — efforts and ideas, roughly prioritized. Discipline: **one delta per cloud run.**
Status tags: 🔵 in-flight · 🟢 ready-to-build · 🟡 idea · ✅ done · ⏸ parked.
Current focus: **⭐ COMET FIX (2026-06-13) — the train/eval sim gap was MISSING COMETS in torch_env, now FIXED**
(byte-faithful to kaggle, `docs/train-eval.md` + `docs/training.md`). This was the root blocker. p2rev9 was relaunched
on the faithful engine and **validated the fix works live** (pin WR recalibrated down), then killed. **Next: comet
FEATURES (is_comet + expiry, train/eval/export parity) then a from-scratch run.** Boards ruled out (self-play = LB =
panel, WR A/B confirmed). Prior framing below (pool pivot) still valid but now sits on a faithful sim.
Current focus: **Phase 2 — the POOL/CURRICULUM pivot** (p2rev9 LIVE, 2026-06-13). 4 reward/mask levers all left
`planets@50=6` → the ceiling is STRUCTURAL because of **win-starvation** (can't learn from a peeler we never beat;
deb ~5%). Fix the SIGNAL: pin winnable RL selves (rev38/rev53b — cross-eval 27%/37.5%, NOT deb-like) instead of
shaping. See `docs/training.md` Current State + [[feedback_win_starvation]]. VDN/per-slot concluded (back to standard arch).

---

## 🔴 CURRENT PROBLEMS & CONCLUSIONS (2026-06-12 — VALIDATED, read before any run)

Localized with honest, **outcome-split** metrics on **p2rev3 0.5M** (our best vs deb: 3.9%) — full
256 panels, correct masks. The dashboard rebuild (below) is the session's big win; prior diagnoses
were made on confounded numbers.

**THE GAP:** beat weak opponents (zach 87–89%), crushed by strong **peelers** (deb 2–4%, ajay ~0%).

**TWO PROBLEMS, TWO WINDOWS** (deb, p2rev3 0.5M; `ref:winner` open 0.51 / mid 0.47):
- **P-OPEN (secondary) — opening conversion weak.** `open<50 cap/atk` **0.29 lost / 0.43 won** vs
  winner **0.51** — below even in our wins. Mechanism: fragments fired *under* the target's defense
  (6 ships at a 43-ship neutral → annihilated). Real, but we still win 87% vs zach with it, so not
  the thing that loses to deb.
- **P-HOLD (DOMINANT) — mid-game peel.** The opening ramp is **identical won/lost through @50**
  (2/4/6); divergence is **@50→100**: won **6→9/10**, lost **6→4** (→0.9, eliminated). And mid-game
  *conversion is FINE* — `mid50-100 cap/atk` **0.47 = winner 0.47**. We convert mid-game well but
  **can't hold**: churn 16, **peel-rate WON 0.61** vs winner **0.43**. Holding, not conversion, is the
  mid-game failure, and it degrades *specifically vs a peeler* (zach WON peel-rate 0.31; deb 0.61).

**REINFORCEMENT (the hold mechanism) — validated against winner replays:**
- **`garrison_floor=0` is CORRECT — keep it.** Winners hard-commit: send 100% of source (median),
  frac-left mean ~7–8%, **no absolute or fractional reserve** (even 30+ ship sources drain; "keep 20%"
  is NOT a winner habit). `floor=10` blocked 86–91% of winner-style reinforces → it was the wrong default.
- **`--reinforce-forward-only` ≈ FINE — not the blocker (corrected).** Our direction **71% fwd / 20%
  rear** ≈ winner **57/26** (centroid). The mask uses *nearest-enemy*, which doesn't force
  centroid-100%-forward, so we already pull back ~20%. Low priority to change.
- **The real reinforce gap = RATE/TIMING, back-loaded.** reinf-by-step `<50:0.05 · 50-100:0.19 ·
  >100:0.42` vs winner `0.29 · 0.41 · 0.31` — too little early/mid, too much late.
- **Quantity ≠ the fix.** floor 10→0 (p2rev4) raised reinforce (mid 0.11→0.19) but peel did NOT drop
  (0.61→0.64), churn got WORSE (16→23). More reinforce fed the churn war, not holding ⇒ we reinforce
  **incorrectly** (wrong target/timing), not too little.

**THE VALIDATED LEVER → next run: a PEELER (deb-class) in the EXTERNAL pool.** The collapse happens
*only* vs opponents that peel (zach can't, deb does). Self-play + fast hammers never punish bad holding,
so holding-under-pressure is never learned. A peeler in the pool makes it a learned skill. Other reinforce
knobs (floor/forward-only) are already right; the missing ingredient is *training pressure*.

**RULED OUT (do NOT re-chase):** spray/`fire_frac` (empire-size + win/loss confound; won-game ≈ Isaiah
0.17 — [[feedback_firefrac_winloss_confound]]); ship-head credit (joint AND per-slot both undercommit —
user-confirmed); reward-shaping fire/target (fire-tax→fire=0 Nash, target-value→rev49 carpet-bomb,
target-KL→rev54 crater); early_capture as the collapse cause (was OFF in p2rev3; eval doesn't use the
reward; collapse is opponent-specific so it can't be a reward schedule); floor / forward-only as the
reinforce fix (floor correct, forward-only ≈ fine).

**METRICS REBUILT (trustworthy now — `eval.py game_conversion`/`_fmt_conversion`, docs/metrics.md):**
all conversion metrics **outcome-split (WON/LOST)** — `planets@`, `cap/atk` by phase
(**open<50 / mid50-100 / whole**), **peel-rate** (renamed from lost-cap), `fire_frac`/`launch_rate`;
**reinf by step** (<50/50-100/>100), **by empire size**, and **direction** (fwd/rear vs centroid);
`underkill` flagged NON-discriminating (winners ~0.40 too); `ref:winner` convention (refs prefixed
`ref:`, our values bare). See [[feedback_opening_capatk_discriminator]].

---

## ✅ SUPERSEDED (historical) — was "NEXT RUN: SUFFICIENT-COMMIT MASK"

> This plan ran as **p2rev6 and FAILED** (planets@50 flat at 6 — a veto removes fragments but doesn't teach
> concentration). Then defense_coef (p2rev7) + early_capture (p2rev8) also failed → all four levers left
> `planets@50=6` → the win-starvation finding → **pivot to pool-seed-RL (p2rev9 LIVE)**. Kept below for history.

### original plan (set 2026-06-12 from the held-out panel trend)

p2rev5's 12-ckpt held-out panel trend (live-instance block below) shows deb-in-pool did NOT move the two
outcome levers: **opening conversion `opnWON` flat ~0.38 vs winner 0.51**, and `peelW` ~0.6 vs 0.43.
Reinforce *direction/rate* are already winner-like vs Ajay (fwd 58%, rf>100 0.24) ⇒ **threat head is NOT
indicated.** The dominant unmoved gap is the OPENING → build the **sufficient-commit mask** (P-OPEN lever
#2 below): veto an attack launch whose `ships ≤ target's current defense` (exact for neutrals, the opening
targets) → fragments impossible by construction, forces concentration. Structural mask, NO reward tax
(avoids the rev41–45 fire=0 Nash). Resume p2rev5's best ckpt; ONE delta = the mask. Secondary: arrest the
early-reinforce decline (rf<50 0.15→0.09 vs winner 0.29). Threat head → parked (revisit only if a later
held-out panel shows reinforce *mis-targeting* vs a real opponent, which 6M did not).

### ~~peeler-in-pool (deb)~~ — DONE as p2rev5 (concluding; verdict above)

- **From-scratch (snowball-BC warmstart)** so holding-under-peeling is learned natively.
- **ONE delta vs the p2rev4 config: add debatreya_1300 to the external pool** (the peeler).
- **Keep validated defaults:** `garrison_floor 0`, `gate 3`, `floor=0` masks at eval/export.
  **DROP `--reinforce-forward-only` for the peeler run** (decided 2026-06-12): the run's whole point is
  learning to *hold* a peeled rear planet, and forward-only blocks pulling ships *back* to defend it
  (the "secondary blocker", 23%→42% of intents). Eval/export masks must match → also no forward-only.
- Selection: held-out WR vs a DIFFERENT strong opp (ajay/producer) — deb is now in training, so it's
  no longer the clean held-out yardstick. Watch `peel-rate WON` ↓ and `planets@50→100` stops collapsing.
- Open (decide at launch): whether to *also* turn on `early_capture` for P-OPEN (2nd delta) — or keep
  this run clean (deb-only) and address the opening separately.

### P-OPEN levers (opening under-commitment) — ranked

Disease: in the opening we fire **fragments *under* the target's defense** (replay: 6 ships at a 43-ship
neutral → annihilated) and **split forces** across several neutrals → each too small to capture →
under-expansion. Winners hard-commit (frac-sent 1.00, send everything at *one* target). Already ruled
out as an architecture bug (ship-head credit, joint AND per-slot, both undercommit) → it's **learned**.

**⭐ P-OPEN ↔ P-HOLD coupling (2026-06-12, user insight) — NOT universal, two regimes.** `planets@50 = 6`
is **invariant** across p2rev5/6/7 in our wins AND losses (winner 9). So: **(regime A) same 6-planet board,
holding alone decides** — won 6→10 vs lost 6→3 @50→100; here a pure holding fix (defense_coef) gets us far,
P-OPEN is NOT the blocker. **(regime B) opening gates holding** — a 6-planet empire is intrinsically harder to
hold than a 9-planet one (less depth to stage from, less production to *fund* reinforcement, peeler concentrates
on a smaller frontier, no margin for loss); here holding has a **ceiling set by the opening** — you can't hold
what you didn't expand to. **Implication:** holding (p2rev7) is a real lever, but its payoff plateaus once
regime-B dominates → **falsifiable read: if p2rev7's `peel-rate WON` improves then plateaus while `planets@50`
stays 6, that's the ceiling** → opening becomes the priority. The gap is **QUANTITY (6→9), not commitment** (the
sufficient-commit/commitment lever FAILED, #2) → revisit the quantity gradient (#3).

1. 🟢 **deb-in-pool may fix it for free (check FIRST).** deb is a strong *expander*, not just a peeler —
   it out-expands a fragment-dribbler (replay: 9 vs our 5 by step 48). So the peeler run punishes opening
   under-expansion *and* mid-game peeling with the same gradient. **Read `open<50 cap/atk` (WON) after the
   deb run before adding a dedicated opening lever** — it may already lift toward 0.51.
2. ❌ **Sufficient-commit MASK — TESTED & FAILED (p2rev6, CONCLUDED 2026-06-12).** `open<50 cap/atk WON` FLAT
   ~0.34 over all 9 ckpts, `planets@50` stuck at 6 → the veto removed fragments but the agent fired *less*, did
   NOT concentrate ([[feedback_veto_mask_removes_not_teaches]]) ⇒ **commitment was not the bottleneck; quantity
   is** (see #3). Relaxed 0.6 / neutrals-only is a low-priority remaining variant. Original spec kept below.
   Veto an attack launch whose
   `ships ≤ target's current defense × factor` → fragments impossible by construction, forces concentration
   (attack only a target you can actually take, else accumulate first). Structural like the reinforce
   masks — **no reward tax** (avoids the rev41–45 fire=0 Nash). **Aligns with measured winner behavior**
   (hard-commit). Training-only, internalized, eval/export parity like floor/forward-only.
   **Impl:** `--sufficient-commit-factor` (1.0 strict / 0.6 relaxed / 0 off) — post-decode veto in
   `torch_env.py _apply_actions` (alongside garrison_floor); parity in `action_mask.py`, `eval.py`,
   `export_agent.py` (bakes `_SUFFICIENT_COMMIT_FACTOR`); test `tests/test_sufficient_commit.py` (veto/fire/off,
   passing). Applies to BOTH self-play seats (near-no-op for strong opponents, who hard-commit). **Launch ready:**
   `gpu_run_artifacts/p2rev6/{run_remote,launch}_p2rev6_jarvis.sh` — resume p2rev5 best + ONE delta (factor 1.0),
   else identical to p2rev5 (deb pool, gate3/floor0/no-forward). ⚠️ update RESUME_CKPT to p2rev5's FINAL best
   after it concludes. Watch: `open<50 cap/atk` (WON) should lift toward 0.51; guard against forced-passive
   opening (if no neutral is takeable from the start garrison → relax to 0.6 or neutrals-only).
   Risks: (a) strict — blocks sub-threshold *softening* waves, but winners barely use those; (b) enemy
   planets reinforce in transit so "sufficient at launch" can fall short on arrival — **exact for neutrals**
   (the opening expansion targets, which don't regrow), approximate for enemy planets; (c) untested —
   masks have surprised us before. A relaxed variant (`ships > defense × 0.6`, or neutrals-only) is the
   fallback if it over-constrains.
3. 🟡 **early_capture / first_strike — REVISIT as the QUANTITY lever (reframed 2026-06-12), conditional.** Was
   parked as "rewards count not commitment" — but the sufficient-commit (commitment) lever FAILED and the data
   says **the gap IS count: `planets@50` 6 vs winner 9.** `early_capture 0.30` was literally the rev28 breakthrough
   that took the agent passive→expanding; the p2rev5/6/7 base runs it at **0** (no opening-expansion gradient at
   all). So re-introduce it as a dedicated **P-OPEN delta** (consider **always-on, not annealed**) with success =
   `planets@50` → 9, NOT as a holding lever. **TRIGGER (user, 2026-06-12): try this IF the opening doesn't improve**
   (i.e. p2rev7/holding plateaus against the regime-B ceiling above). Watch it doesn't regress to carpet-bomb
   (rev49) or 1-ship spray — pair with `--min-ship-bin`. The eventual winning config likely needs BOTH a 9-planet
   opening AND good holding, solved as separate clean deltas; opening arguably first since it *gates* regime B.
4. ⏸ **Capture-efficiency reward — graveyard.** Penalizing wasted/inefficient fires = a conditional
   fire-tax → passivity / fire=0 Nash. Do not. The mask (#2) is the non-tax way to encode the same intent.

### Reinforce-targeting levers

1. 🟡 **K-nearest own-target mask (NEW idea 2026-06-12) — structural "bucket-brigade" staging.** An own
   (reinforce) target is legal only if it is among the **K nearest** owned planets to the source (K=2–3);
   ships then cascade rear→front in short hops over successive turns instead of any one planet shipping far.
   Same structural family as `--reinforce-forward-only` / `--sufficient-commit-factor` — a legality **MASK,
   NO reward / NO target-ranking shaping** (biasing the target distribution via reward/KL = rev49 carpet-bomb /
   rev54 crater; the mask is the non-shaping way to encode "prefer near"). Motivation: (a) replay winners
   target by **distance rank ~0.3** (prefer near, NOT by production value); (b) short hops keep ships ON
   planets (defensible) vs long transits (idle/vulnerable); (c) **it sidesteps the reinforce credit-assignment
   void** — reinforce earns no reward at arrival (value is delayed+counterfactual: only "avoided a future
   loss"), so the agent can't efficiently *learn* the cascade is good; a mask just *makes* it stage in short
   forward hops. **Distinct from forward-only:** "nearest" does NOT force forward, so it still permits pulling
   ships BACK to defend a peeled rear planet (the exact behavior forward-only blocked → why we dropped it).
   Impl: slots next to the gate/forward block in `torch_env.py` (~line 636, the `allow_reinforce` target-mask
   branch) — per source slot, keep own-targets only among the K-smallest src→tgt distances; parity in
   `action_mask.py` / `eval.py` / `export_agent.py` like the other masks; test in `tests/`. Caution: real
   boards are scattered (not a clean front line) so K=1 is too rigid → start **K=2–3**. ⚠️ Risk per
   [[feedback_veto_mask_removes_not_teaches]]: banning far-reinforce may just make it reinforce *less*, not
   cascade — pair with a reason the cascade pays off. Sequencing: a single delta AFTER p2rev6 concludes.

2. 🟡 **Empire-gate threshold (`reinforce_gate_min_planets`) — UNTUNED knob + a misleading metric (user, 2026-06-12).**
   The gate is **3** ("expand first, never reinforce a tiny empire" — replay ramp reinforce@1=0.00, @2=0.10,
   @9-12=0.30); a "felt logical" pick, never swept. **Metric caveat to fix FIRST:** with gate=3, reinforce is
   MASKED at 1–2 planets, so the eval's **"reinf@2-3" bin is diluted by the gated 2s — the true reinf@3 is ~2× the
   reported number** (0.07 bin ≈ 0.13–0.14 @3). Stop comparing our *gated* "2-3" against the winner's *ungated*
   @2 → **split the @2/@3 bins in `eval.py game_conversion`** so reinf@3 reads cleanly. **On lowering 3→2:** it
   would match the replay's @2=0.10 (winners reinforce a little at 2 planets; we ban it), BUT it **fights our
   under-expansion problem** — at 2 planets the right move is to grab the 3rd, and gate=3 *forces* that; lowering
   gives a reason to reinforce-not-expand exactly when smallest. AND the gate isn't the early-reinforce lever
   anyway (the `<50` deficit is downstream of slow expansion; per-empire rates are already ~winner when legal).
   **Verdict: PARK the threshold sweep** — gate=3 is defensible / maybe-right-for-us; revisit only if we become a
   fast expander (planets@50→9) and early reinforce is STILL gated-short. Winner's @2=0.10 is affordable once
   you're already expanding fast — we're not there. (Do the cheap metric-split regardless.)

### Pool / anti-drift levers (training pressure, NOT reward)

1. 🔵 **pool-seed-RL + deb — LIVE as p2rev9 (2026-06-13).** Cross-eval validated the pins are winnable (p2rev5 4M
   beats rev38 27% / rev53b 37.5%, vs deb ~5%) → real win-gradient. Config: pins rev38+rev53b, deb 0.25→0.10, 35%
   self, clean reward (early_capture/defense OFF). See the LIVE block above. Original rationale below.
   **Motivation:** the improve-then-degrade pattern (held-out metric peaks ~500k–1M then drifts down) is the
   self-play **Nash reforming around a misaligned objective** (the documented "SSDR transient / Nash reforms
   ~2M" — [[feedback_selfplay_collapse_metrics]], [[project_phase2_reinforcement]]). The fix is MORE
   *asymmetric non-self* pressure so the degenerate behavior actually loses games. BUT you can't just crank the
   external fraction: deb/orbit_lite is CPU-slow (~11 ms/call, [[feedback_orbit_lite_is_slow]]) so SPS already
   654→~250 at just ~0.19 deb; pure-external also over-fits one **sim-gap archetype** ([[project_train_eval_sim_gap]])
   AND starves the win-signal (we beat deb ~4% → high-deb = nearly all-loss = no win-gradient + passivity
   collapse). **The throughput-viable version:** add **`--pool-seed-rl`** (GPU-fast, sim-gap-IMMUNE, diverse RL
   champions — same fast path as self-play) **+ keep deb as the peeler external.** Pin **rev38** (first-strike
   aggressor → opening pressure) + **rev53b** (selective) [± more]; **KEEP rev31/rev32b HELD-OUT** as the
   cross-eval forgetting detector (do NOT pin them). Optionally bump external **0.25→~0.4** but keep **~30–40%
   self-play** (the matched-difficulty auto-curriculum + winnable games — do NOT go pure-external). Both halves
   needed: pinned-RL = cheap diverse strong pressure; deb = the specific peeler that punishes bad holding.
   **BUILT + TESTED** (`opponent_pool.py add_pinned_rl`, `tests/test_pool_pinned.py`; the p2rev4 GCP script
   already wired `--pool-seed-rl rev38_5M.pt,rev53b_10M.pt`). **Read:** held-out trend PAST 2M — if it stops
   drifting (holds/climbs past the early peak) we changed the attractor (the rare durable-win signal); if it
   still peaks-then-falls, the pressure still isn't enough. Sequencing: a single composite "pool" delta off the
   p2rev5 4M base (or whichever base wins the p2rev6-vs-p2rev7 A/B).

---

## ✅ CONCLUDED — p2rev6 on Jarvis A100 (sufficient-commit mask DID NOT lift opening conversion; box destroyed 2026-06-12)

- **VERDICT (ended at 7.8M, box 425850 destroyed):** the sufficient-commit mask (factor 1.0) **failed to move
  its own target metric.** `open<50 cap/atk WON` was **FLAT ~0.33–0.35 across all 9 held-out Ajay checkpoints**
  (500k→6.8M; winner 0.51) — at or *below* the p2rev5 4M base (~0.40). WR flat-to-declining (5.5→1.6). **Why:**
  `planets@50` stayed **6** throughout (winner 9) — opening *expansion never sped up*. The veto removes fragment
  launches but the agent responded by firing *fewer*, NOT by *concentrating* → a veto-mask removes the bad
  behavior but doesn't supply the good one (aggregation). No degeneracy (fire_frac WON 0.17, garrison low).
  Best ckpt is noise-level, none beat the p2rev5 4M base → **future runs still resume p2rev5 4M.** 14 ckpts +
  log harvested to `gpu_run_artifacts/p2rev6/`. **Narrow remaining hope for the IDEA (new run, not this one):**
  relaxed factor **0.6 / neutrals-only** (1.0 may over-constrain, blocking softening waves). Lower priority than
  p2rev7 (defense_coef) + the K-nearest-mask idea. **DECISION RULE confirmed:** a delta's target metric flat over
  many checkpoints = it isn't working; end + harvest (don't burn GPU to 10M hoping a pinned metric unsticks).

- **Jarvis A100-80GB ON-DEMAND, machine id `425850`, IP `217.18.55.147`, 28 cores. ⚠️ ON-DEMAND (not
  spot) — DESTROY when done (it bills ~₹2x/hr while Running, even after the run auto-stops at 10M).**
  ⚠️ **The local `gpu_run_artifacts/p2rev6/{instance.txt,.watch_env,launch.out}` initially referenced a
  DEAD spot box (425837 / 217.18.55.30) — a prior spot launch that died; the on-demand box was re-created.
  Always reconcile with `jl list` before trusting those files. instance.txt now fixed to 425850/.147.**
- **DESTROY:** `jl destroy 425850 --yes` (after final ckpts sync; needs venv + key — see JARVIS_RUNBOOK §Auth).
- **Run = the SUFFICIENT-COMMIT MASK delta** (`docs/phase2.md` §P-OPEN #2): resume **p2rev5 4M**
  (`seed_checkpoints/torch_step_4194304_p2rev5_20260612_083540.pt` — p2rev5's held-out Ajay WR peak, 5.9%)
  + **ONE delta `--sufficient-commit-factor 1.0`** (veto attack launches whose ships ≤ target defense →
  fragments impossible, forces concentration). Everything else IDENTICAL to p2rev5: deb-only pool @0.25,
  gate3 / **garrison-floor 0** / **no forward-only**, early_capture OFF, speed 0.3, expansion 0.03,
  fire-entropy 0.005, LR 5e-5, gae-lambda 0.99, 256 envs / rollout 128 / ppo-epochs 2 / 8 heuristic workers.
  Script: `gpu_run_artifacts/p2rev6/run_remote_p2rev6_jarvis.sh` (cwd `/home`).
- **TARGETS the held-out panel's dominant unmoved gap: opening conversion** — `open<50 cap/atk (WON)`
  stuck ~0.38 vs winner 0.51 across all of p2rev5's 12 ckpts. WATCH: `open<50 cap/atk (WON)` should lift
  toward 0.51. GUARD: if the opening goes forced-passive (no neutral takeable from the start garrison),
  relax factor 1.0→0.6 (or neutrals-only). Threat head = PARKED (held-out panel showed reinforce already
  winner-like vs Ajay — not the indicated lever).
- **LAUNCH NOTES (2026-06-12, took over a half-finished setup):** (1) `kaggle_environments` was missing on
  the box (a pause/resume wiped conda site-packages while `/home` persisted) → reinstalled via
  `pip install -q kaggle-environments && bash setup/install_orbit_wars.sh`. (2) tmux does NOT survive SSH
  disconnect on this image + a `pkill -f train_torch.py` over SSH self-kills the shell → left duplicate runs;
  cleaned to ONE tmux-managed run. Verify single run: `pgrep -c -f "[t]rain_torch"` == 1. iter 1 healthy
  (EV 0.69, clip 0.095, KL 0.008, SPS ~600–755 once de-duped). See memory `feedback_ssh_pkill_selfkill`.
- **Watchers via the CONTROLLER** repointed to .147: `bash gpu_run_artifacts/run_watchers.sh start p2rev6
  jarvis 217.18.55.147` (tore down the stale .30 watchers). Held-out **Ajay** full panel per ckpt, masks
  **gate3 / floor 0 / NO forward-only** (match training) → `gpu_run_artifacts/p2rev6/eval_ajay_1200.csv` +
  `eval_logs/`. `… status` / `… stop` as usual. First eval = step 500k.
- Monitor: `ssh -i ~/.ssh/jarvis-labs-key root@217.18.55.147 "grep '^iter' /home/train_gpu_phase1_p2rev6_*.log | tail"`.

---

## ✅ CONCLUDED — p2rev9 comet-faithful relaunch (KILLED @1.28M 2026-06-13, box DELETED, no billing)

- **⭐ This was the comet-fixed validation relaunch** (resume p2rev5 4M + POOL pivot on the FIXED torch_env).
  **Verdict: the fix works live** — iter 1 ran past the step-50 comet spawn cleanly, comets active in the full
  loop (SPS ~485), and **in-training pin WR dropped with comets** (rev38 ema 0.60→0.53, rev53b 0.70→0.60),
  difficulty recalibrating toward kaggle reality. Harvested 524k+1M ckpts → `gpu_run_artifacts/p2rev9/`; old
  comet-free artifacts → `p2rev9_cometfree_old/`. Killed after confirming the fix (it was a validation run, not a
  result run). **Next = the comet FEATURE phase (is_comet + expiry, parity) then from-scratch.** Original live
  block kept below for reference.
- **GCP L4 `g2-standard-8`, name `orbit-wars-p2rev9`, IP `8.231.115.108`, zone asia-south1-b. DELETED.**
- **DELETE (manual):** `gcloud compute instances delete orbit-wars-p2rev9 --zone=asia-south1-b`.
- **Run = the POOL pivot (NOT a reward delta).** Resume **p2rev5 4M** + ONE delta = the pool: `--pool-seed-rl
  rev38_5M.pt,rev53b_10M.pt` (winnable RL champions) + **deb demoted 0.25→0.10** (`--pool-external-fraction 0.15`
  × `--pool-fraction 0.65` ≈ 10%) + **35% self-play** (pool-fraction 0.65). Reward = p2rev5 CLEAN baseline:
  **`--early-capture-coef 0` + defense_coef OFF** (both p2rev7/p2rev8 levers DROPPED). gate3/floor0/no-forward,
  speed 0.3, expansion 0.03, LR 5e-5, 256 envs/128 rollout/ppo-2, 4 workers. Script:
  `gpu_run_artifacts/p2rev9/run_remote_p2rev9_gcp.sh`.
- **WHY:** win-starvation ([[feedback_win_starvation]]) — 4 reward/mask levers all left `planets@50` at 6 because
  we can't learn from a peeler (deb) we never beat. **Cross-eval 2026-06-13: p2rev5 4M beats rev38 27% / rev53b
  37.5%** (vs deb ~5%) = matched difficulty, a real transferable win-gradient; rev38 (aggressor) punishes
  under-expansion at a beatable difficulty. Also confirmed p2rev7 1M (27%/25%) — picked p2rev5 4M (cleaner reward
  lineage, no defense_coef baggage, slightly more winnable vs rev53b).
- **WATCH (held-out Ajay panel):** **`planets@50` climbs toward 9** (pins' target) + **WR climbs from the 27-37%
  base**. Secondary `peel-rate WON` (deb). GUARD: clip<0.25 (resume into new pool = warmup); ship0 ~0. KILL read:
  `planets@50` flat over many ckpts = pins didn't break it either → the ceiling is deeper than the pool.
- **Controller watcher** (sync + held-out Ajay per ckpt): `bash gpu_run_artifacts/run_watchers.sh start p2rev9 gcp
  orbit-wars-p2rev9.asia-south1-b.orbit-wars-rl` → `gpu_run_artifacts/p2rev9/`. iter 1 healthy (EV 0.70 warmup, KL
  0.007, clip 0.08, both pins + deb loaded, GPU 100%, SPS ~580 pre-warmup). GCP ckpt dir is a real dir (no symlink
  sync bug). ~6-7h to 10M (deb only 0.10 → less CPU than p2rev7).
- Monitor: `gcloud compute ssh orbit-wars-p2rev9 --zone=asia-south1-b -- "grep '^iter' ~/orbit_wars_rl/train_gpu_phase1_p2rev9_*.log | tail"`.

---

## ✅ CONCLUDED — p2rev8 on Jarvis A100-80GB SPOT (KILLED @5.77M 2026-06-13, box 425956 destroyed)

**VERDICT: FAILED — early_capture 0.2 did NOT move `planets@50` (dead flat at 6 over all 9 held-out ckpts;
winner 9), opening conversion flat ~0.37, WR flat ~4.5%.** Killed together with p2rev7 on the **win-starvation
finding** (`docs/training.md` Current State): we can't learn to beat a peeler we almost never beat — early_capture's
within-game expansion reward has no terminal deb-win that *requires* 9 planets to anchor it. `planets@50=6` is now
invariant across FOUR levers → opening ceiling is STRUCTURAL, not reward-tunable. 11 ckpts harvested → 5.77M.
**Next = pool-seed-RL pivot, not another reward delta** (see pool levers below). Original live block kept for
config reference:



- **Jarvis A100-80GB SPOT, machine id `425956`, IP `217.18.55.39`, 28 cores. ⚠️ SPOT — DESTROY, never pause
  (preemption may lose data; the sync watcher below is mandatory).**
- **DESTROY:** `jl destroy 425956 --yes` (from the launch box; needs venv + key — JARVIS_RUNBOOK §Auth).
- **Run = the QUANTITY/opening lever** (`docs/next-steps.md` P-OPEN #3): resume **p2rev7 1M**
  (`torch_step_1048576_p2rev7_20260612_144503.pt`) + **ONE delta `--early-capture-coef 0.2`** (0→0.2,
  always-on / anneal-frac 0). Everything else IDENTICAL to p2rev7 incl. **`--defense-coef 0.02` carried
  forward** (the 1M base was trained with it → keeping it is what makes early_capture the single delta).
  deb-only pool @0.25, gate3 / floor0 / no-forward, speed 0.3, expansion 0.03, LR 5e-5, 256 envs / 8 workers.
  Script: `gpu_run_artifacts/p2rev8/run_remote_p2rev8_jarvis.sh`.
- **CLEAN A/B:** p2rev7 continues on GCP from the SAME 1M point with defense_coef ONLY; p2rev8 branches here
  adding early_capture, so the divergence past 1M isolates early_capture's marginal effect.
- **WHY:** `planets@50` is stuck at **6** (winner 9) across p2rev5/6/7 in BOTH wins and losses — the QUANTITY
  gap. The sufficient-commit MASK (p2rev6) failed to lift it (veto removes fragments, supplies no
  concentration → [[feedback_veto_mask_removes_not_teaches]]). early_capture rewards the successful-capture
  OUTCOME (clamped count delta, NOT rev49's production-weighted delta → carpet-bomb) so failed fragments pay
  0 → positive gradient toward real commitment. It was the rev28 breakthrough (passive→expanding).
- **WATCH (held-out panel):** `planets@50` should climb toward 9 + `open<50 cap/atk WON` toward 0.51.
  **GUARD:** `ship0` must stay ~0 (1-ship spray canary — NOT using min-ship-bin; early_capture self-punishes
  probes since they capture nothing); `fire_frac`/`shipspp@50` must NOT balloon (carpet-bomb). KILL read:
  `planets@50` flat over 3–4 ckpts = failed like the mask.
- **Watchers via the CONTROLLER:** `bash gpu_run_artifacts/run_watchers.sh start p2rev8 jarvis 217.18.55.39`
  (sync + auto held-out **Ajay** full panel per ckpt, masks gate3/floor0/no-forward) → `gpu_run_artifacts/p2rev8/`.
  iter 1 healthy (EV 0.57 resume-warmup, KL 0.008, clip 0.08, single proc, `Early capture coeff: 0.2` confirmed).
- Monitor: `ssh -i ~/.ssh/jarvis-labs-key root@217.18.55.39 "grep '^iter' /home/train_gpu_phase1_p2rev8_*.log | tail"`.
  ⚠️ At 10M the run auto-stops but the SPOT box keeps billing — `jl destroy 425956 --yes` after final ckpts sync.

---

## ✅ CONCLUDED — p2rev7 on GCP L4 (KILLED @4M 2026-06-13, instance orbit-wars-p2rev7 deleted)

**VERDICT: hold metric moved but it's a self-play MIRROR artifact, not transferable.** `peel WON` declined
0.64→0.52 (toward winner 0.43, first lineage result to move it; flood guard GREEN) BUT **WR DECLINED 5.9→3.1**
and `planets@50` slipped 6→5 (`mid cap/atk` rose to 0.71) — defense_coef + perpetual-deb-loss = good at holding a
SMALL empire, expands less (conservatism). The peel gain is self-play copies out-holding *each other*, which is why
it does NOT lift WR vs deb. Killed on the win-starvation finding (`docs/training.md`). 7 ckpts harvested → 3.67M.
Original live block kept for config reference:



- **GCP L4 `g2-standard-8` (23 GB, 8 vCPU), zone asia-south1-b, name `orbit-wars-p2rev7`, IP 34.100.224.184,
  RUNNING + BILLING (~$1.13/hr, on-demand).**
- **DESTROY (manual, no auto):** `gcloud compute instances delete orbit-wars-p2rev7 --zone=asia-south1-b`
  (DELETE not stop — stopped instances still bill for disk; [[feedback_gcp_instance_cleanup]]).
- **Run = the defense_coef test** (`docs/phase2.md` flood history): resume **p2rev5 4M** (same base as p2rev6)
  + **ONE delta `--defense-coef 0.02`**, `--sufficient-commit-factor` OFF, else IDENTICAL to p2rev6
  (deb-only pool @0.25, gate3 / floor0 / no-forward, early_capture 0, speed 0.3, expansion 0.03, LR 5e-5,
  4 heuristic-workers for the L4's 8 cores). Script: `gpu_run_artifacts/p2rev7/run_remote_p2rev7_gcp.sh`.
- **WHY (the untested combo):** defense_coef is the one reward term that gives reinforcing/holding a
  near-immediate signal (avoid production lost) → fills the reinforce credit-assignment void. rev58 said it's
  the *flood pump*, BUT that was a **symmetric, pool-LESS mirror**; Tier-1 then dropped defense_coef AND added
  the aggressive pool in one move, so **defense_coef + peeler-pool was never isolated**. The deb pool should
  now punish hoarding. **A/B framing:** p2rev6 = sufficient-commit (opening), p2rev7 = defense_coef (hold),
  both single-delta off the SAME p2rev5 4M base → read which lever moves the gap.
- **WATCH:** **peel-rate WON should FALL** (winner 0.43 vs our ~0.6) — the payoff. **FLOOD GUARD:** flood =
  reinf rate up **WITH** volume exploding (`p90`/`shipspp@`); healthy = reinf up, volume flat, peel down.
  Magnitude 0.02 (rev58's 0.03 floods pool-less; started smaller). If it floods → it's the pump again (pool
  insufficient); if too conservative (game length rises, attacks drop) → lower it.
- **MONITORING = the CONTROLLER (since 2026-06-12, after p2rev6 concluded and freed it).** p2rev7 now owns the
  single-run controller: sync + auto held-out **Ajay** panel per checkpoint, masks gate3/floor0/no-forward (NO
  sufficient-commit — matches training). Procs were 51273/51274 → `gpu_run_artifacts/p2rev7/{logs,checkpoints,eval_ajay_1200.csv}`.
  **Restart if the laptop bounces:** `bash gpu_run_artifacts/run_watchers.sh start p2rev7 gcp orbit-wars-p2rev7.asia-south1-b.orbit-wars-rl`
  (`… status` / `… stop`). The launch script's original standalone sync loop (PID 41464) was RETIRED to avoid
  double-sync. ⚠️ If a SECOND concurrent run starts, it can't use the controller (single-run) — give it the
  standalone-sync + manual-eval treatment instead, and DON'T `start` the controller for it (that tears down
  p2rev7's). Cross-eval vs the 4M baseline is still MANUAL + needs a solo window (thrashes swap alongside the panels).
- iter 1→3 healthy (EV 0.69→0.84, KL low, clip 0.10→0.20 warmup, `Defense coeff: 0.02` confirmed).
  **SPS ~250** (deb/orbit_lite is CPU-bound on 8 cores; ~½ the A100's rate) → ~11 h for 10M, first 500k ckpt ~1 h.
- Monitor: `gcloud compute ssh orbit-wars-p2rev7 --zone=asia-south1-b -- "grep '^iter' ~/orbit_wars_rl/train_gpu_phase1_p2rev7_*.log | tail"`.

---

## ✅ CONCLUDED — p2rev5 on Jarvis A100-80GB SPOT (launched 2026-06-12 ~08:22 UTC; verdict → p2rev6 sufficient-commit mask)

- **Jarvis A100-80GB SPOT, machine id `425730`, IP `217.18.55.11`, 28 cores. ⚠️ SPOT — DESTROY, never pause
  (preemption may lose data; that's why the sync watcher below is mandatory).**
- **DESTROY:** `JL_API_KEY=$JARVIS_API_KEY jl destroy 425730 --yes` (from the launch box; needs `jl` + key).
- **Run = the validated peeler-in-pool delta** (`docs/phase2.md`): resume **p2rev3 0.5M**
  (`seed_checkpoints/torch_step_524288_p2rev3_20260611_153903.pt` — the documented **best-vs-deb (3.9%)**,
  before Nash reform erased holding; uploaded under its ORIGINAL name, no `*_resume.pt` rename) +
  **debatreya_1300 as the sole external pool opponent**
  (the peeler, `--pool-external-fraction 0.25`, `--heuristic-workers 8`). 256 envs / rollout 128.
  Masks: gate3, **garrison-floor 0**, **forward-only DROPPED** (decided 2026-06-12 — the run's point is
  learning to *hold* a peeled rear planet; forward-only blocks pulling ships back to defend). early_capture
  OFF, defense_coef 0, speed 0.3, expansion 0.03, fire-entropy 0.005. Script:
  `gpu_run_artifacts/p2rev5/run_remote_p2rev5_jarvis.sh` (cwd `/home`; box log `/home/train_gpu_phase1_p2rev5_*.log`).
- **⚠️ TORCH FIX APPLIED AT LAUNCH:** the box image shipped torch `+cu130` vs driver 570 (CUDA 12.8) →
  `cuda.is_available()=False` → it silently trained on **CPU** (A100 idle, no iter 1). Reinstalled
  `torch/vision/audio 2.11.0+cu128` (`cuda=True`) and relaunched. Setup now self-heals this
  (`setup/install_orbit_wars.sh` cu128 guard). Watch for it on any fresh Jarvis box.
- **Watchers via the CONTROLLER** `gpu_run_artifacts/run_watchers.sh` (NEW 2026-06-12 — fixes the recurring
  "stale prior-run watcher" bug): `run_watchers.sh start p2rev5 217.18.55.11` launches sync + held-out eval,
  tearing down ALL existing watchers first; each watcher self-terminates when `.active_run` changes. `… status`
  = active run + live procs; `… stop` = kill all. Sync pulls log + `torch_step_*/pool_step_*/torch_best_*`
  every 120s → `gpu_run_artifacts/p2rev5/{logs,checkpoints}`. Restart with `start` if the laptop bounces.
  Per-run ad-hoc `*_watch.sh`/`sync_watcher.sh` are DEPRECATED (removed) — see CLAUDE.md GPU rules.
- **⚠️ EVAL/EXPORT MASKS = gate3 / floor 0 / NO forward-only** (must match training). `cross_eval/run_cross_eval.sh`
  updated; `export_agent.py` defaults flipped to forward_only=False/floor=0. Old p2rev1-3 exports need the
  explicit `--reinforce-forward-only --reinforce-garrison-floor 10`.
- **EARLY READ (2026-06-12, 5-pt held-out Ajay trend @ ~4M train):** WR dropped off the 500k high then
  STABILIZED in the ~3% band — `500k 4.7 · 1M 2.7 · 1.5M 2.0 · 2M 3.1 · 2.5M 3.1` (not collapsing, but Ajay
  is a noisy non-LB-predictive guardrail). **peel_WON DEAD FLAT ~0.63 across all 5** (winner 0.43) → the hold
  gap is NOT closing. Training diag drifts the WRONG way: late reinforce `>100` stuck 0.73–0.80 (winner 0.31),
  `garrfrac@50` 0.59→0.71 + `shipspp@50` 38→42 (over-garrison INTENSIFYING), **`H_tgt` rising 1.12→1.71** (the
  "where-head going uniform" canary) ⇒ reinforcement is **mis-TARGETED, not under-rate** — reproduces the
  06-11 finding, points at the **THREAT HEAD** (self-supervised P(lost within K)) as the principled fix.
  Training mechanically healthy (EV ~0.85, KL low, clip ~0.19). **DECISION (user, 2026-06-12): let it run,
  re-evaluate at the 6M checkpoint** before committing to cut+threat-head.
- **6M RE-CHECK (2026-06-12) — the 4M pessimism was PREMATURE; run turned a corner ~3M.** 11-pt Ajay trend:
  WR troughed at 1.5M (2.0%) then RECOVERED/climbed — `3M 3.5 · 3.5M 4.3 · 4M 5.9 · 4.7M 4.3 · 5.2M 4.7 ·
  5.77M 5.1` → now a ~4.5–5% band (peak 5.9, above the 500k base, near best-ever Ajay; rev38 3.1%→994 LB).
  Health improved: `peel_WON` 0.63→~0.57 mid-run (wobbled back to 0.61–0.63 last 2 pts), `fire_frac`
  0.44→0.32 (less spray), `shipspp@50` 42→33 (over-garrison REVERSED), self-play `pl@100` 8→9, `owned`→10.8.
  **BUT unresolved:** `H_tgt` STILL rising → 2.22 (highest), late reinforce `>100` still 0.79 — so the WR
  gains come from better expansion/conversion/less-spray, NOT from fixed reinforce *targeting*. **DECISION
  (user): CONTINUE to 10M + harvest; threat-head = the NEXT run** for the unresolved H_tgt/targeting. Train
  ~6.7M, healthy (EV 0.87, KL low, clip crept 0.19→0.23 — minor, watch). ⚠️ At 10M the run auto-stops but the
  SPOT box keeps billing — DESTROY it (`jl destroy 425730 --yes`) after final ckpts sync.
- **⭐ FULL HELD-OUT PANEL TREND (2026-06-12, 12 ckpts 500k→6.3M) — CORRECTS the threat-head conclusion.**
  The 4M/6M notes above leaned on the SELF-PLAY training diag (H_tgt, garrfrac 0.71, reinf>100 0.79); those
  are mirror-game artifacts (self-play runs to 500 steps → heavy late reinforce). The **held-out Ajay panel
  logs** (`eval_logs/*.log`, parsed via `/tmp/parse_panels.py`) tell a different, trustworthy story. Trend
  (WON-game, decision-grade):
  ```
  metric      500k→6.3M trend          vs winner    verdict
  opnWON      ~0.38 FLAT (no trend)    0.51         ✗ opening conversion STUCK — the dominant gap
  peelW       0.64→0.56 (marginal)     0.43         ✗ mid-game hold barely moved
  WR%         osc ~4.5 (peak 5.9@4M)   —            ~ not a sustained climb
  rf<50       0.15→0.09 (WORSENING)    0.29         ✗ early reinforce DECLINING, away from winner
  fwd%        49→58                    57           ✓ reinforce DIRECTION fixed (cosmetic)
  garf50/spp  0.52→0.44 / 29→23        0.54 / 22    ✓ less hoarding (cosmetic)
  rf>100      ~0.24 (NOT 0.79!)        0.31         ✓ vs Ajay reinforce is moderate/under, NOT a flood
  ```
  **Conclusion:** deb-in-pool moved reinforce-SHAPE metrics (direction, hoarding) but NOT the two outcome
  levers — **opening conversion** and **hold**. Vs the real opponent reinforce is already winner-like (fwd
  58%, rate moderate) ⇒ **the THREAT HEAD is NOT the indicated next lever.** The flat-stuck signal is
  **opening conversion (P-OPEN)** → **the SUFFICIENT-COMMIT MASK is the next run** (veto fragment launches
  `ships ≤ target defense`; §"P-OPEN levers" #2). Early-reinforce decline (rf<50 0.15→0.09) is a secondary
  concern. **Lesson: read held-out PANEL logs, not self-play diag, for behaviour conclusions** (the diag is
  a mirror artifact — cf. [[project_train_eval_sim_gap]], [[feedback_selfplay_collapse_metrics]]).
- **EVAL via the controller (2026-06-12):** the `_eval` loop auto full-256-panels each `torch_step_*p2rev5*.pt`
  vs HELD-OUT **candidate_ajay_1200** (deb is now IN training so no longer held-out; Ajay = documented primary
  metric), masks gate3/floor0/NO-forward-only → `gpu_run_artifacts/p2rev5/eval_ajay_1200.csv` + `eval_logs/`.
  Runs LOCALLY. The stale p2rev4 eval watchers (deb/zach/cross, wrong dir+opp+masks) were KILLED by the
  controller's tear-down. First eval: step 524288 in progress. (Ajay only; add a zach/ladder opp later if wanted.)
- **TWO restarts on 2026-06-12 (current log `_083532`):** (1) LR HALVED 0.0001→0.00005 — the first run
  (from p2rev4 1.5M) drifted: `clip_frac` climbed monotonically 0.157→**0.265** by iter 10 under LR 1e-4
  (KL low 0.006–0.038, EV 0.74–0.90 = LR-too-hot, not KL blowup). (2) BASE SWITCHED p2rev4 1.5M → **p2rev3
  0.5M** — p2rev4 1.5M descended from p2rev3 *4M* (drifted) AND regressed during p2rev4 (churn 16→23,
  peel 0.61→0.64); p2rev3 0.5M is the best-vs-deb point, the coherent base for a deb-in-pool run meant to
  *preserve* the early holding skill Nash later erases. Restart iter 1: resumed from the p2rev3 0.5M ckpt,
  `peak_lr=5e-05`, clip 0.077→ plateau **~0.28** by iter 11. **DECISION 2026-06-12: clip ~0.28 is BENIGN —
  let it run, do NOT chase it with more restarts.** Evidence: (a) the 1e-4→5e-5 LR halve did NOT lower the
  plateau (0.265→0.284) ⇒ LR is not the driver; (b) KL stays low (0.004–0.027, << 0.05 target), EV healthy
  (0.72–0.90), estop=0 — clip-high + KL-low + EV-stable = a subset of actions moving (p2rev3 0.5M re-learning
  hard-commit under floor=0 vs the deb peeler), NOT divergence. If we ever DO want to dampen it the indicated
  low-KL levers are entropy_coef (0.05→0.01) or ppo_epochs 2→1 — NOT LR. **WATCH instead: held-out WR + EV/KL
  actual degradation** (act only if those move). Selection PURE: held-out WR vs a DIFFERENT strong opp
  (ajay/producer — deb is now IN training, no longer the clean held-out).

---

## ⚠️ ~~LIVE INSTANCE~~ TERMINATED — p2rev4 on GCP L4 (instance deleted 2026-06-12; checkpoints synced locally through 1.5M)

- **GCP L4 `g2-standard-8` (23GB), zone asia-south1-b, name `orbit-wars-p2rev4`, IP 34.100.181.138,
  RUNNING + BILLING (~$1.13/hr).** Running **p2rev4** in tmux `training`. The ONE training delta vs p2rev3:
  **`--reinforce-garrison-floor 10 → 0`** (unblock reinforcement — veto probe showed the floor blocked 62%
  of wanted reinforces). Resume of the p2rev3 4M checkpoint (`seed_checkpoints/p2rev4_resume.pt`).
  **256 envs / 4 workers** (fits L4 23GB with `expandable_segments` — no OOM). GCP-adapted start script:
  `gpu_run_artifacts/p2rev4/start_p2rev4_gcp.sh` (repo-root paths, no `--terminate-on-done` → manual delete).
  iter 1 healthy: EV 0.726, clip 0.154, SPS 622, no OOM. Launched ~18:46 UTC 06-11.
- **⚠️ Predecessor Jarvis A100 SPOT (IP 217.18.55.74) — user stopped it; VERIFY it is DESTROYED
  (`jl list`/`jl destroy`), else it keeps billing ₹179/hr. Could not check from this box (no `jl`/key).**
- **Baseline to beat (p2rev3 4M vs deb, 32g, masks floor10):** WR 6.25%, `cap/atk-launch 0.462`,
  `planets@ 2/4/6/3` (collapses), `lost-cap 0.91`, **`launch_rate 0.068` / `fire_frac 0.34` (≈2× Isaiah
  0.036/0.17 — the spray, now measured)**, reinf ramp suppressed mid-empire (.08–.12 vs ref .30).
- **WATCH:** EV recovery + clip<0.25 (resume warmup), then the PAYOFF metrics — **`lost-cap` should FALL /
  `median-hold` should RISE** (reinforcement now unblocked → can hold) and **`reinf@13+` should climb**.
  NEW guardrail: **`launch_rate`/`fire_frac` should NOT balloon** past ~0.07/0.34 (floor=0 → more reinforce
  launches → flood risk; spray may instead FALL if better holding cuts the frantic recapture churn).
  Caveat: floor=0 lets hard-commit reinforces strip a source to 0 → if `lost-cap` WORSENS, that's over-drain
  → add a small floor back (2-3). Selection PURE: held-out win-rate.
- **Eval stack LIVE (2026-06-12, re-pointed at GCP):** launcher sync watcher (PID 34615) pulls
  checkpoints+log from the box → `gpu_run_artifacts/p2rev4/` every 180s. Three local eval watchers
  poll that dir (PY = `orbit_wars_rl/.venv`): deb full-256-panel every ckpt (`eval_debatreya.csv`),
  zach full-panel @2M bucket (`eval_zach.csv`), cross-eval 48g @3M bucket. First ckpt (500k) lands
  ~13min after iter1. Evals run LOCALLY (never on the training box).
- **⚠️ EVAL MASKS MUST MATCH TRAINING = floor 0 for p2rev4** (gate>=3, forward-only, **garrison-floor
  0**). Watchers originally inherited floor=10 from p2rev3 → mismeasured the floor-0 policy (vetoes the
  reinforces it was trained to make); FIXED 2026-06-12 in all 3 watcher scripts + run_cross_eval.sh.
  **Same gotcha at EXPORT:** export_agent defaults floor=10 — any p2rev4 export MUST pass
  `--reinforce-garrison-floor 0` (see [[feedback_phase2_export_reinforce_parity]]).
- SSH: `gcloud compute ssh orbit-wars-p2rev4 --zone=asia-south1-b`.
  Tail: `... -- "grep '^iter' ~/orbit_wars_rl/train_gpu_phase1_p2rev4_*.log | tail"`.
- **TERMINATE when done (manual — no auto):** `gcloud compute instances delete orbit-wars-p2rev4 --zone=asia-south1-b`.

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
