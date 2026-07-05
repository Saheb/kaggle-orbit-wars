# Next steps

Living doc — crisp, prioritized. Discipline: **one delta per cloud run**, eval (vs Ajay) is the arbiter.
Full session history archived in [`archive/docs/next-steps-legacy.md`](../archive/docs/next-steps-legacy.md).

**Current focus:** UNFILTERED Jake BC seed is built + validated (best BC ever); two PPO runs in flight from it
against a producer-pressure pool. Wall sharpened: we under-deploy *available* force **by choice**, not capacity.

---

## 🟢 Now — UNFILTERED Jake BC seed built + 2 PPO runs LIVE (2026-06-20)

**The unfiltered seed (DONE).** Rebuilt the Jake BC on **all 442 Jake wins** (full corpus, not the 267
attack-heavy "decisive" subset) with **NO move filters** — every "hopeless/late/already-safe/reverse-edge"
judgment OFF (those are *our* heuristics, and they were deleting real Jake reinforces). Only sub-1% technical
drops (unresolved-target / source-not-in-slots). Built at 15-dim globals (game-phase) + 22 pairwise.
Dataset: `gpu_run_artifacts/bc_rp_20dim/replay_action_bc_jake_unfiltered_allfeat.pkl` (33,051 samples).
- **Results (best BC ever): Zach 77.3% · Ajay 8.2% · random 100%** (prior BC seeds were 0/256 vs planners).
- Recovered Jake's reinforce-heaviness that filtering destroyed: **reinf@<50 0.10 → 0.21–0.25** (ref 0.29);
  reinforce ramp by empire size 0.00→0.66, smooth/monotonic (real Jake, not noise).
- **Gate settled at 2** (Jake reinforces 0% at 1 planet → gate2 deletes nothing; gate3 would delete 133 real
  reinforces). `--fire-pos-weight 6.2` (13.87% fire rate). Checkpoint:
  `gpu_run_artifacts/bc_rp_20dim/checkpoints/bc_jake_unfiltered_pw6.2_20260620.pt`.

**Two PPO runs LIVE from this seed** (Jarvis A100-80GB spot; same recipe as `jake_h1012` but unfiltered seed):
prod-share 1.0 reward, critic-warmup bridge (BC has no value head), IL-anchor to the seed, lr 1e-4, gate2 + cd3,
external pool = **producer_h12 (42%) + h1166_peak (39%) + lb1084 (37.5%)** @ 60% PFSP (h12 = the wall pressure).
1. **`jake_unfilt_h12`** — r128/ppo2/512env (jake_h1012 baseline regime). Instance 430878.
2. **`jake_unfilt_h12_r32`** — r32/ppo1/1024env (light/fast-update A/B). Instance 430882.
Bar = beat `jake_h1012`'s Ajay trajectory; wall metric = out-massed% ↓. Scripts in `gpu_run_artifacts/jake_unfilt_h12{,_r32}/`.
**DESTROY both boxes when done** (`jl destroy 430878 --yes`, `jl destroy 430882 --yes`).

**UPDATE (2026-06-21) — re-anchor + pool swap at 6M.** Both legs hit a WR-climbing-but-out-massed-flat plateau
(Ajay r32 26.2% / h12 18.0% @ 4M, open<50 cap/atk → winner 0.51; out-massed dead flat 91–95% — no gradient
targets it, `decis 0.00`). Decision: at each leg's 6M checkpoint, **resume + re-anchor IL-ref to that same 6M
checkpoint** (BC is now outgrown: Ajay 8.2% vs live 26%, il_kl rising 0.33→0.54 = the BC anchor is a drag;
corrpack3 self-anchor precedent). And **swap the stale heuristics**: drop lb1166 + lb1152, add **h14 + deb**,
keep h12 → new external pool = **{producer_h12, producer_h14, debatreya_1300}**.
- **r32 DONE (217.18.55.16) — anchored from 5M, NOT 6M.** First tried 6M (`torch_step_6094848`) but the full
  panel confirmed it **collapsed: Ajay 11.7%** (out-massed 97%, planets@50 WON 7) vs **5M's 25.4%** (out-massed
  94%, open<50 cap/atk 0.57 = its peak) — WR more than halved, so anchor from 5M.
  NOTE: critic-warmup did NOT catch the collapse (6M recovered EV 0.43→0.92 in 3 rollouts, identical to 5M) —
  it's blind to policy-side collapse (critic still explains the degenerate policy's returns). **The Ajay panel
  is the real re-anchor gate; eval the candidate ckpt before committing.** Killed the from-6M relaunch at iter 9 (no checkpoints produced → nothing to clean) and
  re-launched off **`torch_step_5079040`** (resume + IL re-anchor both on 5M), fresh pool {h12,h14,deb}
  (4 members), log `train_gpu_phase1_jake_unfilt_h12_r32_reanchor5M_nowarmup_*.log`. **Critic-warmup REMOVED**
  on re-anchor: it's only for BC seeds (no value head); a PPO resume already has a trained critic (warmup
  auto-completed in 3 rollouts → vestigial). Drop `--critic-warmup-*` on all resume/re-anchor launches. Pool siblings for
  BOTH 5M and 6M moved to `…pt.preswap_bak` (resume **appends** externals, never removes — MUST move the
  sibling aside or you get 5 externals). deb (`candidate_debatreya_1300.py`) is self-contained (no orbit_lite);
  h14 already on box. Both staged + import-checked on both boxes.
- **h12 DONE (217.18.55.39) — anchored from 6M.** Unlike r32, h12's 6M was its **best** (Ajay 4M 18.0 → 5M
  dip 14.1 → **6M 21.5%**, open<50 cap/atk 0.52), so panel-eval picked 6M. Re-launched off `torch_step_6291456`
  (resume + IL re-anchor on 6M), fresh pool {h12,h14,deb}, no warmup, r128/ppo2/512env regime, log
  `train_gpu_phase1_jake_unfilt_h12_reanchor6M_nowarmup_*.log`, pid 9381. 6M pool sibling moved to `.preswap_bak`.
  (Lesson applied: eval BOTH candidates and anchor from the winner — r32→5M, h12→6M.)
- **deb REMOVED from r32 (2026-06-21) — win-starvation.** At 1M steps the pool snapshot showed deb WR
  **0.18 / ema 0.16 but pfsp_w 0.713** — PFSP (up-weights opponents you lose to) handed deb 71% of the external
  slice while we beat it ~16%, starving the gradient. h14 (WR 0.36) is the learnable sweet-spot. Re-launched r32
  from 5M with pool **{h12, h14}** only (log `…_reanchor5M_nodeb_*.log`, pid 9453). deb may be too hard *yet* —
  revisit once we clear h14. **h12 SAME — deb dropped too:** confirmed identical starvation at ~1M (deb WR 0.19 /
  pfsp_w 0.676 vs h14 0.38); pulled the 3-opp run's 1M ckpt locally (`torch_step_1048576_…_215921.pt`) for the
  record, then re-launched from 6M with {h12, h14} (log `…_reanchor6M_nodeb_*.log`, pid 11487). Both runs now
  on **{h12, h14}**, no deb, no warmup.
- **GOTCHA — step counter resets:** `--resume` loads **weights only**, so each re-launch is a *fresh 10M-step
  phase* (log starts at `steps 32,768`), new timestamp suffix on checkpoints. `eval_ajay_1200.csv` gets a
  **second arc** climbing from low again (policy is strong, so it starts near 26%, not 3.5%); watcher matches
  on run-name so it auto-syncs/evals the new arc. Critic EV dips early (warmup recalibrating to deb/h14).
- **Watch:** out-massed% ↓ (the prize, now that deb/h14 forward-projection pressure is in the pool) and that
  open<50 cap/atk doesn't regress from the re-anchor. Tripwire: pool WR ↑ while Ajay flat = overfit.

**Open:** validate the reward-attribution change (`torch_env._fleet_target_idx`, lead-collision) — these runs
are the fresh-run check. Recompute stale top-player refs (Isaiah/TonyK in [`metrics.md`](metrics.md)).

**Keep `reinforce_forward_only = False`** — it would ban ~21% of Jake's real reinforces. Lever is the *policy*
reinforcing forward early, not a mask. **cd3/min-ship-bin inference A/B = flat** (7.0 vs 8.2 Ajay; veto removes,
doesn't teach; min-ship-bin can't fix ship0 — it's garrison-limited 1-ship sends, not head miscalibration).

---

## 🔵 The wall — curriculum, not reward proxies

Root cause **sharpened (2026-06-20, single-game replay dig-in vs Ajay)**: we don't lack force — we
**under-deploy *available* force by choice.** Of single-source contested launches that fall short of the
floor, **72%** had ≥1 *other* owned planet with spare garrison **in range** (avg 2.4 such sources), and
**65%** of those would have crossed the floor if we'd piled them on. Only 28% are real capacity limits.
Multi-source concentration *works when we use it* (76% cross, 81% hold-in-wins) but we deploy it on only
**7%** of contested targets and **cap at 2 sources** (winners go 4–5). So the fix isn't more force or a new
feature — it's a policy habit: **concentrate from more sources, more often, sustained.** The loss signature:
total garrison can exceed the enemy's yet we lose because it's *dispersed* (e.g. 518 garr vs 80, eliminated
by a single concentrated 269-ship wave). Tools: `gpu_run_artifacts/{analyze_one_game,multi_source_events,multi_source_why}.py`.
Prior framing (out-massed on defendable captures; opening `<50` cross ~0.23; survives PPO phase4e≈BC) still holds.

- **🟢 Curriculum / opponent-relative — make holding-via-concentration NECESSARY to win.** Boards where a
  capture is only holdable if you concentrate the defensive response; a contested opening where single-planet
  probes die so only multi-source survives. Preferred over a defensive hold-floor reward (= decmass mirror →
  same decoupling). Design notes: [`docs/peeler-curriculum.md`](peeler-curriculum.md),
  [`docs/scenario-curriculum.md`](scenario-curriculum.md).
- **The lever is sizing/sufficiency as a SIGNAL, not a reward proxy** — match send to the capture floor (both
  directions: stop under-committing big rich neutrals AND over-committing small ones). Full reasoning:
  [`docs/targeting-vs-sufficiency.md`](targeting-vs-sufficiency.md).

### Ruled out (don't re-litigate — see links for the autopsies)
- ❌ **Reward proxies** (decmass / caputil / consolidation / defense_coef / production) — exhausted, gameable.
  [`docs/outmass-limits.md`](outmass-limits.md)
- ❌ **Targeting / over-extension / action grammar** — we aggregate and target at winner rates.
  [`docs/targeting-vs-sufficiency.md`](targeting-vs-sufficiency.md)
- ⏸ **Per-planet / VDN value head** — only revisit *paired with a concentration signal*.
  [`docs/conclusions.md`](conclusions.md)

---

## Reference
- Metrics & what to trust (incl. the 2026-06-20 fix): [`docs/metrics.md`](metrics.md)
- Run/eval commands: [`docs/commands.md`](commands.md) · Training state/config: [`docs/training.md`](training.md)
- LB submissions log: [`docs/submissions.md`](submissions.md) · GPU: [`docs/GCP_RUNBOOK.md`](GCP_RUNBOOK.md) · [`docs/JARVIS_RUNBOOK.md`](JARVIS_RUNBOOK.md)
