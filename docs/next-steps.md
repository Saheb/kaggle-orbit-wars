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
