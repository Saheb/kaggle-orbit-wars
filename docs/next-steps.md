# Next steps

Living doc — crisp, prioritized. Discipline: **one delta per cloud run**, eval (vs Ajay) is the arbiter.
Full session history archived in [`archive/docs/next-steps-legacy.md`](../archive/docs/next-steps-legacy.md).

**Current focus:** rebuild the BC seed on **corrected target labels** (the 2026-06-20 fix), then attack the
force-concentration wall with **curriculum**, not reward proxies.

---

## 🟢 Now — corrected-label BC seed (2026-06-20 metric fix)

The launch→target resolver was angle-only (≈56–66% correct). Fixed to lead-collision (≈96–98%) across
eval, the training reward, and the **BC label builder**. This corrupted what we were steering by *and* what
the BC learned. Details + corrected references: [`docs/metrics.md`](metrics.md) (top banner).

1. **🟢 Rebuild the Jake BC dataset + retrain.** `build_replay_action_bc.py` now emits correct target labels
   (55.7%→95.7%) and save/attack split (75.5%→98.0%), so the proactive/save-quality/attack-value filters
   finally curate the right moves. Diff the new label set vs old; retrain; check **open<50 cap/atk-launch
   moves off 0.56 → Jake's 0.70** and **@4-6 reinforce share off 0.10 → 0.52**. Builder details:
   [`docs/replay-action-bc.md`](replay-action-bc.md).
2. **🟢 Validate the reward-attribution change** (`torch_env._fleet_target_idx`, now lead-collision). It
   alters the decisive-mass reward + incoming features — needs a **fresh run + SPS check** before trusting.
   Parity test: `orbit_wars_rl/tests/test_fleet_target_lead.py`.
3. **🟡 Recompute the stale top-player refs** (Isaiah/TonyK rows in [`metrics.md`](metrics.md)) via
   `conversion_from_replays.py` — they're still on the old resolver. Jake already corrected.

**Keep `reinforce_forward_only = False`** — confirmed: it would ban ~21% of Jake's real reinforces (27% late
game). The lever is the *policy* reinforcing forward more early, not a mask. [`metrics.md`](metrics.md) banner.

---

## 🔵 The wall — curriculum, not reward proxies

Root cause (unchanged): we get **out-massed** on defendable captures; the opening land-grab is under-floor
(`<50` cross ~0.23). Now *quantified honestly*: opening capture efficiency is the real gap (ours 0.54 vs
Jake 0.70) and it **survives PPO** (phase4e ≈ BC seed) → not a PPO-tuning problem.

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
