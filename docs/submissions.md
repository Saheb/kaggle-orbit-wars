# Orbit Wars — Submission Log

Ordered by submission date. Checkpoint paths relative to repo root.

---

| Sub ID | Date | Checkpoint | LB Score | Zach Panel | Ajay Panel | Loss Seeds | Key delta |
|--------|------|-----------|----------|------------|------------|-----------|-----------|
| 53076736 | 2026-05-27 | `archive/old_checkpoints/141208.pt` (Phase 0) | **894.0** | 55.5% HB | — | — | Phase 0 baseline, target-decode fix |
| 53205031 | 2026-05-31 | *(rev7 1M)* | 743.7 | 50.8% | — | — | Phase 1 first working export |
| 53206201 | 2026-05-31 | *(rev10d 1M)* | 791.6 | — | — | — | rollout=512 |
| 53207108 | 2026-05-31 | *(rev10d 2M)* | 724.6 | — | — | — | passive drift hurt 2M |
| 53233951 | 2026-06-01 | *(rev15 2M)* | 796.3 | 69.9% | — | — | expansion-coef + win-margin |
| 53322890 | 2026-06-03 | *(rev28 27M)* | 843.9 | 77.3% | — | 0/21 | early-capture-coef 0.30 breakthrough |
| 53330956 | 2026-06-03 | *(rev30 11M)* | 866.3 | 84.8% | — | 15/21 | symmetric capture + exp decay floor |
| 53336058 | 2026-06-03 | `gpu_run_artifacts/gcp_rev31/checkpoints/torch_step_10485760_rev31_20260603_173151.pt` | **918.8** | 84.8% | — | 16/21 | First Strike 2×t<50 — all-time record |
| 53352108 | 2026-06-04 | `gpu_run_artifacts/gcp_rev31/checkpoints/torch_step_26411008_rev31_20260603_173151.pt` | 917.0 | 84.4% | — | 18/21 | Rev31 26M |
| 53358128 | 2026-06-04 | `gpu_run_artifacts/gcp_rev31/checkpoints/torch_step_26411008_rev31_20260603_173151.pt` | 875.7 | 84.4% | — | 18/21 | Rev31 26M + invalid-target mask fix (regressed) |
| 53359633 | 2026-06-04 | `gpu_run_artifacts/gcp_rev31/checkpoints/torch_step_31490048_rev31_20260603_173151.pt` | 914.5 | 84.4% | 2.3% | 17/21 | Rev31 31M — peak of run |
| 53367932 | 2026-06-04 | `gpu_run_artifacts/gcp_rev32b/checkpoints/torch_step_6815744_rev32b_20260604_112217.pt` | 874.3 | **88.7%** | 0.8% | 20/21 | Rev32b — First Strike 4×t<20, new Zach record |
| 53410563 | 2026-06-06 | `gpu_run_artifacts/rev38/checkpoints/torch_step_5242880_rev38_20260605_181635.pt` | **950.5** | 89.1% | 2.7% | — | Rev38 5M — pairwise=15 features (roi_20/roi_50/enemy_contest), rollout=128 — new record |
| 53416454 | 2026-06-06 | `gpu_run_artifacts/rev38/checkpoints/torch_step_6291456_rev38_20260605_181635.pt` | 924.5 | — | 3.1% | — | Rev38 6M — best Ajay panel yet (8/256) |
| 53428737 | 2026-06-06 | `gpu_run_artifacts/rev46/checkpoints/torch_step_4194304_rev46_20260606_143300.pt` | 854.6 | 86.7% | 3.1% | — | Rev46 4M — same config as Rev38, peak before carpet-bomb collapse |
| 53451535 | 2026-06-07 | `gpu_run_artifacts/rev38/checkpoints/torch_step_5242880_rev38_20260605_181635.pt` (re-export) | **993.9** | **92.6%** | 2.7% | — | **Rev38 5M + FIXED intercept aimer — ALL-TIME LB RECORD.** Identical policy to the 950.5 record, only the inference aimer changed (aim-benchmark 73%→95%; Zach 89.1→92.6%). Aimer fix alone = +46.6 LB over 947.3 (same checkpoint, old aimer). |
| 53471121 | 2026-06-08 | `gpu_run_artifacts/rev53b/checkpoints/torch_step_10485760_rev53b_20260607_181202.pt` | **933.0** (later 953.2) | — | **10.9%** | — | **Rev53b 13.6M eff — heuristic-pool selectivity + fixed aimer. LB REGRESSION despite Ajay 10.9% (~3.5× prior panel record).** Ajay panel went up 4×, LB went DOWN ~60 pts vs rev38 5M+aimer (both fixed aimer). ⇒ Ajay/1166 heuristic-ladder panel is NOT LB-predictive — same trap as Zach/srcs_multi. |
| 53527873 | 2026-06-10 | `gpu_run_artifacts/hellburner_spot/checkpoints/torch_step_1048576_rev54_20260609_155722.pt` | pending | — | 5.5% | — | Rev54 1M — early_capture training-wide anneal + metric cleanup. Panel: 5.5% Ajay / 4.7% 1300. This is the rev55 resume point (REINFORCE/reinforcement-lever run). target-decode + fixed aimer. |
| 53574885 | 2026-06-11 | `gpu_run_artifacts/p2rev2/checkpoints/torch_step_8912896_p2rev2_20260611_105500.pt` | pending | — | — | — | **FIRST Phase-2 (reinforce-enabled) submission.** p2rev2 8.91M champion: reinforce gate≥3 + forward-staging + garrison-floor10, defense_coef dropped, aggressive pool. **Export fix:** now bakes reinforce-discipline parity (gate/forward/floor) into the inference mask — without it the agent reinforces <3 planets / backward / drains source = self-sabotage (these three params are NOT stored in the checkpoint, must be passed at export). Also fixed a pre-existing export crash (`os` not imported in template; surfaced by features.py module-level `os.environ` read). target-decode. Sanity: 4/4 vs zach, 10/10 vs random. |
| 53681203 | 2026-06-14 | `gpu_run_artifacts/rev38/checkpoints/torch_step_5242880_rev38_20260605_181635.pt` (re-export) | 905.4 | 95.7% | 6.2% | — | rev38 5M re-export (compass calibration). ⚠️ **905.4 vs the 967.6 record for the SAME ckpt** — the gap is the inference aimer (this re-export ≠ the `_newaim` record build); confirms the fixed aimer is worth ~60 LB. |
| 53681209 | 2026-06-14 | `submission_stageb3_11M.py` (stageb3 11.0M) | 778.0 | — | 5.5% | — | Phase-3 stack (comets + game-phase + reinforce gate2/floor0, from-scratch). **778 ≪ rev38 → the Phase-3/Stage-B direction is BELOW rev38 on LB**, corroborating the panel-sweep finding. |
| 53716089 | 2026-06-15 | `gpu_run_artifacts/corrpack2/checkpoints/torch_step_2539520_corrpack2_20260615_093652.pt` | **891.2** | 94.5% | 11.7% | — | **corrpack2 @2.5M (post-resume PEAK)** — correctness-pack#2 (per-episode pool attribution) lineage, reinforce gate2/floor0/clamp, target-decode + fixed aimer. Run-peak panel (best Ajay of the run); **62.5% vs rev38_5M H2H, 11.3% vs deb.** Sanity 10/10 vs random, 4/4 vs zach. The pre-collapse peak corrpack3 re-anchors from. 891.2 < rev38 993.9 → reaffirms panel≠LB. |
| 53722560 | 2026-06-15 | `gpu_run_artifacts/corrpack3e/checkpoints/torch_step_4718592_corrpack3e_20260615_175903.pt` | pending | **98.8%** | **18.0%** | — | **corrpack3e 4.7M — best-ever panel (Zach 98.8% / Ajay 18.0%).** Self-anchor re-anchor from corrpack2 2.5M peak + LR 1e-4; reinforce gate2/floor0/no-forward; target-decode + fixed aimer. ⚠️ wall INTACT (out-massed ~96%) → numeric panel record, NOT structural; watch for the rev53b-style panel→LB regression. Sanity 10/10 vs random, 4/4 vs zach. Base for Lever A (decmass1). |
| 53751394 | 2026-06-16 | `gpu_run_artifacts/revedge1/checkpoints/torch_step_4718592_revedge1_20260616_095906.pt` | pending | — | **23.8%** | — | **revedge1 4.72M — strongest Ajay panel (23.8%), the Probe-A baseline.** corrpack3e + reverse-edge reinforce cooldown=3 (one delta). ⚠️ **First export to BAKE the reverse-edge cooldown** — `export_agent.py` now reads `reverse_edge_cooldown` from ckpt cfg, embeds `reinforce_cooldown.py` + aliases, applies the per-game ping-pong veto (mirrors `build_agent_fn`); without it the submission ≠ the panel agent. Also fixed: action_mask's `\`-continuation cooldown import broke `_strip_imports` (now single-line). target-decode, gate2/floor0/no-forward. Sanity 4/4 zach, 10/10 random. Panel→LB read for the strongest lineage agent (wall still INTACT, out-massed ~95%). |
| 53958174 | 2026-06-23 | `gpu_run_artifacts/phase4fs/checkpoints/torch_step_7340032_phase4fs_20260622_144408.pt` (2p) + Producer-Lite v2 (4p) | pending | — | — | — | **HYBRID 2p/4p submission (tar.gz: `main.py` + `neural_agent.py` + `producer_v2.py` + `orbit_lite/`).** `main.py` detects player count from distinct live-board owners at step 0 (`initial_planets` is all-neutral in raw obs — must count `planets`): **2p → phase4fs 7.34M neural** (gate2/no-forward/floor0, sufficient-commit 1.0, reverse-edge cd3, game-phase 15-global, target-decode), **4p → producer_v2** (its own 2p/4p CONFIG presets). ⚠️ **EXPORT BUG FIXED:** phase4fs carries COMA `q_*` Q-head params; the exported `_Model` doesn't define them so `_get_model` `load_state_dict` raised every step → kaggle swallowed it → 0 moves (sanity was 0/4 zach, 1/6 random). Fix in `export_agent.py`: strip `q_*`/`value_pp_*` before base64 + load `strict=False`. Post-fix sanity 4/4 zach; isolated-tarball test 2p win + 4p win vs 3 random. Build dir: `submission_2p4p/`. |

---

## Notes

- All Phase 1 exports use `--target-decode` flag
- Sanity check: always run 10/10 vs random before submitting
- Ajay panel added 2026-06-04 as harder eval metric (orbit_lite intercept aiming gives Ajay structural fleet-speed advantage)
- Zach panel saturating ~88-89%; Ajay panel is now the signal that matters
- Top 10 LB target: **above 1500** (~1153 is only top-100; #1 Isaiah ≈ 1751). Gap from current record **993.9** = **~500+ points** (goal 900→1500+).
- **LB record = rev38 5M + fixed aimer = 993.9** (sub 53451535). Best *lineage* to build on is rev38, not rev53b.
- **Ajay/1166 panel demoted (2026-06-09):** rev53b proved the heuristic-ladder panel is not LB-predictive (10.9% Ajay → 933 LB, below rev38's 2.7% → 993.9). Treat panels as guardrails, not the objective; the only honest LB signal is submitting.
