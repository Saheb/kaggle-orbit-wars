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
| 53451535 | 2026-06-07 | `gpu_run_artifacts/rev38/checkpoints/torch_step_5242880_rev38_20260605_181635.pt` (re-export) | pending | **92.6%** | TBD | — | **Rev38 5M + FIXED intercept aimer** — identical policy to the 950.5 record, only the inference aimer changed (aim-benchmark 73%→95%; Zach 89.1→92.6%). Tests aimer fix alone vs 950.5 |

---

## Notes

- All Phase 1 exports use `--target-decode` flag
- Sanity check: always run 10/10 vs random before submitting
- Ajay panel added 2026-06-04 as harder eval metric (orbit_lite intercept aiming gives Ajay structural fleet-speed advantage)
- Zach panel saturating ~88-89%; Ajay panel is now the signal that matters
- Top 10 LB target: ~1153 (gap from current ~918 = ~235 points)
