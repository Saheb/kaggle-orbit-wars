# Orbit Wars — Submission Log

Ordered by submission date. Checkpoint paths relative to repo root.

---

| # | Date | Checkpoint | LB Score | Zach Panel | Ajay Panel | Loss Seeds | Key delta |
|---|------|-----------|----------|------------|------------|-----------|-----------|
| 1 | 2026-05-31 | `archive/old_checkpoints/141208.pt` (Phase 0) | **894.0** | — | — | — | Phase 0 baseline |
| 2 | 2026-05-31 | *(rev7 1M)* | 750.3 | — | — | — | Phase 1 first submission (broken export) |
| 3 | 2026-05-31 | *(rev10d 1M)* | 772.4 | — | — | — | rollout=512 |
| 4 | 2026-05-31 | *(rev10d 2M)* | 600.0 | — | — | — | passive drift hurt 2M |
| 5 | 2026-06-01 | *(rev15 2M)* | 796.0 | — | — | — | expansion-coef + win-margin |
| 6 | 2026-06-02 | *(rev28 27M)* | 843.9 | 77.3% | — | 0/21 | early-capture-coef 0.30 breakthrough |
| 7 | 2026-06-02 | *(rev30 11M)* | 867.4 | 84.8% | — | 15/21 | symmetric capture + exp decay floor |
| 8 | 2026-06-03 | `gpu_run_artifacts/gcp_rev31/checkpoints/torch_step_10485760_rev31_20260603_173151.pt` | **913.2** | 84.8% | — | 16/21 | First Strike 2×t<50 — new all-time record |
| 9 | 2026-06-03 | `gpu_run_artifacts/gcp_rev31/checkpoints/torch_step_26411008_rev31_20260603_173151.pt` | pending | 84.4% | — | 18/21 | Rev31 continuation 26M |
| 10 | 2026-06-04 | `gpu_run_artifacts/gcp_rev31/checkpoints/torch_step_31490048_rev31_20260603_173151.pt` | pending | 84.4% | 2.3% | 17/21 | Rev31 31M — peak of run |
| 11 | 2026-06-04 | `gpu_run_artifacts/gcp_rev32b/checkpoints/torch_step_6815744_rev32b_20260604_112217.pt` | pending | **88.7%** | 0.8% | 20/21 | Rev32b — First Strike 4×t<20, new Zach record |

---

## Notes

- All Phase 1 exports use `--target-decode` flag
- Sanity check: always run 10/10 vs random before submitting
- Ajay panel added 2026-06-04 as harder eval metric (orbit_lite intercept aiming gives Ajay structural fleet-speed advantage)
- Zach panel saturating ~88-89%; Ajay panel is now the signal that matters
- Top 10 LB target: ~1153 (gap from current ~913 = ~240 points)
