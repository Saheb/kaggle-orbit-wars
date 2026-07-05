# Final Kaggle Submissions — Preserved Artifacts

The two final daily submissions (2026-06-24), recovered from the `orbit-audit` worktree
(`/Users/saheb/home/orbit-audit`) on 2026-07-05 before the post-competition cleanup.
These are the ONLY artifacts the cleaned codebase promises to keep runnable.

| File | What it is |
|---|---|
| `submission_presres05.tar.gz` | Kaggle sub **53991660** — exact submitted bundle. 2p = presres1 0.5M neural ("decisive"), 4p = ajay (orbit_lite ProducerLite). Self-contained: `main.py` router + `neural_agent.py` (base64 weights) + `ffa_agent.py` + `orbit_lite/`. |
| `submission_stgpr1.tar.gz` | Kaggle sub **53991662** — exact submitted bundle. 2p = stgpr1 0.5M ("spray"), 4p = ajay. Same structure. |
| `presres1_0.5M_backfilled_resolver.pt` | The checkpoint sub 53991660 was exported from: `torch_step_524288_presres1` with `pressure_precise_resolver` backfilled into cfg (the ppo.py cfg_blob persistence bug meant the raw save lacked the flag). Use THIS one to reproduce the export. |
| `presres1_0.5M_raw.pt` | Same training step, raw save without the backfilled flag (kept for provenance). The cleaned code's blessed-config guard treats the absent flag as OFF and REFUSES this file — use the backfilled copy (or the `pre-cleanup-2026-07` tag). |
| `stgpr1_0.5M.pt` | The checkpoint sub 53991662 was exported from: `torch_step_524288_stgpr1`. |

Notes:
- The tarballs run standalone under kaggle_environments — no repo code needed. Prefer them
  as frozen reference opponents.
- Feature semantics for all three .pt files: resolver ON, gate2/floor0/no-forward,
  reverse-edge cd3, game-phase 15-global, target-decode. Under the cleaned (blessed-config)
  code, `presres1_0.5M_backfilled_resolver.pt` and `stgpr1_0.5M.pt` pass the guards and
  load/eval/export directly; `presres1_0.5M_raw.pt` is refused (absent resolver flag = OFF).
  For anything the guards refuse, use the `pre-cleanup-2026-07` git tag.
- Panel numbers at submission time: presres1 0.5M — Ajay 53.9% (256g); stgpr1 0.5M — Ajay
  57.4% but spray/churn-inflated. See docs/submissions.md entries 53991660/53991662.
