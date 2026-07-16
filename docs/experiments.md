# Experiment Queue

One line per experiment, in rough priority order. One change per run; record hypothesis in
`docs/training.md` before launching, verdict after. Details live in `docs/writeup_lessons.md`.

| Frontier | Status | Decisive evidence | Decision |
|---|---|---|---|
| Exact-marginal binary 40.108M | **Ajay baseline** | 80.5% Ajay · 3.9% Yijie | Retain; best Ajay checkpoint |
| Target counterfactual 45.711M | Complete | 74.2% Ajay · **5.9% Yijie** | No overall promotion; best Yijie read |
| Target+source counterfactual + L4 25.068M | Complete | 75.8% Ajay · 3.9% Yijie | Added source channels active; no promotion |
| Forced projected hold | **Rejected** | 1/16 vs 13/16 all-in on paired Ajay slice | Underprices the opponent response |
| Submitted-agent cross-eval | Audit | Old 256/256 claims invalid; corrected gates 12/16 and 14/16 | Full panels must be rerun |
| Best-checkpoint anchor | **Next** | Preserve a proven policy while testing one structural delta | Add anchor + promotion gate |

## Active validation

- **Submitted-agent cross-eval integrity audit** — rerun the 256-game panels against the exact
  tarball payloads. The old wrappers errored before acting, so their 100% results are void. The
  evaluator now requires `DONE/DONE` and hashes the tracked archives before play.

## Next in line

1. **Best-ckpt anchor + promotion gate** — KL/value-CE anchor to previous best, adopt on >70% h2h; the missing piece for stable 100M+ self-play (Isaiah).
2. **noop-KL A/B** — same-seed/same-steps on-vs-off; mechanism verified, improvement still confounded.
3. **Learned middle commitment** — only if the policy chooses it. A forced 24-step
   no-new-launch projected-hold decoder failed 1/16 versus 13/16 all-in on a paired Ajay slice;
   it underprices the opponent's next launch and must not be used as the execution contract.
4. **Combat-preview scalars** — endpoint owner/ships/flip-margin per planet (Jake); cheap add-on to the timeline, covers the one thing it doesn't hand over (margin).
5. **Conv1d timeline encoder** — SimJeg's 1D-CNN into the planet token; only if timeline signal looks bottlenecked by the linear projection.
6. **Surrender / early-truncation** — cut compute on decided games (Jake: 60–70% of turns); sample-density multiplier, not raw SPS.
7. **Capacity jump** — `--entity-dim`/`--num-layers` up (0.5M → 5–20M); needs H100/H200 (update-bound: bigger model = proportionally slower).
8. **Exploiters** — train a fresh model purely to beat the main one, fold into the league (Ender/rank-55: +15–17pp first-place).

## Parked / conditional

- **Zero-pad warm-start** — presres1 + zero-padded timeline columns; only as a fast confounded side-signal, never the primary run.
- **4p variant work** (separate models, per-player value heads — Jake) — after 2p is competitive.
