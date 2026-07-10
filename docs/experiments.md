# Experiment Queue

One line per experiment, in rough priority order. One change per run; record hypothesis in
`docs/training.md` before launching, verdict after. Details live in `docs/writeup_lessons.md`.

## Running

- **tl100m** — projected-timeline features, from scratch, 100M sparse self-play on L4; verdict = Ajay back-half trend (wandb `gfiwzpf4`).

## Next in line

1. **Best-ckpt anchor + promotion gate** — KL/value-CE anchor to previous best, adopt on >70% h2h; the missing piece for stable 100M+ self-play (Isaiah).
2. **noop-KL A/B** — same-seed/same-steps on-vs-off; mechanism verified, improvement still confounded.
3. **fire=0 target-credit fix** — mask target log-prob from the PPO joint when fire=0; keeps our unique target-first edge, kills the ~90% no-op gradient noise.
4. **Intent ship sizing** — 100% / capture-defend / maintain bins + resolved-size table as features (Jake); timeline.py already does the math.
5. **Combat-preview scalars** — endpoint owner/ships/flip-margin per planet (Jake); cheap add-on to the timeline, covers the one thing it doesn't hand over (margin).
6. **Conv1d timeline encoder** — SimJeg's 1D-CNN into the planet token; only if tl100m shows signal but looks bottlenecked by the linear projection.
7. **Surrender / early-truncation** — cut compute on decided games (Jake: 60–70% of turns); sample-density multiplier, not raw SPS.
8. **Capacity jump** — `--entity-dim`/`--num-layers` up (0.5M → 5–20M); needs H100/H200 (update-bound: bigger model = proportionally slower).
9. **Ender counterfactual timeline** — target head sees the projection *assuming the candidate fleet launches*; his fix for launching too-early/too-late.
10. **Exploiters** — train a fresh model purely to beat the main one, fold into the league (Ender/rank-55: +15–17pp first-place).

## Parked / conditional

- **Zero-pad warm-start** — presres1 + zero-padded timeline columns; only as a fast confounded side-signal, never the primary run.
- **4p variant work** (separate models, per-player value heads — Jake) — after 2p is competitive.
