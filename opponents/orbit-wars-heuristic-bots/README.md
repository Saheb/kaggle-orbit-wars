# Orbit Wars - Heuristic Bots

My personal archive of the heuristic bots I iterated and submitted to the [Kaggle Orbit Wars competition](https://www.kaggle.com/competitions/orbit-wars). Every file here is one of my own submissions across ~14 major versions and dozens of patches. Sharing as baselines.

## The Heuristic Ceiling

My heuristic line tops out around **LB 1166**. I climbed from ~LB 500 to a peak of ~1166 (v13.3 full stack), and after that every architectural attempt I tried — self-play parameter tuning, formula-based thresholds, additional state-derived gates, personality mode arbitration — landed in the same 1080–1160 band.

I only broke past it by switching paradigms to RL self-play (LB 1349 with my `rl_v12_upd559`). RL bots are not included in this archive; this is the heuristic-only ceiling I personally hit.

## Bots (my submission lineage)

| # | File | Final LB | Notes |
|---|------|----------|-------|
| 01 | `01_v11_3_lb1012_first_above_1000.py` | 1012.2 | My first version to clear LB 1000. |
| 02 | `02_v12_6d_lb1043_simple_clean.py` | 1043.0 | Clean expand/attack agent. |
| 03 | `03_v12_7m_lb1084_4p_relative_gap_hammer.py` | 1084.6 | 4P attack trigger keyed to the gap between me and the strongest other player. |
| 04 | `04_v12_8ez_lb1016_psm_baseline.py` | 1016.9 | Baseline with personality-mode arbitration (patient / opportunistic / pressure). |
| 05 | `05_v12_9_v19_lb1050_F_stack_aggregation.py` | 1050.9 | Per-turn fleet-intent aggregation, three-bucket dispatch, candidate diversity scoring. |
| 06 | `06_v13_2y_lb1055_reinforce_front_stack.py` | 1055.9 | Reinforce-front behavior + wider opening + smarter reinforce-target selection. |
| 07 | `07_v13_3_P2_P3_lb1141_correctness_patches.py` | 1141.5 | Added aim sanity-check stand-pat (2P only) + fail-tolerant order cleanup. |
| 08 | `08_v13_3_R8_full_stack_lb1166_PEAK_HEURISTIC.py` | 1166.7 | **My peak heuristic.** v13.2y + 15 patches. The biggest single LB jump was teaching the evaluator that the enemy can reactive-snipe my landings. |
| 09 | `09_v14_0_cmaes_lb1114_overfit_reverted.py` | 1114.4 | CMA-ES tuned 36 constants on self-play. +22pp local validation lift, but −52 LB. Reverted to v13.3. |
| 10 | `10_v14_1d_lb1135_phase3_stop_expand.py` | 1135.5 | Stop-expanding gates (production lag, enemy tempo, stockpile ready, 4P production lead), evacuate-doomed-planets, production-aware coordinated attack. |
| 11 | `11_v14_1n_lb1138_doom_evac_mega_hammer.py` | 1138.3 | Phase 3.5 stop-expand gates + evac-then-attack-fallback in 4P + chained coordinated attack with fresh-capture inheritance. |
| 12 | `12_v14_4c_lb1099_2p_focus_enemy_bias.py` | 1099.8 | 2P bias toward targeting the enemy directly (close distance, attack threshold, large attack threshold). Local gains didn't transfer to LB. |
| 13 | `13_main_k_v1_lb1099_math_aware_rewrite.py` | 1099.5 | `main_k.py` — fast-elimination logic + 15 formula-based thresholds (replacing hardcoded constants) + full 4P enemy intelligence. |
| 14 | `14_main_k_v2_lb1152_LAST_HEURISTIC.py` | 1152.8 | `main_k_v2.py` — informed by replay analysis of top-10% bots (2631 episodes from bovard's dataset), ~40 formula-based thresholds, patches K9 through K16. My last pure-heuristic submission before pivoting to RL. |

## Notes from My Iteration

- **Self-play A/B punishes "do less" patches.** Many skip-rules that lost in my self-play A/B harness actually helped on the LB. A mirror-match harness systematically penalizes restraint because both sides restrain symmetrically.
- **Personality mode arbitration was net-zero for me.** Direct A/B at n=384 of `PERSONALITY_ENABLED=False` vs `True` showed exact parity in 2P and 4P. 4 of 5 published top Planet Wars bots I studied have no mode arbitration at all.
- **CMA-ES on self-play overfit my harness.** My v14.0 was the cleanest case: +22pp local validation lift, −52 LB. If you tune, validate on a held-out opponent set you've never optimized against.
- **Fixing the evaluator beat tuning aggression.** My biggest single LB jump (~+25 LB) came from making the evaluator account for the enemy's reactive snipe — the bot had been launching into a trap because its projection ignored that response.
- **My local A/B → LB conversion ran ~3-5 LB per pp.** At n=192 the 4P confidence interval is ±6-8pp, so single A/Bs are barely above noise. I had to gate-and-retest at n=384.
- **Engine facts that took me longer than I'd like to confirm:** every planet produces 1-5 ships per turn (production=0 never exists, despite some external research claiming "white = zero-prod"); win condition is sum of ships per player, not planet count; in 4P, comet hits and sun-burn dominate variance at the noise floor.
- **Reading other bots' source paid off.** I read the source of 5 published top Planet Wars bots (Melis, Khramtsov, zvold, Rendon, Lucas) and surfaced 6 recurring techniques my line didn't have: speculative apply/undo, asymmetric MIN-TURN-TO-DEPART, stale-order cleanup with `wereCleaned` flags, knapsack opening, reinforcer/front classification, and zero mode arbitration.
