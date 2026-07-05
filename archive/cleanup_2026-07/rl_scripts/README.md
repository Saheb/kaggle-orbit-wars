# scripts/ — one-off analysis & tooling

Scripts here are **not** part of the core train/eval pipeline (that lives one level up in
`orbit_wars_rl/`). They are diagnostics, probes, audits and dataset builders written for
specific investigations — kept for reference and reproducibility.

Run any of them as a module from the repo root:

```bash
python -m orbit_wars_rl.scripts.<name> --help
```

79 scripts, grouped by purpose:

### Replay / dataset builders (BC, preferences, returns)
`build_ajay_action_preferences`, `build_ajay_dagger_replays`, `build_conversion_bc`, `build_producer_action_bc`, `build_producer_action_preferences`, `build_producer_target_bc`, `build_producerv2_head_labels`, `build_replay_action_bc`, `build_replay_returns`, `build_short_horizon_action_preferences`, `build_snowball_bc`, `build_tempo_bc`, `collect_ajay_bc`, `conversion_from_replays`, `pretrain_new_features`, `targeted_opening_bc`

### Audits & validation
`audit_ar_action_lists`, `audit_lossmode`, `audit_replay_head_labels`, `audit_scenario_agents`, `review_submission_targets`, `validate_head_audit_candidates`, `validate_training`

### Analysis
`analyze_action_list_replays`, `analyze_deb_wins_p2rev2`, `analyze_joint_action_ranker`, `behavior_analysis`, `board_dist_ab`, `board_prod_stats`, `count_ajay_srcs`, `diagnose_opening`, `mechanics_trend`, `value_analysis`

### Probes
`expansion_probe`, `feature_parity_comet_probe`, `feature_parity_gamephase_probe`, `garrison_floor_probe`, `probe_aggregation`, `probe_overextension`, `q_head_offline_probe`, `ship_commit_probe`, `sim_gap_probe`

### Autopsies
`expansion_autopsy`, `hold_autopsy`, `transition_autopsy`

### Counterfactuals / what-ifs
`ajay_fire_spare`, `force_fire_counterfactual`, `frozen_vs_deb_torch`, `frozen_vs_pins_torch`, `reward_redteam`, `ship0_counterfactual`, `ship0_why`, `value_spare_diagnostic`

### Single-game replay tools
`fetch_analyze_top_replays`, `find_comeback_replays`, `onegame_deep`, `onegame_html`, `onegame_launches`, `onegame_load`, `onegame_reward`, `onegame_scan`, `replay_wins`

### Joint action ranker (parked AR-adjacent track)
`eval_joint_opening`, `joint_action_ranker`, `train_joint_action_ranker`

### Q-head probes
`q_head_opportunity_gate`

### Eval variants / checkpoint utils
`bc_frac`, `check_phase4_parity`, `ckpt_info`, `compare_tempo_checkpoints`, `eval_ffa_checkpoint`, `eval_vec`, `merge_panel_shards`, `migrate_pairwise_dim`, `plot_train_log`, `step_firep`

### Other
`test_seed6462`, `train_producer_action_preferences`, `train_producerv2_head_labels`

