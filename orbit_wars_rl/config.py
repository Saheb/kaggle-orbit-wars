from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class EnvConfig:
    board_size: float = 100.0
    sun_radius: float = 10.0
    center: Tuple[float, float] = (50.0, 50.0)
    max_speed: float = 6.0
    episode_steps: int = 500
    max_planets: int = 48
    max_fleets: int = 256
    comet_speed: float = 4.0
    rotation_radius_limit: float = 50.0
    num_players: int = 2
    max_moves_per_turn: int = 8
    win_margin_coeff: float = 0.0   # terminal bonus: winner gets +1 + α*(my_score/total_score)


@dataclass
class ModelConfig:
    max_entities: int = 64
    max_owned_planets: int = 16
    # 20 base channels + 96 projected-timeline channels
    # (timeline.TIMELINE_DIM: 4 channels × 24 steps). Eval infers the width from
    # planet_proj.weight when loading an older checkpoint.
    planet_feature_dim: int = 116
    fleet_feature_dim: int = 13
    # Global features: 11 base + 4 game-phase channels (phase one-hot
    # early/mid/late + normalized steps-to-next-comet-spawn) + 48 projected
    # economy-delta channels (timeline.GLOBAL_ECON_DIM: production/material
    # delta × 24 steps). Eval infers the width from global_proj.weight when
    # loading a checkpoint from before the economy series (15-wide).
    global_feature_dim: int = 63
    entity_dim: int = 96
    num_heads: int = 4
    num_layers: int = 3
    mlp_expansion: int = 3
    num_angle_bins: int = 144
    num_ship_bins: int = 32
    # How to decode a ship-bin index into an absolute ship count:
    #   "absolute" — bin → SHIP_COUNTS[bin]  (32-entry hybrid linear-log table)
    #   "fraction" — bin → round(FRACTION_BIN_VALUES[bin] * max_ships)
    #   "binary"   — fire head chooses NOOP/COMMIT; ships are resolved deterministically
    # MUST match the BC label scheme that produced the checkpoint.
    # Default "absolute" preserves legacy checkpoint behaviour.
    ship_bin_mode: str = "absolute"
    pairwise_feature_dim: int = 36   # 22 base + 4 intent + 6 target-CF + 4 source-CF
    max_planets: int = 48            # for target_head output size; matches EnvConfig
    # Target-conditioned fire/ship residual output init scale. The phase4 name is
    # retained in the checkpoint and CLI contract. 0.0 starts the residual at zero.
    phase4_residual_init_std: float = 0.0
    # Target-decode discipline. These are persisted in checkpoints so train/eval/export
    # do not silently disagree about own-target legality or attack concentration vetoes.
    allow_reinforce: bool = False
    reinforce_gate_min_planets: int = 0
    reinforce_forward_only: bool = False
    reinforce_garrison_floor: float = 0.0
    reverse_edge_cooldown: int = 0
    sufficient_commit_factor: float = 0.0
    dropout: float = 0.0
    # Value head input width. 0 = auto (2*entity_dim concat head). Eval sets this
    # to entity_dim when loading an older mean-pool checkpoint.
    value_head_in: int = 0

    def __post_init__(self):
        # Intent sizing emits target-relative semantics rather than absolute counts.
        # Force the head width so model, PPO, and Q-head construction agree.
        if self.ship_bin_mode == "intent":
            from action_mask import NUM_INTENTS
            self.num_ship_bins = NUM_INTENTS


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    # Multiplier applied only to target-conditioned fire/ship residual parameters.
    # The phase4 name is retained for checkpoint and CLI compatibility.
    phase4_residual_lr_mult: float = 1.0
    lr_warmup_steps: int = 5000
    lr_decay: str = "cosine"
    total_env_steps: int = 500_000_000
    batch_size: int = 2048
    num_minibatches: int = 4
    # Two epochs is the throughput-oriented default; see docs/perf.md. Override
    # per run with --ppo-epochs when additional sample reuse is worth the cost.
    ppo_epochs: int = 2
    clip_eps: float = 0.2
    gamma: float = 0.995
    gae_lambda: float = 0.95
    entropy_coef_fire: float = 0.01
    entropy_coef_target: float = 0.02   # entropy bonus on the target head (was misnamed entropy_coef_angle)
    entropy_coef_ships: float = 0.01
    # No-op KL bias: pull the batch-mean launch rate toward a low
    # prior so the policy saves ships instead of spraying. 0 = off. Adds to (not replaces)
    # the fire entropy bonus. See docs/writeup_lessons.md Lesson 3.
    noop_kl_coef: float = 0.0
    noop_target_launch_rate: float = 0.10   # target mean fire probability the KL anchors to
    # Ship-size KL-to-prior: replace the uniform-seeking ship entropy bonus with a
    # KL from the ship-count distribution toward a full-send-biased prior over the 32 bins
    # (w_i ∝ SHIP_COUNTS[i] ** ship_kl_prior_exp). Unlike noop_kl (batch-MEAN launch RATE), this
    # shapes each per-draw SIZE distribution — it starves the 1-3 ship spray tail while keeping
    # small bins learnable (reward can still buy a genuine probe). 0 = off. When ON, set
    # entropy_coef_ships=0 because this replaces rather than stacks with entropy.
    ship_kl_coef: float = 0.0
    ship_kl_prior_exp: float = 1.0   # prior w_i ∝ SHIP_COUNTS[i]**exp; 1.0=linear-in-count, higher=more full-send-biased
    # Best-checkpoint ANCHOR (Isaiah #1 / Yijie #13; docs/training.md "The recipe"): KL from the
    # live policy to the frozen previous-best, plus a value-CE term. Unanchored self-play has
    # nothing pulling it back toward known-good play, so it drifts (the noopkl2 0% collapse);
    # anchoring converts that drift into bounded oscillation near the best, and the promotion
    # gate (train_torch --anchor-promote-*) ratchets the best upward. 0 = off (both terms).
    # Costs one extra no-grad forward per minibatch.
    anchor_kl_coef: float = 0.0
    anchor_value_coef: float = 0.0
    kl_target: float = 0.05   # KL early-stop threshold per epoch; inf = disabled
    value_coef: float = 0.5
    # Critic-only warmup (for BC warmstarts: trained policy + UNtrained critic).
    # Before normal PPO, freeze the trunk + policy heads and train ONLY the value
    # head until explained-variance reaches critic_warmup_ev (so PPO never trusts a
    # random critic's advantages and unlearns the BC policy). 0 = disabled. Adaptive
    # threshold self-skips on a warm-critic resume (EV already high → 0 warmup steps).
    critic_warmup_ev: float = 0.0
    critic_warmup_max_updates: int = 30   # safety cap if EV never reaches the threshold
    # Note: env reward-shaping coefficients are CLI args wired directly to VecTorchEnv
    # (see train_torch.py) — PPOConfig is not the right owner for them.
    max_grad_norm: float = 0.5
    clip_value: bool = True
    normalize_advantages: bool = True


@dataclass
class SelfPlayConfig:
    opponent_pool_size: int = 8
    opponent_sample_prob_old: float = 0.3
    eval_interval_steps: int = 10_000_000
    eval_num_games: int = 32
    checkpoint_interval_steps: int = 10_000_000
    num_env_workers: int = 4


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    seed: int = 42
    wandb_project: str = "orbit-wars-rl"
    wandb_entity: str = ""
    device: str = ""  # auto-detect

    def __post_init__(self):
        if not self.device:
            import torch
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
