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
    planet_feature_dim: int = 20
    fleet_feature_dim: int = 13
    global_feature_dim: int = 11
    entity_dim: int = 96
    num_heads: int = 4
    num_layers: int = 3
    mlp_expansion: int = 3
    num_angle_bins: int = 144
    num_ship_bins: int = 32
    # Mask ship bins below this index to -inf in model forward (never sampled).
    # For fraction-head (10 bins on [0.1..1.0]), set to 1 to remove the
    # "10%-of-source" bin that PPO collapses to in cold-start self-play.
    min_ship_bin: int = 0
    # How to decode a ship-bin index into an absolute ship count:
    #   "absolute" — bin → SHIP_COUNTS[bin]  (32-entry hybrid linear-log table)
    #   "fraction" — bin → round(FRACTION_BIN_VALUES[bin] * max_ships)
    # MUST match the BC label scheme that produced the checkpoint.
    # Default "absolute" preserves legacy checkpoint behaviour.
    ship_bin_mode: str = "absolute"
    pairwise_feature_dim: int = 15   # see features.PAIRWISE_FEATURE_DIM
    max_planets: int = 48            # for target_head output size; matches EnvConfig
    dropout: float = 0.0
    # Value head input width. 0 = auto (2*entity_dim, new concat head).
    # Set to entity_dim when loading pre-Phase-1 checkpoints (old mean-pool head).
    value_head_in: int = 0
    # Optional supervised auxiliary head: per owned planet P(lost within K steps).
    use_threat_head: bool = False


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    lr_warmup_steps: int = 5000
    lr_decay: str = "cosine"
    total_env_steps: int = 500_000_000
    batch_size: int = 2048
    num_minibatches: int = 4
    ppo_epochs: int = 4
    clip_eps: float = 0.2
    gamma: float = 0.995
    gae_lambda: float = 0.95
    entropy_coef_fire: float = 0.01
    entropy_coef_target: float = 0.02   # entropy bonus on the target head (was misnamed entropy_coef_angle)
    entropy_coef_ships: float = 0.01
    bc_coef: float = 0.0
    # IL regularization: KL penalty between current policy and a frozen
    # reference (typically the BC warmstart). Computed on PPO rollout states.
    # Anchors the policy to teacher competence — prevents drift to degenerate
    # local optima in self-play. Decays linearly to 0 over training so the
    # policy can eventually exceed the teacher.
    #   il_lambda:        peak coefficient (0 = disabled)
    #   il_decay_frac:    fraction of training over which lambda decays to 0
    il_lambda: float = 0.0
    il_decay_frac: float = 0.8
    kl_target: float = 0.05   # KL early-stop threshold per epoch; inf = disabled
    value_coef: float = 0.5
    # Note: shaping_coef is intentionally NOT here — it is passed as --shaping-coef
    # CLI arg directly to TorchEnv (see train_torch.py). PPOConfig is not the
    # right owner for an environment reward shaping parameter.
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
class BCConfig:
    num_trajectories: int = 5000
    num_steps: int = 5000
    learning_rate: float = 3e-4
    batch_size: int = 128    # 128 < typical dataset so each epoch has multiple steps


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    bc: BCConfig = field(default_factory=BCConfig)
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
