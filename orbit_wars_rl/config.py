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


@dataclass
class ModelConfig:
    max_entities: int = 64
    max_owned_planets: int = 10
    planet_feature_dim: int = 18
    fleet_feature_dim: int = 9
    global_feature_dim: int = 10
    entity_dim: int = 96
    num_heads: int = 4
    num_layers: int = 3
    mlp_expansion: int = 3
    num_angle_bins: int = 144
    num_ship_bins: int = 32
    dropout: float = 0.0


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
    entropy_coef_angle: float = 0.02
    entropy_coef_ships: float = 0.01
    bc_coef: float = 0.0
    kl_target: float = 0.05   # KL early-stop threshold per epoch; inf = disabled
    value_coef: float = 0.5
    shaping_coef: float = 0.01
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
