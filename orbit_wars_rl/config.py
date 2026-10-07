from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    max_planets: int = 48
    num_players: int = 2


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
    # early/mid/late + normalized steps-to-next-comet-spawn) = 15. With
    # --global-econ, + 48 projected economy-delta channels
    # (timeline.GLOBAL_ECON_DIM: production/material delta × 24 steps) = 63.
    # OPT-IN: an unvalidated feature must not ride along in an unrelated arm.
    # Eval/export infer the width from global_proj.weight.
    global_feature_dim: int = 15
    entity_dim: int = 96
    num_heads: int = 4
    num_layers: int = 3
    mlp_expansion: int = 3
    num_angle_bins: int = 144
    num_ship_bins: int = 32
    # How to decode a ship-bin index into an absolute ship count:
    #   "absolute" — bin → SHIP_COUNTS[bin]  (32-entry hybrid linear-log table)
    #   "binary"   — fire head chooses NOOP/COMMIT; ships are resolved deterministically
    # MUST match the BC label scheme that produced the checkpoint.
    # Default "absolute" preserves legacy checkpoint behaviour.
    ship_bin_mode: str = "absolute"
    # Binary-mode commit gates. "full" = legacy (capture_required affordability + maintain/
    # defend_ok); "minimal" = COMMIT is all-in at ANY target, gated only on having
    # MIN_BINARY_COMMIT_SHIPS. Measured, "full" removes 80.2% of the action space and makes
    # pre-emptive reinforcement inexpressible — see docs/training.md "THE REINFORCEMENT LEGALITY
    # WALL". Persisted in the checkpoint: eval/export MUST mask the same way training did.
    # Default "minimal" (2026-10) for NEW runs; a checkpoint WITHOUT the key is legacy "full"
    # (eval/export/resume all apply that rule when they read a checkpoint).
    binary_commit_gates: str = "minimal"
    pairwise_feature_dim: int = 36   # 22 base + 4 intent + 6 target-CF + 4 source-CF
    max_planets: int = 48            # for target_head output size; matches EnvConfig
    # Target-decode discipline. These are persisted in checkpoints so train/eval/export
    # do not silently disagree about own-target legality or attack concentration vetoes.
    allow_reinforce: bool = False
    reinforce_gate_min_planets: int = 0
    reinforce_forward_only: bool = False
    reinforce_garrison_floor: float = 0.0
    reverse_edge_cooldown: int = 0
    sufficient_commit_factor: float = 0.0
    dropout: float = 0.0


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    lr_warmup_steps: int = 5000
    total_env_steps: int = 500_000_000
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
    # Note: env reward-shaping coefficients are CLI args wired directly to VecTorchEnv
    # (see train_torch.py) — PPOConfig is not the right owner for them.
    max_grad_norm: float = 0.5
    clip_value: bool = True
    normalize_advantages: bool = True


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
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
