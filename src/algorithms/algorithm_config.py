from dataclasses import dataclass


@dataclass
class AlgorithmConfig:
    n_timesteps: int = 1_000_000
    lr: float = 3e-4


@dataclass
class PPOHyperparams(AlgorithmConfig):
    value_lr: float = 3e-4
    gamma: float = 0.99
    lamda: float = 0.95
    importance_ratio_clip: float = 0.2
    batch_size: int = 64
    buffer_size: int = 2048
    gradient_epochs: int = 10
    entropy_loss_weight: float = 0.0
