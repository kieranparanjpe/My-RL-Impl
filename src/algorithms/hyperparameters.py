from dataclasses import dataclass


@dataclass
class Hyperparameters:
    n_timesteps: int = 1000000
    lr: float = 3e-4
    record_stats : int = 2048
