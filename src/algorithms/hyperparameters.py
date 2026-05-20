from dataclasses import dataclass


@dataclass
class Hyperparameters:
    n_timesteps: int = 1000
    lr: float = 0.001
