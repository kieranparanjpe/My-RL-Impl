from __future__ import annotations

from dataclasses import dataclass

import torch

from ml_commons.networks import NetworkConfig


@dataclass
class CategoricalPolicyConfig(NetworkConfig):
    pass


@dataclass
class GaussianPolicyConfig(NetworkConfig):
    pass


@dataclass
class BetaPolicyConfig(NetworkConfig):
    action_range: tuple[float, float] = (-1.0, 1.0)


PolicyConfig = CategoricalPolicyConfig | GaussianPolicyConfig | BetaPolicyConfig

torch.serialization.add_safe_globals(
    [CategoricalPolicyConfig, GaussianPolicyConfig, BetaPolicyConfig]
)
