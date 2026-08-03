from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rl_commons.config import RunInfo, ConfigLoader
from rl_commons.mdp import MdpConfig

from src.algorithms.algorithm_config import AlgorithmConfig, PPOHyperparams
from src.algorithms.network_config import ActivationId, ACTIVATION_MAP, NetworkConfig, ValueFunctionConfig
from src.algorithms.policies.policy_configs import (
    CategoricalPolicyConfig, GaussianPolicyConfig, BetaPolicyConfig, PolicyConfig,
)

# Re-export so callers that import RunInfo/ConfigLoader from src.config continue to work.
__all__ = ["RunInfo", "ConfigLoader", "RunConfig", "load_config", "load_grid_configs"]


@dataclass
class RunConfig:
    algorithm: PPOHyperparams = field(default_factory=PPOHyperparams)
    policy: PolicyConfig = field(default_factory=CategoricalPolicyConfig)
    value_fn: ValueFunctionConfig = field(default_factory=ValueFunctionConfig)
    mdp: MdpConfig = field(default_factory=MdpConfig)


_ALGORITHM_REGISTRY: dict[str, type[AlgorithmConfig]] = {
    "base": AlgorithmConfig,
    "ppo": PPOHyperparams,
}

_POLICY_REGISTRY: dict[str, type[PolicyConfig]] = {
    "categorical": CategoricalPolicyConfig,
    "single_gaussian": GaussianPolicyConfig,
    "single_beta": BetaPolicyConfig,
}


def load_config(path: str, algorithm_id: str, policy_id: str) -> RunConfig:
    sections = {
        "algorithm": _ALGORITHM_REGISTRY[algorithm_id],
        "policy": _POLICY_REGISTRY[policy_id],
        "value_fn": ValueFunctionConfig,
        "mdp": MdpConfig,
    }
    return RunConfig(**ConfigLoader.load_single(path, sections))


def load_grid_configs(path: str, algorithm_id: str, policy_id: str) -> list[RunConfig]:
    sections = {
        "algorithm": _ALGORITHM_REGISTRY[algorithm_id],
        "policy": _POLICY_REGISTRY[policy_id],
        "value_fn": ValueFunctionConfig,
        "mdp": MdpConfig,
    }
    return [RunConfig(**d) for d in ConfigLoader.load_grid(path, sections)]
