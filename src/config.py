from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product

from pydantic import TypeAdapter

# Sub-package configs — imported here so callers can use src.config as a one-stop import.
from src.mdp.mdp_config import MdpConfig
from src.algorithms.algorithm_config import AlgorithmConfig, PPOHyperparams
from src.algorithms.network_config import ActivationId, ACTIVATION_MAP, NetworkConfig, ValueFunctionConfig
from src.algorithms.policies.policy_configs import (
    CategoricalPolicyConfig, GaussianPolicyConfig, BetaPolicyConfig, PolicyConfig,
)


@dataclass
class RunInfo:
    environment_id: str
    algorithm_id: str
    policy_id: str
    grid_index: int | None
    time: datetime

    def _env_and_time(self):
        return f"{self.environment_id}@{self.time:%Y-%m-%d-%H-%M-%S}"

    def group(self):
        return None if self.grid_index is None else self._env_and_time()

    def run_name(self) -> str:
        return self._env_and_time() if self.grid_index is None else f"{self._env_and_time()}_RUN-{self.grid_index}"

    def local_folder_path(self, folder_name: str) -> str:
        if self.grid_index is None:
            return f"{folder_name}/{self.environment_id}/{self.run_name()}"
        else:
            return f"{folder_name}/{self.environment_id}/{self.group()}/{self.run_name()}"


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

_SECTIONS = ("algorithm", "policy", "value_fn", "mdp")


class ConfigLoader:

    @classmethod
    def load_single(cls, path: str, algorithm_id: str, policy_id: str) -> RunConfig:
        data = cls._load_data(path)
        return cls._parse_single(data, algorithm_id, policy_id)

    @classmethod
    def load_grid(cls, path: str, algorithm_id: str, policy_id: str) -> list[RunConfig]:
        data = cls._load_data(path)
        return cls._parse_grid(data, algorithm_id, policy_id)

    @classmethod
    def _parse_single(cls, data: dict, algorithm_id: str, policy_id: str) -> RunConfig:
        algo_cls = _ALGORITHM_REGISTRY[algorithm_id]
        policy_cls = _POLICY_REGISTRY[policy_id]

        algorithm = TypeAdapter(algo_cls).validate_python(data.get("algorithm", {}))
        policy = TypeAdapter(policy_cls).validate_python(data.get("policy", {}))
        value_fn = TypeAdapter(ValueFunctionConfig).validate_python(data.get("value_fn", {}))
        mdp = TypeAdapter(MdpConfig).validate_python(data.get("mdp", {}))

        return RunConfig(algorithm=algorithm, policy=policy, value_fn=value_fn, mdp=mdp)

    @classmethod
    def _parse_grid(cls, data: dict, algorithm_id: str, policy_id: str) -> list[RunConfig]:
        """
        Values that are lists become grid dimensions; scalars are fixed.
        The final grid is the cartesian product across all dimensions from all sections.
        """
        grid_dims: list[tuple[str, str, list]] = []  # (section, key, values)
        fixed: dict[str, dict] = {s: {} for s in _SECTIONS}

        for section in _SECTIONS:
            for key, val in data.get(section, {}).items():
                if isinstance(val, list):
                    grid_dims.append((section, key, val))
                else:
                    fixed[section][key] = val

        all_keys = [(s, k) for s, k, _ in grid_dims]
        all_vals = [v for _, _, v in grid_dims]

        configs = []
        for combo in product(*all_vals):
            section_dicts: dict[str, dict] = {s: dict(fixed[s]) for s in _SECTIONS}
            for (section, key), value in zip(all_keys, combo):
                section_dicts[section][key] = value
            configs.append(cls._parse_single(section_dicts, algorithm_id, policy_id))

        return configs

    @classmethod
    def _load_data(cls, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
