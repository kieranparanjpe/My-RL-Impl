from dataclasses import dataclass
from pydantic import TypeAdapter
import json
from itertools import product


@dataclass
class Hyperparameters:
    n_timesteps: int = 1000000
    lr: float = 3e-4

@dataclass
class PPOHyperparams(Hyperparameters):
    value_lr: float = 3e-4
    gamma: float = 0.99
    lamda: float = 0.95
    importance_ratio_clip: float = 0.2
    batch_size: int = 64
    buffer_size: int = 2048
    gradient_epochs: int = 10
    entropy_loss_weight: float = 0.0


class HyperparameterLoader:
    REGISTRY = {
        "base": Hyperparameters,
        "ppo": PPOHyperparams,
    }

    @classmethod
    def _parse_single(cls, data, default_type):
        hp_type = data.pop("type", default_type)
        hp_cls = cls.REGISTRY[hp_type]
        return TypeAdapter(hp_cls).validate_python(data)

    @classmethod
    def _parse_grid(cls, data, default_type):
        keys = data.keys()
        values = data.values()

        hyperparameter_grid = [dict(zip(keys, v)) for v in product(*values)]

        return [cls._parse_single(h, default_type) for h in hyperparameter_grid]

    @classmethod
    def load_single(cls, path : str, default_type : str):
        data = cls._load_data(path)
        return cls._parse_single(data, default_type)

    @classmethod
    def load_grid(cls, path : str, default_type : str):
        data = cls._load_data(path)
        return cls._parse_grid(data, default_type)

    @classmethod
    def _load_data(cls, path : str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

