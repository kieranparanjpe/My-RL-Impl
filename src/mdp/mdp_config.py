from dataclasses import dataclass, field
from typing import Any


@dataclass
class MdpConfig:
    normalise_obs: bool = True
    normalise_reward: bool = True
    reward_norm_gamma: float = 0.99
    make_kwargs : dict[str, Any] = field(default_factory={})
