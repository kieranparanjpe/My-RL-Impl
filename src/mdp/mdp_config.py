from dataclasses import dataclass


@dataclass
class MdpConfig:
    normalise_obs: bool = True
    normalise_reward: bool = True
    reward_norm_gamma: float = 0.99
