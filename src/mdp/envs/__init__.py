import gymnasium
import gym_pusht  # noqa: F401 — ensures gym_pusht envs are registered first

from .pusht_shaped import PushTShaped

gymnasium.register(
    id="myPushT-v0",
    entry_point="src.mdp.envs.pusht_shaped:PushTShaped",
    max_episode_steps=300,
)
