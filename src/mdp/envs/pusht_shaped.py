import numpy as np
from gym_pusht.envs.pusht import PushTEnv

ARENA_SIZE = 512.0


class PushTShaped(PushTEnv):
    """PushT with dense reward shaping.

    Adds two distance-based penalty terms to the base coverage reward:
      - end-effector → block distance (encourages the agent to approach the block)
      - block → goal centroid distance (encourages pushing the block toward the goal)

    Both terms are normalised by the arena size so they stay small relative to
    the base coverage reward which lives in [0, 1].
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        ee_pos = np.array(self.agent.position)
        block_pos = np.array(self.block.position)
        goal_pos = np.array(self.goal_pose[:2])

        ee_to_block = np.linalg.norm(ee_pos - block_pos) / ARENA_SIZE
        block_to_goal = np.linalg.norm(block_pos - goal_pos) / ARENA_SIZE

        shaped = -0.5 * ee_to_block - 0.5 * block_to_goal

        return obs, reward + shaped, terminated, truncated, info
