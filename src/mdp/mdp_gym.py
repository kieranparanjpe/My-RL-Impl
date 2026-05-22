import torch
import gymnasium as gym
from .mdp_base import Mdp

class MdpGym(Mdp):

    def __init__(self, environment_id : str, device : torch.device = torch.device('cpu'), render_mode=None):
        super().__init__(device)

        self.env = gym.make(environment_id, render_mode=render_mode)

    @property
    def obs_dimension(self) -> int:
        return self.env.observation_space.shape[0]
    
    @property
    def discrete(self) -> bool:
        return isinstance(self.env.action_space, gym.spaces.Discrete)

    @property
    def action_dimension(self) -> int:
        if self.discrete:
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]

    def reset(self) -> torch.Tensor:
        obs, _ = self.env.reset()
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool]:
        # 1. Convert PyTorch action back to NumPy/integer for Gymnasium to understand
        if self.discrete:
            raw_action = int(action.item())
        else:
            raw_action = action.cpu().numpy()

        # 2. Run the actual physics step
        next_obs, reward, terminated, truncated, _ = self.env.step(raw_action)
        
        # 3. Process the outputs into generic forms
        done = terminated or truncated
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        
        return next_obs_tensor, float(reward), bool(done)
        
    def close(self):
        self.env.close()

