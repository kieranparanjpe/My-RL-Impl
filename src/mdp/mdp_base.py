from abc import ABC, abstractmethod
import torch

class Mdp(ABC):

    def __init__(self, device):
        self.device = device

    @property
    @abstractmethod
    def obs_dimension(self) -> int:
        """Returns the size of the observation."""
        pass

    @property
    @abstractmethod
    def action_dimension(self) -> int:
        """Returns the number of actions."""
        pass

    @property
    @abstractmethod
    def discrete(self) -> bool:
        """Returns whether the actions are continuous or discrete."""
        pass

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """Resets the MDP and returns the initial observation"""
        pass

    @abstractmethod
    def step(self, action : torch.Tensor) -> tuple[torch.Tensor, float, bool]:
        """
        Advances the MDP obs by executing an action.
        Returns: (next_obs_tensor, reward_float, terminal_obs)
        """
        pass

    @abstractmethod
    def close(self):
        """
        Closes the mdp environment.
        """
        pass
