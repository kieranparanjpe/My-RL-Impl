from abc import ABC, abstractmethod

import torch

from .policies.policy import Policy

class Algorithm(ABC):

    def __init__(self, hyperparameters, policy : Policy, obs_dimension : int, action_dimension : int,
                 is_discrete : bool = False):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.policy = policy
        self.obs_dimension = obs_dimension
        self.action_dimension = action_dimension
        self.is_discrete = is_discrete

    @abstractmethod
    def sample_action(self, obs : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Take an action given a obs and return it. Should be called before stepping environment."""
        pass

    @abstractmethod
    def update_and_observe(self, initial_obs : torch.Tensor, next_obs : torch.Tensor, action : torch.Tensor, action_log_prob : float, reward : float, done : bool):
        """Update and observe next steps based on environment's current obs after stepping. May include gradient updates, buffer updates, etc."""
        pass

