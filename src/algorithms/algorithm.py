from abc import ABC, abstractmethod

import torch

from mdp import Mdp
from policy import Policy

class Algorithm(ABC):

    def __init__(self, hyperparameters, policy : Policy, mdp : Mdp):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.policy = policy
        self.mdp = mdp

    @abstractmethod
    def take_action(self, obs : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Take an action given a obs and return it. Should be called before stepping environment."""
        pass

    @abstractmethod
    def update_and_observe(self, initial_obs : torch.Tensor, next_obs : torch.Tensor, action : torch.Tensor, action_log_prob : float, reward : float, done : bool):
        """Update and observe next steps based on environment's current obs after stepping. May include gradient updates, buffer updates, etc."""
        pass

