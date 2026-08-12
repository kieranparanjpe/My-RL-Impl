from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import torch

from rl_commons.mdp import MdpTerminationState
from rl_commons.log import Logger
from rl_commons.policies.policy import Policy

class Algorithm(ABC):

    def __init__(self, hyperparameters, policy : Policy, obs_dimension : int, action_dimension : int,
                 discrete : bool = False, logger : Optional[Logger]=None,
                 device : torch.device = torch.device('cpu')):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.policy = policy
        self.obs_dimension = obs_dimension
        self.action_dimension = action_dimension
        self.discrete = discrete
        self.logger = logger
        self.device = device

    @abstractmethod
    def sample_action(self, obs : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Take an action given a obs and return it. Should be called before stepping environment."""
        pass

    @abstractmethod
    def update_and_observe(self, initial_obs : torch.Tensor, next_obs : torch.Tensor, action : torch.Tensor,
           action_log_prob : float, reward : float, termination_state : MdpTerminationState, timestep : int) -> bool:
        """Update and observe next steps based on environment's current obs after stepping. May include gradient updates, buffer updates, etc. Returns True if a policy update occurred."""
        pass

