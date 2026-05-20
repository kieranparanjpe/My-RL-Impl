from abc import ABC, abstractmethod
import torch

class Policy(ABC, torch.nn.Module):

    def __init__(self, input_size : int, number_actions : int):
        super().__init__()
        self.input_size = input_size
        self.number_actions = number_actions

    @abstractmethod
    def sample(self, obs : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the policy to get an action. Returns the action and its log probability"""
        pass

    def log_probability_of_action(self, obs : torch.Tensor, action : torch.Tensor) -> float:
        pass
