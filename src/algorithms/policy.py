from abc import ABC, abstractmethod
import torch

class Policy(ABC, torch.nn.Module):

    def __init__(self, input_size : int, number_actions : int):
        super().__init__()
        self.input_size = input_size
        self.number_actions = number_actions

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """Sample the policy to get an action"""
        pass
