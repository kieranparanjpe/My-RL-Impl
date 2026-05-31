from abc import ABC, abstractmethod
import torch

class Policy(ABC, torch.nn.Module):

    def __init__(self, input_size : int, number_actions : int):
        super().__init__()
        self.input_size = input_size
        self._number_actions = number_actions

    @abstractmethod
    def forward(self, observation : torch.Tensor) -> torch.distributions.Distribution:
        pass

    @abstractmethod
    def log_probability(self, action : torch.Tensor, distribution : torch.distributions.Distribution) -> torch.Tensor:
        pass

    def sample_action(self, distribution : torch.distributions.Distribution) -> torch.Tensor:
        return distribution.sample()

    def entropy(self, distribution : torch.distributions.Distribution) -> torch.Tensor:
        return distribution.entropy().sum(-1)



