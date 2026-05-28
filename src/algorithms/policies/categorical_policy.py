import torch
from typing_extensions import override

from .policy import Policy


class CategoricalPolicy(Policy):

    def __init__(self, input_size : int, number_actions : int):
        super().__init__(input_size, number_actions)

        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, number_actions)

    def forward(self, observation : torch.Tensor) -> torch.distributions.Distribution:
        """Converts from the observation to categorical distribution over discrete actions."""
        x = self.fc1(observation)
        x = torch.nn.functional.relu(x)

        x = self.fc2(x)
        x = torch.nn.functional.relu(x)

        x = self.fc3(x)

        return torch.distributions.Categorical(logits=x)

    @override
    def log_probability(self, action : torch.Tensor, distribution : torch.distributions.Categorical) -> torch.Tensor:
        # the action will have shape [B, 1], but we need to feed the categorical dist [B]:
        log_probabilities = distribution.log_prob(action.squeeze()).unsqueeze(-1)
        # the log probs returned will have shape [B], so we unsqueeze to make them [B, 1]
        return log_probabilities

    @override
    def entropy(self, distribution : torch.distributions.Distribution) -> torch.Tensor:
        return distribution.entropy()