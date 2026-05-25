from typing import override

import torch

from .policy import Policy


class SingleBetaPolicy(Policy):
    """Single Beta Policy class. Actions are clamped between -1 and 1."""
    def __init__(self, input_size : int, number_actions : int):
        super().__init__(input_size, number_actions)

        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, number_actions * 2)

    @override
    def _get_action(self, distribution : torch.distributions.Distribution) -> torch.Tensor:
        raw_action = super()._get_action(distribution)
        # action currently bounded between [0, 1] -> we want to make it between [-1, 1]
        return raw_action * 2 - 1

    def log_probability(self, action: torch.Tensor, distribution: torch.distributions.Distribution) -> torch.Tensor:
        # we need to sum the log probabilities together because
        # for an n dimensional action a with independent elements, p(a) = p(a1) * p(a2) * ... * p(an) => log(p(a)) =
        # log(p(a1)) + ... + log(p(an))

        raw_action = (action + 1) / 2
        log_prob = distribution.log_prob(raw_action).sum(-1)
        log_prob -= action.shape[-1] * torch.log(torch.tensor(2.0))
        return log_prob

    def forward(self, observation : torch.Tensor) -> torch.distributions.Distribution:
        """Converts from the observation to number_actions beta distributions to represent one per action dim."""
        x = self.fc1(observation)
        x = torch.nn.functional.relu(x)

        x = self.fc2(x)
        x = torch.nn.functional.relu(x)

        x = self.fc3(x)

        x = torch.nn.functional.softplus(x)
        alphas, betas = x.chunk(2, dim=-1)

        return torch.distributions.Beta(alphas, betas)