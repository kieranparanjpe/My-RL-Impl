import torch

from .policy import Policy


class SingleGaussianPolicy(Policy):

    def __init__(self, input_size : int, number_actions : int):
        super().__init__(input_size, number_actions)

        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, number_actions * 2)

    def log_probability(self, action : torch.Tensor, distribution : torch.distributions.Distribution) -> torch.Tensor:
        # we need to sum the log probabilities together because
        # for an n dimensional action a with independent elements, p(a) = p(a1) * p(a2) * ... * p(an) => log(p(a)) =
        # log(p(a1)) + ... + log(p(an))
        return distribution.log_prob(action).sum(-1).unsqueeze(-1)

    def forward(self, observation : torch.Tensor) -> torch.distributions.Distribution:
        """Converts from the observation to number_actions normal distributions to represent one per action dim."""
        x = self.fc1(observation)
        x = torch.nn.functional.relu(x)

        x = self.fc2(x)
        x = torch.nn.functional.relu(x)

        x = self.fc3(x)

        means, raw_stds = x.chunk(2, dim=-1)
        stds = torch.nn.functional.softplus(raw_stds)

        return torch.distributions.Normal(means, stds)