import torch

from src.algorithms.policies.policy_configs import GaussianPolicyConfig
from .policy import Policy


class SingleGaussianPolicy(Policy):

    def __init__(self, input_size : int, number_actions : int, config : GaussianPolicyConfig = GaussianPolicyConfig()):
        super().__init__(input_size, number_actions)
        self._trunk, trunk_out = config.build_trunk(input_size)
        self._head = torch.nn.Linear(trunk_out, number_actions * 2)

    def log_probability(self, action : torch.Tensor, distribution : torch.distributions.Distribution) -> torch.Tensor:
        # we need to sum the log probabilities together because
        # for an n dimensional action a with independent elements, p(a) = p(a1) * p(a2) * ... * p(an) => log(p(a)) =
        # log(p(a1)) + ... + log(p(an))
        return distribution.log_prob(action).sum(-1).unsqueeze(-1)

    def forward(self, observation : torch.Tensor) -> torch.distributions.Distribution:
        """Converts from the observation to number_actions normal distributions to represent one per action dim."""
        x = self._head(self._trunk(observation))

        means, raw_stds = x.chunk(2, dim=-1)
        stds = torch.nn.functional.softplus(raw_stds)

        return torch.distributions.Normal(means, stds)
