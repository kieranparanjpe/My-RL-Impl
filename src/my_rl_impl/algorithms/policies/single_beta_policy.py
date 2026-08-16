from typing import override

import torch

from my_rl_impl.algorithms.policies.policy_configs import BetaPolicyConfig
from rl_commons.policies.policy import Policy


class SingleBetaPolicy(Policy):
    """Single Beta Policy class. Actions are clamped between -1 and 1."""
    def __init__(self, input_size : int, number_actions : int, config : BetaPolicyConfig = BetaPolicyConfig()):
        super().__init__(input_size, number_actions)
        self.config = config
        self._trunk, trunk_out = config.build_trunk(input_size)
        self._head = torch.nn.Linear(trunk_out, number_actions * 2)

        self.action_scale = float(config.action_range[1] - config.action_range[0])
        self.action_shift = float(config.action_range[0])

    @override
    def sample_action(self, distribution : torch.distributions.Distribution) -> torch.Tensor:
        raw_action = super().sample_action(distribution)
        # action currently bounded between [0, 1] -> we want to make it between action range
        return raw_action * self.action_scale + self.action_shift

    def log_probability(self, action: torch.Tensor, distribution: torch.distributions.Distribution) -> torch.Tensor:
        # we need to sum the log probabilities together because
        # for an n dimensional action a with independent elements, p(a) = p(a1) * p(a2) * ... * p(an) => log(p(a)) =
        # log(p(a1)) + ... + log(p(an))

        # then we also need to correct for the transformation of the random variable, following for u ~ p(u),
        # a = f(u), p(a) = p(u) * |du/da|

        raw_action = ((action - self.action_shift) / self.action_scale).clamp(1e-6, 1 - 1e-6) # clamp for stability
        log_prob = distribution.log_prob(raw_action).sum(-1, keepdim=True)
        log_prob -= action.shape[-1] * torch.log(torch.tensor(self.action_scale, device=action.device))

        return log_prob

    def forward(self, observation : torch.Tensor) -> torch.distributions.Distribution:
        """Converts from the observation to number_actions beta distributions to represent one per action dim."""
        x = self._head(self._trunk(observation))

        # keep between 1.02 and 500.0
        x = torch.nn.functional.softplus(x) + 1e-2 + 1
        x = x.clamp(max=500.0)

        alphas, betas = x.chunk(2, dim=-1)
        dist = torch.distributions.Beta(alphas, betas)
        return dist


def _build_single_beta_policy(obs_dimension: int, action_dimension: int, config) -> Policy:
    cfg = config if isinstance(config, BetaPolicyConfig) else BetaPolicyConfig()
    return SingleBetaPolicy(obs_dimension, action_dimension, cfg)


Policy.register('single_beta', _build_single_beta_policy)
