import torch
from typing_extensions import override

from src.algorithms.policies.policy_configs import CategoricalPolicyConfig
from rl_commons.policies.policy import Policy


class CategoricalPolicy(Policy):

    def __init__(self, input_size : int, number_actions : int, config : CategoricalPolicyConfig = CategoricalPolicyConfig()):
        super().__init__(input_size, number_actions)
        self.config = config
        self._trunk, trunk_out = config.build_trunk(input_size)
        self._head = torch.nn.Linear(trunk_out, number_actions)

    def forward(self, observation : torch.Tensor) -> torch.distributions.Distribution:
        """Converts from the observation to categorical distribution over discrete actions."""
        x = self._head(self._trunk(observation))

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


def _build_categorical_policy(obs_dimension: int, action_dimension: int, config) -> Policy:
    cfg = config if isinstance(config, CategoricalPolicyConfig) else CategoricalPolicyConfig()
    return CategoricalPolicy(obs_dimension, action_dimension, cfg)


Policy.register('categorical', _build_categorical_policy)
