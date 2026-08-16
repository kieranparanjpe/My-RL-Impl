import torch

from my_rl_impl.algorithms.policies.categorical_policy import CategoricalPolicy
from my_rl_impl.algorithms.policies.policy_configs import CategoricalPolicyConfig


def test_forward_returns_categorical_over_action_dim():
    policy = CategoricalPolicy(4, 3, CategoricalPolicyConfig(hidden_sizes=[8]))

    dist = policy.forward(torch.randn(5, 4))

    assert isinstance(dist, torch.distributions.Categorical)
    assert dist.logits.shape == (5, 3)


def test_log_probability_shape():
    policy = CategoricalPolicy(4, 3, CategoricalPolicyConfig(hidden_sizes=[8]))
    dist = policy.forward(torch.randn(5, 4))
    action = dist.sample().unsqueeze(-1)  # shape [5, 1], mirrors real usage

    log_prob = policy.log_probability(action, dist)

    assert log_prob.shape == (5, 1)


def test_entropy_matches_distribution_entropy_directly():
    policy = CategoricalPolicy(4, 3, CategoricalPolicyConfig(hidden_sizes=[8]))
    dist = policy.forward(torch.randn(5, 4))

    assert torch.equal(policy.entropy(dist), dist.entropy())
