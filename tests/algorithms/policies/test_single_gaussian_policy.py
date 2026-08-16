import torch

from my_rl_impl.algorithms.policies.single_gaussian_policy import SingleGaussianPolicy
from my_rl_impl.algorithms.policies.policy_configs import GaussianPolicyConfig


def test_forward_returns_normal_with_positive_std():
    policy = SingleGaussianPolicy(4, 2, GaussianPolicyConfig(hidden_sizes=[8]))

    dist = policy.forward(torch.randn(5, 4))

    assert isinstance(dist, torch.distributions.Normal)
    assert dist.loc.shape == (5, 2)
    assert (dist.scale > 0).all()


def test_log_probability_sums_over_action_dims():
    policy = SingleGaussianPolicy(4, 2, GaussianPolicyConfig(hidden_sizes=[8]))
    dist = torch.distributions.Normal(torch.zeros(5, 2), torch.ones(5, 2))
    action = torch.zeros(5, 2)

    log_prob = policy.log_probability(action, dist)

    assert log_prob.shape == (5, 1)
    assert torch.allclose(log_prob.squeeze(-1), dist.log_prob(action).sum(-1))
