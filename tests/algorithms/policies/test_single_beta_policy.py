import torch

from my_rl_impl.algorithms.policies.single_beta_policy import SingleBetaPolicy
from my_rl_impl.algorithms.policies.policy_configs import BetaPolicyConfig


def test_forward_returns_beta_with_valid_concentration():
    policy = SingleBetaPolicy(4, 2, BetaPolicyConfig(hidden_sizes=[8]))

    dist = policy.forward(torch.randn(5, 4))

    assert isinstance(dist, torch.distributions.Beta)
    assert (dist.concentration0 > 0).all()
    assert (dist.concentration1 > 0).all()


def test_sample_action_rescales_into_action_range():
    policy = SingleBetaPolicy(4, 2, BetaPolicyConfig(hidden_sizes=[8], action_range=(-3.0, 5.0)))
    dist = policy.forward(torch.randn(100, 4))

    action = policy.sample_action(dist)

    assert (action >= -3.0).all()
    assert (action <= 5.0).all()


def test_log_probability_matches_change_of_variables_formula():
    # log_probability inverts the range-rescale then subtracts the transform's
    # Jacobian correction (num_action_dims * log(scale)) - pin down both pieces.
    policy = SingleBetaPolicy(4, 2, BetaPolicyConfig(hidden_sizes=[8], action_range=(-1.0, 1.0)))
    dist = torch.distributions.Beta(torch.full((3, 2), 2.0), torch.full((3, 2), 2.0))

    raw_action = dist.sample()  # in (0, 1)
    action = raw_action * policy.action_scale + policy.action_shift

    log_prob = policy.log_probability(action, dist)

    expected = dist.log_prob(raw_action).sum(-1, keepdim=True) - action.shape[-1] * torch.log(
        torch.tensor(policy.action_scale))
    assert torch.allclose(log_prob, expected, atol=1e-5)
