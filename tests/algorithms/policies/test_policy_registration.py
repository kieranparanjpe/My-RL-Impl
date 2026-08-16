from my_rl_impl.algorithms.policies import (
    BetaPolicyConfig, CategoricalPolicy, GaussianPolicyConfig,
    Policy, SingleBetaPolicy, SingleGaussianPolicy,
)


def test_categorical_registered():
    assert isinstance(Policy.build_policy("categorical", 4, 2), CategoricalPolicy)


def test_single_gaussian_registered():
    assert isinstance(Policy.build_policy("single_gaussian", 4, 2), SingleGaussianPolicy)


def test_single_beta_registered():
    assert isinstance(Policy.build_policy("single_beta", 4, 2), SingleBetaPolicy)


def test_wrong_config_type_falls_back_to_matching_default():
    policy = Policy.build_policy("categorical", 4, 2, config=GaussianPolicyConfig())
    assert isinstance(policy, CategoricalPolicy)


def test_matching_config_type_is_used_as_is():
    config = BetaPolicyConfig(action_range=(-2.0, 2.0))
    policy = Policy.build_policy("single_beta", 4, 2, config=config)
    assert policy.action_shift == -2.0
