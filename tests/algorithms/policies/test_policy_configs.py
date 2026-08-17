from rl_commons.policies.policy import Policy
from my_rl_impl.algorithms.policies.categorical_policy import CategoricalPolicy
from my_rl_impl.algorithms.policies.single_beta_policy import SingleBetaPolicy
from my_rl_impl.algorithms.policies.single_gaussian_policy import SingleGaussianPolicy
from my_rl_impl.algorithms.policies.policy_configs import BetaPolicyConfig, CategoricalPolicyConfig, GaussianPolicyConfig


def test_categorical_config_survives_weights_only_load(tmp_path):
    config = CategoricalPolicyConfig(hidden_sizes=[8])
    policy = CategoricalPolicy(4, 2, config)
    path = tmp_path / "p.pth"
    policy.save(str(path))

    loaded = Policy.load(str(path), policy_id="categorical")
    assert loaded.config == config


def test_gaussian_config_survives_weights_only_load(tmp_path):
    config = GaussianPolicyConfig(hidden_sizes=[8])
    policy = SingleGaussianPolicy(4, 2, config)
    path = tmp_path / "p.pth"
    policy.save(str(path))

    loaded = Policy.load(str(path), policy_id="single_gaussian")
    assert loaded.config == config


def test_beta_config_survives_weights_only_load(tmp_path):
    config = BetaPolicyConfig(hidden_sizes=[8], action_range=(-2.0, 2.0))
    policy = SingleBetaPolicy(4, 2, config)
    path = tmp_path / "p.pth"
    policy.save(str(path))

    loaded = Policy.load(str(path), policy_id="single_beta")
    assert loaded.config == config
