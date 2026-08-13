from rl_commons.policies.policy import Policy
from src.algorithms.policies.categorical_policy import CategoricalPolicy
from src.algorithms.policies.single_beta_policy import SingleBetaPolicy
from src.algorithms.policies.single_gaussian_policy import SingleGaussianPolicy
from src.algorithms.policies.policy_configs import BetaPolicyConfig, CategoricalPolicyConfig, GaussianPolicyConfig


def test_categorical_config_survives_weights_only_load(tmp_path):
    config = CategoricalPolicyConfig(hidden_sizes=[8])
    policy = CategoricalPolicy(4, 2, config)
    path = tmp_path / "p.pth"
    policy.save(str(path), config=config)

    checkpoint = Policy.load_checkpoint(str(path))
    assert checkpoint["config"] == config


def test_gaussian_config_survives_weights_only_load(tmp_path):
    config = GaussianPolicyConfig(hidden_sizes=[8])
    policy = SingleGaussianPolicy(4, 2, config)
    path = tmp_path / "p.pth"
    policy.save(str(path), config=config)

    checkpoint = Policy.load_checkpoint(str(path))
    assert checkpoint["config"] == config


def test_beta_config_survives_weights_only_load(tmp_path):
    config = BetaPolicyConfig(hidden_sizes=[8], action_range=(-2.0, 2.0))
    policy = SingleBetaPolicy(4, 2, config)
    path = tmp_path / "p.pth"
    policy.save(str(path), config=config)

    checkpoint = Policy.load_checkpoint(str(path))
    assert checkpoint["config"] == config
