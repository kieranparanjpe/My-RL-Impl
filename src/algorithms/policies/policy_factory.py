from src.algorithms.policies.policy_configs import CategoricalPolicyConfig, GaussianPolicyConfig, BetaPolicyConfig, PolicyConfig
from rl_commons.policies.policy import Policy
from .categorical_policy import CategoricalPolicy
from .single_beta_policy import SingleBetaPolicy
from .single_gaussian_policy import SingleGaussianPolicy


class PolicyFactory:

    @classmethod
    def build_policy(cls, policy_id: str, obs_dimension: int, action_dimension: int,
                     config: PolicyConfig | None = None) -> Policy:
        if policy_id == 'single_gaussian':
            cfg = config if isinstance(config, GaussianPolicyConfig) else GaussianPolicyConfig()
            return SingleGaussianPolicy(obs_dimension, action_dimension, cfg)
        elif policy_id == 'categorical':
            cfg = config if isinstance(config, CategoricalPolicyConfig) else CategoricalPolicyConfig()
            return CategoricalPolicy(obs_dimension, action_dimension, cfg)
        elif policy_id == 'single_beta':
            cfg = config if isinstance(config, BetaPolicyConfig) else BetaPolicyConfig()
            return SingleBetaPolicy(obs_dimension, action_dimension, cfg)

        raise ValueError(f"Policy not found: {policy_id}")