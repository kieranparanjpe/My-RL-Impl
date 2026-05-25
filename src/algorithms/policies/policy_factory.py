from .policy import Policy
from .categorical_policy import CategoricalPolicy
from .single_beta_policy import SingleBetaPolicy
from .single_gaussian_policy import SingleGaussianPolicy

class PolicyFactory:

    @classmethod
    def build_policy(cls, policy_id : str, obs_dimension : int, action_dimension : int) -> Policy:
        if policy_id == 'single_gaussian':
            return SingleGaussianPolicy(obs_dimension, action_dimension)
        elif policy_id == 'categorical':
            return CategoricalPolicy(obs_dimension, action_dimension)
        elif policy_id == 'single_beta':
            return SingleBetaPolicy(obs_dimension, action_dimension)

        raise ValueError(f"Policy not found {policy_id}")