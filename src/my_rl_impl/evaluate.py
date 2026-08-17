import argparse

from rl_commons.execution import BaseEvaluator
from rl_commons.mdp import MdpTerminationState, MdpConfig
from my_rl_impl.algorithms.policies import Policy


class Evaluator(BaseEvaluator):

    def __init__(self, environment_id, policy_id, policy_weights_path):
        super().__init__(task_id=environment_id, mdp_config=MdpConfig(normalise_obs=False, normalise_reward=False))

        policy = Policy.load(policy_weights_path, policy_id=policy_id,
                             obs_dimension=self._mdp.obs_dimension,
                             action_dimension=self._mdp.action_dimension)

        if policy.obs_norm_stats is not None:
            self._mdp.enable_obs_normalization(policy.obs_norm_stats)

        self.policy = policy.to(self.device)

    def _run(self):
        last_observation = self._mdp.reset()

        while not self._stop.is_set():
            distribution = self.policy.forward(last_observation)
            action = self.policy.sample_action(distribution)

            next_observation, reward, termination_state = self._mdp.step(action)

            if termination_state is not MdpTerminationState.IN_PROGRESS:
                last_observation = self._mdp.reset()
            else:
                last_observation = next_observation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", "-e", help="Environment Id to run", default="CartPole-v1")
    parser.add_argument("--policy", "-p", help="Policy Id to use", default="categorical")
    parser.add_argument("--weights", "-w", help="Path to weights to load into policy")

    args = parser.parse_args()

    evaluator = Evaluator(args.environment, args.policy, args.weights)
    evaluator.evaluate()
