import argparse

from rl_commons.execution import BaseEvaluator
from rl_commons.mdp import MdpTerminationState, MdpConfig
from src.algorithms.policies import PolicyFactory


class Evaluator(BaseEvaluator):

    def load_checkpoint(self, path):
        import torch
        return torch.load(path, weights_only=False) if path else {}

    def __init__(self, environment_id, policy_id, policy_weights_path):
        checkpoint = self.load_checkpoint(policy_weights_path)
        policy_state_dict = checkpoint.get("policy")

        norm_stats_raw = checkpoint.get("norm_stats")
        obs_rms_stats = (
            norm_stats_raw["obs_mean"].numpy(),
            norm_stats_raw["obs_var"].numpy(),
        ) if norm_stats_raw is not None else None

        mdp_config = MdpConfig(
            normalise_obs=(obs_rms_stats is not None),
            normalise_reward=False,
        )

        super().__init__(task_id=environment_id, mdp_config=mdp_config, obs_rms_stats=obs_rms_stats)

        if isinstance(policy_state_dict, dict):
            self.policy = PolicyFactory.build_policy(policy_id, int(self._mdp.obs_dimension),
                                                     int(self._mdp.action_dimension)).to(self.device)
            if policy_state_dict:
                self.policy.load_state_dict(policy_state_dict)
        elif policy_state_dict is not None:
            self.policy = policy_state_dict.to(self.device)

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
