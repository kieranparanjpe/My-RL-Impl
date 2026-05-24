import torch
import argparse
from src.algorithms.policies import PolicyFactory
from src.mdp import MdpGym


class Evaluator:

    def __init__(self, environment_id, policy_id, policy_weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mdp = MdpGym(environment_id, self.device, render_mode='human')

        self.policy = PolicyFactory.build_policy(policy_id, self.mdp.obs_dimension, self.mdp.action_dimension).to(self.device)

        state_dict = torch.load(policy_weights_path, weights_only=True)
        self.policy.load_state_dict(state_dict)

    def evaluate(self):
        last_observation = self.mdp.reset()
        while True:
            action, _, _ = self.policy.sample(last_observation)

            next_observation, reward, done = self.mdp.step(action)

            if done:
                last_observation = self.mdp.reset()
            else:
                last_observation = next_observation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", "-e", help="Environment Id to run", default="CartPole-v1")
    parser.add_argument("--policy", "-p", help="Policy Id to use", default="categorical")
    parser.add_argument("--weights", "-w", help="Path to weights to load into policy")

    args = parser.parse_args()

    evaluator = Evaluator(args.environment, args.policy,args.weights)
    evaluator.evaluate()