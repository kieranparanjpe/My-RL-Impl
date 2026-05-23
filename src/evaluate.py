import torch
from tqdm.auto import tqdm

from algorithms.policies import PolicyFactory
from mdp import MdpGym


class Evaluator:

    def __init__(self, environment_id, policy_id, policy_weights_path, n_timesteps):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mdp = MdpGym(environment_id, self.device, render_mode='human')

        self.policy = PolicyFactory.build_policy(policy_id, self.mdp.obs_dimension, self.mdp.action_dimension).to(self.device)

        state_dict = torch.load(policy_weights_path, weights_only=True)
        self.policy.load_state_dict(state_dict)

        self.n_timesteps = n_timesteps


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
    evaluator = Evaluator("CartPole-v1", "categorical",
                          "saved_policies/CartPole-v1/CartPole-v1@2026-05-22-19-48-17/policy_2047.pth", 100000)
    evaluator.evaluate()