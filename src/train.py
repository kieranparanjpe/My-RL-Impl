import torch
from tqdm.auto import tqdm
from src.algorithms import Hyperparameters, PPO
from src.algorithms.policies.categorical_policy import CategoricalPolicy
from src.algorithms.policies.single_gaussian_policy import SingleGaussianPolicy
from src.algorithms.ppo import PPOHyperparams
from src.mdp.mdp_gym import MdpGym


class Trainer:

    def __init__(self, environment_id, algorithm_id : str, policy_id : str, hyperparameters : Hyperparameters):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyperparameters = hyperparameters
        self.mdp = MdpGym(environment_id, self.device, render_mode=None)

        if policy_id == 'single_gaussian':
            self.policy = SingleGaussianPolicy(self.mdp.obs_dimension, self.mdp.action_dimension)
        elif policy_id == 'categorical':
            self.policy = CategoricalPolicy(self.mdp.obs_dimension, self.mdp.action_dimension)

        if algorithm_id == 'ppo':
            self.algorithm = PPO(self.hyperparameters, self.policy, self.mdp.obs_dimension,
                                 self.mdp.action_dimension, self.mdp.is_discrete)

    def train(self):
        last_observation = self.mdp.reset()
        reward_sum = 0
        episode_number = 0
        episode_length = 0
        for timestep in tqdm(range(self.hyperparameters.n_timesteps)):
            action, log_prob_action = self.algorithm.sample_action(last_observation)

            next_observation, reward, done = self.mdp.step(action)

            self.algorithm.update_and_observe(last_observation, next_observation, action, log_prob_action, reward, done)

            episode_length += 1
            reward_sum += reward

            if done:
                print(f"\n------Timestep: [{timestep}] | Episode: [{episode_number}]------")
                print(f"Total reward for last episode: "
                      f"{reward_sum}")
                reward_sum = 0
                episode_length = 0



            if done:
                last_observation = self.mdp.reset()


if __name__ == "__main__":
    trainer = Trainer("CartPole-v1", "ppo", "categorical", PPOHyperparams())
    trainer.train()
