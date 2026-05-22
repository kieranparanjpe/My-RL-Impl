import torch
from tqdm.auto import tqdm
import wandb
from datetime import datetime

from .logger import Logger
from src.algorithms import Hyperparameters, PPO
from src.algorithms.policies.categorical_policy import CategoricalPolicy
from src.algorithms.policies.single_gaussian_policy import SingleGaussianPolicy
from src.algorithms.ppo import PPOHyperparams
from src.mdp.mdp_gym import MdpGym
import os


class Trainer:

    def __init__(self, environment_id, algorithm_id : str, policy_id : str, hyperparameters : Hyperparameters, logging=True, save_policy=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.save_policy = save_policy
        self.hyperparameters = hyperparameters

        self.environment_id = environment_id
        now = datetime.now()
        self.run_name = f"{self.environment_id}@{now:%Y-%m-%d-%H-%M-%S}"
        self.wandb_run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="kieranparanjpe-mcgill-university",
            # Set the wandb project where this run will be logged.
            project="RL_Project1",
            name=self.run_name,
            tags=[f"{algorithm_id}", f"{policy_id}", f"{environment_id}"],
            job_type="train",
            # Track hyperparameters and run metadata.
            config=self.hyperparameters.__dict__,
        ) if logging else None

        self.logger = Logger(self.wandb_run, {
            "charts/episodic_return": 0.0,
            "charts/episode_length": 0,
            "global_step": 0,
        })

        if self.save_policy:
            self.save_policy_folder = self.create_policy_folder()

        self.mdp = MdpGym(environment_id, self.device, render_mode=None)

        if policy_id == 'single_gaussian':
            self.policy = SingleGaussianPolicy(self.mdp.obs_dimension, self.mdp.action_dimension).to(self.device)
        elif policy_id == 'categorical':
            self.policy = CategoricalPolicy(self.mdp.obs_dimension, self.mdp.action_dimension).to(self.device)

        if algorithm_id == 'ppo':
            self.algorithm = PPO(self.hyperparameters, self.policy, self.mdp.obs_dimension,
                                 self.mdp.action_dimension, self.mdp.discrete,
                                 wandb_run=self.wandb_run, device=self.device)

    def create_policy_folder(self):
        directory_path = f"saved_policies/{self.environment_id}/{self.run_name}"

        os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def _save_policy(self, timestep):
        torch.save(self.policy.state_dict(), f'{self.save_policy_folder}/policy_{timestep}.pth')

    def train(self):
        last_observation = self.mdp.reset()
        for timestep in tqdm(range(self.hyperparameters.n_timesteps)):
            action, log_prob_action = self.algorithm.sample_action(last_observation)

            next_observation, reward, done = self.mdp.step(action)

            updated_policy = self.algorithm.update_and_observe(last_observation, next_observation, action, log_prob_action, reward,
                                              done, timestep)

            if updated_policy and self.save_policy:
                self._save_policy(timestep)

            self.logger.add_log_data({
                "charts/episodic_return": reward,
                "charts/episode_length": 1,
            })
            if done:
                self.logger.set_log_data({'global_step': timestep})

                self.logger.log_data()
                self.logger.reset_fields("charts/episodic_return", "charts/episode_length")

                last_observation = self.mdp.reset()
            else:
                last_observation = next_observation

if __name__ == "__main__":
    trainer = Trainer("CartPole-v1", "ppo", "categorical", PPOHyperparams(), logging=True, save_policy=True)
    trainer.train()
