from dataclasses import dataclass

import torch
from tqdm.auto import tqdm
import wandb
from datetime import datetime

from logger import Logger
from src.algorithms import Hyperparameters, PPO
from src.algorithms.policies.categorical_policy import CategoricalPolicy
from src.algorithms.policies.single_gaussian_policy import SingleGaussianPolicy
from src.algorithms.ppo import PPOHyperparams
from src.mdp.mdp_gym import MdpGym



class Trainer:

    def __init__(self, environment_id, algorithm_id : str, policy_id : str, hyperparameters : Hyperparameters):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        now = datetime.now()
        self.wandb_run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="kieranparanjpe-mcgill-university",
            # Set the wandb project where this run will be logged.
            project="RL_Project1",
            name=f"{environment_id}@{now:%Y-%m-%d-%H-%M-%S}",
            tags=[f"{algorithm_id}", f"{policy_id}", f"{environment_id}"],
            job_type="train",
            # Track hyperparameters and run metadata.
            config=hyperparameters.__dict__,
        )

        self.hyperparameters = hyperparameters
        self.mdp = MdpGym(environment_id, self.device, render_mode=None)

        if policy_id == 'single_gaussian':
            self.policy = SingleGaussianPolicy(self.mdp.obs_dimension, self.mdp.action_dimension)
        elif policy_id == 'categorical':
            self.policy = CategoricalPolicy(self.mdp.obs_dimension, self.mdp.action_dimension)

        if algorithm_id == 'ppo':
            self.algorithm = PPO(self.hyperparameters, self.policy, self.mdp.obs_dimension,
                                 self.mdp.action_dimension, self.mdp.is_discrete, wandb_run=self.wandb_run)


    def train(self):
        log_data = Logger(self.wandb_run, {
            "charts/episodic_return": 0.0,
            "charts/episode_length": 0,
            "global_step": 0,
        })

        last_observation = self.mdp.reset()
        for _ in tqdm(range(self.hyperparameters.n_timesteps)):
            action, log_prob_action = self.algorithm.sample_action(last_observation)

            next_observation, reward, done = self.mdp.step(action)

            self.algorithm.update_and_observe(last_observation, next_observation, action, log_prob_action, reward, done)

            log_data.add_log_data({
                "charts/episodic_return": reward,
                "charts/episode_length": 1,
                "global_step": 1,
            })

            if done:
                log_data.log_data()
                log_data.reset_fields("charts/episodic_return", "charts/episode_length")



            if done:
                last_observation = self.mdp.reset()

if __name__ == "__main__":
    trainer = Trainer("CartPole-v1", "ppo", "categorical", PPOHyperparams())
    trainer.train()
