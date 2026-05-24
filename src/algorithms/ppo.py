from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch
if TYPE_CHECKING:
    import wandb
from torch.utils.data import DataLoader

from src import Logger
from .algorithm import Algorithm
from .value_function import ValueFunction
from .policies.policy import Policy
from .hyperparameters import Hyperparameters
from .replay_buffer import ReplayBuffer


@dataclass
class PPOHyperparams(Hyperparameters):
    value_lr : float = 3e-4
    gamma: float = 0.99
    lamda: float = 0.95
    importance_ratio_clip : float = 0.2
    batch_size: int = 64
    buffer_size: int = 2048
    gradient_epochs : int = 10

class PPO(Algorithm):
    def __init__(self, hyperparameters : Hyperparameters, policy : Policy, obs_dimension : int, action_dimension :
    int, discrete : bool = False, wandb_run : Optional[wandb.Run]=None, device : torch.device = torch.device('cpu')):
        super().__init__(hyperparameters, policy, obs_dimension, action_dimension, discrete, wandb_run=wandb_run,
                         device=device)

        self.logger = Logger(self.wandb_run, {
            "losses/policy_loss": 0.0,
            "losses/value_loss": 0.0,
            "losses/policy_entropy": 0.0,
            "global_step": 0
        })

        self.value = ValueFunction(policy.input_size).to(self.device)

        self.replay_buffer = ReplayBuffer(self.hyperparameters.buffer_size, obs_dimension,
                                          1 if discrete else action_dimension,
                                          device=self.device, discrete=discrete)

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hyperparameters.lr, eps=1e-5)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.hyperparameters.value_lr)


    def sample_action(self, obs):
        with torch.no_grad():
            action, log_probability, _ = self.policy.sample(obs)
        return action, log_probability

    def update_and_observe(self, initial_obs, next_obs, action, action_log_prob : torch.Tensor, reward, done, timestep) -> bool:
        with torch.no_grad():
            reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
            done = torch.tensor(done, device=self.device, dtype=torch.bool)

        self.replay_buffer.append(initial_obs, action, reward, action_log_prob, next_obs, done)

        if self.replay_buffer.is_full():
            self.gae_backwards()
            self.update_gradients(timestep)
            self.replay_buffer.reset()
            return True
        return False

    def td_error(self, initial_obs, next_obs, reward, terminal_mask):
        return reward + self.hyperparameters.gamma * terminal_mask * self.value(next_obs) - self.value(initial_obs)

    def gae_backwards(self):
        """Populate the advantages and value targets in the replay buffer backwards."""
        with torch.no_grad():
            next_advantage = 0
            for k in range(self.replay_buffer.size-1, -1, -1):
                obs, _, reward, _, next_obs, _, _, next_terminal = self.replay_buffer[k]

                # the terminal mask is important so that we do not look past the episode boundary when doing gae
                terminal_mask = ~next_terminal

                td_error = self.td_error(obs, next_obs, reward, terminal_mask)
                advantage = (self.hyperparameters.gamma * self.hyperparameters.lamda * terminal_mask *next_advantage) + td_error
                self.replay_buffer.insert_advantage(k, advantage)

                value_target = advantage + self.value(obs)
                self.replay_buffer.insert_value_target(k, value_target)

                next_advantage = advantage

    def loss(self, new_policy_log_prob, old_policy_log_prob: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
        ratio = torch.exp(new_policy_log_prob - old_policy_log_prob)
        clipped = torch.min(
            ratio * advantage,
            torch.clamp(ratio,
                        1 - self.hyperparameters.importance_ratio_clip,
                        1 + self.hyperparameters.importance_ratio_clip)
            * advantage)
        return -torch.mean(clipped)


    def update_gradients(self, timestep : int):
        dataloader = DataLoader(self.replay_buffer, batch_size=self.hyperparameters.batch_size, shuffle=True)

        value_criterion = torch.nn.MSELoss()


        for iteration in range(self.hyperparameters.gradient_epochs):
            for batch in dataloader:
                obs, action, reward, old_policy_log_prob, next_obs, advantage, value_target, next_terminal = batch
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                new_policy_log_prob, distribution = self.policy.log_probability_of_action(obs, action)

                policy_loss = self.loss(new_policy_log_prob, old_policy_log_prob, advantage)
                value_loss = value_criterion(self.value(obs), value_target)

                policy_loss.backward()
                value_loss.backward()

                self.policy_optimizer.step()
                self.value_optimizer.step()

                with torch.no_grad():
                    entropy = distribution.entropy().sum()
                    batch_length = obs.size(0)
                    total_samples = len(dataloader) * self.hyperparameters.gradient_epochs * batch_length

                    self.logger.sum_log_data({
                        "losses/policy_loss": policy_loss.item() * batch_length / total_samples,
                        "losses/value_loss": value_loss.item() * batch_length/ total_samples,
                        "losses/policy_entropy": entropy.item() / total_samples,
                    })

        self.logger.set_log_data({'global_step': timestep})
        self.logger.log_data()
        self.logger.reset_fully()


