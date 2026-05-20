from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import DataLoader

from algorithm import Algorithm
from mdp import Mdp
from policy import Policy
from hyperparameters import Hyperparameters
from replay_buffer import ReplayBuffer


@dataclass
class PPOHyperparams(Hyperparameters):
    gamma: float = 0.99
    lamda: float = 0.95
    objective_clip : float = 0.2
    batch_size: int = 64
    buffer_size: int = 50000
    gradient_updates : int = 100

class PPO(Algorithm):
    def __init__(self, hyperparameters : PPOHyperparams, policy : Policy, mdp : Mdp, value : torch.nn.Module):
        super().__init__(hyperparameters, policy, mdp)
        self.value = value

        self.replay_buffer = ReplayBuffer(self.hyperparameters.buffer_size, mdp.obs_dimension, mdp.action_dim, device=next(policy.parameters()).device)


    def take_action(self, obs):
        action, log_probability = self.policy.sample(obs)
        return action, log_probability

    def update_and_observe(self, initial_obs, next_obs, action, action_log_prob : torch.Tensor, reward, done):
        reward = torch.tensor(reward, device=initial_obs.device, dtype=torch.float32)
        td_error = self.td_error(initial_obs, next_obs, reward)

        if self.replay_buffer.is_full():
            self.gae(self.replay_buffer.size - 2, 0)
            self.update_gradients()
            self.replay_buffer.reset()

        self.replay_buffer.append(initial_obs, action, reward, action_log_prob, td_error)

    def td_error(self, initial_obs, next_obs, reward):
        return reward + self.hyperparameters.gamma * self.value(next_obs) - self.value(initial_obs)

    def gae(self, k, next_advantage):
        if k < 0:
            return
        obs, _, _, _, td_error, _, _ = self.replay_buffer[k]

        advantage = (self.hyperparameters.gamme * self.hyperparameters.lamda * next_advantage) + td_error
        self.replay_buffer.insert_advantage(k, advantage)

        value_target = advantage - self.value(obs)
        self.replay_buffer.insert_value_target(k, value_target)

        self.gae(k - 1, advantage)

    def loss(self, obs : torch.Tensor, action: torch.Tensor, old_policy_log_prob: torch.Tensor, advantage: torch.Tensor):
        ratio = torch.exp(self.policy.log_probability_of_action(obs, action) - old_policy_log_prob)
        clipped = torch.min(
            ratio * advantage,
            torch.clamp(ratio,
                        1 - self.hyperparameters.clip,
                        1 + self.hyperparameters.clip)
            * advantage)
        return -torch.mean(clipped)


    def update_gradients(self):
        dataloader = DataLoader(self.replay_buffer, batch_size=self.hyperparameters.batch_size, shuffle=True)

        policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hyperparameters.lr, eps=1e-5)
        value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.hyperparameters.lr)

        value_criterion = torch.nn.MSELoss()

        for iteration in range(self.hyperparameters.gradient_updates):
            for batch in dataloader:
                obs, action, reward, old_policy_log_prob, td_error, advantage, value_target = batch
                policy_optimizer.zero_grad()
                value_optimizer.zero_grad()

                policy_loss = self.loss(obs, action, old_policy_log_prob, advantage)
                value_loss = value_criterion(self.value(obs), value_target)

                policy_loss.backward()
                value_loss.backward()

                policy_optimizer.step()
                value_optimizer.step()


