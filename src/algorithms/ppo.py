from __future__ import annotations
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.mdp import MdpTerminationState
from src.log import Logger
from .algorithm import Algorithm
from .value_function import ValueFunction
from .policies.policy import Policy
from .hyperparameters import Hyperparameters
from .replay_buffer import ReplayBuffer

class PPO(Algorithm):
    def __init__(self, hyperparameters : Hyperparameters, policy : Policy, obs_dimension : int, action_dimension :
    int, discrete : bool = False, logger : Optional[Logger]=None, device : torch.device = torch.device('cpu')):
        super().__init__(hyperparameters, policy, obs_dimension, action_dimension, discrete, logger=logger,
                         device=device)

        self.logger.add_elements({
            "losses/policy_loss": 0.0,
            "losses/value_loss": 0.0,
            "losses/policy_entropy": 0.0,
        })

        self._value = ValueFunction(policy.input_size).to(self.device)

        self._replay_buffer = ReplayBuffer(self.hyperparameters.buffer_size, obs_dimension,
                                           1 if discrete else action_dimension,
                                           device=self.device, discrete=discrete)

        self._policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hyperparameters.lr, eps=1e-5)
        self._value_optimizer = torch.optim.Adam(self._value.parameters(), lr=self.hyperparameters.value_lr)


    def sample_action(self, obs):
        with torch.no_grad():
            dist = self.policy.forward(obs)
            action = self.policy.sample_action(dist)
            log_probability = self.policy.log_probability(action, dist)
        return action, log_probability

    def update_and_observe(self, initial_obs, next_obs, action, action_log_prob : torch.Tensor,
           reward : float, termination_state : MdpTerminationState, timestep : int) -> bool:
        with torch.no_grad():
            reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
            termination_state = torch.tensor(termination_state.value, device=self.device, dtype=torch.int32)

        self._replay_buffer.append(initial_obs, action, reward, action_log_prob, next_obs, termination_state)

        if self._replay_buffer.is_full():
            self._gae_backwards()
            self._replay_buffer.standardize_advantages()
            self._update_gradients(timestep)
            self._replay_buffer.reset()
            return True
        return False

    def _td_error(self, initial_obs, next_obs, reward, terminal_mask):
        return reward + self.hyperparameters.gamma * terminal_mask * self._value(next_obs) - self._value(initial_obs)

    def _gae_backwards(self):
        """Populate the advantages and value targets in the replay buffer backwards."""
        with torch.no_grad():
            next_advantage = 0
            for k in range(self._replay_buffer.size-1, -1, -1):
                obs, _, reward, _, next_obs, _, _, next_terminal = self._replay_buffer[k]

                # the terminal mask is to ensure we only bootstrap for truncated and done states.
                # the done mask is important so that we do not look past the episode boundary when doing gae
                terminal_mask = next_terminal != MdpTerminationState.TERMINATED.value
                done_mask = next_terminal == MdpTerminationState.IN_PROGRESS.value

                td_error = self._td_error(obs, next_obs, reward, terminal_mask)
                advantage = (self.hyperparameters.gamma * self.hyperparameters.lamda * done_mask *next_advantage) + td_error
                self._replay_buffer.insert_advantage(k, advantage)

                value_target = advantage + self._value(obs)
                self._replay_buffer.insert_value_target(k, value_target)

                next_advantage = advantage

    def _loss(self, new_policy_log_prob, old_policy_log_prob: torch.Tensor, advantage: torch.Tensor, entropy : torch.Tensor) -> torch.Tensor:
        new_policy_log_prob = torch.clip(new_policy_log_prob, -40.0, 40.0) # clip for safety
        old_policy_log_prob = torch.clip(old_policy_log_prob, -40.0, 40.0) # clip for safety

        log_ratio = (new_policy_log_prob - old_policy_log_prob).clip(-10.0, 10.0) # clip for safety.
        ratio = torch.exp(log_ratio)
        clipped = torch.min(
            ratio * advantage,
            torch.clamp(ratio,
                        1 - self.hyperparameters.importance_ratio_clip,
                        1 + self.hyperparameters.importance_ratio_clip)
            * advantage)

        objective = clipped.mean()
        if self.hyperparameters.entropy_loss_weight != 0.0:
            objective += self.hyperparameters.entropy_loss_weight * entropy.mean()
        return -objective


    def _update_gradients(self, timestep : int):
        dataloader = DataLoader(self._replay_buffer, batch_size=self.hyperparameters.batch_size, shuffle=True)

        value_criterion = torch.nn.MSELoss()

        for iteration in range(self.hyperparameters.gradient_epochs):
            for batch in dataloader:
                obs, action, reward, old_policy_log_prob, next_obs, advantage, value_target, next_terminal = batch
                self._policy_optimizer.zero_grad()
                self._value_optimizer.zero_grad()

                policy_distribution = self.policy.forward(obs)
                new_policy_log_prob = self.policy.log_probability(action, policy_distribution)
                entropy = self.policy.entropy(policy_distribution)

                policy_loss = self._loss(new_policy_log_prob, old_policy_log_prob, advantage, entropy)
                value_loss = value_criterion(self._value(obs), value_target)

                policy_loss.backward()
                value_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5, error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(self._value.parameters(), 0.5, error_if_nonfinite=True)

                self._policy_optimizer.step()
                self._value_optimizer.step()

                with torch.no_grad():
                    batch_length = obs.size(0)
                    total_samples = len(dataloader) * self.hyperparameters.gradient_epochs * batch_length

                    self.logger.sum_log_data({
                        "losses/policy_loss": policy_loss.item() * batch_length / total_samples,
                        "losses/value_loss": value_loss.item() * batch_length/ total_samples,
                        "losses/policy_entropy": entropy.sum().item() / total_samples,
                    })

        self.logger.set_log_data({"global_step": timestep})
        self.logger.log_data("losses/policy_loss", "losses/value_loss", "losses/policy_entropy", "global_step")
        self.logger.reset("losses/policy_loss", "losses/value_loss", "losses/policy_entropy")


