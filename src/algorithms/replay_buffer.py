import warnings

import torch
from torch.utils.data import Dataset


class ReplayBuffer(Dataset):

    def __init__(self, capacity : int, obs_size : int, action_size : int, 
                 device : torch.device = torch.device("cpu"),
                 discrete : bool = False):
        # Convert your data into PyTorch tensors if they aren't already
        self._observations = torch.empty((capacity, obs_size), dtype=torch.float32, device=device)
        self._next_observations = torch.empty((capacity, obs_size), dtype=torch.float32, device=device)
        if discrete:
            self._actions = torch.empty((capacity, action_size), dtype=torch.int64, device=device)
        else:
            self._actions = torch.empty((capacity, action_size), dtype=torch.float32, device=device)

        self._rewards = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self._old_policy_log_probs = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self._advantages = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self._value_targets = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self._next_terminals = torch.empty((capacity, 1), dtype=torch.int32, device=device)
        self.device = device


        if capacity < 2:
            warnings.warn(f"Replay buffer capacity is {capacity} < 2.")
        self._capacity = capacity
        self.size = 0


    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_full():
            warnings.warn(f"Using unfinished replay buffer of size {self.size} < {self._capacity - 1}.")

        return (self._observations[idx], self._actions[idx], self._rewards[idx], self._old_policy_log_probs[idx],
                self._next_observations[idx], self._advantages[idx], self._value_targets[idx], self._next_terminals[idx])

    def is_full(self):
        return self.size >= self._capacity

    def append(self, observation : torch.Tensor, action : torch.Tensor, reward : torch.Tensor, old_policy_log_prob :
    torch.Tensor, next_observation : torch.Tensor, next_termination_state : torch.Tensor) -> bool:
        """Appends to the ReplayBuffer. Returns false if the buffer is full."""
        if self.is_full():
            return False
        self._observations[self.size] = observation
        self._actions[self.size] = action
        self._rewards[self.size] = reward
        self._old_policy_log_probs[self.size] = old_policy_log_prob
        self._next_observations[self.size] = next_observation
        self._next_terminals[self.size] = next_termination_state
        self.size += 1

        return True

    def standardize_advantages(self):
        self._advantages = (self._advantages - self._advantages.mean()) / (self._advantages.std() + 1e-8)

    def insert_advantage(self, idx : int, advantage : torch.Tensor):
        self._advantages[idx] = advantage

    def insert_value_target(self, idx : int, value_target : torch.Tensor):
        self._value_targets[idx] = value_target

    def reset(self):
        self.size = 0
