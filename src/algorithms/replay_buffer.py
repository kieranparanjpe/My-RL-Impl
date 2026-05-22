import warnings

import torch
from torch.utils.data import Dataset


class ReplayBuffer(Dataset):

    def __init__(self, capacity : int, obs_size : int, action_size : int, 
                 device : torch.device = torch.device("cpu"),
                 is_discrete : bool = False):
        # Convert your data into PyTorch tensors if they aren't already
        self.observations = torch.empty((capacity, obs_size), dtype=torch.float32, device=device)
        self.next_observations = torch.empty((capacity, obs_size), dtype=torch.float32, device=device)
        if is_discrete:
            self.actions = torch.empty(capacity, dtype=torch.int64, device=device)
        else:
            self.actions = torch.empty((capacity, action_size), dtype=torch.float32, device=device)

        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.old_policy_log_probs = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.advantages = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.value_targets = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.next_terminals = torch.empty((capacity, 1), dtype=torch.bool, device=device)
        self.device = device


        if capacity < 2:
            warnings.warn(f"Replay buffer capacity is {capacity} < 2.")
        self.capacity = capacity
        self.size = 0


    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_full():
            warnings.warn(f"Using unfinished replay buffer of size {self.size} < {self.capacity - 1}.")

        return (self.observations[idx], self.actions[idx], self.rewards[idx], self.old_policy_log_probs[idx],
                self.next_observations[idx], self.advantages[idx], self.value_targets[idx], self.next_terminals[idx])

    def is_full(self):
        return self.size >= self.capacity

    def append(self, observation : torch.Tensor, action : torch.Tensor, reward : torch.Tensor, old_policy_log_prob :
    torch.Tensor, next_observation : torch.Tensor, next_terminal : torch.Tensor) -> bool:
        """Appends to the ReplayBuffer. Returns false if the buffer is full."""
        if self.is_full():
            return False
        self.observations[self.size] = observation
        self.actions[self.size] = action
        self.rewards[self.size] = reward
        self.old_policy_log_probs[self.size] = old_policy_log_prob
        self.next_observations[self.size] = next_observation
        self.next_terminals[self.size] = next_terminal
        self.size += 1

        return True

    def insert_advantage(self, idx : int, advantage : torch.Tensor):
        self.advantages[idx] = advantage

    def insert_value_target(self, idx : int, value_target : torch.Tensor):
        self.value_targets[idx] = value_target

    def reset(self):
        self.size = 0