import torch

from .policy import Policy


class SingleGaussianPolicy(Policy):

    def __init__(self, input_size : int, number_actions : int):
        super().__init__(input_size, number_actions)

        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, number_actions * 2)

    def forward(self, observation : torch.Tensor) -> torch.distributions.Distribution:
        """Converts from the observation to number_actions normal distributions to represent one per action dim."""
        x = self.fc1(observation)
        x = torch.nn.functional.relu(x)

        x = self.fc2(observation)
        x = torch.nn.functional.relu(x)

        x = self.fc3(x)

        means = x[:, self.number_actions]
        stds = torch.nn.functional.relu(x[:, self.number_actions:]) + 1e-6

        return torch.distributions.Normal(means, stds)