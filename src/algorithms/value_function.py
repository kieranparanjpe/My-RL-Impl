import torch

class ValueFunction(torch.nn.Module):

    def __init__(self, input_size : int):
        super().__init__()

        self._fc1 = torch.nn.Linear(input_size, 64)
        self._fc2 = torch.nn.Linear(64, 64)
        self._fc3 = torch.nn.Linear(64, 1)

    def forward(self, observation : torch.Tensor) -> torch.Tensor:
        x = self._fc1(observation)
        x = torch.nn.functional.relu(x)

        x = self._fc2(x)
        x = torch.nn.functional.relu(x)

        x = self._fc3(x)

        return x
