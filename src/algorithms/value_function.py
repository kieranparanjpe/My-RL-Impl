import torch

from src.algorithms.value_function_config import ValueFunctionConfig


class ValueFunction(torch.nn.Module):

    def __init__(self, input_size : int, config : ValueFunctionConfig = ValueFunctionConfig()):
        super().__init__()
        trunk, out_size = config.build_trunk(input_size)
        self._net = trunk
        self._head = torch.nn.Linear(out_size, 1)

    def forward(self, observation : torch.Tensor) -> torch.Tensor:
        return self._head(self._net(observation))
