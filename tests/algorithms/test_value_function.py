import torch

from src.algorithms.value_function import ValueFunction
from src.algorithms.value_function_config import ValueFunctionConfig


def test_forward_shape():
    vf = ValueFunction(4, ValueFunctionConfig(hidden_sizes=[8]))

    output = vf(torch.randn(5, 4))

    assert output.shape == (5, 1)
