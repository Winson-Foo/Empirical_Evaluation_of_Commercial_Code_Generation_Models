import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """Implementation of a multi-layer perceptron."""

    def __init__(
        self,
        input_shape,
        output_shape,
        hidden_sizes,
        activation=F.relu,
        activate_output=False,
    ):
        """
        Args:
            input_shape: tuple, list or torch.Size, the shape of the input.
            output_shape: tuple, list or torch.Size, the shape of the output.
            hidden_sizes: iterable of ints, the hidden-layer sizes.
            activation: callable, the activation function.
            activate_output: bool, whether to apply the activation to the output.
        """
        super().__init__()
        self.input_shape = torch.Size(input_shape)
        self.output_shape = torch.Size(output_shape)
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.activate_output = activate_output

        if len(hidden_sizes) == 0:
            raise ValueError("List of hidden sizes can't be empty.")

        layers = []
        layers.append(nn.Linear(np.prod(input_shape), hidden_sizes[0]))
        layers.append(activation())

        for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(activation())

        layers.append(nn.Linear(hidden_sizes[-1], np.prod(output_shape)))

        if activate_output:
            layers.append(activation())

        self.model = nn.Sequential(*layers)
        
    def forward(self, inputs):
        if inputs.shape[1:] != self.input_shape:
            raise ValueError(f"Expected inputs of shape {self.input_shape}, got {inputs.shape[1:]}.")

        inputs = inputs.reshape(-1, np.prod(self.input_shape))
        outputs = self.model(inputs)
        outputs = outputs.reshape(-1, *self.output_shape)

        return outputs