import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List, Callable

class MLP(nn.Module):
    """A standard multi-layer perceptron."""

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        hidden_sizes: List[int],
        activation: Callable = F.relu,
        activate_output: bool = False,
    ) -> None:
        """
        Args:
            in_shape: tuple, list or torch.Size, the shape of the input.
            out_shape: tuple, list or torch.Size, the shape of the output.
            hidden_sizes: iterable of ints, the hidden-layer sizes.
            activation: callable, the activation function.
            activate_output: bool, whether to apply the activation to the output.
        """
        super().__init__()
        self.in_shape = torch.Size(in_shape)
        self.out_shape = torch.Size(out_shape)
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.activate_output = activate_output

        if len(hidden_sizes) == 0:
            raise ValueError("List of hidden sizes can't be empty.")

        # Define input layer
        self.input_layer = nn.Linear(torch.tensor(in_shape).prod(), hidden_sizes[0])
        
        # Define hidden layers
        hidden_layers = []
        for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            hidden_layers.append(nn.Linear(in_size, out_size))
        self.hidden_layers = nn.ModuleList(hidden_layers)
        
        # Define output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], torch.tensor(out_shape).prod())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1:] != self.in_shape:
            raise ValueError(
                "Expected inputs of shape {}, got {}.".format(
                    self.in_shape, inputs.shape[1:]
                )
            )

        inputs = inputs.reshape(-1, torch.tensor(self.in_shape).prod())
        outputs = self.input_layer(inputs)
        outputs = self.activation(outputs)

        for hidden_layer in self.hidden_layers:
            outputs = hidden_layer(outputs)
            outputs = self.activation(outputs)

        outputs = self.output_layer(outputs)
        
        if self.activate_output:
            outputs = self.activation(outputs)
        
        outputs = outputs.reshape(-1, *self.out_shape)

        return outputs