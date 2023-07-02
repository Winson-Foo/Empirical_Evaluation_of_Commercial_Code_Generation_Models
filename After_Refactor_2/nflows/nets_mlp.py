import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """A standard multi-layer perceptron."""

    def __init__(
        self,
        in_shape,
        out_shape,
        hidden_sizes,
        activation=F.relu,
        activate_output=False,
    ):
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

        self._initialize_layers()

    def _initialize_layers(self):
        self.input_layer = nn.Linear(torch.prod(self.in_shape), self.hidden_sizes[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(self.hidden_sizes[:-1], self.hidden_sizes[1:])
        ])
        self.output_layer = nn.Linear(self.hidden_sizes[-1], torch.prod(self.out_shape))

    def forward(self, inputs):
        if inputs.shape[1:] != self.in_shape:
            raise ValueError(
                f"Expected inputs of shape {self.in_shape}, got {inputs.shape[1:]}."
            )

        inputs = inputs.reshape(-1, torch.prod(self.in_shape))
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