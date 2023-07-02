import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from nflows.utils import torchutils


class MaskedLinear(nn.Linear):
    """A linear module with a masked weight matrix."""

    def __init__(
        self,
        input_degrees,
        output_features,
        autoregressive_features,
        random_mask,
        output_layer,
        bias=True,
    ):
        super().__init__(
            in_features=len(input_degrees), out_features=output_features, bias=bias
        )
        mask, output_degrees = self._get_mask_and_degrees(
            input_degrees=input_degrees,
            output_features=output_features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            output_layer=output_layer,
        )
        self.register_buffer("mask", mask)
        self.register_buffer("degrees", output_degrees)

    @classmethod
    def _get_mask_and_degrees(
        cls, input_degrees, output_features, autoregressive_features, random_mask, output_layer,
    ):
        if output_layer:
            output_degrees = torchutils.tile(
                _get_input_degrees(autoregressive_features),
                output_features // autoregressive_features,
            )
            mask = (output_degrees[..., None] > input_degrees).float()

        else:
            if random_mask:
                min_input_degree = torch.min(input_degrees).item()
                min_input_degree = min(min_input_degree, autoregressive_features - 1)
                output_degrees = torch.randint(
                    low=min_input_degree,
                    high=autoregressive_features,
                    size=[output_features],
                    dtype=torch.long,
                )
            else:
                output_degrees = torch.arange(output_features) % (autoregressive_features - 1) + 1
            mask = (output_degrees[..., None] >= input_degrees).float()

        return mask, output_degrees

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MaskedFeedforwardBlock(nn.Module):
    """A feedforward block based on a masked linear module.
    
    The number of output features is taken to be equal to the number of input features.
    """

    def __init__(
        self,
        input_degrees,
        autoregressive_features,
        context_features=None,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        super().__init__()
        num_features = len(input_degrees)

        # Batch norm.
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_features, eps=1e-3)
        else:
            self.batch_norm = None

        # Masked linear.
        self.linear = MaskedLinear(
            input_degrees=input_degrees,
            output_features=num_features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            output_layer=False,
        )
        self.degrees = self.linear.degrees

        # Activation and dropout.
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, inputs, context=None):
        if self.batch_norm:
            inputs = self.batch_norm(inputs)
        temps = self.linear(inputs)
        temps = self.activation(temps)
        outputs = self.dropout(temps)
        return outputs


class MaskedResidualBlock(nn.Module):
    """A residual block containing masked linear modules."""

    def __init__(
        self,
        input_degrees,
        autoregressive_features,
        context_features=None,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        if random_mask:
            raise ValueError("Masked residual block can't be used with random masks.")
        super().__init__()
        num_features = len(input_degrees)

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, num_features)

        # Batch norm.
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(num_features, eps=1e-3) for _ in range(2)]
            )

        # Masked linear.
        linear_0 = MaskedLinear(
            input_degrees=input_degrees,
            output_features=num_features,
            autoregressive_features=autoregressive_features,
            random_mask=False,
            output_layer=False,
        )
        linear_1 = MaskedLinear(
            input_degrees=linear_0.degrees,
            output_features=num_features,
            autoregressive_features=autoregressive_features,
            random_mask=False,
            output_layer=False,
        )
        self.linear_layers = nn.ModuleList([linear_0, linear_1])
        self.degrees = linear_1.degrees
        if torch.all(self.degrees >= input_degrees).item() != 1:
            raise RuntimeError(
                "In a masked residual block, the output degrees can't be"
                " less than the corresponding input degrees."
            )

        # Activation and dropout
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

        # Initialization.
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, a=-1e-3, b=1e-3)
            init.uniform_(self.linear_layers[-1].bias, a=-1e-3, b=1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if context is not None:
            temps += self.context_layer(context)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        return inputs + temps


class MADE(nn.Module):
    """Implementation of MADE."""

    def __init__(
        self,
        input_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        output_multiplier=1,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        if use_residual_blocks and random_mask:
            raise ValueError("Residual blocks can't be used with random masks.")
        super().__init__()

        # Initial layer.
        self.initial_layer = MaskedLinear(
            input_degrees=_get_input_degrees(input_features),
            output_features=hidden_features,
            autoregressive_features=input_features,
            random_mask=random_mask,
            output_layer=False,
        )

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, hidden_features)

        self.use_residual_blocks = use_residual_blocks
        self.activation = activation
        # Residual blocks.
        blocks = []
        if use_residual_blocks:
            block_constructor = MaskedResidualBlock
        else:
            block_constructor = MaskedFeedforwardBlock
        prev_out_degrees = self.initial_layer.degrees
        for _ in range(num_blocks):
            blocks.append(
                block_constructor(
                    input_degrees=prev_out_degrees,
                    autoregressive_features=input_features,
                    context_features=context_features,
                    random_mask=random_mask,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
            )
            prev_out_degrees = blocks[-1].degrees
        self.blocks = nn.ModuleList(blocks)

        # Final layer.
        self.final_layer = MaskedLinear(
            input_degrees=prev_out_degrees,
            output_features=input_features * output_multiplier,
            autoregressive_features=input_features,
            random_mask=random_mask,
            output_layer=True,
        )

    def forward(self, inputs, context=None):
        temps = self.initial_layer(inputs)
        if context is not None:
            temps += self.activation(self.context_layer(context))
        if not self.use_residual_blocks:
            temps = self.activation(temps)
        for block in self.blocks:
            temps = block(temps, context)
        outputs = self.final_layer(temps)
        return outputs