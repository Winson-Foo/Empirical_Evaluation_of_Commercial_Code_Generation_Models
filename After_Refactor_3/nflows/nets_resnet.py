import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features,
        context_features,
        activation=F.relu,
        dropout_prob=0.0,
        use_batch_norm=False,
        zero_initialization=True
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_prob)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        inputs_temp = inputs
        if self.use_batch_norm:
            inputs_temp = self.batch_norm_layers[0](inputs_temp)
        inputs_temp = self.activation(inputs_temp)
        inputs_temp = self.linear_layers[0](inputs_temp)
        if self.use_batch_norm:
            inputs_temp = self.batch_norm_layers[1](inputs_temp)
        inputs_temp = self.activation(inputs_temp)
        inputs_temp = self.dropout(inputs_temp)
        inputs_temp = self.linear_layers[1](inputs_temp)
        if context is not None:
            inputs_temp = F.glu(torch.cat((inputs_temp, self.context_layer(context)), dim=1), dim=1)
        return inputs + inputs_temp


class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.relu,
        dropout_prob=0.0,
        use_batch_norm=False
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_prob=dropout_prob,
                    use_batch_norm=use_batch_norm
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, inputs, context=None):
        if context is None:
            inputs_temp = self.initial_layer(inputs)
        else:
            inputs_temp = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            inputs_temp = block(inputs_temp, context=context)
        outputs = self.final_layer(inputs_temp)
        return outputs


class ConvResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        context_channels=None,
        activation=F.relu,
        dropout_prob=0.0,
        use_batch_norm=False,
        zero_initialization=True
    ):
        super().__init__()
        self.activation = activation

        if context_channels is not None:
            self.context_layer = nn.Conv2d(
                in_channels=context_channels,
                out_channels=channels,
                kernel_size=1,
                padding=0
            )
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm2d(channels, eps=1e-3) for _ in range(2)]
            )
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, padding=1) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_prob)
        if zero_initialization:
            init.uniform_(self.conv_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.conv_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        inputs_temp = inputs
        if self.use_batch_norm:
            inputs_temp = self.batch_norm_layers[0](inputs_temp)
        inputs_temp = self.activation(inputs_temp)
        inputs_temp = self.conv_layers[0](inputs_temp)
        if self.use_batch_norm:
            inputs_temp = self.batch_norm_layers[1](inputs_temp)
        inputs_temp = self.activation(inputs_temp)
        inputs_temp = self.dropout(inputs_temp)
        inputs_temp = self.conv_layers[1](inputs_temp)
        if context is not None:
            inputs_temp = F.glu(torch.cat((inputs_temp, self.context_layer(context)), dim=1), dim=1)
        return inputs + inputs_temp


class ConvResidualNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        context_channels=None,
        num_blocks=2,
        activation=F.relu,
        dropout_prob=0.0,
        use_batch_norm=False
    ):
        super().__init__()
        self.context_channels = context_channels
        self.hidden_channels = hidden_channels
        if context_channels is not None:
            self.initial_layer = nn.Conv2d(
                in_channels=in_channels + context_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                padding=0
            )
        else:
            self.initial_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                padding=0
            )
        self.blocks = nn.ModuleList(
            [
                ConvResidualBlock(
                    channels=hidden_channels,
                    context_channels=context_channels,
                    activation=activation,
                    dropout_prob=dropout_prob,
                    use_batch_norm=use_batch_norm
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=1, padding=0
        )

    def forward(self, inputs, context=None):
        if context is None:
            inputs_temp = self.initial_layer(inputs)
        else:
            inputs_temp = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            inputs_temp = block(inputs_temp, context)
        outputs = self.final_layer(inputs_temp)
        return outputs


def main():
    batch_size, channels, height, width = 100, 12, 64, 64
    inputs = torch.rand(batch_size, channels, height, width)
    context = torch.rand(batch_size, channels // 2, height, width)
    net = ConvResidualNet(
        in_channels=channels,
        out_channels=2 * channels,
        hidden_channels=32,
        context_channels=channels // 2,
        num_blocks=2,
        dropout_prob=0.1,
        use_batch_norm=True
    )
    print(get_num_parameters(net))
    outputs = net(inputs, context)
    print(outputs.shape)

if __name__ == "__main__":
    main()