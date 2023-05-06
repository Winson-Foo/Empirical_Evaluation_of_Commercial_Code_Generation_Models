from .network_utils import *


class ConvBody(nn.Module):
    def __init__(self, in_channels, conv_kernel_sizes, conv_output_sizes, fc_output_size, gate=F.relu,
                 noisy_linear=False):
        super(ConvBody, self).__init__()
        self.feature_dim = fc_output_size
        self.convs = nn.ModuleList([
            layer_init(nn.Conv2d(in_size, out_size, kernel_size=ksize, stride=stride))
            for in_size, out_size, ksize, stride in zip([in_channels] + conv_output_sizes[:-1], conv_output_sizes, conv_kernel_sizes, [4, 2, 1])
        ])
        self.noisy_linear = noisy_linear
        if noisy_linear:
            self.fc = NoisyLinear(conv_output_sizes[-1], fc_output_size)
        else:
            self.fc = layer_init(nn.Linear(conv_output_sizes[-1], fc_output_size))

        self.gate = gate

    def reset_noise(self):
        if self.noisy_linear:
            self.fc.reset_noise()

    def forward(self, x):
        for conv in self.convs:
            x = self.gate(conv(x))
        x = x.view(x.size(0), -1)
        x = self.gate(self.fc(x))
        return x


class NatureConvBody(ConvBody):
    def __init__(self, in_channels=4, noisy_linear=False):
        super(NatureConvBody, self).__init__(
            in_channels=in_channels,
            conv_kernel_sizes=[8, 4, 3],
            conv_output_sizes=[32, 64, 64],
            fc_output_size=512,
            noisy_linear=noisy_linear
        )


class DDPGConvBody(ConvBody):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__(
            in_channels=in_channels,
            conv_kernel_sizes=[3, 3],
            conv_output_sizes=[32, 32],
            fc_output_size=39 * 39 * 32,
        )


class FCBody(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=(64, 64), gate=F.relu, noisy_linear=False):
        super(FCBody, self).__init__()
        dims = (input_size,) + hidden_units
        if noisy_linear:
            self.layers = nn.ModuleList(
                [NoisyLinear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])]
            )
        else:
            self.layers = nn.ModuleList(
                [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])]
            )

        self.fc = layer_init(nn.Linear(dims[-1], output_size))
        self.gate = gate
        self.feature_dim = output_size
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            for layer in self.layers:
                layer.reset_noise()

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        x = self.fc(x)
        return x


class DummyBody(nn.Module):
    def __init__(self, input_size, output_size):
        super(DummyBody, self).__init__()
        self.fc = layer_init(nn.Linear(input_size, output_size))
        self.feature_dim = output_size

    def forward(self, x):
        return self.fc(x)