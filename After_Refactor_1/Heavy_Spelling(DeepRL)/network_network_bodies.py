import torch.nn.functional as F
from .network_utils import *

class NatureConvBody(nn.Module):
    """Convolutional neural network body for the DQN agent.

    Args:
        in_channels (int): Number of input channels. Default is 4.
        noisy_linear (bool): Whether to use NoisyLinear layer. Default is False.

    Returns:
        torch.Tensor: Feature representation of the input image.
    """

    def __init__(self, in_channels=4, noisy_linear=False):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        if noisy_linear:
            self.fc4 = NoisyLinear(7 * 7 * 64, self.feature_dim)
        else:
            self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        """Reset the noise parameters of the NoisyLinear layer."""
        if self.noisy_linear:
            self.fc4.reset_noise()

    def forward(self, x):
        """Forward pass of the convolutional neural network body.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Feature representation of the input image.
        """
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y


class DDPGConvBody(nn.Module):
    """Convolutional neural network body for the DDPG agent.

    Args:
        in_channels (int): Number of input channels. Default is 4.

    Returns:
        torch.Tensor: Feature representation of the input image.
    """

    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        """Forward pass of the convolutional neural network body.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Feature representation of the input image.
        """
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y


class FCBody(nn.Module):
    """Fully connected neural network body for the agents.

    Args:
        state_dim (int): Dimensions of the input state.
        hidden_units (tuple): Number of hidden units in each layer. Default is (64, 64).
        gate (function): Activation function used in each layer. Default is F.relu.
        noisy_linear (bool): Whether to use NoisyLinear layer. Default is False.

    Returns:
        torch.Tensor: Feature representation of the input state.
    """

    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, noisy_linear=False):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        if noisy_linear:
            self.layers = nn.ModuleList(
                [NoisyLinear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        else:
            self.layers = nn.ModuleList(
                [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = gate
        self.feature_dim = dims[-1]
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        """Reset the noise parameters of the NoisyLinear layers."""
        if self.noisy_linear:
            for layer in self.layers:
                layer.reset_noise()

    def forward(self, x):
        """Forward pass of the fully connected neural network body.

        Args:
            x (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Feature representation of the input state.
        """
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class DummyBody(nn.Module):
    """Dummy neural network body for debugging and testing.

    Args:
        state_dim (int): Dimensions of the input state.

    Returns:
        torch.Tensor: Feature representation of the input state.
    """

    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        """Forward pass of the dummy neural network body.

        Args:
            x (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Feature representation of the input state.
        """
        return x