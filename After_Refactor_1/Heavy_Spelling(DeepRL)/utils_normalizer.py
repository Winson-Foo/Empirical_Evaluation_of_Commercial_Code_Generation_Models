import numpy as np
import torch
from baselines.common.running_mean_std import RunningMeanStd

class BaseNormalizer:
    """
    A base class for data normalization.

    Args:
        read_only (bool): Whether the normalization should be read-only.
    """

    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        """Set the normalization to read-only mode."""
        self.read_only = True

    def unset_read_only(self):
        """Set the normalization to read-write mode."""
        self.read_only = False

    def state_dict(self):
        """Get the current state of the normalization."""
        return None

    def load_state_dict(self, state_dict):
        """Load a saved state for the normalization."""
        return

class MeanStdNormalizer(BaseNormalizer):
    """
    A class for normalizing data using running mean and standard deviation.

    Args:
        read_only (bool): Whether the normalization should be read-only.
        clip (float): The maximum value for data after normalization.
        epsilon (float): A small value used to avoid division by zero.
    """

    def __init__(self, read_only=False, clip=10.0, epsilon=1e-8):
        BaseNormalizer.__init__(self, read_only)
        self.rms = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        """
        Normalize the input data using running mean and standard deviation.

        Args:
            x (ndarray or Tensor): The data to normalize.

        Returns:
            The normalized data.
        """
        x = np.asarray(x)
        if self.rms is None:
            self.rms = RunningMeanStd(shape=(1,) + x.shape[1:])
        if not self.read_only:
            self.rms.update(x)
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
                       -self.clip, self.clip)

    def state_dict(self):
        """Get the current state of the normalization."""
        return {'mean': self.rms.mean,
                'var': self.rms.var}

    def load_state_dict(self, state_dict):
        """Load a saved state for the normalization."""
        self.rms.mean = state_dict['mean']
        self.rms.var = state_dict['var']

class RescaleNormalizer(BaseNormalizer):
    """
    A class for rescaling data by a fixed coefficient.

    Args:
        coef (float): The rescaling coefficient.
    """

    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)
        self.coef = coef

    def __call__(self, x):
        """
        Rescale the input data by a fixed coefficient.

        Args:
            x (ndarray or Tensor): The data to rescale.

        Returns:
            The rescaled data.
        """
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return self.coef * x


class ImageNormalizer(RescaleNormalizer):
    """A class for normalizing image data."""

    def __init__(self):
        RescaleNormalizer.__init__(self, 1.0 / 255)

class SignNormalizer(BaseNormalizer):
    """A class for normalizing data using the sign function."""

    def __call__(self, x):
        """
        Normalize the input data using the sign function.

        Args:
            x (ndarray or Tensor): The data to normalize.

        Returns:
            The normalized data.
        """
        return np.sign(x)