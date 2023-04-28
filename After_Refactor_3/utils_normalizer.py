import numpy as np
import torch
from baselines.common.running_mean_std import RunningMeanStd


class Normalizer:
    """
    The base normalizer class. All other normalizer classes should inherit from this class.
    """
    def __init__(self, read_only=False):
        """
        Initializes the normalizer.

        :param read_only: Whether the normalizer is read only or not.
        """
        self.read_only = read_only

    def set_read_only(self):
        """Sets the normalizer to read only mode."""
        self.read_only = True

    def unset_read_only(self):
        """Sets the normalizer to write mode."""
        self.read_only = False

    def state_dict(self):
        """Returns the state of the normalizer."""
        return None

    def load_state_dict(self, _):
        """Loads the state of the normalizer."""
        return


class MeanStdNormalizer(Normalizer):
    """
    A normalizer that normalizes inputs based on their mean and standard deviation.
    """
    def __init__(self, read_only=False, clip=10.0, epsilon=1e-8):
        """
        Initializes the normalizer.

        :param read_only: Whether the normalizer is read only or not.
        :param clip: The maximum value that the normalized inputs can have.
        :param epsilon: A small value added to standard deviation to prevent division by zero.
        """
        super().__init__(read_only)
        self.rms = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        """
        Normalizes the input based on the mean and standard deviation.

        :param x: The input to be normalized.
        :return: The normalized input.
        """
        x = np.asarray(x)
        if self.rms is None:
            self.rms = RunningMeanStd(shape=(1,) + x.shape[1:])
        if not self.read_only:
            self.rms.update(x)
        return self._normalize(x)

    def _normalize(self, x):
        """
        Normalizes the input based on the mean and standard deviation.

        :param x: The input to be normalized.
        :return: The normalized input.
        """
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
                       -self.clip, self.clip)

    def state_dict(self):
        """Returns the state of the normalizer."""
        return {'mean': self.rms.mean,
                'var': self.rms.var}

    def load_state_dict(self, saved):
        """Loads the state of the normalizer."""
        self.rms.mean = saved['mean']
        self.rms.var = saved['var']


class ImageNormalizer(Normalizer):
    """
    A normalizer that rescales image inputs by a factor of 1/255.
    """
    def __init__(self):
        """Initializes the normalizer."""
        super().__init__()

    def __call__(self, x):
        """
        Rescales the input image by a factor of 1/255.

        :param x: The input image to be rescaled.
        :return: The rescaled input image.
        """
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return x / 255


class SignNormalizer(Normalizer):
    """
    A normalizer that applies np.sign function to the input.
    """
    def __init__(self):
        """Initializes the normalizer."""
        super().__init__()

    def __call__(self, x):
        """
        Applies np.sign function to the input.

        :param x: The input to have np.sign applied.
        :return: The input with np.sign applied.
        """
        return np.sign(x)