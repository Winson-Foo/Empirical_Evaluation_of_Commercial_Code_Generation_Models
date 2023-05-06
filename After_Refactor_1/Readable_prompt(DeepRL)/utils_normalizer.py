import numpy as np
import torch
from baselines.common.running_mean_std import RunningMeanStd


class BaseNormalizer:
    """
    Base class for all normalizers
    """
    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        """
        Set the read only flag to True
        """
        self.read_only = True

    def unset_read_only(self):
        """
        Unset the read only flag to False
        """
        self.read_only = False

    def state_dict(self):
        """
        Returns the state dictionary
        """
        return None

    def load_state_dict(self, _):
        """
        Load the saved state dictionary
        """
        return


class MeanStdNormalizer(BaseNormalizer):
    """
    Class for mean std normalizer
    """
    def __init__(self, read_only=False, clip=10.0, epsilon=1e-8):
        BaseNormalizer.__init__(self, read_only=read_only)
        self.rms = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        x = np.asarray(x)
        if self.rms is None:
            self.rms = RunningMeanStd(shape=(1,) + x.shape[1:])
        if not self.read_only:
            self.rms.update(x)
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
                       -self.clip, self.clip)

    def state_dict(self):
        return {'mean': self.rms.mean,
                'var': self.rms.var}

    def load_state_dict(self, saved):
        self.rms.mean = saved['mean']
        self.rms.var = saved['var']


class RescaleNormalizer(BaseNormalizer):
    """
    Class for rescale normalizer
    """
    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)
        self.coef = coef

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return self.coef * x


class ImageNormalizer(RescaleNormalizer):
    """
    Class for image normalizer
    """
    def __init__(self):
        RescaleNormalizer.__init__(self, coef=1.0 / 255)


class SignNormalizer(BaseNormalizer):
    """
    Class for sign normalizer
    """
    def __call__(self, x):
        return np.sign(x)