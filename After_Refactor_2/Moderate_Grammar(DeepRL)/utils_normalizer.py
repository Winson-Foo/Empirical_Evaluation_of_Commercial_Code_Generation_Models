import numpy as np
import torch
from baselines.common.running_mean_std import RunningMeanStd


class BaseNormalizer:
    """Base class for normalizers."""
    def __init__(self, read_only: bool = False):
        self.read_only = read_only

    def set_read_only(self) -> None:
        self.read_only = True

    def unset_read_only(self) -> None:
        self.read_only = False

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        pass


class MeanStdNormalizer(BaseNormalizer):
    """Normalizer using mean and standard deviation."""
    def __init__(self, read_only: bool = False, clip: float = 10.0, epsilon: float = 1e-8):
        super().__init__(read_only=read_only)
        self.rms = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x)
        if self.rms is None:
            self.rms = RunningMeanStd(shape=(1,) + x.shape[1:])
        if not self.read_only:
            self.rms.update(x)
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
                       -self.clip, self.clip)

    def state_dict(self) -> dict:
        return {'mean': self.rms.mean,
                'var': self.rms.var}

    def load_state_dict(self, state_dict: dict) -> None:
        self.rms.mean = state_dict['mean']
        self.rms.var = state_dict['var']


class RescaleNormalizer(BaseNormalizer):
    """Normalize by rescaling using a coefficient."""
    def __init__(self, coef: float = 1.0):
        super().__init__()
        self.coef = coef

    def __call__(self, x) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return self.coef * x


class ImageNormalizer(RescaleNormalizer):
    """Normalize images by rescaling with 1/255."""
    def __init__(self):
        super().__init__(coef=1.0 / 255)


class SignNormalizer(BaseNormalizer):
    """Normalize by taking the sign."""
    def __call__(self, x) -> np.ndarray:
        return np.sign(x)