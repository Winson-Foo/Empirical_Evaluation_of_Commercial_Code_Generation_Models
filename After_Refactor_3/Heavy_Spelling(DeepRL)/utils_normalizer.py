import numpy as np
from typing import List, Union


class BaseNormalizer:
    def __init__(self, read_only: bool = False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, saved):
        pass


class MeanStdNormalizer(BaseNormalizer):
    def __init__(self, read_only: bool = False, clip: float = 10.0, epsilon: float = 1e-8):
        super().__init__(read_only)
        self.rms = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x: Union[List, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)
        if self.rms is None:
            # Initialize RunningMeanStd if not already initialized
            self.rms = RunningMeanStd(shape=(1,) + x.shape[1:])
        if not self.read_only:
            # Update RunningMeanStd statistics if not read-only
            self.rms.update(x)
        # Normalize and clip the input using the RunningMeanStd statistics
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon), -self.clip, self.clip)

    def state_dict(self) -> dict:
        return {'mean': self.rms.mean, 'var': self.rms.var}

    def load_state_dict(self, saved: dict):
        self.rms.mean = saved['mean']
        self.rms.var = saved['var']


class RescaleNormalizer(BaseNormalizer):
    def __init__(self, coef: float = 1.0):
        super().__init__()
        self.coef = coef

    def __call__(self, x: Union[List, np.ndarray]) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        return self.coef * x


class ImageNormalizer(RescaleNormalizer):
    def __init__(self):
        super().__init__(1.0 / 255)


class SignNormalizer(BaseNormalizer):
    def __call__(self, x: Union[List, np.ndarray]) -> np.ndarray:
        return np.sign(x)