import numpy as np
import torch
from typing import List, Tuple

class BaseNormalizer:
    def __init__(self, read_only: bool = False):
        self._read_only = read_only

    @property
    def read_only(self):
        return self._read_only
    
    @read_only.setter
    def read_only(self, value):
        self._read_only = value
    
    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return

class RunningMeanStdNormalizer(BaseNormalizer):
    def __init__(self, shape: Tuple[int, ...], clip: float = 10.0, epsilon: float = 1e-8):
        super().__init__()
        self._rms = RunningMeanStd(shape=shape)
        self._clip = clip
        self._epsilon = epsilon

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)

        if not self.read_only:
            self._rms.update(x)

        normalized_x = (x - self._rms.mean) / np.sqrt(self._rms.var + self._epsilon)
        normalized_x = np.clip(normalized_x, -self._clip, self._clip)

        return normalized_x
    
    def state_dict(self):
        return {'mean': self._rms.mean,
                'var': self._rms.var}

    def load_state_dict(self, saved):
        self._rms.mean = saved['mean']
        self._rms.var = saved['var']

class RescaleNormalizer(BaseNormalizer):
    def __init__(self, coef: float = 1.0):
        super().__init__()
        self._coef = coef

    def __call__(self, x):
        x = np.asarray(x)
        return self._coef * x

class ImageNormalizer(RescaleNormalizer):
    def __init__(self):
        super().__init__(1.0 / 255)

class SignNormalizer(BaseNormalizer):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x)