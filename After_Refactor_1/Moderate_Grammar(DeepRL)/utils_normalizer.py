import numpy as np
import torch
from baselines.common.running_mean_std import RunningMeanStd

class Normalizer:
    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict):
        pass


class MeanStdNormalizer(Normalizer):
    def __init__(self, clip: float = 10.0, epsilon: float = 1e-8):
        super().__init__()
        self.rms = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if self.rms is None:
            shape = (1,) + x.shape[1:] if len(x.shape) > 1 else (1,)
            self.rms = RunningMeanStd(shape=shape)
        if not self.read_only:
            self.rms.update(x)
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
                       -self.clip, self.clip)

    def state_dict(self) -> dict:
        if self.rms:
            return {'mean': self.rms.mean, 'var': self.rms.var}
        else:
            return {}

    def load_state_dict(self, state_dict: dict):
        if self.rms and 'mean' in state_dict and 'var' in state_dict:
            self.rms.mean = state_dict['mean']
            self.rms.var = state_dict['var']


class RescaleNormalizer(Normalizer):
    def __init__(self, coef: float = 1.0):
        super().__init__()
        self.coef = coef

    def __call__(self, x) -> np.ndarray:
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return self.coef * x


class ImageNormalizer(RescaleNormalizer):
    def __init__(self):
        super().__init__(1.0 / 255)


class SignNormalizer(Normalizer):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x)