import numpy as np
import torch

class BaseNormalizer:
    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return

class RunningMeanStdNormalizer(BaseNormalizer):
    def __init__(self, update=True, clip=10.0, epsilon=1e-8):
        self.update = update
        self.clip = clip
        self.epsilon = epsilon
        self.rms = RunningMeanStd()

    def __call__(self, x):
        x = np.asarray(x)
        if self.update:
            self.rms.update(x)
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
                       -self.clip, self.clip)

    def state_dict(self):
        return {'mean': self.rms.mean,
                'var': self.rms.var,
                'count': self.rms.count}

    def load_state_dict(self, saved):
        self.rms.mean = saved['mean']
        self.rms.var = saved['var']
        self.rms.count = saved['count']

class RescaleNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        self.coef = coef

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return self.coef * x

class ImageNormalizer(RescaleNormalizer):
    def __init__(self):
        super().__init__(1.0 / 255)

class SignNormalizer(BaseNormalizer):
    def __call__(self, x):
        return np.sign(x)