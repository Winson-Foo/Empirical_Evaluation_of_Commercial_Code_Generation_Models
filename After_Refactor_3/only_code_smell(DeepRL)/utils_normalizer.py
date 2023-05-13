import numpy as np
from baselines.common.running_mean_std import RunningMeanStd


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


class MeanStdNormalizer(BaseNormalizer):
    def __init__(self, clip=10.0, epsilon=1e-8):
        super().__init__()
        self.rms = RunningMeanStd(shape=None)
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        x = np.asarray(x)
        self.rms.update(x)
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon), -self.clip, self.clip)

    def state_dict(self):
        return {'mean': self.rms.mean, 'var': self.rms.var}

    def load_state_dict(self, saved):
        self.rms.mean = saved['mean']
        self.rms.var = saved['var']


class RescaleNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        super().__init__()
        self.coef = coef

    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        return self.coef * x


class ImageNormalizer(RescaleNormalizer):
    def __init__(self):
        super().__init__(coef=1.0 / 255)


class SignNormalizer(BaseNormalizer):
    def __call__(self, x):
        return np.sign(x)