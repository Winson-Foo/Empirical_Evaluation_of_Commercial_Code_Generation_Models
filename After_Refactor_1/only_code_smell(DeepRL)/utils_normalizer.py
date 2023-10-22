import numpy as np
from abc import ABC, abstractmethod


class BaseNormalizer(ABC):
    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, saved):
        pass


class MeanStdNormalizer(BaseNormalizer):
    def __init__(self, read_only=False, clip=10.0, epsilon=1e-8):
        super().__init__(read_only)
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
    def __init__(self, coef=1.0):
        super().__init__()
        self.coef = coef

    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        return self.coef * x


class ImageNormalizer(RescaleNormalizer):
    def __init__(self):
        super().__init__(1.0 / 255)


class SignNormalizer(BaseNormalizer):
    def __call__(self, x):
        return np.sign(x)


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0

    def update(self, x):
        x = np.asarray(x)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count
        new_count = total_count
        self.mean = new_mean
        self.var = new_var
        self.count = new_count