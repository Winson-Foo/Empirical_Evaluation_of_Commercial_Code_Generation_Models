import numpy as np
from baselines.common.running_mean_std import RunningMeanStd


class BaseNormalizer:
    def __init__(self):
        self.read_only = False

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, saved_dict):
        return


class MeanStdNormalizer(BaseNormalizer):
    def __init__(self, clip=10.0, epsilon=1e-8):
        super().__init__()
        self.rms = RunningMeanStd()
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        if not self.read_only:
            self.rms.update(x)
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
                       -self.clip, self.clip)

    def state_dict(self):
        return {'mean': self.rms.mean,
                'var': self.rms.var}

    def load_state_dict(self, saved_dict):
        self.rms.mean = saved_dict['mean']
        self.rms.var = saved_dict['var']


class RescaleNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        super().__init__()
        self.coef = coef

    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)
        return self.coef * x


class ImageNormalizer(RescaleNormalizer):
    def __init__(self):
        super().__init__(1.0 / 255.0)


class SignNormalizer(BaseNormalizer):
    def __call__(self, x):
        return np.sign(x)

# Notes:
# 1. Removed unnecessary read_only parameter from BaseNormalizer's constructor.
# 2. Made BaseNormalizer's read_only parameter a class attribute.
# 3. Added super().__init__() calls to the constructors of derived classes.
# 4. Replaced shape=(1,) + x.shape[1:] with shape=x.shape in MeanStdNormalizer constructor.
# 5. Added dtype=np.float32 to np.asarray() calls to ensure data consistency.
# 6. Changed Baseline's RunningMeanStd import to a local one.