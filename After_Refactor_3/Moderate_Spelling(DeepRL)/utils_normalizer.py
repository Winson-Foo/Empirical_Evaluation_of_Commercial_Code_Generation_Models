import numpy as np
from typing import Tuple

class BaseNormalizer:
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    def state_dict(self) -> Tuple:
        pass

    def load_state_dict(self, state: Tuple) -> None:
        pass


class MeanStdNormalizer(BaseNormalizer):
    def __init__(self, clip: float = 10.0, epsilon: float = 1e-8):
        super().__init__()
        self.rms = RunningMeanStd()
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.rms.update(x)
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
                       -self.clip, self.clip)

    def state_dict(self) -> Tuple:
        return self.rms.mean, self.rms.var

    def load_state_dict(self, state: Tuple) -> None:
        self.rms.mean = state[0]
        self.rms.var = state[1]


class RescaleNormalizer(BaseNormalizer):
    def __init__(self, coef: float):
        super().__init__()
        self.coef = coef

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.coef * x


class ImageNormalizer(RescaleNormalizer):
    def __init__(self):
        super().__init__(1.0 / 255)


class SignNormalizer(BaseNormalizer):
    def __init__(self):
        super().__init__()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x)