import numpy as np


class RandomProcess:
    def resetStates(self):
        pass


class ContinuousRandomProcess(RandomProcess):
    def __init__(self, size: tuple, std: float, theta: float = .15, dt: float = 1e-2, x0: np.ndarray = None):
        self.size = size
        self.std = std
        self.theta = theta
        self.mu = 0
        self.dt = dt
        self.x0 = x0 if x0 is not None else np.zeros(self.size)
        self.resetStates()

    def calcNoise(self) -> np.ndarray:
        return self.std * np.sqrt(self.dt) * np.random.randn(*self.size)

    def resetStates(self):
        self.xPrev = self.x0


class GaussianProcess(ContinuousRandomProcess):
    def sample(self) -> np.ndarray:
        return np.random.randn(*self.size) * self.calcNoise()


class OrnsteinUhlenbeckProcess(ContinuousRandomProcess):
    def sample(self) -> np.ndarray:
        x = self.xPrev + \
            self.theta * (self.mu - self.xPrev) * self.dt + \
            self.calcNoise()
        self.xPrev = x
        return x