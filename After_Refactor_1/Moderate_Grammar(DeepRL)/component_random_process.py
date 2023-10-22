import numpy as np
from typing import Optional

class RandomProcess:
    def reset_states(self) -> None:
        pass

class GaussianProcess(RandomProcess):
    def __init__(self, dimension: int, std: float):
        self.dimension = dimension
        self.std = std

    def sample(self) -> np.ndarray:
        return np.random.randn(self.dimension) * self.std

class OrnsteinUhlenbeckProcess(RandomProcess):
    def __init__(self, dimension: int, std: float, theta: float = 0.15, dt: float = 1e-2, x0: Optional[np.ndarray] = None):
        self.dimension = dimension
        self.theta = theta
        self.mu = np.zeros(dimension)
        self.std = std
        self.dt = dt
        self._x_prev = x0 if x0 is not None else np.zeros(self.dimension)
        self.reset_states()

    def sample(self) -> np.ndarray:
        noise = np.random.randn(self.dimension)
        x = self._x_prev + self.theta * (self.mu - self._x_prev) * self.dt + self.std * np.sqrt(self.dt) * noise
        self._x_prev = x
        return x

    def reset_states(self) -> None:
        self._x_prev = np.zeros(self.dimension)