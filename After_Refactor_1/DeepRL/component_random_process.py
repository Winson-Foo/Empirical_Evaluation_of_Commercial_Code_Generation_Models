import numpy as np

class RandomProcess:
    def reset_states(self) -> None:
        """Reset the internal state of the process."""
        pass

class GaussianProcess(RandomProcess):
    def __init__(self, shape: tuple[int], std: float):
        self.shape = shape
        self.std = std

    def sample(self) -> np.ndarray:
        """Generate a sample from the Gaussian process."""
        return np.random.randn(*self.shape) * self.std

class OrnsteinUhlenbeckProcess(RandomProcess):
    def __init__(
        self, shape: tuple[int], std: float, theta: float = 0.15, dt: float = 0.01, x0: np.ndarray = None
    ):
        self.theta = theta
        self.mu = 0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.shape = shape
        self.reset_states()

    def sample(self) -> np.ndarray:
        """Generate a sample from the Ornstein-Uhlenbeck process."""
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.std * np.sqrt(
            self.dt
        ) * np.random.randn(*self.shape)
        self.x_prev = x
        return x

    def reset_states(self) -> None:
        """Reset the internal state of the process."""
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.shape)