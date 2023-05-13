import numpy as np


class RandomProcess:
    """Base class for random processes."""

    def reset_states(self) -> None:
        """Reset the internal state of the process."""


class GaussianProcess(RandomProcess):
    """A Gaussian random process."""

    def __init__(self, size: tuple[int], std: float):
        """
        Initialize a new Gaussian process.

        Args:
            size: The shape of the data to generate.
            std: The standard deviation of the Gaussian distribution.
        """
        self.size = size
        self.std = std

    def sample(self) -> np.ndarray:
        """Generate a sample from the Gaussian process."""
        return np.random.randn(*self.size) * self.std


class OrnsteinUhlenbeckProcess(RandomProcess):
    """An Ornstein-Uhlenbeck process."""

    def __init__(
        self,
        size: tuple[int],
        std: float,
        theta: float = 0.15,
        dt: float = 1e-2,
        x0: np.ndarray | None = None,
    ):
        """
        Initialize a new Ornstein-Uhlenbeck process.

        Args:
            size: The shape of the data to generate.
            std: The standard deviation of the noise.
            theta: The rate of mean reversion.
            dt: The time step size.
            x0: The initial state of the process.
        """
        self.theta = theta
        self.mu = 0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self) -> np.ndarray:
        """Generate a sample from the Ornstein-Uhlenbeck process."""
        x = self.previous_state + self.theta * (self.mu - self.previous_state) * self.dt + (
            self.std * np.sqrt(self.dt) * np.random.randn(*self.size)
        )
        self.previous_state = x
        return x

    def reset_states(self) -> None:
        """Reset the internal state of the process."""
        self.previous_state = self.x0 if self.x0 is not None else np.zeros(self.size)