# ornstein_uhlenbeck_process.py
import numpy as np

class OrnsteinUhlenbeckProcess:
    """An Ornstein-Uhlenbeck process for generating correlated noise."""

    def __init__(self, size: tuple, std: float, theta: float = 0.15,
                 dt: float = 1e-2, x0: np.ndarray = None) -> None:
        """Initialize the Ornstein-Uhlenbeck process.

        Args:
            size (tuple): The size of the noise array to generate.
            std (float): The standard deviation of the noise.
            theta (float): The rate of mean reversion.
            dt (float): The time step size.
            x0 (np.ndarray): The initial value of the noise.
        """
        self.theta = theta
        self.mu = 0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self) -> np.ndarray:
        """Generate a sample of noise."""
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.std() * np.sqrt(
            self.dt) * np.random.randn(*self.size)
        self.x_prev = x
        return x

    def reset_states(self) -> None:
        """Reset the state of the process."""
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)