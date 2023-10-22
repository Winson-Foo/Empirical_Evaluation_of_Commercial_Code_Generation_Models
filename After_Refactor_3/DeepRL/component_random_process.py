import numpy as np


class RandomProcess:
    """
    This is a parent class for Gaussian and Ornstein-Uhlenbeck process.
    """

    def reset_states(self):
        """
        Resets the state of the process.
        """
        pass


class GaussianProcess(RandomProcess):
    """
    This class represents a Gaussian process.

    Attributes:
    -----------
    size : tuple
        The shape of the output array.
    std : float
        Standard deviation of the normal distribution to sample from.
    
    Methods:
    --------
    sample() -> ndarray:
        Returns an array of shape `size` sampled from a normal distribution
        with zero mean and `std` standard deviation.

    reset_states() -> None:
        Resets the state of the process.
    """

    def __init__(self, size: tuple, std: float):
        self.size = size
        self.std = std

    def sample(self) -> np.ndarray:
        return np.random.randn(*self.size) * self.std

    def reset_states(self) -> None:
        pass


class OrnsteinUhlenbeckProcess(RandomProcess):
    """
    This class represents an Ornstein-Uhlenbeck process.

    Attributes:
    -----------
    size : tuple
        The shape of the output array.
    std : float
        Standard deviation of the normal distribution to sample from.
    theta : float
        The rate at which the process reverts to the mean.
    dt : float
        The time step between successive samples.
    x0 : ndarray or None
        The initial state of the process. If None, the process starts from zero.

    Methods:
    --------
    sample() -> ndarray:
        Returns an array of shape `size` sampled from an Ornstein-Uhlenbeck process.

    reset_states() -> None:
        Resets the state of the process.
    """

    def __init__(self, size: tuple, std: float, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = 0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self) -> np.ndarray:
        """
        Returns an array of shape `size` sampled from an Ornstein-Uhlenbeck process.
        """
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.std * np.sqrt(
            self.dt) * np.random.randn(*self.size)
        self.x_prev = x
        return x

    def reset_states(self) -> None:
        """
        Resets the state of the process.
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)