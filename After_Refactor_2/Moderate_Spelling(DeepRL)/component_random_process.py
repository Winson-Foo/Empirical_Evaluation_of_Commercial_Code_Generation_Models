import numpy as np

class RandomProcess:
    """Abstract base class for all random processes."""
    def reset_states(self):
        """Reset the internal state of the process."""
        pass

    def sample(self):
        """Return a random sample from the process."""
        pass

class GaussianProcess(RandomProcess):
    """A Gaussian process."""
    def __init__(self, size, std):
        self.size = size
        self.std = std

    def sample(self):
        """Return a random sample from the Gaussian process."""
        return np.random.randn(*self.size) * self.std

class OrnsteinUhlenbeckProcess(RandomProcess):
    """An Ornstein-Uhlenbeck process."""
    def __init__(self, size, std, theta=0.15, dt=1e-2, x0=None):
        self.size = size
        self.std = std
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset_states()

    def reset_states(self):
        """Reset the internal state of the process."""
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

    def sample(self):
        """Return a random sample from the OU process."""
        x = self.x_prev + self.theta * (self.std - self.x_prev) * self.dt + np.sqrt(self.dt) * self.std * np.random.randn(*self.size)
        self.x_prev = x
        return x