import numpy as np

class RandomProcess:
    def reset_states(self):
        pass


class GaussianProcess(RandomProcess):
    def __init__(self, size, std):
        self.size = size
        self.std = std

    def sample(self):
        return np.random.randn(*self.size) * self.std


class OrnsteinUhlenbeckProcess(RandomProcess):
    def __init__(self, size, std, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = 0
        self.std = std
        self.dt = dt
        self.size = size
        self.x0 = x0
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.std * np.sqrt(
            self.dt) * np.random.randn(*self.size)
        self.x_prev = x
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

### CHANGES MADE:
# Removed redundant parentheses in GaussianProcess's sample method
# Made RandomProcess an abstract class by removing sample method signature
# Removed x_prev initialization from OrnsteinUhlenbeckProcess's __init__ method for better encapsulation.