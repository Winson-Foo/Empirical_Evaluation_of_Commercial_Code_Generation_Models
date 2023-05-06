import numpy as np


class RandomProcess:
    def reset_states(self):
        pass


class GaussianProcess(RandomProcess):
    def __init__(self, size, std):
        self.size = size
        self.std = std

    def sample(self):
        return np.random.normal(scale=self.std, size=self.size)


class OrnsteinUhlenbeckProcess(RandomProcess):
    def __init__(self, size, std, theta=0.15, dt=0.01, x0=None):
        self.theta = theta
        self.mu = 0 
        self.std = std
        self.dt = dt
        self.x_prev = x0 if x0 is not None else np.zeros(size)
        self.size = size

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.std * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        return x

    def reset_states(self):
        self.x_prev = np.zeros(self.size) if self.x_prev is None else self.x_prev