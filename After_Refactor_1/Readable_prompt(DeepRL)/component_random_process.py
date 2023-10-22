import numpy as np


class RandomProcess:
    def reset_states(self):
        pass


class GaussianProcess(RandomProcess):
    def __init__(self, size, std_deviation):
        self.size = size
        self.std_deviation = std_deviation

    def sample(self):
        return np.random.randn(*self.size) * self.std_deviation


class OrnsteinUhlenbeckProcess(RandomProcess):
    def __init__(self, size, std_deviation, theta=0.15, delta_time=1e-2, initial_x=None):
        self.theta = theta
        self.mean = 0
        self.std_deviation = std_deviation
        self.delta_time = delta_time
        self.initial_x = initial_x
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.previous_x + self.theta * (self.mean - self.previous_x) * self.delta_time + self.std_deviation() * np.sqrt(
            self.delta_time) * np.random.randn(*self.size)
        self.previous_x = x
        return x

    def reset_states(self):
        self.previous_x = self.initial_x if self.initial_x is not None else np.zeros(self.size)