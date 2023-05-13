import numpy as np


class RandomProcess:
    def reset_states(self):
        pass


class GaussianProcess(RandomProcess):
    def __init__(self, size, std):
        self.size = size
        self.std_deviation = std
        
    def sample(self):
        return np.random.randn(*self.size) * self.std_deviation


class OrnsteinUhlenbeckProcess(RandomProcess):
    def __init__(self, size, std_deviation, theta=0.15, dt=0.01, initial_value=None):
        self.theta = theta
        self.mean = 0
        self.std_deviation = std_deviation
        self.dt = dt
        self.initial_value = initial_value
        self.size = size
        self._reset_states()
        
    def sample(self):
        current_value = self.previous_value + self.theta * (self.mean - self.previous_value) * self.dt + self.std_deviation * np.sqrt(self.dt) * np.random.randn(*self.size)
        self.previous_value = current_value
        return current_value
    
    def reset_states(self):
        self._reset_states()
    
    def _reset_states(self):
        self.previous_value = self.initial_value if self.initial_value is not None else np.zeros(self.size)