import numpy as np

class RewardNormalizer:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def __call__(self, reward):
        reward = (reward - self.mean) / (self.std + 1e-8)
        return reward

    def update(self, reward):
        self.mean = (reward.mean() + self.mean * self.count) / (self.count + 1)
        self.std = (reward.std() + self.std * self.count) / (self.count + 1)
        self.count += 1

class StateNormalizer:
    def __init__(self, state_size):
        self.mean = np.zeros(state_size)
        self.std = np.ones(state_size)
        self.count = 0

    def __call__(self, state):
        state = (state - self.mean) / (self.std + 1e-8)
        return state

    def update(self, state):
        self.mean = (self.mean * self.count + state.mean(axis=0)) / (self.count + 1)
        self.std = (self.std * self.count + state.std(axis=0)) / (self.count + 1)
        self.count += 1