"""
Module containing StandardNormal distribution.
"""

import torch

class StandardNormal:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def log_prob(self, inputs, context=None):
        # Compute log probability of inputs under the standard normal distribution.
        pass

    def sample(self, num_samples, context=None):
        # Sample from the standard normal distribution.
        pass

    def sample_and_log_prob(self, num_samples, context=None):
        # Sample from the standard normal distribution and compute log probability.
        pass
    
    def mean(self, context=None):
        # Compute the mean of the standard normal distribution.
        pass