import numpy as np


class RandomProcess(object):
    """
    Base class for random processes.
    """

    def reset_states(self):
        """
        Resets any state that the process may have accumulated during sampling.
        """
        pass


class GaussianProcess(RandomProcess):
    """
    Gaussian process for adding noise to action selections.
    """

    def __init__(self, size, standard_deviation):
        """
        Creates a new GaussianProcess instance.

        :param size: The size of the noise vector to generate.
        :param standard_deviation: The standard deviation of the noise to add.
        """
        self.size = size
        self.standard_deviation = standard_deviation

    def sample(self):
        """
        Generates a sample of noise from the Gaussian process.

        :return: A vector of noise with the specified size and standard deviation.
        """
        return np.random.randn(*self.size) * self.standard_deviation


class OrnsteinUhlenbeckProcess(RandomProcess):
    """
    Ornstein-Uhlenbeck process for adding temporally correlated noise to action selections.
    """

    def __init__(self, size, standard_deviation, theta=0.15, time_delta=1e-2, x0=None):
        """
        Creates a new OrnsteinUhlenbeckProcess instance.

        :param size: The size of the noise vector to generate.
        :param standard_deviation: The standard deviation of the process noise.
        :param theta: The rate at which the process reverts back to the mean.
        :param time_delta: The time step between successive samples.
        :param x0: The initial state of the process.
        """
        self.theta = theta
        self.mu = 0
        self.standard_deviation = standard_deviation
        self.time_delta = time_delta
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        """
        Generates a sample of noise from the Ornstein-Uhlenbeck process.

        :return: A vector of noise with the specified size and correlation properties.
        """
        # Calculate the new state from the previous state
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.time_delta \
            + self.standard_deviation * np.sqrt(self.time_delta) * np.random.randn(*self.size)

        # Update the previous state with the new state
        self.x_prev = x

        return x

    def reset_states(self):
        """
        Resets the Ornstein-Uhlenbeck process state to the initial value.
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)