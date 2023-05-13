import numpy as np


class RandomProcess:
    """
    Abstract class for generating random processes
    """
    def reset_states(self):
        """
        Resets the states of the random process
        """
        pass


class GaussianProcess(RandomProcess):
    """
    Generates a Gaussian process
    """
    def __init__(self, dimension, standard_deviation):
        """
        Initializes the Gaussian process

        :param dimension: Dimension of the process
        :param standard_deviation: Standard deviation of the process
        """
        self.dimension = dimension
        self.standard_deviation = standard_deviation

    def sample(self):
        """
        Generates a sample from the Gaussian process

        :return: A sample from the process
        """
        return np.random.randn(*self.dimension) * self.standard_deviation


class OrnsteinUhlenbeckProcess(RandomProcess):
    """
    Generates an Ornstein-Uhlenbeck process
    """
    def __init__(self, dimension, standard_deviation, theta=0.15, time_step=1e-2, initial_state=None):
        """
        Initializes the Ornstein-Uhlenbeck process

        :param dimension: Dimension of the process
        :param standard_deviation: Standard deviation of the process
        :param theta: Rate of reversion to the mean
        :param time_step: Time step
        :param initial_state: Initial state of the process
        """
        self.theta = theta
        self.mean = 0
        self.standard_deviation = standard_deviation
        self.time_step = time_step
        self.previous_state = initial_state
        self.dimension = dimension
        self.reset_states()

    def sample(self):
        """
        Generates a sample from the Ornstein-Uhlenbeck process

        :return: A sample from the process
        """
        current_state = (self.previous_state + 
                         self.theta * (self.mean - self.previous_state) * self.time_step +
                         self.standard_deviation * np.sqrt(self.time_step) * np.random.randn(*self.dimension))
        self.previous_state = current_state
        return current_state

    def reset_states(self):
        """
        Resets the states of the Ornstein-Uhlenbeck process
        """
        self.previous_state = self.initial_state if self.initial_state is not None else np.zeros(self.dimension)