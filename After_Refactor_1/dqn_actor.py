from ..networks import *
from ..components import *
from ..utils import *
from .base_actor import BaseActor


class DQNActor(BaseActor):
    """
    An actor for a DQN agent.
    """

    def __init__(self, config):
        """
        Initialize the actor.

        Args:
        - config: a configuration object
        """
        BaseActor.__init__(self, config)
        self.config = config
        self.start()

    def compute_q_values(self, prediction):
        """
        Compute the Q-values from the neural network's output.

        Args:
        - prediction: a dictionary containing the neural network's output

        Returns:
        - A numpy array of Q-values
        """
        q_values = to_np(prediction['q'])
        return q_values

    def _transition(self):
        """
        Take a step in the environment and store the resulting transition in the buffer.

        Returns:
        - A list containing a transition (state, action, reward, next_state, done, info)
        """
        if self._state is None:
            self._state = self._task.reset()

        if self.config.noisy_linear:
            self._network.reset_noise()

        with self.config.lock:
            prediction = self._network(self.config.state_normalizer(self._state))

        q_values = self.compute_q_values(prediction)

        if self.config.noisy_linear:
            epsilon = 0
        elif self._total_steps < self.config.exploration_steps:
            epsilon = 1
        else:
            epsilon = self.config.random_action_prob()

        action = epsilon_greedy(epsilon, q_values)

        next_state, reward, done, info = self._task.step(action)

        entry = [self._state, action, reward, next_state, done, info]
        self._total_steps += 1
        self._state = next_state

        return entry