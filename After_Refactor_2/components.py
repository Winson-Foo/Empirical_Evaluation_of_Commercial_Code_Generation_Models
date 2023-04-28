
import numpy as np
import torch


class ReplayBuffer:
    """
    A replay buffer that stores transitions and allows for sampling batches of transitions for training.
    
    :param max_size: The maximum number of transitions that can be stored in the buffer.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size

        self.states = np.empty((max_size, 1))
        self.actions = np.empty((max_size, 1))
        self.rewards = np.empty((max_size, 1))
        self.next_states = np.empty((max_size, 1))
        self.masks = np.empty((max_size, 1))

        self.index = 0
        self.size = 0

    def add(self, transition: dict) -> None:
        """
        Adds a transition to the buffer.

        :param transition: A dictionary containing the state, action, reward, next state, and mask of the transition.
        """

        self.states[self.index] = transition["state"]
        self.actions[self.index] = transition["action"]
        self.rewards[self.index] = transition["reward"]
        self.next_states[self.index] = transition["next_state"]
        self.masks[self.index] = transition["mask"]

        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> dict:
        """
        Samples a batch of transitions from the buffer.

        :param batch_size: The size of the batch to sample.
        :return: A dictionary containing the sampled transitions.
        """

        indices = np.random.randint(0, self.size, batch_size)

        return dict(
            state=self.states[indices],
            action=self.actions[indices],
            reward=self.rewards[indices],
            next_state=self.next_states[indices],
            mask=self.masks[indices],
        )

    def size(self) -> int:
        """
        Returns the number of transitions stored in the buffer.

        :return: The number of transitions stored in the buffer.
        """

        return self.size


class RandomProcess:
    """
    A random process that generates noise to encourage exploration during training.
    """

    def reset_states(self) -> None:
        """
        Resets the internal state of the random process.
        """

        pass

    def sample(self) -> np.ndarray:
        """
        Samples a single noise value.

        :return: A numpy array containing the sampled noise value.
        """

        pass


class Config:
    """
    A configuration object containing parameters for the agent and its components.
    """

    def __init__(self, task_fn, state_normalizer, reward_normalizer, network_fn, replay_buffer_fn, random_process_fn,
                 logger, warm_up=1000, target_network_mix=0.001, discount=0.99):
        self.task_fn = task_fn
        self.state_normalizer = state_normalizer
        self.reward_normalizer = reward_normalizer
        self.network_fn = network_fn
        self.replay_buffer_fn = replay_buffer_fn
        self.random_process_fn = random_process_fn
        self.logger = logger
        self.warm_up = warm_up
        self.target_network_mix = target_network_mix
        self.discount = discount