import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
from collections import deque
from skimage.io import imsave
from utils import *


class BaseAgent:
    def __init__(self, config):
        """
        Initializes the Base Agent with the provided configuration.

        Args:
        config: An object of the Config class.
        """
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_index = 0

    def close(self):
        """
        Closes the task.
        """
        close_obj(self.task)

    def save(self, filename):
        """
        Saves the model by serializing the state_dict of the network and the state_normalizer.

        Args:
        filename: Name with which the model will be saved.
        """
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename):
        """
        Loads the model to the network by de-serializing the state_dict. The state_normalizer is loaded
        from the pickled object.

        Args:
        filename: Name of the model to be loaded.
        """
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, state):
        """
        Evaluate each state.

        Args:
        state: a state in the state-space.

        Raises:
        NotImplementedError.
        """
        raise NotImplementedError

    def eval_episode(self):
        """
        Completes a single episode of the environment.

        Returns:
        Total rewards obtained after completion.
        """
        env = self.config.eval_env
        state = env.reset()
        while True:
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            if ret is not None:
                break
        return ret

    def eval_episodes(self):
        """
        Evaluates the agent over multiple episodes.

        Returns:
        A dictionary with the mean and standard deviation of episodic return of the agent.
        """
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
        self.logger.info('steps %d, episodic_return_test %.2f(%.2f)' % (
            self.total_steps, np.mean(episodic_returns), np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        ))
        self.logger.add_scalar('episodic_return_test', np.mean(episodic_returns), self.total_steps)
        return {
            'episodic_return_test': np.mean(episodic_returns),
        }

    def record_online_return(self, info, offset=0):
        """
        Records the episodic return of the agent in the dictionary and the logger.

        Args:
        info: An object of type dict or tuple containing the episodic return.
        offset: Time step from which the recording needs to start.

        Raises:
        NotImplementedError.
        """
        if isinstance(info, dict):
            ret = info['episodic_return']
            if ret is not None:
                self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
                self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps + offset, ret))
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError

    def switch_task(self):
        """
        Switches the task when the max step limit is reached.

        Raises:
        IndexError if the index exceeds the length of tasks.
        """
        config = self.config
        if not config.tasks:
            return
        segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
        if self.total_steps > segs[self.task_index + 1]:
            self.task_index += 1
            self.task = config.tasks[self.task_index]
            self.states = self.task.reset()
            self.states = config.state_normalizer(self.states)

    def record_episode(self, dir, env):
        """
        Records an episode of the environment.

        Args:
        dir: Directory where the recorded images are to be stored.
        env: An instance of the OpenAI gym environment.

        Returns:
        None.
        """
        mkdir(dir)
        steps = 0
        state = env.reset()
        while True:
            self.record_obs(env, dir, steps)
            action = self.record_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            steps += 1
            if ret is not None:
                break

    def record_step(self, state):
        """
        Records a single step in the episode.

        Args:
        state: Current state in the state space.

        Raises:
        NotImplementedError.
        """
        raise NotImplementedError

    # For DMControl
    def record_obs(self, env, dir, steps):
        """
        Records the observations in the environment and stores it in the directory.

        Args:
        env: An instance of the OpenAI gym environment.
        dir: Directory where the images are to be stored.
        steps: Current step count.

        Returns:
        None.
        """
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave('%s/%04d.png' % (dir, steps), obs)


class BaseActor(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, config):
        """
        Initializes the Base Actor with the provided configuration.

        Args:
        config: An object of the Config class.
        """
        mp.Process.__init__(self)
        self.config = config
        self.__pipe, self.__worker_pipe = mp.Pipe()

        self.state = None
        self.task = None
        self.network = None
        self.total_steps = 0
        self.__cache_len = 2

        if not config.async_actor:
            self.start = lambda: None
            self.step = self._sample
            self.close = lambda: None
            self._set_up()
            self.task = config.task_fn()

    def _sample(self):
        """
        Collect samples from the worker and create a list of transitions for each SGD step.

        Returns:
        transitions: List of transitions for each SGD step.
        """
        transitions = []
        for _ in range(self.config.sgd_update_frequency):
            transition = self._transition()
            if transition is not None:
                transitions.append(transition)
        return transitions

    def run(self):
        """
        Starts running the worker.
        """
        self._set_up()
        config = self.config
        self.task = config.task_fn()

        cache = deque([], maxlen=2)
        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.STEP:
                if not len(cache):
                    cache.append(self._sample())
                    cache.append(self._sample())
                self.__worker_pipe.send(cache.popleft())
                cache.append(self._sample())
            elif op == self.EXIT:
                self.__worker_pipe.close()
                return
            elif op == self.NETWORK:
                self.network = data
            else:
                raise NotImplementedError

    def _transition(self):
        """
        Repeatedly samples actions from the policy and stores the results.

        Returns:
        transition: One transition of the episode.
        """
        raise NotImplementedError

    def _set_up(self):
        """
        Placeholder method.
        """
        pass

    def step(self):
        """
        Sends a step signal to the worker and returns the transitions.

        Returns:
        transitions: List of transitions.
        """
        self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self):
        """
        Sends a signal to the worker to stop and closes pipes.
        """
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()