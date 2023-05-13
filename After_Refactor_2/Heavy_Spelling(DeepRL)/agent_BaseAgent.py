import torch
import numpy as np
import torch.multiprocessing as mp
from collections import deque
from skimage.io import imsave


class BaseAgent:
    """
    Base class for RL agents.
    """
    def __init__(self, config):
        """
        Initializes the agent.
        :param config: Configuration object containing the agent's settings.
        """
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0

    def close(self):
        """
        Closes the agent's task object.
        """
        close_obj(self.task)

    def save(self, filename):
        """
        Saves the agent's network object and state normalizer object to files.
        :param filename: Prefix to use for the filenames to save the files with.
        """
        torch.save(self.network.state_dict(), f"{filename}.model")
        with open(f"{filename}.stats", "wb") as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename):
        """
        Loads the agent's network object and state normalizer object from files.
        :param filename: Prefix to use for the filenames to load the files from.
        """
        state_dict = torch.load(f"{filename}.model", map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open(f"{filename}.stats", "rb") as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, state):
        """
        Perform a single step of evaluation given a state.
        :param state: Current state.
        :return: Action to take.
        """
        raise NotImplementedError

    def eval_episode(self):
        """
        Run an evaluation episode to test the agent's performance.
        :return: Total reward accumulated during the episode.
        """
        env = self.config.eval_env
        state = env.reset()
        while True:
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]["episodic_return"]
            if ret is not None:
                break
        return ret

    def eval_episodes(self):
        """
        Run multiple evaluation episodes and calculate their average performance.
        :return: Dictionary containing the average episodic return.
        """
        episodic_returns = []
        for _ in range(self.config.eval_episodes):
            total_reward = self.eval_episode()
            episodic_returns.append(np.sum(total_reward))
        mean_return = np.mean(episodic_returns)
        std_error = np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        self.logger.info(f"steps {self.total_steps}, episodic_return_test {mean_return:.2f}({std_error:.2f})")
        self.logger.add_scalar("episodic_return_test", mean_return, self.total_steps)
        return {"episodic_return_test": mean_return}

    def record_online_return(self, info, offset=0):
        """
        Record the current online return during training.
        :param info: Information containing the current episodic return.
        :param offset: Offset to use for the logger.
        """
        if isinstance(info, dict):
            ret = info["episodic_return"]
            if ret is not None:
                self.logger.add_scalar("episodic_return_train", ret, self.total_steps + offset)
                self.logger.info(f"steps {self.total_steps + offset}, episodic_return_train {ret}")
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError

    def switch_task(self):
        """
        Switch to the next sub-task in the agent's task list.
        """
        if not self.config.tasks:
            return
        segs = np.linspace(0, self.config.max_steps, len(self.config.tasks) + 1)
        if self.total_steps > segs[self.task_ind + 1]:
            self.task_ind += 1
            self.task = self.config.tasks[self.task_ind]
            self.states = self.task.reset()
            self.states = self.config.state_normalizer(self.states)

    def record_episode(self, dir, env):
        """
        Record an entire episode and save it to disk.
        :param dir: Directory to save the frames to.
        :param env: Environment object to record the episode on.
        """
        mkdir(dir)
        steps = 0
        state = env.reset()
        while True:
            self.record_obs(env, dir, steps)
            action = self.record_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]["episodic_return"]
            steps += 1
            if ret is not None:
                break

    def record_step(self, state):
        """
        Record a single step during an episode.
        :param state: Current state.
        :return: Action to take.
        """
        raise NotImplementedError

    def record_obs(self, env, dir, steps):
        """
        Record the current observation frame during an episode.
        :param env: Environment object to capture the frame from.
        :param dir: Directory to save the frame to.
        :param steps: Current step number.
        """
        env = env.env.envs[0]
        obs = env.render(mode="rgb_array")
        imsave(f"{dir}/{steps:04d}.png", obs)


class BaseActor(mp.Process):
    """
    Base class for actor objects.
    """
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, config):
        """
        Initializes the actor object.
        :param config: Configuration object containing the actor's settings.
        """
        mp.Process.__init__(self)
        self.config = config
        self.__pipe, self.__worker_pipe = mp.Pipe()

        self._state = None
        self._task = None
        self._network = None
        self._total_steps = 0
        self.__cache_len = 2

        if not config.async_actor:
            self.start = lambda: None
            self.step = self._sample
            self.close = lambda: None
            self._set_up()
            self._task = config.task_fn()

    def _sample(self):
        """
        Sample a sequence of transitions from the actor's task.
        :return: List of sampled transitions.
        """
        transitions = []
        for _ in range(self.config.sgd_update_frequency):
            transition = self._transition()
            if transition is not None:
                transitions.append(transition)
        return transitions

    def run(self):
        """
        Run the actor process's main loop.
        """
        self._set_up()
        config = self.config
        self._task = config.task_fn()

        cache = deque([], maxlen=self.__cache_len)
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
                self._network = data
            else:
                raise NotImplementedError

    def _transition(self):
        """
        Sample a single transition from the actor's task.
        :return: Transition tuple.
        """
        raise NotImplementedError

    def _set_up(self):
        """
        Set up the actor's initial state and/or connections.
        """
        pass

    def step(self):
        """
        Perform a single step of the actor's task.
        :return: List of sampled transitions.
        """
        self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self):
        """
        Close the actor process's connections.
        """
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        """
        Set the actor's network object.
        :param net: Network object to use.
        """
        if not self.config.async_actor:
            self._network = net
        else:
            self.__pipe.send([self.NETWORK, net])