import torch
import numpy as np
from ..utils import *
import torch.multiprocessing as mp
from collections import deque
from skimage.io import imsave

class BaseAgent:
    """
    Base class for all RL agents.
    """
    def __init__(self, config):
        """
        Initializes the agent.
        """
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0

    def close(self):
        """
        Closes the current task.
        """
        close_obj(self.task)

    def save(self, filename):
        """
        Saves the agent's state and statistics to a file.
        """
        torch.save(self.network.state_dict(), f"{filename}.model")
        with open(f"{filename}.stats", "wb") as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename):
        """
        Loads the agent's state and statistics from a file.
        """
        state_dict = torch.load(f"{filename}.model", map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open(f"{filename}.stats", "rb") as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, state):
        """
        Calculates the action to take in a given state during evaluation.
        """
        raise NotImplementedError

    def eval_episode(self):
        """
        Runs an episode of the current task during evaluation.
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
        Runs multiple episodes of the current task during evaluation and logs the overall return.
        """
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
        mean_return, std_return = np.mean(episodic_returns), np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        self.logger.info(f"steps {self.total_steps}, episodic_return_test {mean_return:.2f}({std_return:.2f})")
        self.logger.add_scalar('episodic_return_test', mean_return, self.total_steps)
        return {'episodic_return_test': mean_return}

    def record_online_return(self, info, offset=0):
        """
        Records the online return for the current task.
        """
        if isinstance(info, dict):
            ret = info['episodic_return']
            if ret is not None:
                self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
                self.logger.info(f"steps {self.total_steps + offset}, episodic_return_train {ret}")
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError

    def switch_task(self):
        """
        Switches to the next task when current task is completed.
        """
        config = self.config
        if not config.tasks:
            return
        segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
        if self.total_steps > segs[self.task_ind + 1]:
            self.task_ind += 1
            self.task = config.tasks[self.task_ind]
            self.states = self.task.reset()
            self.states = config.state_normalizer(self.states)

    def record_episode(self, dir, env):
        """
        Records a video of the current task.
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
        Records and returns the action taken to reach the given state.
        """
        raise NotImplementedError

    def record_obs(self, env, dir, steps):
        """
        Records an observation during video recording.
        """
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave(f"{dir}/{steps:04d}.png", obs)

class BaseActor(mp.Process):
    """
    Base class for all RL actors.
    """
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, config):
        """
        Initializes the actor.
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
        Samples transitions from current task and returns them.
        """
        transitions = []
        for _ in range(self.config.sgd_update_frequency):
            transition = self._transition()
            if transition is not None:
                transitions.append(transition)
        return transitions

    def run(self):
        """
        Processes the current task and communicates with the main process.
        """
        self._set_up()
        config = self.config
        self._task = config.task_fn()

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
                self._network = data
            else:
                raise NotImplementedError

    def _transition(self):
        """
        Samples a single transition from the current task and returns it.
        """
        raise NotImplementedError

    def _set_up(self):
        pass

    def step(self):
        """
        Sends a signal to the worker process to process the current task and returns the results.
        """
        self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self):
        """
        Sends a signal to the worker process to exit.
        """
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        """
        Sets the network for the current task.
        """
        if not self.config.async_actor:
            self._network = net
        else:
            self.__pipe.send([self.NETWORK, net])