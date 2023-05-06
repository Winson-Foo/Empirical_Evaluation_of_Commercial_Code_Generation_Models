import torch
import numpy as np
import torch.multiprocessing as mp
from collections import deque
from skimage.io import imsave

from ..utils import get_logger, close_obj, mkdir


class BaseAgent:
    """Base class for reinforcement learning agents."""
    
    def __init__(self, config):
        """Construct an instance of BaseAgent.
        
        Args:
            config (Config): Configuration object containing hyperparameters and settings.
        """
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0

    def close(self):
        """Close the environment or connection to it."""
        close_obj(self.task)

    def save(self, filename):
        """Save the agent's learned parameters and statistics to disk.
        
        Args:
            filename (str): Name of the file to save to (without extension).
        """
        torch.save(self.network.state_dict(), f'{filename}.model')
        with open(f'{filename}.stats', 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename):
        """Load the agent's learned parameters and statistics from disk.
        
        Args:
            filename (str): Name of the file to load from (without extension).
        """
        state_dict = torch.load(f'{filename}.model', map_location=torch.device('cpu'))
        self.network.load_state_dict(state_dict)
        with open(f'{filename}.stats', 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, state):
        """Choose an action based on a state during evaluation.
        
        Args:
            state (np.ndarray): Current state of the environment.
            
        Returns:
            int: Chosen action for the given state.
        """
        raise NotImplementedError

    def eval_episode(self):
        """Run a single episode of the evaluation environment.
        
        Returns:
            float: The total reward obtained during the episode.
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
        """Run multiple episodes of the evaluation environment and compute their average total reward.
        
        Returns:
            dict: Results of the evaluation, as a dictionary of metric names and their values.
        """
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
        mean_return = np.mean(episodic_returns)
        std_return = np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        self.logger.info(f'steps {self.total_steps}, episodic_return_test {mean_return:.2f} ({std_return:.2f})')
        self.logger.add_scalar('episodic_return_test', mean_return, self.total_steps)
        return {
            'episodic_return_test': mean_return,
        }

    def record_online_return(self, info, offset=0):
        """Record the return obtained during a single training episode.
        
        Args:
            info (dict): Dictionary containing training information, including the episodic return.
            offset (int): Number of steps to add to the current step count.
        """
        if isinstance(info, dict):
            ret = info['episodic_return']
            if ret is not None:
                self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
                self.logger.info(f'steps {self.total_steps + offset}, episodic_return_train {ret}')
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError

    def switch_task(self):
        """Switch to the next task in the list of training tasks."""
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
        """Record a video of a single episode of the training environment.
        
        Args:
            dir (str): Directory to save the video in.
            env (gym.Env): Environment to record the video of.
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
        """Choose an action based on a state during recording of a training episode.
        
        Args:
            state (np.ndarray): Current state of the environment.
            
        Returns:
            int: Chosen action for the given state.
        """
        raise NotImplementedError

    def record_obs(self, env, dir, steps):
        """Record an observation of the environment during training.
        
        Args:
            env (gym.Env): Environment to record an observation of.
            dir (str): Directory to save the observation in.
            steps (int): Index of the current time step.
        """
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave(f'{dir}/{steps:04d}.png', obs)


class BaseActor(mp.Process):
    """Base class for parallel actor processes that interact with the environment."""
    
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, config):
        """Construct an instance of BaseActor.
        
        Args:
            config (Config): Configuration object containing hyperparameters and settings.
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

    def run(self):
        """Main loop of the actor process."""
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

    def step(self):
        """Take a step in the environment."""
        self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self):
        """Send a signal to the actor process to exit."""
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        """Set the network used by the actor for action selection."""
        if not self.config.async_actor:
            self._network = net
        else:
            self.__pipe.send([self.NETWORK, net])

    def _sample(self):
        """Sample a transition from the environment."""
        transitions = []
        for _ in range(self.config.sgd_update_frequency):
            transition = self._transition()
            if transition is not None:
                transitions.append(transition)
        return transitions

    def _transition(self):
        """Advance the environment and produce a transition tuple containing the previous state, action, reward, next state and done flag.
        
        Returns:
            tuple or None: A transition tuple (state, action, reward, next_state, done) if the episode is not done, else None.
        """
        raise NotImplementedError

    def _set_up(self):
        """Perform any necessary set up before running the main loop."""
        pass