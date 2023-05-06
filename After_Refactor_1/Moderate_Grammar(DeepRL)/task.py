# file: task.py
import numpy as np
import gym

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from env_utils import create_env

class Task:
    def __init__(self,
                 env_name: str,
                 num_envs: int = 1,
                 single_process: bool = True,
                 log_dir: str = None,
                 episode_life: bool = True,
                 seed: int = None):

        if seed is None:
            seed = np.random.randint(int(1e9))
        if log_dir is not None:
            mkdir(log_dir)

        self.envs = [create_env(env_name, seed, episode_life) for i in range(num_envs)]
        self.env = gym.wrappers.FlattenObservation(gym.vector.VectorEnv(self.envs))
        self.name = env_name
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            raise ValueError('unknown action space')

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)