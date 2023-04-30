import os
import gym
import numpy as np
import torch

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.atari_wrappers import FrameStack as FrameStack_
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv

from ..utils import *


def make_env(env_id, seed, rank, episode_life=True):
    """
    Creates a new gym environment with the given ID, seed, and rank.

    Args:
        env_id (str): The ID of the gym environment.
        seed (int): The random seed for this environment instance.
        rank (int): The rank of this environment instance.
        episode_life (bool, optional): Whether to use an episodic life in
            the environment (used only for Atari environments). Defaults to True.

    Returns:
        The created gym environment.

    """
    def _thunk():
        random_seed(seed)
        if env_id.startswith("dm"):
            import dm_control2gym
            _, domain, task = env_id.split('-')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        env = OriginalReturnWrapper(env)
        if is_atari:
            env = wrap_deepmind(env,
                                episode_life=episode_life,
                                clip_rewards=False,
                                frame_stack=False,
                                scale=False)
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3:
                env = TransposeImage(env)
            env = FrameStack(env, 4)

        return env

    return _thunk


class OriginalReturnWrapper(gym.Wrapper):
    """
    Wraps a gym environment to add an episodic return to the `info` dictionary.

    Args:
        env (gym.Env): The gym environment to wrap.

    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class TransposeImage(gym.ObservationWrapper):
    """
    Transposes the image observation of a gym environment.

    Args:
        env (gym.Env): The gym environment to wrap.

    """
    def __init__(self, env=None):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class LazyFrames(object):
    """
    Optimizes the memory usage of repeated frames in observations.

    Args:
        frames (list): A list of observation frames.

    """
    def __init__(self, frames):
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


class FrameStack(FrameStack_):
    """
    Stacks the observation frames of a gym environment into a single array.

    Args:
        env (gym.Env): The gym environment to wrap.
        k (int): The number of frames to stack.

    """
    def __init__(self, env, k):
        super().__init__(env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class DummyVecEnv(VecEnv):
    """
    A dummy implementation of the VecEnv class that works for a single process.

    Args:
        env_fns (list): A list of factory functions for creating gymn environments.

    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        data = []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(self.actions[i])
            if done:
                obs = self.envs[i].reset()
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        return obs, np.asarray(rew), np.asarray(done), info

    def reset(self):
        return [env.reset() for env in self.envs]


class Task:
    """
    A high-level wrapper for a gym environment with some additional features.

    Args:
        name (str): The ID of the gym environment.
        num_envs (int, optional): The number of environment instances to create. Defaults to 1.
        single_process (bool, optional): Whether to run the environment instances in a single process or in multiple processes. Defaults to True.
        log_dir (str, optional): The directory to save log files to. Defaults to None.
        episode_life (bool, optional): Whether to use an episodic life in the environment (used only for Atari environments). Defaults to True.
        seed (int, optional): The random seed for this environment instance. If None, a random seed will be generated. Defaults to None.

    Attributes:
        env (VecEnv or SubprocVecEnv): The wrapped gym environment.
        name (str): The ID of the gym environment.
        observation_space (gym.spaces.space.Space): The observation space of the gym environment.
        state_dim (int): The dimensionality of the observation space.
        action_space (gym.spaces.space.Space): The action space of the gym environment.
        action_dim (int): The dimensionality of the action space.

    """
    def __init__(self,
                 name,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=None):
        if seed is None:
            seed = np.random.randint(int(1e9))
        if log_dir is not None:
            mkdir(log_dir)
        envs = [make_env(name, seed, i, episode_life) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
        else:
            Wrapper = SubprocVecEnv
        self.env = Wrapper(envs)
        self.name = name
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    def reset(self):
        """Resets the wrapped gym environment."""
        return self.env.reset()

    def step(self, actions):
        """
        Advances the wrapped gym environment with the given actions.

        Args:
            actions (numpy.ndarray): The actions to take in the environment.

        Returns:
            The new observation state, the rewards received, the flags indicating
            whether the episode is done, and additional information.

        """
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)


if __name__ == '__main__':
    task = Task('Hopper-v2', 5, single_process=False)
    state = task.reset()
    while True:
        action = np.random.rand(task.observation_space.shape[0])
        next_state, reward, done, _ = task.step(action)
        print(done)