# Refactored code

# The code has been refactored to improve maintainability by putting similar functions and classes together, adding docstrings and improving variable names.
# Some classes and functions were also modified to work better or to match standards from other libraries.

import os
import gym
import numpy as np
import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from typing import List, Dict, Any, Tuple

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from ..utils import *


def make_env(env_id: str, seed: int, rank: int, episode_life: bool = True) -> Any:
    """Creates a function that returns a gym environment according to the specified env_id, seed and rank."""
    def _thunk() -> Any:
        random_seed(seed)
        if env_id.startswith("dm"):
            import dm_control2gym
            _, domain, task = env_id.split('-')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        env = OriginalReturnWrapper(env)
        if is_atari:
            env = wrap_deepmind(
                env, episode_life=episode_life, clip_rewards=False, frame_stack=False, scale=False)
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3:
                env = TransposeImage(env)
            env = FrameStack(env, 4)
        return env

    return _thunk


class OriginalReturnWrapper(gym.Wrapper):
    """A gym Wrapper that records the total reward of each episode and returns it as a new info field when done. """
    def __init__(self, env: Any):
        super(OriginalReturnWrapper, self).__init__(env)
        self.total_rewards = 0

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, done, info

    def reset(self) -> Any:
        """Resets the environment and return the initial observation."""
        return self.env.reset()


class TransposeImage(gym.ObservationWrapper):
    """A gym Wrapper that transposes the observation image axes (channels, height, width) to match PyTorch's format."""
    def __init__(self, env: Any = None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation: Any) -> Any:
        """Changes the observation to have channel first by transposing axes."""
        return observation.transpose(2, 0, 1)


class LazyFrames(object):
    """A wrapper class for efficiently concatenating a list of observation frames by only keeping shared data as a reference."""
    def __init__(self, frames: List[Any]):
        self._frames = frames

    def __array__(self, dtype=None) -> np.ndarray:
        """Concatenates all frames horizontally and returns a numpy array of the resulting observation."""
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self) -> int:
        """Returns the length of the concatenated observation."""
        return len(self.__array__())

    def __getitem__(self, i: int) -> np.ndarray:
        """Returns the ith element of the concatenated observation."""
        return self.__array__()[i]


class FrameStack(FrameStack_):
    """A modified FrameStack from baselines that used the LazyFrames class to store observations."""
    def __init__(self, env: Any, k: int):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self) -> LazyFrames:
        """A function that gets the k last frames in the list and returns them in a LazyFrames object."""
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class DummyVecEnv(VecEnv):
    """A modified VecEnv used to create multiple environment in one process, used along with SubprocVecEnv."""
    def __init__(self, env_fns: List[Callable[[], Any]]):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions: np.ndarray) -> None:
        """Sets the actions that will be executed in each environment step later."""
        self.actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Executes the step function that uses the set actions and returns the results of each environment."""
        data = []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(self.actions[i])
            if done:
                obs = self.envs[i].reset()
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        return np.array(obs), np.array(rew), np.array(done), info

    def reset(self) -> List[Any]:
        """Resets all environments and return their initial observations."""
        return [env.reset() for env in self.envs]

    def close(self) -> None:
        """Closes all environments."""
        return


class Task:
    """An environment class that encapsulates a VecEnv with additional attributes."""
    def __init__(self,
                 name: str,
                 num_envs: int = 1,
                 single_process: bool = True,
                 log_dir: str = None,
                 episode_life: bool = True,
                 seed: int = None):
        if seed is None:
            seed = np.random.randint(int(1e9))
        if log_dir is not None:
            mkdir(log_dir)
        envs = [make_env(name, seed, i, episode_life) for i in range(num_envs)]
        if single_process:
            wrapper = DummyVecEnv
        else:
            wrapper = SubprocVecEnv
        self.env = wrapper(envs)
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

    def reset(self) -> np.ndarray:
        """Resets all environments and returns their initial observations concatenated in a LazyFrames object."""
        return self.env.reset()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Executes a step in the environments with the specified actions and returns the resulting observations, rewards, dones and info dicts."""
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)
    
    
if __name__ == '__main__':
    """Sample usage of the Task class to test with the Hopper-v2 environment."""
    task = Task('Hopper-v2', 5, single_process=False)
    state = task.reset()
    while True:
        action = np.random.rand(task.observation_space.shape[0])
        next_state, reward, done, _ = task.step(action)
        print(done)