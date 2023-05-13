import os
import gym
import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv

from ..utils import mkdir, random_seed

try:
    import roboschool
except ImportError:
    pass


# Utility functions

def make_env(env_id, seed, rank, episode_life=True):
    """
    Create a function to generate environment instances with given parameters.

    Args:
        env_id: str, the id of the environment to create.
        seed: int, the random seed for the environment.
        rank: int, the rank of the environment in a vectorized environment.
        episode_life: bool, whether to use episode life for Atari environments.

    Returns:
        The generated environment instance.
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
            env = gym.wrappers.AtariPreprocessing(env, episode_life=episode_life)
        env.seed(seed + rank)
        return env

    return _thunk


class Task:
    """
    Wrapper class for a vectorized RL environment.
    """
    def __init__(self,
                 env_id,
                 num_envs=1,
                 use_single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=None):
        """
        Initialize a Task instance with given parameters.

        Args:
            env_id: str, the id of the environment to create.
            num_envs: int, the number of environments to create.
            use_single_process: bool, whether to use a single process for the environments.
            log_dir: str, the path to the directory for logging data.
            episode_life: bool, whether to use episode life for Atari environments.
            seed: int, the random seed for the environment.

        Returns:
            None.
        """
        if seed is None:
            seed = np.random.randint(int(1e9))
        if log_dir is not None:
            mkdir(log_dir)
        envs = [make_env(env_id, seed, i, episode_life) for i in range(num_envs)]
        if use_single_process:
            self.env = DummyVecEnv(envs)
        else:
            self.env = SubprocVecEnv(envs)
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.observation_space.shape))

        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    def reset(self):
        """
        Reset the environment to its initial state.

        Args:
            None.

        Returns:
            The current state of the environment.
        """
        return self.env.reset()

    def step(self, actions):
        """
        Take an action in the environment.

        Args:
            actions: np.ndarray, the action to take.

        Returns:
            The next state, the reward, a flag indicating whether the episode is done, and additional information.
        """
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)


# Wrapper classes for environment modifications

class TransposeImage(gym.ObservationWrapper):
    """
    Transpose the image observation of the environment.
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


class OriginalReturnWrapper(gym.Wrapper):
    """
    Wrap the environment to return the total episodic reward as additional information.
    """
    def __init__(self, env):
        super().__init__(env)
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


class FrameStack(gym.Wrapper):
    """
    Stack multiple frames of the observation together.
    """
    def __init__(self, env, num_frames):
        super().__init__(env)
        self.frames = []
        self.num_frames = num_frames

        obs_shape = env.observation_space.shape
        self.observation_space = Box(
            env.observation_space.low[0, 0, 0] * num_frames,
            env.observation_space.high[0, 0, 0] * num_frames,
            [obs_shape[2] * num_frames, obs_shape[1], obs_shape[0]],
            dtype=env.observation_space.dtype)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        if len(self.frames) > self.num_frames:
            self.frames.pop(0)
        obs = LazyFrames(self.frames)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.num_frames):
            self.frames.append(obs)
        obs = LazyFrames(self.frames)
        return obs


# Vectorized environment classes

class DummyVecEnv(VecEnv):
    """
    A simple in-memory vectorized environment that runs all environments synchronously in the same process.
    """
    def __init__(self, env_fns):
        env = env_fns[0]()
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.envs = [fn() for fn in env_fns]
        self.actions = None
        self.obs = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs = []
        rewards = []
        dones = []
        infos = []
        for i in range(self.num_envs):
            ob, reward, done, info = self.envs[i].step(self.actions[i])
            if done:
                ob = self.envs[i].reset()
            obs.append(ob)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return obs, np.asarray(rewards), np.asarray(dones), infos

    def reset(self):
        obs = []
        for env in self.envs:
            ob = env.reset()
            obs.append(ob)
        self.obs = obs
        return self.obs

    def close(self):
        for env in self.envs:
            env.close()


class LazyFrames(object):
    """
    This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
    buffers.

    This object should only be converted to numpy array before being passed to the model.

    Inspired by the implementation used in OpenAI Baselines.
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