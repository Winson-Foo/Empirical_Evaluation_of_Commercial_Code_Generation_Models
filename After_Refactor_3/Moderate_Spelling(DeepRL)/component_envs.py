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


def random_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class EnvFactory:
    """
    Factory class that creates an OpenAI Gym environment based on the environment ID.
    """
    def __init__(self, env_id: str, seed: int, rank: int, episode_life: bool):
        """
        Creates a factory that can create the specified OpenAI Gym environment.

        Args:
            env_id (str): The ID of the OpenAI Gym environment.
            seed (int): The seed used to initialize the random number generator.
            rank (int): The rank of the environment process.
            episode_life (bool): Flag indicating whether to use episode life in Atari environments.
        """
        self.env_id = env_id
        self.seed = seed
        self.rank = rank
        self.episode_life = episode_life

    def create_environment(self) -> gym.Env:
        """
        Creates the OpenAI Gym environment.

        Returns:
            gym.Env: The created environment.
        """
        if self.env_id.startswith("dm"):
            import dm_control2gym
            _, domain, task = self.env_id.split('-')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(self.env_id)

        return env

    def create_wrapped_environment(self) -> gym.Env:
        """
        Creates the wrapped OpenAI Gym environment with additional wrappers for Atari environments.

        Returns:
            gym.Env: The created wrapped environment.
        """
        env = self.create_environment()

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)

        if is_atari:
            env = make_atari(self.env_id)

            env = wrap_deepmind(env,
                                episode_life=self.episode_life,
                                clip_rewards=False,
                                frame_stack=False,
                                scale=False)

            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3:
                env = TransposeImage(env)

            env = FrameStack(env, 4)

        env.seed(self.seed + self.rank)
        env = OriginalReturnWrapper(env)

        return env


class OriginalReturnWrapper(gym.Wrapper):
    """
    A gym.Wrapper that adds episodic returns information to the environment.
    """
    def __init__(self, env):
        """
        Initializes the wrapper.

        Args:
            env (gym.Env): The environment to wrap.
        """
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        """
        Executes a step on the environment.

        Args:
            action: The action to execute.

        Returns:
            The observation, reward, done flag, and episodic return information.
        """
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, done, info

    def reset(self):
        """
        Resets the environment.

        Returns:
            The initial observation.
        """
        return self.env.reset()


class TransposeImage(gym.ObservationWrapper):
    """
    A gym.ObservationWrapper that transposes the image observation from (height,width,channels) to (channels,height,width).
    """
    def __init__(self, env=None):
        """
        Initializes the wrapper.

        Args:
            env (gym.Env): The environment to wrap.
        """
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        """
        Transposes the image observation.

        Args:
            observation: The input observation.

        Returns:
            The transposed observation.
        """
        return observation.transpose(2, 0, 1)


class LazyFrames(object):
    """
    An object that ensures that common frames between the observations are only stored once.
    This object should only be converted to numpy array before being passed to the model.
    """
    def __init__(self, frames):
        """
        Initializes the object.

        Args:
            frames: A list of frames that make up the observation.
        """
        self._frames = frames

    def __array__(self, dtype=None):
        """
        Converts the object to a numpy array.

        Args:
            dtype: The data type of the numpy array.

        Returns:
            Numpy array containing the frames.
        """
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        """
        Returns the length of the object.

        Returns:
            The length of the object.
        """
        return len(self.__array__())

    def __getitem__(self, i):
        """
        Returns the item at the given index.

        Args:
            i: The index of the item to return.

        Returns:
            The item at the given index.
        """
        return self.__array__()[i]


class FrameStack(FrameStack_):
    """
    A gym.Wrapper that stacks multiple consecutive frames to form the observation.
    """
    def __init__(self, env, k):
        """
        Initializes the wrapper.

        Args:
            env (gym.Env): The environment to wrap.
            k (int): The number of frames to stack.
        """
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        """
        Returns the stacked observation.

        Returns:
            The stacked observation.
        """
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class DummyVecEnv(VecEnv):
    """
    A VecEnv that runs multiple environments in a single process.
    """
    def __init__(self, env_fns):
        """
        Initializes the VecEnv.

        Args:
            env_fns: A list of environment factory functions.
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        """
        Asyncronously executes an action on the environment.

        Args:
            actions: A list of actions to execute.
        """
        self.actions = actions

    def step_wait(self):
        """
        Waits for the asyncronous step to complete and returns the results.

        Returns:
            Tuples of observations, rewards, done flags, and episode infos.
        """
        data = []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(self.actions[i])
            if done:
                obs = self.envs[i].reset()
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        return obs, np.asarray(rew), np.asarray(done), info

    def reset(self):
        """
        Resets the environment.

        Returns:
            A list of the initial observations.
        """
        return [env.reset() for env in self.envs]

    def close(self):
        """
        Closes the environment.
        """
        return


class Task:
    """
    An object representing the task to be completed by the agent.
    """
    def __init__(self, name: str, num_envs: int = 1, single_process: bool = True, log_dir: str = None, episode_life: bool = True, seed: int = None):
        """
        Initializes the Task object.

        Args:
            name (str): The ID of the OpenAI Gym environment.
            num_envs (int): The number of environments to run in parallel.
            single_process (bool): Whether to run all environments in a single process.
            log_dir (str): The directory where the log files will be stored.
            episode_life (bool): Flag indicating whether to use episode life in Atari environments.
            seed (int): The seed used to initialize the random number generator.
        """
        if seed is None:
            seed = np.random.randint(int(1e9))
        if log_dir is not None:
            mkdir(log_dir)

        # Create the environment factory
        env_factory = EnvFactory(name, seed, 0, episode_life)

        # Create the list of environment factory functions
        env_fns = [env_factory.create_wrapped_environment for _ in range(num_envs)]

        # Create the VecEnv
        if single_process:
            wrapper = DummyVecEnv
        else:
            wrapper = SubprocVecEnv

        self.env = wrapper(env_fns)

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
        """
        Resets the environment.

        Returns:
            A list of the initial observations.
        """
        return self.env.reset()

    def step(self, actions):
        """
        Executes an action on the environment.

        Args:
            actions: A list of actions to execute.

        Returns:
            Tuples of observations, rewards, done flags, and episode infos.
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