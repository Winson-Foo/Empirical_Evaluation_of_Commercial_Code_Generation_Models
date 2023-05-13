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

# Create a Gym environment using the provided ID
def make_gym_environment(env_id, seed, rank, episode_life=True):
    def _thunk():
        random_seed(seed)
        gym_env = gym.make(env_id)
        gym_env.seed(seed + rank)
        gym_env = OriginalReturnWrapper(gym_env)
        if hasattr(gym.envs, 'atari') and isinstance(gym_env.unwrapped, gym.envs.atari.atari_env.AtariEnv):
            atari_env = make_atari(env_id)
            atari_env = wrap_deepmind(
                atari_env,
                episode_life=episode_life,
                clip_rewards=False,
                frame_stack=False,
                scale=False
            )
            obs_shape = atari_env.observation_space.shape
            if len(obs_shape) == 3:
                atari_env = TransposeImage(atari_env)
            env = FrameStack(atari_env, 4)
        else:
            env = gym_env

        return env

    return _thunk

class OriginalReturnWrapper(gym.Wrapper):
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
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)

class FrameStack(FrameStack_):
    def __init__(self, env, k):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

# Wrapper class for a set of Gym environments
class Task:
    def __init__(self, env_id, num_envs=1, single_process=True, log_dir=None, episode_life=True, seed=None):
        if seed is None:
            seed = np.random.randint(int(1e9))
        if log_dir is not None:
            mkdir(log_dir)
        envs = [make_gym_environment(env_id, seed, i, episode_life) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
        else:
            Wrapper = SubprocVecEnv
        self.env = Wrapper(envs)
        self.env_id = env_id
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    # Reset the environment
    def reset(self):
        return self.env.reset()

    # Step through the environment with the given actions
    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)

# A vectorized environment that runs multiple environments in parallel in separate processes
class SubprocessEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None

    # Asynchronously send actions to the environments
    def step_async(self, actions):
        self.actions = actions

    # Wait for all environments to complete the current step
    def step_wait(self):
        data = []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(self.actions[i])
            if done:
                obs = self.envs[i].reset()
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        return obs, np.asarray(rew), np.asarray(done), info

    # Reset all environments
    def reset(self):
        return [env.reset() for env in self.envs]

    # Close all environments
    def close(self):
        return