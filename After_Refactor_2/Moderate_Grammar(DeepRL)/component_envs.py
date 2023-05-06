import os
import gym
import numpy as np
import torch

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.atari_wrappers import FrameStack as FrameStack_

from ..utils import *


def create_env(env_id, seed, rank, episode_life=True):
    """Create environment for training"""
    def env_fn():
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
            env = wrap_deepmind(env,
                                episode_life=episode_life,
                                clip_rewards=False,
                                frame_stack=False,
                                scale=False)
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3:
                env = TransposeImage(env)
            env = FrameStack(env, 4)
        env.seed(seed + rank)
        return env
    return env_fn


class OriginalReturnWrapper(gym.Wrapper):
    """A Wrapper around OpenAI Gym environments that returns episodic_return as part of the info dictionary"""
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['episodic_return'] = reward if done else None
        return obs, reward, done, info


class TransposeImage(gym.ObservationWrapper):
    """Transpose the image dimensions of an observation"""
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
    """Stack consecutive frames of an observation together"""
    def __init__(self, env, num_frames):
        FrameStack_.__init__(self, env, num_frames)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class DummyVecEnv(VecEnv):
    """A simple implementation of VecEnv that uses a single process"""
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)

    def step_async(self, actions):
        pass

    def step_wait(self):
        obs, rewards, dones, infos = [], [], [], []
        for env in self.envs:
            ob, rew, done, info = env.step(self.actions[env.env_idx])
            if done:
                ob = env.reset()
            obs.append(ob)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)
        return obs, np.asarray(rewards), np.asarray(dones), infos

    def reset(self):
        return [env.reset() for env in self.envs]

    def close(self):
        pass


class Task:
    """A helper class that wraps a vectorized environment and provides the observation and action space dimensions"""
    def __init__(self,
                 env_id,
                 num_envs=1,
                 num_frames=4,
                 episode_life=True,
                 seed=None):
        if seed is None:
            seed = np.random.randint(int(1e9))
        env_fns = [create_env(env_id, seed, i, episode_life) for i in range(num_envs)]
        self.env = DummyVecEnv(env_fns)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.observation_dim = self.observation_space.shape[0] * num_frames

        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'Unknown action space'

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)


if __name__ == '__main__':
    task = Task('Hopper-v2', num_envs=5)
    state = task.reset()
    while True:
        action = np.random.rand(task.observation_dim)
        next_state, reward, done, _ = task.step(action)
        print(done)