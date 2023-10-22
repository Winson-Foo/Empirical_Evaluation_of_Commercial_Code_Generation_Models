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

try:
    import roboschool
except ImportError:
    roboschool = None

class TaskEnv(gym.Wrapper):
    def __init__(self, env, episode_life=True, clip_rewards=False,
                 frame_stack=False, scale=False):
        gym.Wrapper.__init__(self, env)
        self.is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if self.is_atari:
            self.env = wrap_deepmind(env,
                                     episode_life=episode_life,
                                     clip_rewards=clip_rewards,
                                     frame_stack=frame_stack,
                                     scale=scale)
            obs_shape = self.observation_space.shape
            if len(obs_shape) == 3:
                self.env = TransposeImage(self.env)
            self.env = FrameStack(self.env, 4)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            info['episodic_return'] = info['episode_reward']
        else:
            info['episodic_return'] = None
        return obs, reward, done, info


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


# The original LayzeFrames doesn't work well
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
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
    def __init__(self, env, num_frames):
        FrameStack_.__init__(self, env, num_frames)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class VectorEnv(VecEnv):
    def __init__(self, env_fns, num_envs, episode_life=True, clip_rewards=False,
                 frame_stack=False, scale=False):
        self.envs = [TaskEnv(fn(), episode_life, clip_rewards, frame_stack, scale) for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, num_envs, env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        data = [env.step(action) for env, action in zip(self.envs, self.actions)]
        obs, rew, done, info = zip(*data)
        return obs, np.asarray(rew), np.asarray(done), info

    def reset(self):
        return [env.reset() for env in self.envs]

class Task:
    def __init__(self, name, num_envs=1, single_process=True, log_dir=None,
                 episode_life=True, clip_rewards=False, frame_stack=False, scale=False, seed=None):
        if seed is None:
            seed = np.random.randint(int(1e9))
        if log_dir is not None:
            mkdir(log_dir)
        env_fns = [make_env(name, seed, i, episode_life) for i in range(num_envs)]
        if single_process:
            self.env = VectorEnv(env_fns, num_envs, episode_life, clip_rewards, frame_stack, scale)
        else:
            self.env = SubprocVecEnv(env_fns)
        self.name = name
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.action_space = self.env.action_space
        self.action_dim = get_action_dim(self.action_space)

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)

def get_action_dim(action_space):
    if isinstance(action_space, Discrete):
        return action_space.n
    elif isinstance(action_space, Box):
        return action_space.shape[0]
    else:
        assert 'unknown action space'

if __name__ == '__main__':
    task = Task('Hopper-v2', 5, single_process=False)
    state = task.reset()
    while True:
        action = np.random.rand(task.observation_space.shape[0])
        next_state, reward, done, _ = task.step(action)