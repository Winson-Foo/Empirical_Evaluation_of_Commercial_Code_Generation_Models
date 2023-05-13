# file: env_utils.py
import gym
import numpy as np

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

def create_env(env_id, seed, episode_life=True):
    env = gym.make(env_id)
    env.seed(seed)
    is_atari = hasattr(gym.envs, 'atari') and isinstance(
        env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
    if is_atari:
        env = make_atari(env_id)
    env = OriginalReturnWrapper(env)
    if is_atari:
        env = wrap_deepmind(env,
                            episode_life=episode_life,
                            clip_rewards=False,
                            frame_stack=False,
                            scale=False)
        env = TransposeImage(env)
        env = VecFrameStack(env, 4)

    return env

class OriginalReturnWrapper(gym.Wrapper):
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

class TransposeImage(gym.ObservationWrapper):
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