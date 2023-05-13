import os
from typing import List, Tuple, Any
from pathlib import Path

import gym
import numpy as np
import torch

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.atari_wrappers import FrameStack as FrameStack_
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv

from ..utils import mkdir, random_seed


try:
    import roboschool
except ImportError:
    pass


ENV_CATEGORIES = {
    "atari": gym.envs.atari.atari_env.AtariEnv,
    "dm": gym.envs.robotics,
}


def create_environment_manager(
    env_id: str,
    num_envs: int = 1,
    single_process: bool = True,
    log_dir: str = None,
    episode_life: bool = True,
    seed: int = None,
) -> "EnvironmentManager":
    if seed is None:
        seed = np.random.randint(int(1e9))
    if log_dir is not None:
        mkdir(log_dir)
    envs = [create_environment_with_seed_and_rank(env_id, seed, i, episode_life) for i in range(num_envs)]
    if single_process:
        VectorizedEnvironment = DummyVecEnv
    else:
        VectorizedEnvironment = SubprocVecEnv
    return EnvironmentManager(VectorizedEnvironment(envs))


def create_environment_with_seed_and_rank(env_id: str, seed: int, rank: int, episode_life: bool = True) -> gym.Env:
    random_seed(seed)
    env_category, *env_params = env_id.split("-")
    if env_category in ENV_CATEGORIES:
        env_module = ENV_CATEGORIES[env_category]
        if len(env_params) == 1:
            domain_name = env_params[-1]
            task_name = None
        elif len(env_params) == 2:
            domain_name, task_name = env_params
        else:
            raise ValueError(f"Invalid env_id: {env_id}")
        env = env_module.make(domain_name=domain_name, task_name=task_name)
    else:
        env = gym.make(env_id)
    is_atari = isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
    if is_atari:
        env = make_atari(env_id)
        env = wrap_deepmind(
            env,
            episode_life=episode_life,
            clip_rewards=False,
            frame_stack=False,
            scale=False,
        )
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3:
            env = TransposeImage(env)
        env = FrameStack(env, k=4)
        env.observation_space = Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=(obs_shape[-1], obs_shape[0], obs_shape[1]),
            dtype=env.observation_space.dtype,
        )
    env.seed(seed + rank)
    env = OriginalReturnWrapper(env)
    return env


class EnvironmentManager:
    def __init__(self, env: VecEnv):
        self.env = env

        self.name = env.spec.id
        self.observation_space = env.observation_space
        self.state_dim = int(np.prod(env.observation_space.shape))

        self.action_space = env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert "unknown action space"

    def reset(self) -> np.ndarray:
        return self.env.reset()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)


class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.total_rewards = 0

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        if done:
            info["episodic_return"] = self.total_rewards
            self.total_rewards = 0
        else:
            info["episodic_return"] = None
        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        return self.env.reset()


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            low=np.min(self.observation_space.low),
            high=np.max(self.observation_space.high),
            shape=(obs_shape[-1], obs_shape[0], obs_shape[1]),
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation.transpose(2, 0, 1)


class FrameStack(FrameStack_):
    def __init__(self, env: gym.Env, k: int):
        super().__init__(env, k)

    def _get_ob(self) -> LazyFrames:
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames: List[np.ndarray]):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None) -> np.ndarray:
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self) -> int:
        return len(self.__array__())

    def __getitem__(self, i) -> Any:
        return self.__array__()[i]


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns: List):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        super().__init__(
            num_envs=len(env_fns),
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        self.actions = None

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        data = []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(self.actions[i])
            if done:
                obs = self.envs[i].reset()
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        return obs, np.asarray(rew), np.asarray(done), info

    def reset(self) -> List[np.ndarray]:
        return [env.reset() for env in self.envs]

    def close(self) -> None:
        return


if __name__ == "__main__":
    env_manager = EnvironmentManager("Hopper-v2", num_envs=5, single_process=False)
    state = env_manager.reset()
    while True:
        action = np.random.rand(env_manager.observation_space.shape[0])
        next_state, reward, done, _ = env_manager.step(action)
        print(done)