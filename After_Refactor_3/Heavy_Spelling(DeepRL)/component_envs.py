import numpy as np
import gym

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.atari_wrappers import FrameStack as BaselinesFrameStack
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv

from .utils import mkdir, random_seed


class OriginalReturnWrapper(gym.Wrapper):
    """
    Wraps a gym environment and adds 'episodic_return' info to the training info dict.
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


class TransposeImage(gym.ObservationWrapper):
    """
    Transpose image observation from (H, W, C) to (C, H, W).
    """

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class FrameStack(BaselinesFrameStack):
    """
    Stack frames together along the time dimension.
    """
    
    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    """
    This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
    buffers.

    This object should only be converted to numpy array before being passed to the model.
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


def make_env(env_id, seed, rank, episode_life=True):
    """
    Create a function that returns an environment.
    """
    
    def _thunk():
        random_seed(seed)
        
        env = gym.make(env_id)
        if env_id.startswith("dm"):
            import dm_control2gym
            _, domain, task = env_id.split('-')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        
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
        env = OriginalReturnWrapper(env)

        return env

    return _thunk


class DummyVecEnv(VecEnv):
    """
    Vectorized environment that runs all environments sequentially in a single process.
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        super().__init__(len(env_fns), env.observation_space, env.action_space)
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

    def close(self):
        pass


class Task:
    """
    Wrapper for a single or multiple environments.
    """

    def __init__(self,
                 env_id,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=None):
        if seed is None:
            seed = np.random.randint(int(1e9))
        if log_dir is not None:
            mkdir(log_dir)

        envs = [make_env(env_id, seed, i, episode_life) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
        else:
            Wrapper = SubprocVecEnv
        
        self.env = Wrapper(envs)
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.action_space = self.env.action_space
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, gym.spaces.Box):
            self.action_dim = self.action_space.shape[0]
        else:
            raise ValueError('Unknown action space')
            
    def reset(self):
        return self.env.reset()

    def step(self, actions):
        if isinstance(self.action_space, gym.spaces.Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)