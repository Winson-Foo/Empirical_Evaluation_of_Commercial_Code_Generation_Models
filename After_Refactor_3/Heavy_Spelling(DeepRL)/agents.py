from typing import List, Dict, Any, Tuple
import torch
import numpy as np
from collections import deque
from skimage.io import imsave
from ..utils import get_logger, close_obj, mkdir


class BaseAgent:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0
        self.task = None
        self.states = None
        self.network = None
        self.total_steps = 0

    def close(self) -> None:
        with self.logger:
            close_obj(self.task)

    def save(self, filename: str) -> None:
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename: str) -> None:
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def eval_episode(self) -> float:
        env = self.config.eval_env
        state = env.reset()
        while True:
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            if ret is not None:
                break
        return ret

    def eval_episodes(self) -> Dict[str, float]:
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
        mean_return = np.mean(episodic_returns)
        std_return = np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        self.logger.info('steps %d, episodic_return_test %.2f(%.2f)' % (
            self.total_steps, mean_return, std_return
        ))
        self.logger.add_scalar('episodic_return_test', mean_return, self.total_steps)
        return {
            'episodic_return_test': mean_return,
        }

    def record_online_return(self, info: Union[Dict[str, Any], Tuple[Dict[str, Any], ...]], 
                             offset: int = 0) -> None:
        if isinstance(info, dict):
            ret = info['episodic_return']
            if ret is not None:
                self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
                self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps + offset, ret))
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError

    def switch_task(self) -> None:
        config = self.config
        if not config.tasks:
            return
        segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
        if self.total_steps > segs[self.task_ind + 1]:
            self.task_ind += 1
            self.task = config.tasks[self.task_ind]
            self.states = self.task.reset()
            self.states = config.state_normalizer(self.states)

    def record_episode(self, dir: str, env: Any) -> None:
        mkdir(dir)
        steps = 0
        state = env.reset()
        while True:
            self.record_obs(env, dir, steps)
            action = self.record_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            steps += 1
            if ret is not None:
                break

    def record_step(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # For DMControl
    def record_obs(self, env: Any, dir: str, steps: int) -> None:
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave('%s/%04d.png' % (dir, steps), obs)