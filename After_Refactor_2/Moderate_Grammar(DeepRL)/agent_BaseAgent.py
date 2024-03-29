import torch
import numpy as np
from typing import List, Dict, Union, Tuple
from collections import deque
from skimage.io import imsave


class BaseAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_index = 0

    def close(self):
        close_obj(self.task)

    def save(self, filename: str):
        torch.save(self.network.state_dict(), f"{filename}.model")
        with open(f"{filename}.stats", "wb") as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename: str):
        state_dict = torch.load(f"{filename}.model", map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open(f"{filename}.stats", "rb") as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, state: np.ndarray) -> Union[int, np.ndarray]:
        raise NotImplementedError

    def eval_episode(self) -> np.ndarray:
        env = self.config.eval_env
        state = env.reset()
        while True:
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            if ret is not None:
                break
        return ret

    def eval_episodes(self) -> Dict[str, np.ndarray]:
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
        mean_return = np.mean(episodic_returns)
        std_err_return = np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        self.logger.info(f'steps {self.total_steps}, '
                         f'episodic_return_test {mean_return:.2f}({std_err_return:.2f})')
        self.logger.add_scalar('episodic_return_test', np.mean(episodic_returns), self.total_steps)
        return {'episodic_return_test': np.mean(episodic_returns)}

    def record_online_return(self, info: Union[Dict, Tuple], offset: int = 0):
        if isinstance(info, dict):
            ret = info['episodic_return']
            if ret is not None:
                self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
                self.logger.info(f'steps {self.total_steps + offset}, episodic_return_train {ret}')
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError

    def switch_task(self):
        config = self.config
        if not config.tasks:
            return
        segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
        if self.total_steps <= segs[self.task_index + 1]:
            return
        self.task_index += 1
        self.task = config.tasks[self.task_index]
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

    def record_episode(self, dir: str, env) -> None:
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

    def record_step(self, state: np.ndarray) -> Union[int, np.ndarray]:
        raise NotImplementedError

    # For DMControl
    def record_obs(self, env, dir: str, steps: int) -> None:
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave(f"{dir}/{steps:04d}.png", obs)


class BaseActor(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, config: Dict):
        mp.Process.__init__(self)
        self.config = config
        self.__pipe, self.__worker_pipe = mp.Pipe()

        self._state = None
        self._task = None
        self._network = None
        self._total_steps = 0
        self.__cache_len = 2

        if not config.async_actor:
            self.start = lambda: None
            self.step = self._sample
            self.close = lambda: None
            self._set_up()
            self._task = config.task_fn()

    def run(self):
        self._set_up()
        config = self.config
        self._task = config.task_fn()

        cache = deque([], maxlen=2)
        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.STEP:
                if not len(cache):
                    cache.append(self._sample())
                    cache.append(self._sample())
                self.__worker_pipe.send(cache.popleft())
                cache.append(self._sample())
            elif op == self.EXIT:
                self.__worker_pipe.close()
                return
            elif op == self.NETWORK:
                self._network = data
            else:
                raise NotImplementedError

    def _set_up(self):
        pass

    def step(self) -> List:
        self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self) -> None:
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net: torch.nn.Module) -> None:
        if not self.config.async_actor:
            self._network = net
        else:
            self.__pipe.send([self.NETWORK, net])

    def _sample(self) -> Tuple:
        transitions = []
        for _ in range(self.config.sgd_update_frequency):
            transition = self._transition()
            if transition is not None:
                transitions.append(transition)
        return transitions

    def _transition(self) -> Union[Tuple, None]:
        raise NotImplementedError