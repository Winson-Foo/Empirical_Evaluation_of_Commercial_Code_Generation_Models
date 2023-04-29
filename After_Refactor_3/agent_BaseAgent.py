import torch
import pickle
import torch.multiprocessing as mp
from skimage.io import imsave

from ..utils import close_obj, get_logger
from collections import deque
from typing import Dict, List, Union


class BaseAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0
        self.task = None
        self.states = None
        self.network = None
        self.total_steps = 0

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

    def eval_step(self, state: List[float]) -> List[float]:
        raise NotImplementedError

    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        while True:
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]["episodic_return"]
            if ret is not None:
                break
        return ret

    def eval_episodes(self) -> Dict[str, float]:
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(sum(total_rewards))
        mean = np.mean(episodic_returns)
        std = np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        self.logger.info(
            f"steps {self.total_steps}, episodic_return_test {mean:.2f}({std:.2f})"
        )
        self.logger.add_scalar("episodic_return_test", mean, self.total_steps)
        return {"episodic_return_test": mean}

    def record_online_return(self, info: Union[Dict, Tuple], offset: int = 0):
        if isinstance(info, dict):
            ret = info["episodic_return"]
            if ret is not None:
                self.logger.add_scalar(
                    "episodic_return_train", ret, self.total_steps + offset
                )
                self.logger.info(
                    f"steps {self.total_steps + offset}, episodic_return_train {ret}"
                )
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
        if self.total_steps > segs[self.task_ind + 1]:
            self.task_ind += 1
            self.task = config.tasks[self.task_ind]
            self.states = self.task.reset()
            self.states = config.state_normalizer(self.states)

    def record_episode(self, directory: str, env) -> None:
        mkdir(directory)
        steps = 0
        state = env.reset()
        while True:
            self.record_obs(env, directory, steps)
            action = self.record_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]["episodic_return"]
            steps += 1
            if ret is not None:
                break

    def record_step(self, state) -> List[float]:
        raise NotImplementedError

    # For DMControl
    def record_obs(self, env, directory: str, steps: int) -> None:
        env = env.env.envs[0]
        obs = env.render(mode="rgb_array")
        imsave(f"{directory}/{steps:04d}.png", obs)


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
        self.pipe, self.worker_pipe = mp.Pipe()

        self.state = None
        self.task = None
        self.network = None
        self.total_steps = 0
        self.cache_len = 2

        if not config.async_actor:
            self.start = lambda: None
            self.step = self.sample
            self.close = lambda: None
            self.set_up()
            self.task = config.task_fn()

    def sample(self) -> List[Union[int, float, List]]:
        transitions = []
        for _ in range(self.config.sgd_update_frequency):
            transition = self.transition()
            if transition is not None:
                transitions.append(transition)
        return transitions

    def run(self):
        self.set_up()
        config = self.config
        self.task = config.task_fn()

        cache = deque([], maxlen=self.cache_len)
        while True:
            op, data = self.worker_pipe.recv()
            if op == self.STEP:
                if not len(cache):
                    cache.append(self.sample())
                    cache.append(self.sample())
                self.worker_pipe.send(cache.popleft())
                cache.append(self.sample())
            elif op == self.EXIT:
                self.worker_pipe.close()
                return
            elif op == self.NETWORK:
                self.network = data
            else:
                raise NotImplementedError

    def transition(self):
        raise NotImplementedError

    def set_up(self) -> None:
        pass

    def step(self) -> List[Union[int, float, List]]:
        self.pipe.send([self.STEP, None])
        return self.pipe.recv()

    def close(self) -> None:
        self.pipe.send([self.EXIT, None])
        self.pipe.close()

    def set_network(self, net) -> None:
        if not self.config.async_actor:
            self.network = net
        else:
            self.pipe.send([self.NETWORK, net])