import torch
import numpy as np
import pickle
from collections import deque
from skimage.io import imsave
from typing import Dict, Tuple, List, Union


class BaseAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task = self._create_task()
        self.task_idx = 0

    def _create_task(self):
        raise NotImplementedError

    def close(self):
        close_obj(self.task)

    def save(self, filename: str):
        torch.save(self.network.state_dict(), f'{filename}.model')
        with open(f'{filename}.stats', 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename: str):
        state_dict = torch.load(f'{filename}.model', map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open(f'{filename}.stats', 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, state: np.ndarray) -> int:
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

    def eval_episodes(self) -> Dict:
        episodic_returns = []
        for _ in range(self.config.eval_episodes):
            episodic_returns.append(np.sum(self.eval_episode()))
        avg_return = np.mean(episodic_returns)
        std_return = np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        self.logger.info(f'steps {self.total_steps}, episodic_return_test {avg_return:.2f}({std_return:.2f})')
        self.logger.add_scalar('episodic_return_test', avg_return, self.total_steps)
        return {
            'episodic_return_test': avg_return,
        }

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

    def record_step(self, state: np.ndarray) -> int:
        raise NotImplementedError

    def record_obs(self, env, dir: str, steps: int) -> None:
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave(f'{dir}/{steps:04d}.png', obs)

    @property
    def total_steps(self) -> int:
        raise NotImplementedError


class BaseActor:
    class Operation:
        STEP = 0
        EXIT = 1
        NETWORK = 2

    def __init__(self, config: Dict):
        self.config = config
        self.conn, self.worker_conn = mp.Pipe()
        self.worker = self._create_worker()

        if not config.async_actor:
            self.start = lambda: None
            self.step = self._step_sync
            self.close = lambda: None
        else:
            self.worker.start()

    def _create_worker(self) -> mp.Process:
        raise NotImplementedError

    def _step_sync(self) -> List:
        self.worker_conn.send((self.Operation.STEP, None))
        return self.worker_conn.recv()

    def close(self) -> None:
        self.worker_conn.send((self.Operation.EXIT, None))
        self.worker_conn.close()

    def set_network(self, net: torch.nn.Module) -> None:
        self.worker_conn.send((self.Operation.NETWORK, net))


class Task:
    def __init__(self, config: Dict):
        self.config = config
        self.states = None
        self.env = config.task_fn()
        self.states = self.env.reset()
        self.states = config.state_normalizer(self.states)

    def reset(self):
        self.states = self.env.reset()
        self.states = self.config.state_normalizer(self.states)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        next_states, reward, done, info = self.env.step(action)
        next_states = self.config.state_normalizer(next_states)
        if any(done):
            self.reset()
        return next_states, reward, done, info


class RecordingEpisode:
    def __init__(self, config: Dict, dir: str, env):
        self.config = config
        self.dir = dir
        self.steps = 0
        self.env = env
        self.env.reset()

        self._create_dir()

    def _create_dir(self) -> None:
        mkdir(self.dir)

    def close(self) -> None:
        self.env.close()

    def step(self) -> None:
        self._record_obs()
        action = self._record_step()
        self.steps += 1
        self.env.step(action)

    def _record_obs(self) -> None:
        env = self.env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave(f'{self.dir}/{self.steps:04d}.png', obs)

    def _record_step(self) -> int:
        raise NotImplementedError


class SequentialTask(Task):
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        next_states, reward, done, info = super().step(action)
        return next_states[self.config.task_id], reward, done, info


class SequentialRecordingEpisode(RecordingEpisode):
    def __init__(self, config: Dict, dir: str, env):
        super().__init__(config, dir, env)
        self.task_id = config.task_id

    def _record_step(self) -> int:
        return self.config.action_space.sample()


class AsynchronousActor(BaseActor):
    def _create_worker(self) -> mp.Process:
        return AsynchronousActorWorker(self.config, self.worker_conn)


class AsynchronousActorWorker(mp.Process):
    def __init__(self, config: Dict, conn):
        mp.Process.__init__(self)
        self.config = config
        self.conn = conn
        self.task = Task(config)
        self.network = None
        self.total_steps = 0

    def _transition(self) -> Union[Tuple, None]:
        action = self.config.action_space.sample()
        next_states, reward, done, info = self.task.step(action)
        next_states = self.config.state_normalizer(next_states)
        reward = np.sign(reward)
        self.total_steps += 1
        if self.total_steps % self.config.log_frequency == 0:
            self.conn.send((self.Operation.STATS, self.total_steps))
        if any(done):
            self.task.reset()
            return
        return (self.states, action, reward, next_states, done[0])

    def _set_up(self) -> None:
        pass

    def run(self) -> None:
        self._set_up()
        while True:
            op, data = self.conn.recv()
            if op == self.Operation.STEP:
                transitions = []
                for _ in range(self.config.sgd_update_frequency):
                    transition = self._transition()
                    if transition is not None:
                        transitions.append(transition)
                self.conn.send(transitions)
            elif op == self.Operation.EXIT:
                self.conn.close()
                return
            elif op == self.Operation.NETWORK:
                self.network = data
            else:
                raise NotImplementedError


class SynchronousActor(BaseActor):
    def _create_worker(self) -> mp.Process:
        return SynchronousActorWorker(self.config, self.worker_conn)


class SynchronousActorWorker(mp.Process):
    def __init__(self, config: Dict, conn):
        mp.Process.__init__(self)
        self.config = config
        self.conn = conn
        self.task = Task(config)
        self.network = None
        self.total_steps = 0

    def _transition(self) -> Union[Tuple, None]:
        action = self.config.action_selector.select_action(self.states, self.network)
        next_states, reward, done, info = self.task.step(action)
        next_states = self.config.state_normalizer(next_states)
        reward = np.sign(reward)
        self.total_steps += 1
        if self.total_steps % self.config.log_frequency == 0:
            self.conn.send((self.Operation.STATS, self.total_steps))
        if any(done):
            self.task.reset()
            return
        return (self.states, action, reward, next_states, done[0])

    def run(self) -> None:
        while True:
            op, data = self.conn.recv()
            if op == self.Operation.STEP:
                transitions = self._transition()
                while transitions is None:
                    transitions = self._transition()
                self.conn.send(transitions)
            elif op == self.Operation.EXIT:
                self.conn.close()
                return
            elif op == self.Operation.NETWORK:
                self.network = data
            else:
                raise NotImplementedError


class StatsRecorder:
    def __init__(self, config: Dict):
        self.config = config

    def __enter__(self):
        self.task_idx = 0
        self.total_steps = 0
        self.segs = np.linspace(
            0, self.config.max_steps, len(self.config.tasks) + 1
        )

        self.start_time = time.time()
        self.episode_rewards = np.zeros(len(self.config.tasks))
        self.episode_lengths = np.zeros(len(self.config.tasks))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def update(self, step: int, info: Dict) -> None:
        self.total_steps = step

        if isinstance(info, dict):
            episodic_return = info['episodic_return']
            if episodic_return is not None:
                self.episode_rewards[self.task_idx] += episodic_return
                self.episode_lengths[self.task_idx] += 1
        else:
            for i, info_ in enumerate(info):
                self.update(step, info_, self.task_idx + 1)

        if self.total_steps > self.segs[self.task_idx + 1]:
            avg_episode_reward = (
                    self.episode_rewards[self.task_idx] / self.episode_lengths[self.task_idx]
            )
            self.episode_rewards[self.task_idx] = 0
            self.episode_lengths[self.task_idx] = 0
            self.task_idx += 1
            self.task = self.config.tasks[self.task_idx]
            self.task.reset()

            self.config.action_selector.reset()

            self.config.logger.info(
                f'Task {self.task_idx}, steps {step:.2f}, '
                f'time {(time.time() - self.start_time) / 60:.2f} mins, '
                f'avg_return {avg_episode_reward:.2f}'
            )
            self.config.logger.add_scalar(f'avg_return/task_{self.task_idx}', avg_episode_reward, step)