import torch
import numpy as np
from ..utils import *
import torch.multiprocessing as mp
from collections import deque
from skimage.io import imsave


class BaseAgent:
    def __init__(self, configuration):
        self.configuration = configuration
        self.logger = get_logger(tag=configuration.tag, log_level=configuration.log_level)
        self.task_index = 0

    def close(self):
        close_obj(self.task)

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as file:
            pickle.dump(self.configuration.state_normalizer.state_dict(), file)

    def load(self, filename):
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as file:
            self.configuration.state_normalizer.load_state_dict(pickle.load(file))

    def evaluate_step(self, state):
        raise NotImplementedError

    def evaluate_all_episodes(self):
        evaluator = Evaluator(self.configuration.eval_env, self.evaluate_step, self.logger)
        results = evaluator.evaluate_all(self.configuration.eval_episodes)
        self.logger.info('steps %d, episodic_return_test %.2f(%.2f)' % (
            self.total_steps, np.mean(results), np.std(results) / np.sqrt(len(results))
        ))
        self.logger.add_scalar('episodic_return_test', np.mean(results), self.total_steps)
        return {
            'episodic_return_test': np.mean(results),
        }

    def record_training_return(self, info, offset=0):
        if isinstance(info, dict):
            return_value = info['episodic_return']
            if return_value is not None:
                self.logger.add_scalar('episodic_return_train', return_value, self.total_steps + offset)
                self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps + offset, return_value))
        elif isinstance(info, tuple):
            for index in range(len(info)):
                self.__record_training_return(info[index], index)
        else:
            raise NotImplementedError

    def change_task(self):
        if not self.configuration.tasks:
            return
        
        segs = np.linspace(0, self.configuration.max_steps, len(self.configuration.tasks) + 1)
        if self.total_steps > segs[self.task_index + 1]:
            self.task_index += 1
            self.task = self.configuration.tasks[self.task_index]
            self.states = self.task.reset()
            self.states = self.configuration.state_normalizer(self.states)

    def record_episode(self, directory, env):
        mkdir(directory)
        steps = 0
        state = env.reset()
        while True:
            self.record_obs(env, directory, steps)
            action = self.record_step(state)
            state, reward, done, info = env.step(action)
            return_value = info[0]['episodic_return']
            steps += 1
            if return_value is not None:
                break

    def record_step(self, state):
        raise NotImplementedError

    # For DMControl
    def record_obs(self, env, directory, steps):
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave('%s/%04d.png' % (directory, steps), obs)


class Evaluator:
    def __init__(self, env, evaluate_step, logger):
        self.env = env
        self.evaluate_step = evaluate_step
        self.logger = logger

    def evaluate_all(self, episodes):
        results = []
        for episode in range(episodes):
            total_rewards = self.evaluate_episode()
            results.append(np.sum(total_rewards))
        return results

    def evaluate_episode(self):
        state = self.env.reset()
        while True:
            action = self.evaluate_step(state)
            state, reward, done, info = self.env.step(action)
            return_value = info[0]['episodic_return']
            if return_value is not None:
                break
        return return_value


class BaseActor(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, configuration):
        mp.Process.__init__(self)
        self.configuration = configuration
        self.__pipe, self.__worker_pipe = mp.Pipe()
        self._state = None
        self._task = None
        self._network = None
        self.total_steps = 0
        self.__cache_len = 2

        if not configuration.async_actor:
            self.start = lambda: None
            self.step = self._sample
            self.close = lambda: None
            self._set_up()
            self._task = configuration.task_fn()

    def run(self):
        self._set_up()
        self._task = self.configuration.task_fn()

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

    def _transition(self):
        raise NotImplementedError

    def _set_up(self):
        pass

    def step(self):
        self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        if not self.configuration.async_actor:
            self._network = net
        else:
            self.__pipe.send([self.NETWORK, net])