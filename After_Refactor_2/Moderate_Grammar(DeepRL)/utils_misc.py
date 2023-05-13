import numpy as np
import pickle
import os
import datetime
import torch
import time
from .torch_utils import *
from pathlib import Path


class Agent:
    def __init__(self, config):
        self.config = config
        self.total_steps = 0
        self.logger = get_logger(config.log_level)
        self.initialize()

    def initialize(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def switch_task(self):
        raise NotImplementedError

    def eval_episodes(self):
        raise NotImplementedError

    def save(self, save_path):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class Trainer:
    def __init__(self, agent):
        self.agent = agent

    def run_steps(self):
        config = self.agent.config
        agent_name = self.agent.__class__.__name__
        t0 = time.time()
        while True:
            if config.save_interval and not self.agent.total_steps % config.save_interval:
                self.agent.save('data/%s-%s-%d' % (agent_name, config.tag, self.agent.total_steps))
            if config.log_interval and not self.agent.total_steps % config.log_interval:
                self.agent.logger.info('steps %d, %.2f steps/s' % (self.agent.total_steps, config.log_interval / (time.time() - t0)))
                t0 = time.time()
            if config.eval_interval and not self.agent.total_steps % config.eval_interval:
                self.agent.eval_episodes()
            if config.max_steps and self.agent.total_steps >= config.max_steps:
                self.agent.close()
                break
            self.agent.step()
            self.agent.switch_task()


def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_default_log_dir(name):
    return './log/%s-%s' % (name, get_time_str())


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


def is_plain_type(x):
    for t in [str, int, float, bool]:
        if isinstance(x, t):
            return True
    return False


def generate_tag(params):
    if 'tag' in params.keys():
        return
    game = params['game']
    params.setdefault('run', 0)
    run = params['run']
    del params['game']
    del params['run']
    str = ['%s_%s' % (k, v if is_plain_type(v) else v.__name__) for k, v in sorted(params.items())]
    tag = '%s-%s-run-%d' % (game, '-'.join(str), run)
    params['tag'] = tag
    params['game'] = game
    params['run'] = run


def translate(pattern):
    groups = pattern.split('.')
    pattern = ('\.').join(groups)
    return pattern


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


class HyperParameter:
    def __init__(self, id, param):
        self.id = id
        self.param = dict()
        for key, item in param:
            self.param[key] = item

    def __str__(self):
        return str(self.id)

    def dict(self):
        return self.param


class HyperParameters(Sequence):
    def __init__(self, ordered_params):
        if not isinstance(ordered_params, OrderedDict):
            raise NotImplementedError
        params = []
        for key in ordered_params.keys():
            param = [[key, iterm] for iterm in ordered_params[key]]
            params.append(param)
        self.params = list(itertools.product(*params))

    def __getitem__(self, index):
        return HyperParameter(index, self.params[index])

    def __len__(self):
        return len(self.params)