# file: reinforcement_learning.py

import datetime
import itertools
import logging
import os
import pickle
import random
import time
from collections import OrderedDict, Sequence
from pathlib import Path

import numpy as np
import torch

from .torch_utils import *

# Initialize logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, config):
        self.config = config
        self.total_steps = 0
        self.current_task = None
        self.logger = logger.getChild(self.__class__.__name__)

    def save(self, filename):
        pass  # TODO: Implement save method

    def close(self):
        pass  # TODO: Implement close method

    def eval_episodes(self):
        pass  # TODO: Implement eval_episodes method

    def step(self):
        pass  # TODO: Implement step method

    def switch_task(self):
        pass  # TODO: Implement switch_task method

class Runner:
    def __init__(self, agent):
        self.agent = agent

    def run(self):
        config = self.agent.config
        agent_name = self.agent.__class__.__name__

        t0 = time.time()
        while True:
            if config.save_interval and not self.agent.total_steps % config.save_interval:
                self.agent.save('data/%s-%s-%d' % (agent_name, config.tag, self.agent.total_steps))

            if config.log_interval and not self.agent.total_steps % config.log_interval:
                steps_per_second = config.log_interval / (time.time() - t0)
                self.agent.logger.info('steps %d, %.2f steps/s', self.agent.total_steps, steps_per_second)
                t0 = time.time()

            if config.eval_interval and not self.agent.total_steps % config.eval_interval:
                self.agent.eval_episodes()

            if config.max_steps and self.agent.total_steps >= config.max_steps:
                self.agent.close()
                break

            self.agent.step()
            self.agent.switch_task()

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
    run = params.setdefault('run', 0)
    del params['game']
    del params['run']

    sorted_params = sorted(params.items())
    param_strings = []
    for key, value in sorted_params:
        if is_plain_type(value):
            param_strings.append(f'{key}_{value}')
        else:
            param_strings.append(f'{key}_{value.__name__}')
    tag = f'{game}-{"-".join(param_strings)}-run-{run}'
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

def get_time_string():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")

def get_default_log_dir(name):
    return f'./log/{name}-{get_time_string()}'

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()