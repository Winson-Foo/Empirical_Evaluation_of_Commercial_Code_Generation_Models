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

logger = logging.getLogger(__name__)


class AgentRunner:
    def __init__(self, agent):
        self.agent = agent
        self.config = agent.config
        self.agent_name = agent.__class__.__name__

    def run(self):
        while self.should_continue():
            if self.should_save():
                self.save()
            if self.should_log():
                self.log()
            if self.should_evaluate():
                self.evaluate()
            self.agent.step()
            self.agent.switch_task()
        self.agent.close()

    def should_continue(self):
        return (not self.config.max_steps or self.agent.total_steps < self.config.max_steps)

    def should_save(self):
        return (self.config.save_interval and 
                self.agent.total_steps % self.config.save_interval == 0)

    def save(self):
        dirname = f'data/{self.agent_name}-{self.config.tag}-{self.agent.total_steps}'
        os.makedirs(dirname, exist_ok=True)
        self.agent.save(dirname)

    def should_log(self):
        return (self.config.log_interval and 
                self.agent.total_steps % self.config.log_interval == 0)

    def log(self):
        elapsed_time = time.time() - self.agent.start_time
        steps_per_sec = self.config.log_interval / elapsed_time
        logger.info(f'steps {self.agent.total_steps}, {steps_per_sec:.2f} steps/s')
        self.agent.start_time = time.time()

    def should_evaluate(self):
        return (self.config.eval_interval and 
                self.agent.total_steps % self.config.eval_interval == 0)

    def evaluate(self):
        self.agent.eval_episodes()


def get_time_str():
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')


def get_default_log_dir(name):
    return f'./log/{name}-{get_time_str()}'


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()


def random_sample(indices, batch_size):
    random.shuffle(indices)
    batches = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
    for batch in batches:
        yield batch


def is_plain_type(x):
    for t in [str, int, float, bool]:
        if isinstance(x, t):
            return True
    return False


def generate_tag(params):
    game = params['game']
    params.setdefault('run', 0)
    run = params['run']
    del params['game']
    del params['run']
    sorted_params = sorted(params.items(), key=lambda x: x[0])
    str_params = [f'{k}_{v if is_plain_type(v) else v.__name__}' for k, v in sorted_params]
    tag = f'{game}-{"-".join(str_params)}-run-{run}'
    params['tag'] = tag
    params['game'] = game
    params['run'] = run


def translate(pattern):
    return pattern.replace('.', '\\.')


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
            param = [[key, item] for item in ordered_params[key]]
            params.append(param)
        self.params = list(itertools.product(*params))

    def __getitem__(self, index):
        return HyperParameter(index, self.params[index])

    def __len__(self):
        return len(self.params)