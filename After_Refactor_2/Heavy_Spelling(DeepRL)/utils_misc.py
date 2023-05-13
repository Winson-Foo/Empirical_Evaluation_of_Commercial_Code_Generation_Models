import datetime
import itertools
import os
import pickle
import time
from collections import OrderedDict, Sequence
from pathlib import Path

import numpy as np
import torch

from .torch_utils import *


def run_steps(agent):
    while True:
        if agent.config.save_interval and not agent.total_steps % agent.config.save_interval:
            agent.save('data/%s-%s-%d' % (agent.__class__.__name__, agent.config.tag, agent.total_steps))
        if agent.config.log_interval and not agent.total_steps % agent.config.log_interval:
            log_step(agent)
        if agent.config.eval_interval and not agent.total_steps % agent.config.eval_interval:
            agent.eval_episodes()
        if agent.config.max_steps and agent.total_steps >= agent.config.max_steps:
            agent.close()
            break
        agent.step()
        agent.switch_task()


def log_step(agent):
    steps_per_sec = agent.config.log_interval / (time.time() - agent.last_log_time)
    agent.logger.info('steps %d, %.2f steps/s' % (agent.total_steps, steps_per_sec))
    agent.last_log_time = time.time()


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
    remainder = len(indices) % batch_size
    if remainder:
        yield indices[-remainder:]


def is_plain_type(x):
    return isinstance(x, (str, int, float, bool))


def generate_tag(params):
    if 'tag' in params:
        return
    game = params.pop('game')
    run = params.pop('run')
    str_params = ['%s_%s' % (k, v if is_plain_type(v) else v.__name__) for k, v in sorted(params.items())]
    tag = '%s-%s-run-%d' % (game, '-'.join(str_params), run)
    params['tag'] = tag
    params['game'] = game
    params['run'] = run


def translate(pattern):
    return r'\.'.join(pattern.split('.'))


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


class HyperParameter:
    def __init__(self, id, param):
        self.id = id
        self.param = dict(param)

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