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
    config = agent.config
    agent_name = agent.__class__.__name__
    time_0 = time.time()

    while True:
        if save_interval := config.save_interval and not agent.total_steps % save_interval:
            agent.save(f"data/{agent_name}-{config.tag}-{agent.total_steps}")

        if log_interval := config.log_interval and not agent.total_steps % log_interval:
            steps_second = log_interval / (time.time() - time_0)
            agent.logger.info(f"steps {agent.total_steps}, {steps_second:.2f} steps/s")
            time_0 = time.time()

        if eval_interval := config.eval_interval and not agent.total_steps % eval_interval:
            agent.eval_episodes()

        if max_steps := config.max_steps and agent.total_steps >= max_steps:
            agent.close()
            break

        agent.step()
        agent.switch_task()


def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_default_log_dir(name):
    return f"./log/{name}-{get_time_str()}"


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
    return isinstance(x, (str, int, float, bool))


def generate_tag(params):
    game = params.get('game')
    run = params.get('run', 0)

    if not game:
        raise ValueError("Game parameter is missing.")

    del params['game']
    del params['run']

    sorted_params = [f"{k}_{v if is_plain_type(v) else v.__name__}" for k, v in sorted(params.items())]
    tag = f"{game}-{'-'.join(sorted_params)}-run-{run}"

    params['tag'] = tag
    params['game'] = game
    params['run'] = run


def translate(pattern):
    groups = pattern.split('.')
    pattern = '\\.'.join(groups)
    return pattern


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
            raise TypeError("The ordered_params must be an instance of OrderedDict.")

        params = [OrderedDict(zip(ordered_params.keys(), p)) for p in itertools.product(*ordered_params.values())]
        self.params = [HyperParameter(idx, param) for idx, param in enumerate(params)]

    def __getitem__(self, index):
        return self.params[index]

    def __len__(self):
        return len(self.params)