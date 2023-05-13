# Refactored code

# imports
import datetime
import itertools
import os
import pickle
import random
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .torch_utils import *

# Constants
LOG_DIR = './log/'
DATA_DIR = 'data/'

# Helper functions

def get_time_str():
    """Returns current time in a string format."""
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_default_log_dir(name):
    """Returns directory path for logging."""
    return os.path.join(LOG_DIR, f'{name}-{get_time_str()}')


def mkdir(path):
    """Creates directory for given path."""
    Path(path).mkdir(parents=True, exist_ok=True)


def close_obj(obj):
    """Closes the an object if it has a close method."""
    if hasattr(obj, 'close'):
        obj.close()


def random_sample(indices, batch_size):
    """Returns batches of randomly sampled indices as per given batch size."""
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


def is_plain_type(x):
    """Returns True if given input data type is a plain type. Returns False otherwise."""
    for t in [str, int, float, bool]:
        if isinstance(x, t):
            return True
    return False


def generate_tag(params):
    """Generates a unique tag string based on the given parameters."""
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
    """Replaces all single dots with multiple dots."""
    groups = pattern.split('.')
    pattern = ('\.').join(groups)
    return pattern


def split(a, n):
    """Splits the input data into n number of batches."""
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


class HyperParameter:
    """Hyperparameters class with id and parameter attributes."""

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
    """Hyperparameters container class with Sequence protocol."""

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


def run_steps(agent):
    """Runs the training loop with the given agent configuration."""
    config = agent.config
    agent_name = agent.__class__.__name__
    time_prev = time.time()
    
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save(os.path.join(DATA_DIR, f'{agent_name}-{config.tag}-{agent.total_steps}'))
        if config.log_interval and not agent.total_steps % config.log_interval:
            t_diff = time.time() - time_prev
            agent.logger.info(f'steps {agent.total_steps}, {config.log_interval / t_diff:.2f} steps/s')
            time_prev = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()
        agent.switch_task()