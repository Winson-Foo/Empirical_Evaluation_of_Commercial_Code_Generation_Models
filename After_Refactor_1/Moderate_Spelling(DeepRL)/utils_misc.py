import datetime
import itertools
import os
import random
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import torch

from .torch_utils import *

def run_steps(agent):
    """Run agent until max_steps is reached"""
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save(f"data/{agent_name}-{config.tag}-{agent.total_steps}")
        if config.log_interval and not agent.total_steps % config.log_interval:
            steps_per_sec = config.log_interval / (time.time() - t0)
            agent.logger.info(f"steps {agent.total_steps}, {steps_per_sec:.2f} steps/s")
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()
        agent.switch_task()

def get_time_str():
    """Return current time in string format"""
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")

def get_default_log_dir(name):
    """Format log directory"""
    return f"./log/{name}-{get_time_str()}"

def mkdir(path):
    """Make directories at given path"""
    Path(path).mkdir(parents=True, exist_ok=True)

def random_sample(indices, batch_size):
    """Return random batches of data"""
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[: len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    remainder = len(indices) % batch_size
    if remainder:
        yield indices[-remainder:]

def generate_tag(params):
    """Generate tag using params"""
    if 'tag' in params.keys():
        return
    game = params['game']
    params.setdefault('run', 0)
    run = params['run']
    del params['game']
    del params['run']
    pairs = sorted(params.items())
    str_items = [f"{k}_{v if isinstance(v, (str, int, float, bool)) else v.__name__}" for k, v in pairs]
    tag = f"{game}-{'-'.join(str_items)}-run-{run}"
    params['tag'] = tag
    params['game'] = game
    params['run'] = run

def translate(pattern):
    """Translate pattern string to regex format"""
    groups = pattern.split('.')
    regex = r"\.".join(groups)
    return regex

def split(a, n):
    """Split sequence a into n sub-sequences"""
    k, m = divmod(len(a), n)
    sub_sequences = (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))
    return sub_sequences

class HyperParameter:
    """Hyperparameter object"""
    def __init__(self, id, param_dict):
        self.id = id
        self.param_dict = param_dict

    def __str__(self):
        return str(self.id)

    def get_dict(self):
        return self.param_dict

class HyperParameters(Sequence):
    """Sequence object containing all permutations of hyperparameters"""
    def __init__(self, ordered_params_dict):
        if not isinstance(ordered_params_dict, OrderedDict):
            raise TypeError("OrderedDict expected")
        param_pairs = []
        for key in ordered_params_dict.keys():
            items = [[key, item] for item in ordered_params_dict[key]]
            param_pairs.append(items)
        self.param_permutations = list(itertools.product(*param_pairs))

    def __getitem__(self, index):
        return HyperParameter(index, dict(self.param_permutations[index]))

    def __len__(self):
        return len(self.param_permutations)