import datetime
import itertools
import logging
import os
import random
import time
from collections import OrderedDict, Sequence
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from .torch_utils import *


def run_steps(agent) -> None:
    """Run the agent for a specified number of steps."""
    config = agent.config
    agent_name = agent.__class__.__name__

    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save(os.path.join('data', f'{agent_name}-{config.tag}-{agent.total_steps}'))

        if config.log_interval and not agent.total_steps % config.log_interval:
            logging.info('steps %d, %.2f steps/s' % (agent.total_steps, config.log_interval / (time.time() - t0)))
            t0 = time.time()

        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()

        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break

        agent.step()
        agent.switch_task()


def get_time_str() -> str:
    """Get the current time as a string."""
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')


def get_default_log_dir(name: str) -> str:
    """Get the default log directory for a given name."""
    return os.path.join('log', f'{name}-{get_time_str()}')


def mkdir(path: str) -> None:
    """Create a directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def random_sample(indices: List[int], batch_size: int) -> List[List[int]]:
    """Generate batches of random samples from a list of indices."""
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)

    for batch in batches:
        yield batch

    remainder = len(indices) % batch_size

    if remainder:
        yield indices[-remainder:]


def is_plain_type(x) -> bool:
    """Check if a given value is a plain type (string, int, float, or bool)."""
    for t in [str, int, float, bool]:
        if isinstance(x, t):
            return True

    return False


def generate_tag(params: dict) -> None:
    """Generate a unique tag for the experiment based on the given parameter values."""
    if 'tag' in params:
        return

    game = params['game']
    run = params.setdefault('run', 0)
    del params['game'], params['run']

    str_vals = [f'{k}_{v if is_plain_type(v) else v.__name__}' for k, v in sorted(params.items())]
    tag = f'{game}-{"-".join(str_vals)}-run-{run}'

    params['tag'] = tag
    params['game'] = game
    params['run'] = run


def translate(pattern: str) -> str:
    """Translate a pattern string into a regular expression."""
    return '\\.'.join(pattern.split('.'))


def split(a: List[int], n: int) -> List[List[int]]:
    """Split a given list into n sublists of approximately equal length."""
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


class HyperParameter:
    """A single hyperparameter for an experiment."""

    def __init__(self, id: int, param: dict):
        self.id = id
        self.param = param

    def __str__(self) -> str:
        return str(self.id)

    def dict() -> dict:
        return self.param


class HyperParameters(Sequence):
    """A collection of hyperparameters for an experiment."""

    def __init__(self, ordered_params: OrderedDict):
        if not isinstance(ordered_params, OrderedDict):
            raise NotImplementedError

        params = []
        for key in ordered_params.keys():
            param = [[key, item] for item in ordered_params[key]]
            params.append(param)

        self.params = list(itertools.product(*params))

    def __getitem__(self, index: int) -> HyperParameter:
        return HyperParameter(index, self.params[index])

    def __len__(self) -> int:
        return len(self.params)