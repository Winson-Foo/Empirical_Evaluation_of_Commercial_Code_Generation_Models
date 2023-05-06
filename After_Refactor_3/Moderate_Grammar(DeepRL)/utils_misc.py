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

LOG_DIR = "./log"

def run_steps(agent):
    """Train agent for a specified number of steps."""
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if config.save_interval and agent.total_steps % config.save_interval == 0:
            path = os.path.join(LOG_DIR, f"{agent_name}-{config.tag}-{agent.total_steps}")
            agent.save(path)
        if config.log_interval and agent.total_steps % config.log_interval == 0:
            logging.info("steps %d, %.2f steps/s" % (agent.total_steps, config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval and agent.total_steps % config.eval_interval == 0:
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()
        agent.switch_task()


def get_time_str():
    """Get current datetime as a formatted string."""
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_default_log_dir(name):
    """Create a log directory for the current run."""
    return os.path.join(LOG_DIR, f"{name}-{get_time_str()}")


def mkdir(path):
    """Create a directory if it doesn't already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def random_sample(indices, batch_size):
    """Generate random batches of samples from a list of indices."""
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[: len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


def is_plain_type(x):
    """Determine if x is a primitive data type."""
    for t in [str, int, float, bool]:
        if isinstance(x, t):
            return True
    return False


def generate_tag(params):
    """Generate a unique string for identifying the current run."""
    game = params["game"]
    params.setdefault("run", 0)
    run = params["run"]
    del params["game"]
    del params["run"]
    tags = [f"{k}_{v if is_plain_type(v) else v.__name__}" for k, v in sorted(params.items())]
    tag = f"{game}-{'-'.join(tags)}-run-{run}"
    params["tag"] = tag
    params["game"] = game
    params["run"] = run


class HyperParameter:
    """A single hyperparameter setting."""
    def __init__(self, id, param):
        self.id = id
        self.param = dict(param)

    def __str__(self):
        return str(self.id)

    def dict(self):
        return self.param


class HyperParameters(Sequence):
    """A collection of all possible hyperparameter settings."""
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