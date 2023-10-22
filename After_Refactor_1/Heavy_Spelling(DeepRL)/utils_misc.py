import datetime
import itertools
import os
import pickle
import time
from collections import OrderedDict, Sequence
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from .torch_utils import *

# Constants
DEFAULT_LOG_DIR = './log'
SAVE_DIR = './data'


# Agent Class
class Agent:
    def __init__(self, config):
        self.config = config
        self.logger = create_logger(config)
        self.total_steps = 0
        self.tasks = self.config.tasks
        self.task_idx = 0
        self.is_training = True

    def save(self, save_path):
        torch.save(self.policy.state_dict(), save_path)
        self.logger.info("Saved model at step %s to %s" % (self.total_steps, save_path))

    def step(self):
        raise NotImplementedError

    def switch_task(self):
        if not self.is_training:
            return
        self.task_idx = (self.task_idx + 1) % len(self.tasks)

    def eval_episodes(self):
        raise NotImplementedError

    def close(self):
        pass


def run_agent_simulation(agent: Agent) -> None:
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()

    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            save_agent(agent_name, config.tag, agent.total_steps, agent.policy)
        if config.log_interval and not agent.total_steps % config.log_interval:
            log_step_information(agent, t0)
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()
        agent.switch_task()


def create_logger(config) -> List:
    log_dir = config.log_dir or os.path.join(DEFAULT_LOG_DIR, config.tag)
    mkdir(log_dir)
    handlers = []
    handlers.append(create_logging_stdout_handler(config))
    handlers.append(create_logging_file_handler(log_dir))
    logger = get_logger(__name__, handlers)
    logger.propagate = False
    return logger


def create_logging_stdout_handler(config):
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    return handler


def create_logging_file_handler(log_dir):
    filename = os.path.join(log_dir, 'output.log')
    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    return handler


def save_agent(agent_name: str, tag: str, total_steps: int, policy) -> None:
    save_path = os.path.join(SAVE_DIR, '%s-%s-%d.pth' % (agent_name, tag, total_steps))
    torch.save(policy.state_dict(), save_path)
    logger.info("Saved model at step %s to %s" % (total_steps, save_path))


def log_step_information(agent, t0):
    logger = agent.logger
    total_steps = agent.total_steps
    log_interval = agent.config.log_interval
    elapsed_time = time.time() - t0
    steps_per_sec = log_interval / elapsed_time
    logger.info('steps %d, %.2f steps/s' % (total_steps, steps_per_sec))


def generate_batch_indices(indices: np.ndarray, batch_size: int):
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


def dict_to_str(params: dict) -> str:
    str_params = ['%s_%s' % (k, v if is_plain_type(v) else v.__name__) for k, v in sorted(params.items())]
    return '-'.join(str_params)


def get_default_log_dir(name):
    return os.path.join(DEFAULT_LOG_DIR, '%s-%s' % (name, get_time_str()))


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def random_split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


class HyperParameter:
    def __init__(self, id: int, param: List[Tuple[str, str]]):
        self.id = id
        self.param = dict()
        for key, item in param:
            self.param[key] = item

    def __str__(self):
        return str(self.id)

    def dict(self):
        return self.param


class HyperParameters(Sequence):
    def __init__(self, ordered_params: OrderedDict):
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