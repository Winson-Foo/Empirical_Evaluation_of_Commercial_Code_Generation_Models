import numpy as np
import pickle
import os
import datetime
import torch
import time
from pathlib import Path
import itertools
from collections import OrderedDict, Sequence

from .torch_utils import *

def run_steps(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        should_save = config.save_interval and not agent.total_steps % config.save_interval
        should_log = config.log_interval and not agent.total_steps % config.log_interval
        should_eval = config.eval_interval and not agent.total_steps % config.eval_interval
        should_end = config.max_steps and agent.total_steps >= config.max_steps
        
        if should_save:
            agent.save('data/%s-%s-%d' % (agent_name, config.tag, agent.total_steps))
            
        if should_log:
            elapsed_time = time.time() - t0
            steps_s = config.log_interval / elapsed_time
            agent.logger.info(f'steps {agent.total_steps}, {steps_s:.2f} steps/s')
            t0 = time.time()
            
        if should_eval:
            agent.eval_episodes()
        
        if should_end:
            agent.close()
            break
        
        agent.step()
        agent.switch_task()


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
    all_items = [(k, v if is_plain_type(v) else v.__name__) for k, v in sorted(params.items())]
    str_items = [f'{k}_{v}' for k, v in all_items]
    tag = f'{game}-{str_items}-run-{run}'
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
        self.params = []
        for key in ordered_params.keys():
            param = [[key, item] for item in ordered_params[key]]
            self.params.append(param)

        self.params = list(itertools.product(*self.params))

    def __getitem__(self, index):
        return HyperParameter(index, self.params[index])

    def __len__(self):
        return len(self.params)