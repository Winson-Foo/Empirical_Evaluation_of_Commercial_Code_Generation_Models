import numpy as np
import pickle
import os
import datetime
import torch
import time
from .torch_utils import *
from pathlib import Path
import itertools
from collections import OrderedDict, Sequence


class Agent:
    def __init__(self, config):
        self.config = config
        self.total_steps = 0
        self.logger = get_logger(config)

    def save(self, save_path):
        torch.save(self.model.state_dict(), f'{save_path}.pt')
        data = {'total_steps': self.total_steps}
        with open(f'{save_path}.pkl', 'wb') as f:
            pickle.dump(data, f)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(f'{load_path}.pt'))
        with open(f'{load_path}.pkl', 'rb') as f:
            data = pickle.load(f)
        self.total_steps = data['total_steps']

    def step(self):
        raise NotImplementedError

    def eval_episodes(self):
        raise NotImplementedError

    def switch_task(self):
        raise NotImplementedError

    def close(self):
        pass


def run_steps(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save(os.path.join('data', f'{agent_name}-{config.tag}-{agent.total_steps}'))
        if config.log_interval and not agent.total_steps % config.log_interval:
            agent.logger.info(f'steps {agent.total_steps}, {config.log_interval / (time.time() - t0): .2f} steps/s')
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()
        agent.switch_task()


def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_default_log_dir(name):
    return os.path.join('log', f'{name}-{get_time_str()}')


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
    if 'tag' in params.keys():
        return
    game = params['game']
    params.setdefault('run', 0)
    run = params['run']
    del params['game']
    del params['run']
    str_ = [f'{k}_{v if is_plain_type(v) else v.__name__}' for k, v in sorted(params.items())]
    tag = f'{game}-{"-".join(str_)}-run-{run}'
    params.update({'tag': tag, 'game': game, 'run': run})


def translate(pattern):
    groups = pattern.split('.')
    pattern = ('\.').join(groups)
    return pattern


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


class HyperParameter:
    def __init__(self, id_, param):
        self.id_ = id_
        self.param = {key: item for key, item in param}

    def __str__(self):
        return str(self.id_)

    def dict(self):
        return self.param


class HyperParameters(Sequence):
    def __init__(self, ordered_params):
        if not isinstance(ordered_params, OrderedDict):
            raise NotImplementedError
        self.params = []
        for key in ordered_params.keys():
            param = [[key, iterm] for iterm in ordered_params[key]]
            self.params.append(param)
        self.params = list(itertools.product(*self.params))

    def __getitem__(self, index):
        return HyperParameter(index, self.params[index])

    def __len__(self):
        return len(self.params)