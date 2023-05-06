import datetime
import itertools
import logging
import numpy as np
import pathlib
import time
import torch

from .torch_utils import *

def run_steps(agent):
    while True:
        agent.step()
        agent.switch_task()

        if agent.total_steps % agent.config.log_interval == 0:
            steps_per_sec = agent.config.log_interval / (time.time() - agent.log_start_time)
            agent.logger.info('Steps: %d | Avg steps/s: %.2f', agent.total_steps, steps_per_sec)
            agent.log_start_time = time.time()

        if agent.config.eval_interval and agent.total_steps % agent.config.eval_interval == 0:
            agent.eval_episodes()

        if agent.config.save_interval and agent.total_steps % agent.config.save_interval == 0:
            agent.save(agent.config.save_dir / f'{agent.config.agent_name}-{agent.config.tag}-{agent.total_steps}')

        if agent.config.max_steps and agent.total_steps >= agent.config.max_steps:
            agent.close()
            break

def get_time_str():
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')

def get_default_log_dir(name):
    return pathlib.Path('./log') / f'{name}-{get_time_str()}'

def mkdir(path):
    path.mkdir(parents=True, exist_ok=True)

def random_sample(indices, batch_size):
    indices = np.random.permutation(indices)
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]

def is_plain_type(x):
    return isinstance(x, (str, int, float, bool))

def generate_tag(params):
    game = params.pop('game')
    run = params.pop('run')
    sorted_params = sorted(params.items(), key=lambda x: x[0])
    str_params = [f'{k}_{v if is_plain_type(v) else v.__name__}' for k, v in sorted_params]
    params['tag'] = f'{game}-{"-".join(str_params)}-run-{run}'
    params['game'] = game
    params['run'] = run

def translate(pattern):
    return r'\.'.join(pattern.split('.'))

def split(lst, n):
    k, m = divmod(len(lst), n)
    return (lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n))

class Config:
    def __init__(self, **kwargs):
        self.agent_name = kwargs.pop('agent_name')
        self.game = kwargs.pop('game')
        self.run = kwargs.pop('run')
        self.device = kwargs.pop('device')
        self.save_dir = pathlib.Path(kwargs.pop('save_dir'))
        self.save_interval = kwargs.get('save_interval', None)
        self.log_interval = kwargs.pop('log_interval')
        self.eval_interval = kwargs.pop('eval_interval')
        self.max_steps = kwargs.pop('max_steps')
        self.tag = kwargs.pop('tag', None)
        self.log_start_time = time.time()
        self.total_steps = 0
        
        generate_tag(self.__dict__)

        self.__dict__.update(kwargs)

        if self.save_interval:
            mkdir(self.save_dir)

        self.logger = logging.getLogger(self.tag)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()