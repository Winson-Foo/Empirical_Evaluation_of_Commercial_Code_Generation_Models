import datetime
import itertools
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import List, Tuple, Any, Union

import numpy as np
import torch

from .torch_utils import *


def run_steps(agent) -> None:
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save(f"data/{agent_name}-{config.tag}-{agent.total_steps}")
        if config.log_interval and not agent.total_steps % config.log_interval:
            steps_per_second = config.log_interval / (time.time() - t0)
            agent.logger.info(f"steps {agent.total_steps}, {steps_per_second:.2f} steps/s")
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()
        agent.switch_task()


def get_time_str() -> str:
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_default_log_dir(name: str) -> str:
    return './log/%s-%s' % (name, get_time_str())


def mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def close_obj(obj: Any) -> None:
    if hasattr(obj, 'close'):
        obj.close()


def random_sample(indices: List[int], batch_size: int) -> List[List[int]]:
    random.shuffle(indices)
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    return batches


def is_plain_type(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool))


def generate_tag(params: dict) -> None:
    if 'tag' in params:
        return
    game = params['game']
    params.setdefault('run', 0)
    run = params['run']
    del params['game']
    del params['run']
    str_ = [f"{k}_{v if is_plain_type(v) else v.__name__}" for k, v in sorted(params.items())]
    tag = f"{game}-{'-'.join(str_)}-run-{run}"
    params['tag'] = tag
    params['game'] = game
    params['run'] = run


def translate(pattern: str) -> str:
    return pattern.replace('.', '\.')


def split(lst: List[Any], n: int) -> List[List[Any]]:
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


class HyperParameter:
    def __init__(self, id: int, param: List[Tuple[str, Any]]) -> None:
        self.id = id
        self.param = dict(param)

    def __str__(self) -> str:
        return str(self.id)

    def dict(self) -> dict:
        return self.param


class HyperParameters(Sequence):
    def __init__(self, ordered_params: OrderedDict) -> None:
        if not isinstance(ordered_params, OrderedDict):
            raise TypeError("ordered_params must be an OrderedDict")
        params = []
        for key in ordered_params.keys():
            param = [(key, item) for item in ordered_params[key]]
            params.append(param)
        self.params = list(itertools.product(*params))

    def __getitem__(self, index: int) -> HyperParameter:
        return HyperParameter(index, self.params[index])

    def __len__(self) -> int:
        return len(self.params)