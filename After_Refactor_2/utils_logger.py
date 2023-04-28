# imports
import logging
import os
from typing import Any, Optional
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from .misc import get_time_str


class TensorboardLogger:
    def __init__(self, tag: Optional[str] = 'default', log_level: int = logging.INFO):
        self._log_level = log_level
        self._writer = None
        if tag is not None:
            log_dir = f'./tf_log/logger-{tag}-{get_time_str()}'
            os.makedirs(log_dir, exist_ok=True)
            self._init_vanilla_logger(tag, log_dir)

        self._all_steps = {}

    def _init_vanilla_logger(tag: str, log_dir: str) -> None:
        self._vanilla_logger = logging.getLogger()
        self._vanilla_logger.setLevel(logging.INFO)
        fh = logging.FileHandler(f'./log/{tag}-{get_time_str()}.txt')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
        self._vanilla_logger.addHandler(fh)

    def lazy_init_writer(self):
        if self._writer is None:
            self._writer = SummaryWriter(self._log_dir)

    def to_numpy(self, data: Any) -> Any:
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()
        return data

    def get_step(self, tag: str) -> int:
        if tag not in self._all_steps:
            self._all_steps[tag] = 0
        step = self._all_steps[tag]
        self._all_steps[tag] += 1
        return step

    def add_scalar(self, tag: str, value: Any, step: Optional[int] = None, log_level: int = 0):
        self.lazy_init_writer()
        if log_level > self._log_level:
            return
        value = self.to_numpy(value)
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self._writer.add_scalar(tag, value, step)

    def add_histogram(self, tag: str, values: Any, step: Optional[int] = None, log_level: int = 0):
        self.lazy_init_writer()
        if log_level > self._log_level:
            return
        values = self.to_numpy(values)
        if step is None:
            step = self.get_step(tag)
        self._writer.add_histogram(tag, values, step)