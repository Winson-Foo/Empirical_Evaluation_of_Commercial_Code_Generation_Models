import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, tag=None, log_level=logging.INFO, log_dir='./tf_log'):
        self.tag = tag
        if tag is not None:
            self.log_file = os.path.join(log_dir, f'{tag}-{get_time_str()}.txt')
        else:
            self.log_file = None
        self.log_level = log_level
        self.writer = None
        self.all_steps = {}

        self._configure_logger()

    def _configure_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')

        if self.log_file:
            fh = logging.FileHandler(self.log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def add_scalar(self, tag, value, step=None):
        self._lazy_init_writer()
        if self.log_level <= logging.INFO:
            value = self._to_numpy(value)
            scalar = self._get_scalar(value)
            self.writer.add_scalar(tag, scalar, step)

    def add_histogram(self, tag, values, step=None):
        self._lazy_init_writer()
        if self.log_level <= logging.INFO:
            values = self._to_numpy(values)
            self.writer.add_histogram(tag, values, step)

    def _lazy_init_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(self._get_log_dir())

    def _to_numpy(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return value

    def _get_scalar(self, value):
        if np.isscalar(value):
            return value
        else:
            return float(np.mean(value))

    def _get_step(self, tag):
        if tag not in self.all_steps:
            self.all_steps[tag] = 0
        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step

    def _get_log_dir(self):
        if self.tag:
            return os.path.join('./tf_log', f'logger-{self.tag}-{get_time_str()}')
        else:
            return './tf_log'


def get_time_str():
    return datetime.now().strftime('%Y%m%d_%H%M%S')