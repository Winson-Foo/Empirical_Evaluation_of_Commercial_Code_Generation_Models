# Refactored code

import os
import logging
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from .misc import get_time_str

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')


class Logger:
    """
    This class offers functionality to log Scalars and Histograms that can be viewed using Tensorboard.
    """
    def __init__(self, tag='default', log_level=0):
        self.log_level = log_level
        self.all_steps = {}
        self.writer = None
        self.tag = tag

        self.vanilla_logger = logging.getLogger()
        self.vanilla_logger.setLevel(logging.INFO)
        if tag is not None:
            fh = logging.FileHandler(f'./log/{tag}-{get_time_str()}.txt')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
            fh.setLevel(logging.INFO)
            self.vanilla_logger.addHandler(fh)

    def lazy_init_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(f'./tf_log/logger-{self.tag}-{get_time_str()}')

    @staticmethod
    def to_numpy(v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v

    def get_step(self, tag):
        if tag not in self.all_steps:
            self.all_steps[tag] = 0
        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step

    def add_scalar(self, tag, value, step=None, log_level=0):
        self.lazy_init_writer()
        if log_level > self.log_level:
            return
        value = self.to_numpy(value)
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)

    def add_histogram(self, tag, values, step=None, log_level=0):
        self.lazy_init_writer()
        if log_level > self.log_level:
            return
        values = self.to_numpy(values)
        if step is None:
            step = self.get_step(tag)
        self.writer.add_histogram(tag, values, step)