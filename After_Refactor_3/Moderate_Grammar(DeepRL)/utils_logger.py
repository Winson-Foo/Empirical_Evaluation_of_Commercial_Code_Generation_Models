import logging
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .misc import get_time_str

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')


class Logger:
    def __init__(self, tag="default", log_level=0):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        if tag is not None:
            log_path = f"./log/{tag}-{get_time_str()}.txt"
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
            fh.setLevel(logging.INFO)
            self.logger.addHandler(fh)
        self.writer = None
        self.all_steps = {}
        self.log_level = log_level

    def add_scalar(self, tag, value, step=None):
        if self.is_logging_disabled():
            return
        value = self.to_numpy(value)
        self.ensure_writer_initialized()
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)

    def add_histogram(self, tag, values, step=None):
        if self.is_logging_disabled():
            return
        values = self.to_numpy(values)
        self.ensure_writer_initialized()
        if step is None:
            step = self.get_step(tag)
        self.writer.add_histogram(tag, values, step)

    def get_step(self, tag):
        if tag not in self.all_steps:
            self.all_steps[tag] = 0
        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step

    def ensure_writer_initialized(self):
        if self.writer is None:
            log_dir = f"./tf_log/logger-{tag}-{get_time_str()}"
            self.writer = SummaryWriter(log_dir)

    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v

    def is_logging_disabled(self):
        return self.log_level > 0