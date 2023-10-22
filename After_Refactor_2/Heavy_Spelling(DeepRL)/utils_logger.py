from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import torch
import logging
from .misc import get_time_str


class Logger:
    def __init__(self, vanilla_logger=None, log_dir=None, log_level=logging.INFO):
        self.log_level = log_level
        self.writer = None
        self.all_steps = {}
        self.log_dir = log_dir

        if vanilla_logger is not None:
            self.log = vanilla_logger.info
            self.debug = vanilla_logger.debug
            self.warning = vanilla_logger.warning

    def lazy_init_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)

    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v

    def get_step(self, tag):
        step = self.all_steps.get(tag, 0)
        self.all_steps[tag] = step + 1
        return step

    def add_scalar(self, tag, value, step=None, log_level=logging.INFO):
        self.lazy_init_writer()
        if log_level > self.log_level:
            return
        value = self.to_numpy(value)
        step = step or self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)

    def add_histogram(self, tag, values, step=None, log_level=logging.INFO):
        self.lazy_init_writer()
        if log_level > self.log_level:
            return
        values = self.to_numpy(values)
        step = step or self.get_step(tag)
        self.writer.add_histogram(tag, values, step)


def get_logger(tag='default', log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    if tag is not None:
        fh = logging.FileHandler(f'./log/{tag}-{get_time_str()}.txt')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
        fh.setLevel(log_level)
        logger.addHandler(fh)
    return Logger(logger, f'./tf_log/logger-{tag}-{get_time_str()}', log_level)