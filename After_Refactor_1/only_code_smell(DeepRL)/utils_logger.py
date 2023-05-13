import logging
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
LOGGER_FORMAT = '%(asctime)s - %(name)s - %(levelname)s: %(message)s'
LOG_DIR = './log'
TF_LOG_DIR = './tf_log'

def get_logger(tag='default', log_level=logging.INFO):
    logger = logging.getLogger(tag)
    logger.setLevel(logging.INFO)
    if tag:
        fh = logging.FileHandler(f'{LOG_DIR}/{tag}-{get_time_str()}.txt')
        fh.setFormatter(logging.Formatter(LOGGER_FORMAT))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    return Logger(logger, f'{TF_LOG_DIR}/logger-{tag}-{get_time_str()}', log_level)

class Logger:
    def __init__(self, vanilla_logger, log_dir, log_level=logging.INFO):
        self.vanilla_logger = vanilla_logger
        self.writer = None
        self.log_level = log_level
        self.all_steps = {}
        self.log_dir = log_dir

    def lazy_init_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)

    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v

    def get_step(self, tag):
        self.all_steps[tag] = self.all_steps.get(tag, 0) + 1
        return self.all_steps[tag] - 1

    def add_scalar(self, tag, value, step=None):
        self.lazy_init_writer()
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, self.to_numpy(value), step)

    def add_histogram(self, tag, values, step=None):
        self.lazy_init_writer()
        if step is None:
            step = self.get_step(tag)
        self.writer.add_histogram(tag, self.to_numpy(values), step)

    def info(self, msg, *args, **kwargs):
        if self.vanilla_logger:
            self.vanilla_logger.info(msg, *args, **kwargs)
        print(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        if self.log_level <= logging.DEBUG:
            if self.vanilla_logger:
                self.vanilla_logger.debug(msg, *args, **kwargs)
            print(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if self.vanilla_logger:
            self.vanilla_logger.warning(msg, *args, **kwargs)
        print(msg, *args, **kwargs)