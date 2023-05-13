from typing import Optional
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    A wrapper class for the vanilla logger instance and Tensorboard SummaryWriter instance
    """

    def __init__(
        self, tag: Optional[str] = 'default', log_level: Optional[int] = logging.INFO
    ):
        self.logger = logging.getLogger()
        self.logger.setLevel(log_level)

        if tag is not None:
            log_file_path = Path('./log/') / f"{tag}-{get_time_str()}.txt"
            log_file_path.parent.mkdir(exist_ok=True)
            fh = logging.FileHandler(log_file_path)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
            fh.setLevel(logging.INFO)
            self.logger.addHandler(fh)

        self.writer = None
        self.all_steps = {}
        self.log_level = log_level

    def get_logger(self):
        return self.logger

    def lazy_init_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)

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

    def add_scalar(self, tag, value, step=None, log_level=None):
        log_level = log_level or self.log_level
        self.lazy_init_writer()

        if log_level > self.log_level:
            return

        value = self.to_numpy(value)

        if step is None:
            step = self.get_step(tag)

        if np.isscalar(value):
            value = np.asarray([value])

        self.writer.add_scalar(tag, value, step)

    def add_histogram(self, tag, values, step=None, log_level=None):
        log_level = log_level or self.log_level
        self.lazy_init_writer()

        if log_level > self.log_level:
            return

        values = self.to_numpy(values)

        if step is None:
            step = self.get_step(tag)

        self.writer.add_histogram(tag, values, step)
        
        
def get_logger(tag: Optional[str] = 'default', log_level: Optional[int] = 0):
    """
    Helper function that creates and returns a new Logger instance.
    """
    return Logger(tag, log_level).get_logger()


def get_time_str():
    """
    Helper function to get a formatted string representing the current date and time
    """
    return time.strftime('%Y%m%d-%H%M%S')