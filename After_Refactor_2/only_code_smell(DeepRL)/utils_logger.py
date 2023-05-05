import logging
import os
from typing import Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .misc import get_time_str


class Logger:
    """
    This class provides logging functionality for TensorBoard and text files

    Attributes:
        writer: a SummaryWriter object
        all_steps: a dictionary to keep track of the step number for each recorded tag
        log_dir: the filepath where logs are saved
        log_level: an integer to specify the minimum log level required to record a message
    """
    def __init__(self, log_dir: str, log_level: int = 0):
        self.log_level = log_level
        self.writer = SummaryWriter(log_dir)
        self.all_steps = {}
        self.log_dir = log_dir

    def to_numpy(self, v):
        """
        Convert a tensor to a numpy array

        Args:
            v: a tensor object

        Returns:
            a numpy array
        """
        return np.array(v)

    def get_step(self, tag):
        """
        Return the current step number for a given tag

        Args:
            tag: a string

        Returns:
            an integer
        """
        if tag not in self.all_steps:
            self.all_steps[tag] = 0
        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step

    def add_scalar(self, tag: str, value: Union[int, float], log_level: int = 0):
        """
        Add a scalar value to the SummaryWriter for a given tag

        Args:
            tag: a string
            value: an integer or float value to be recorded
            log_level: an integer specifying the minimum log level required to record a message
        """
        if log_level > self.log_level:
            return
        value = self.to_numpy(value)
        step = self.get_step(tag)
        if not isinstance(value, np.ndarray):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)

    def add_histogram(self, tag, values, log_level=0):
        """
        Add a histogram to the SummaryWriter for a given tag

        Args:
            tag: a string
            values: a numpy array of values to be recorded
            log_level: an integer specifying the minimum log level required to record a message
        """
        if log_level > self.log_level:
            return
        values = self.to_numpy(values)
        step = self.get_step(tag)
        self.writer.add_histogram(tag, values, step)


def get_logger(tag='default', log_level=0):
    """
    A function that creates a Logger object and returns it

    Args:
        tag: a string to identify the logger object, defaults to 'default'
        log_level: an integer specifying the minimum log level required to record a message

    Returns:
        a Logger object
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if tag is not None:
        fh = logging.FileHandler(f'./log/{tag}-{get_time_str()}.txt')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    return Logger(f'./tf_log/logger-{tag}-{get_time_str()}', log_level)