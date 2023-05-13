# constants
DEFAULT_TAG = 'default'

# imports
import os
import numpy as np
import torch
import logging
from torch.utils.tensorboard import SummaryWriter

#Logger class in a new module called logger.py
from logger import Logger

def get_time_string():
    return str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

def get_logger(tag=DEFAULT_TAG, log_level=logging.INFO):
    """
    Returns a logger object with the provided tag name and log level
    
    :param tag: String identifying the logger object
    :param log_level: The logging level INFO, DEBUG, WARNING, etc.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)
    if tag is not None:
        filename = './log/%s-%s.txt' % (tag, get_time_string())
        fh = logging.FileHandler(filename)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
        fh.setLevel(log_level)
        logger.addHandler(fh)
    return Logger(logger, './tf_log/logger-%s-%s' % (tag, get_time_string()), log_level)


class SummaryWriterManager(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, tag, value, step):
        value = self.to_numpy(value)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)

    def add_histogram(self, tag, values, step):
        values = self.to_numpy(values)
        self.writer.add_histogram(tag, values, step)

    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v


class LogStepManager(object):
    def __init__(self):
        self.all_steps = {}

    def get_step(self, tag):
        if tag not in self.all_steps:
            self.all_steps[tag] = 0
        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step


class LoggerFactory(object):
    def __init__(self, tag=DEFAULT_TAG, log_level=logging.INFO):
        self.tag = tag
        self.log_level = log_level
        self.logger = get_logger(tag, log_level)
        self.summary_manager = SummaryWriterManager('./tf_log/logger-%s-%s' % (tag, get_time_string()))
        self.step_manager = LogStepManager()

    def add_scalar(self, tag, value, step=None):
        if step is None:
            step = self.step_manager.get_step(tag)
        self.summary_manager.add_scalar(tag, value, step)

    def add_histogram(self, tag, values, step=None):
        if step is None:
            step = self.step_manager.get_step(tag)
        self.summary_manager.add_histogram(tag, values, step)

    def log(self, message, log_level=logging.INFO):
        """
        Logs the message with the provided log level
        
        :param message: The message to log
        :param log_level: The log level
        """
        if self.log_level <= log_level:
            self.logger.log(log_level, message)