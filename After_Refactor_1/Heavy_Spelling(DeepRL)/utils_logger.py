from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import torch
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

def to_numpy(tensor):
    """
    Converts a tensor to a numpy array.
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().detach().numpy()
    return tensor

class Logger:
    """
    A logger that records scalar and histogram data for visualization in TensorBoard.
    """
    def __init__(self, log_dir, log_level=0):
        """
        Initializes the Logger object.

        Args:
            log_dir: (str) The directory where the log file will be saved.
            log_level: (int) The logging verbosity level.
        """
        self.all_steps = {}
        self.log_dir = log_dir
        self.log_level = log_level
        self.writer = None

    def init_writer(self):
        """
        Initializes the SummaryWriter object for logging data to TensorBoard.
        """
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)

    def get_step(self, log_tag):
        """
        Computes the current step count for a given log tag.

        If the log tag does not exist, initializes the step count to zero.

        Args:
            log_tag: (str) The tag associated with the log data.

        Returns:
            (int) The current step count for the given log tag.
        """
        if log_tag not in self.all_steps:
            self.all_steps[log_tag] = 0
        step = self.all_steps[log_tag]
        self.all_steps[log_tag] += 1
        return step

    def log_scalar(self, log_tag, value, step=None):
        """
        Logs a scalar value to TensorBoard.

        Args:
            log_tag: (str) The tag associated with the log data.
            value: (float) The scalar value to be logged.
            step: (int) The current step count for the log data. 
                  If None, the function will compute the current step count.
        """
        self.init_writer()
        if self.log_level > 0:
            return
        value = to_numpy(value)
        if step is None:
            step = self.get_step(log_tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(log_tag, value, step)

    def log_histogram(self, log_tag, values, step=None):
        """
        Logs a histogram of values to TensorBoard.

        Args:
            log_tag: (str) The tag associated with the log data.
            values: (array or tensor) The array or tensor containing the values
                    to be logged as a histogram.
            step: (int) The current step count for the log data.
                  If None, the function will compute the current step count.
        """
        self.init_writer()
        if self.log_level > 0:
            return
        values = to_numpy(values)
        if step is None:
            step = self.get_step(log_tag)
        self.writer.add_histogram(log_tag, values, step)


def get_logger(logger_tag='default', log_level=0):
    """
    Creates a logger object for recording data to a log file.

    Args:
        logger_tag: (str) The tag associated with the logger object.
        log_level: (int) The logging verbosity level.

    Returns:
        (Logger) A Logger object for recording log data.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger_tag is not None:
        log_file_name = './log/%s-%s.txt' % (logger_tag, get_time_str())
        fh = logging.FileHandler(log_file_name)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    return Logger('./tf_log/logger-%s-%s' % (logger_tag, get_time_str()), log_level)