import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
  def __init__(self, log_dir, log_level=0):
    self.log_level = log_level
    self.writer = None
    self.all_steps = {}
    self.log_dir = log_dir

  def lazy_init_writer(self):
    if self.writer is None:
      self.writer = SummaryWriter(self.log_dir)

  def to_numpy(self, value):
    if isinstance(value, torch.Tensor):
      value = value.cpu().detach().numpy()
    return value

  def get_step(self, tag):
    if tag not in self.all_steps:
      self.all_steps[tag] = 0
    step = self.all_steps[tag]
    self.all_steps[tag] += 1
    return step

  def add_scalar(self, tag, value, step=None):
    self.lazy_init_writer()
    if self.log_level > 0:
      return
    value = self.to_numpy(value)
    if step is None:
      step = self.get_step(tag)
    if np.isscalar(value):
      value = np.asarray([value])
    self.writer.add_scalar(tag, value, step)

  def add_histogram(self, tag, values, step=None):
    self.lazy_init_writer()
    if self.log_level > 0:
      return
    values = self.to_numpy(values)
    if step is None:
      step = self.get_step(tag)
    self.writer.add_histogram(tag, values, step)


def get_logger(tag='default', log_level=0):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  if tag is not None:
    fh = logging.FileHandler(f'./log/{tag}-{get_time_str()}.txt')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
  return Logger(logger, f'./tf_log/logger-{tag}-{get_time_str()}', log_level)