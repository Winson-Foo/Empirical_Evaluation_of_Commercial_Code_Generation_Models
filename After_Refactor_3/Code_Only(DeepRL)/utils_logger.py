from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import torch
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

from .misc import *

class Logger(object):
    def __init__(self, tag, log_level=0):
        self.log_level = log_level
        self.writer = None
        self.pth = './log/%s-%s.txt' % (tag, get_time_str())
        create_dir(self.pth)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.pth)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)
        self.all_steps = {}
        self.writer = SummaryWriter(log_dir='logger-%s-%s' % (tag, get_time_str()))

    def to_numpy(self, v):
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
        if log_level > self.log_level:
            return
        value = self.to_numpy(value)
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)
        self.logger.info('%s: %s' % (tag, value))

    def add_histogram(self, tag, value, step=None, log_level=0):
        if log_level > self.log_level:
            return
        value = self.to_numpy(value)
        if step is None:
            step = self.get_step(tag)
        self.writer.add_histogram(tag, value, step)
        self.logger.info('%s: %s' % (tag, value))