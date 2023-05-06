from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import torch
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')


class Logger:
    
    def __init__(self, log_level=0):
        
        self.log_level = log_level
        self.all_steps = {}
        self.writer = None
        
        # Setting up the logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
    def get_logger(self, tag='default', path_to_logs='./log/'):
        '''
        This function will set up the logs
        '''
        if tag:
            path_to_logs = os.path.join(path_to_logs, f'{tag}-{get_time_str()}.txt')
            fh = logging.FileHandler(path_to_logs)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
            fh.setLevel(logging.INFO)
            self.logger.addHandler(fh)
            
        return self.logger
    
    def lazy_init_writer(self):
        
        if not self.writer:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            
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

    def add_scalar(self, tag, value, step=None):
        
        self.lazy_init_writer()
        if log_level > self.log_level:
            return
        
        value = self.to_numpy(value)
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)

    def add_histogram(self, tag, values, step=None):
        
        self.lazy_init_writer()
        if log_level > self.log_level:
            return
        
        values = self.to_numpy(values)
        if step is None:
            step = self.get_step(tag)
        self.writer.add_histogram(tag, values, step)