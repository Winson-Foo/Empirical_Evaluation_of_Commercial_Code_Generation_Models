# File: config.py

import torch
import os

class Config:
    DEVICE = None

    @staticmethod
    def select_device(gpu_id):
        if gpu_id >= 0:
            Config.DEVICE = torch.device('cuda:%d' % (gpu_id))
        else:
            Config.DEVICE = torch.device('cpu')