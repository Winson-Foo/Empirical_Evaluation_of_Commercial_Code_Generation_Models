from torch.utils.tensorboard import SummaryWriter
import torch
import logging


class Logger:
    def __init__(self, tag=None, log_level=logging.INFO):
        self.tag = tag
        self.log_level = log_level
        self.writer = None
        self.all_steps = {}

        self.vanilla_logger = logging.getLogger()
        self.vanilla_logger.setLevel(log_level)

        if tag is not None:
            log_file = f'./log/{tag}-{get_time_str()}.txt'
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            self.vanilla_logger.addHandler(file_handler)

    def lazy_init_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(f'./tf_log/logger-{self.tag}-{get_time_str()}')

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

    def add_scalar(self, tag, value, step=None, log_level=None):
        log_level = log_level or self.log_level
        if log_level > self.log_level:
            return
        value = self.to_numpy(value)
        step = step or self.get_step(tag)
        if isinstance(value, (int, float)):
            value = [value]
        self.lazy_init_writer()
        self.writer.add_scalar(tag, value, step)

    def add_histogram(self, tag, values, step=None, log_level=None):
        log_level = log_level or self.log_level
        if log_level > self.log_level:
            return
        values = self.to_numpy(values)
        step = step or self.get_step(tag)
        self.lazy_init_writer()
        self.writer.add_histogram(tag, values, step)


def getLogger(tag='default', log_level=logging.INFO):
    logger = Logger(tag=tag, log_level=log_level)
    return logger


def get_time_str():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')