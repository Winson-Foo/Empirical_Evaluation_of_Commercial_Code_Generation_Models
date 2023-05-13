import logging
import os
import pickle


def get_logger(tag, log_level=logging.INFO):
    logger = logging.getLogger(tag)
    logger.setLevel(log_level)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def close_obj(obj):
    if obj is not None:
        obj.close()


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


class StateNormalizer:
    def __init__(self, shape, default_clip_range=(0, 5)):
        self.clip_range = default_clip_range
        self.epsilon = 1e-8
        self.size = np.prod(shape)
        self.sum = np.zeros(self.size)
        self.sum_sq = np.zeros(self.size)
        self.mean = np.zeros(self.size)
        self.std = np.ones(self.size)
        self.t = 0

    def update(self, x):
        x = x.reshape(-1, self.size)
        self.sum += x.sum(axis=0)
        self.sum_sq += (x ** 2).sum(axis=0)
        self.t += x.shape[0]
        self.mean = self.sum / self.t
        self.std = np.sqrt(np.maximum(self.sum_sq / self.t - self.mean ** 2, 0) + self.epsilon)
        self.clip_range = (self.mean - 5 * self.std, self.mean + 5 * self.std)

    def __call__(self, x):
        x = x.reshape(-1, self.size)
        return np.clip((x - self.mean) / self.std, *self.clip_range).reshape(-1, *x.shape[1:])


class Config:
    def __init__(self):
        self.num_workers = 8
        self.sgd_update_frequency = 128
        self.eval_env = None
        self.eval_episodes = 512
        self.task_fn = None
        self.network_fn = None
        self.optimizer_fn = None
        self.state_normalizer = None
        self.discount = 0.99
        self.logger = None
        self.rollout_length = 5
        self.max_steps = 10e6
        self.async_actor = False
        self.tasks = []
        self.tag = ''
        self.log_level = logging.INFO
