import argparse
from typing import List, Dict
from logging import getLogger, basicConfig, DEBUG, ERROR

from .env import Env
from .trainer import Trainer


logger = getLogger(__name__)

class Config:
    """Configuration object"""

    DEVICE = torch.device('cpu')
    NOISY_LAYER_STD = 0.1
    DEFAULT_REPLAY = 'replay'
    PRIORITIZED_REPLAY = 'prioritized_replay'

    def __init__(self, env: Env, config_dict: Dict = None) -> None:
        """Construct the configuration object"""
        self.parser = argparse.ArgumentParser()
        self.env = env
        self.random_process_fn = None
        self.discount = None
        self.target_network_update_freq = None
        self.exploration_steps = None
        self.log_level = 0
        self.history_length = None
        self.double_q = False
        self.tag = 'vanilla'
        self.num_workers = 1
        self.gradient_clip = None
        self.entropy_weight = 0
        self.use_gae = False
        self.gae_tau = 1.0
        self.target_network_mix = 0.001
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
        self.min_memory_size = None
        self.max_steps = 0
        self.rollout_length = None
        self.value_loss_weight = 1.0
        self.iteration_log_interval = 30
        self.categorical_v_min = None
        self.categorical_v_max = None
        self.categorical_n_atoms = 51
        self.num_quantiles = None
        self.optimization_epochs = 4
        self.mini_batch_size = 64
        self.termination_regularizer = 0
        self.sgd_update_frequency = None
        self.random_action_prob = None
        self.__eval_env = None
        self.log_interval = int(1e3)
        self.save_interval = 0
        self.eval_interval = 0
        self.eval_episodes = 10
        self.async_actor = True
        self.tasks = False
        self.replay_type = Config.DEFAULT_REPLAY
        self.decaying_lr = False
        self.shared_repr = False
        self.noisy_linear = False
        self.n_step = 1

        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])

    @property
    def eval_env(self) -> Env:
        return self.__eval_env

    @eval_env.setter
    def eval_env(self, env: Env) -> None:
        self.__eval_env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.task_name = env.name

    def add_argument(self, *args, **kwargs) -> None:
        self.parser.add_argument(*args, **kwargs)

    def merge(self, config_dict: Dict = None) -> None:
        if config_dict is not None:
            for key in config_dict.keys():
                setattr(self, key, config_dict[key])
                

def main() -> None:
    basicConfig(level=ERROR)
    env = Env()
    config = Config(env)
    trainer = Trainer(config, logger)
    trainer.train()
