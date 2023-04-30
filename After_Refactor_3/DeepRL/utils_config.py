import argparse
import torch
from .normalizer import RescaleNormalizer

class Config:
    DEVICE = torch.device('cpu')
    NOISY_LAYER_STD = 0.1
    DEFAULT_REPLAY = 'replay'
    PRIORITIZED_REPLAY = 'prioritized_replay'

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._state_normalizer = RescaleNormalizer()
        self._reward_normalizer = RescaleNormalizer()
        self._eval_env = None
        
        self._set_default_values()

    def _set_default_values(self):
        self.task_fn = None
        self.optimizer_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.network_fn = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.replay_fn = None
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
        self._replay_type = Config.DEFAULT_REPLAY
        self.decaying_lr = False
        self.shared_repr = False
        self.noisy_linear = False
        self.n_step = 1

        self._log_interval = int(1e3)
        self._save_interval = 0
        self._eval_interval = 0
        self._eval_episodes = 10
        self.async_actor = True
        self.tasks = False

    def _parse_arguments(self):
        args = self.parser.parse_args()
        arg_dict = args.__dict__
        for key, value in arg_dict.items():
            if key.startswith('_'):
                continue
            setattr(self, key, value)
    
    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    @property
    def eval_env(self):
        return self._eval_env

    @eval_env.setter
    def eval_env(self, env):
        self._eval_env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.task_name = env.name

    @property
    def replay_type(self):
        return self._replay_type

    @replay_type.setter
    def replay_type(self, value):
        self._replay_type = value.lower()

    @property
    def log_interval(self):
        return self._log_interval

    @log_interval.setter
    def log_interval(self, value):
        self._log_interval = int(value)

    @property
    def save_interval(self):
        return self._save_interval

    @save_interval.setter
    def save_interval(self, value):
        self._save_interval = int(value)

    @property
    def eval_interval(self):
        return self._eval_interval

    @eval_interval.setter
    def eval_interval(self, value):
        self._eval_interval = int(value)

    @property
    def eval_episodes(self):
        return self._eval_episodes

    @eval_episodes.setter
    def eval_episodes(self, value):
        self._eval_episodes = int(value)

    def set_device(self, device):
        self.DEVICE = torch.device(device)

    def set_state_normalizer(self, normalizer):
        self._state_normalizer = normalizer

    def set_reward_normalizer(self, normalizer):
        self._reward_normalizer = normalizer

    def get_state_normalizer(self):
        return self._state_normalizer

    def get_reward_normalizer(self):
        return self._reward_normalizer

    def initialize(self, config_dict=None):
        self._set_default_values()
        self._parse_arguments()
        if config_dict is not None:
            self.merge(config_dict)

    def merge(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)