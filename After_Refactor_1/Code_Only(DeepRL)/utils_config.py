from .normalizer import RescaleNormalizer
import argparse
import torch


class Config:
    DEFAULT_REPLAY = 'replay'
    PRIORITIZED_REPLAY = 'prioritized_replay'

    def __init__(self):
        self.device = torch.device('cpu')
        self.parser = argparse.ArgumentParser()
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
        self.noisy_layer_std = 0.1
        self.double_q = False
        self.history_length = None
        self.num_workers = 1
        self.gradient_clip = None
        self.entropy_weight = 0
        self.use_gae = False
        self.gae_tau = 1.0
        self.target_network_mix = 0.001
        self.log_level = 0
        self.iteration_log_interval = 30
        self.min_memory_size = None
        self.max_steps = 0
        self.rollout_length = None
        self.value_loss_weight = 1.0
        self.categorical_v_min = None
        self.categorical_v_max = None
        self.categorical_n_atoms = 51
        self.num_quantiles = None
        self.optimization_epochs = 4
        self.mini_batch_size = 64
        self.termination_regularizer = 0
        self.sgd_update_frequency = None
        self.random_action_prob = None
        self.eval_episodes = 10
        self.async_actor = True
        self.tasks = False
        self.replay_type = Config.DEFAULT_REPLAY
        self.decaying_lr = False
        self.shared_repr = False
        self.n_step = 1
        self.tag = 'vanilla'
        self.target_network_update_freq = None
        self.exploration_steps = None
        self.num_steps = None
        self.discount = None
        self.task_fn = None
        self.optimizer_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.network_fn = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.replay_fn = None
        self.random_process_fn = None
        self.log_interval = int(1e3)
        self.save_interval = 0
        self.eval_interval = 0
        self.__eval_env = None

    @property
    def eval_env(self):
        return self.__eval_env

    @eval_env.setter
    def eval_env(self, env):
        self.__eval_env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.task_name = env.name

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key.lower(), config_dict[key])