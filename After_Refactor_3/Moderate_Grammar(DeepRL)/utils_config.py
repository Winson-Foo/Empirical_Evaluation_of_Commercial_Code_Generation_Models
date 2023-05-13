import argparse
import torch
from .normalizer import RescaleNormalizer


class Config:
    DEFAULT_REPLAY = 'replay'
    PRIORITIZED_REPLAY = 'prioritized_replay'
    DEVICE = torch.device('cpu')
    NOISY_LAYER_STD = 0.1

    def __init__(self):
        self._init_args()
        self._init_defaults()
        self._parse_args()

    def _init_args(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--log-level', type=int, default=0)
        # Add more arguments here

    def _init_defaults(self):
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

    def _parse_args(self):
        args = self.parser.parse_args()
        config_dict = vars(args)
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])

    @property
    def eval_env(self):
        return self.__eval_env

    @eval_env.setter
    def eval_env(self, env):
        self.__eval_env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.task_name = env.name