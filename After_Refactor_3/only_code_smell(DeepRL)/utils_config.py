import argparse
import torch

from .normalizer import RescaleNormalizer


class Config:
    DEVICE = torch.device('cpu')
    DEFAULT_REPLAY = 'replay'
    PRIORITIZED_REPLAY = 'prioritized_replay'

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Environment and agent property definitions
        self.task = None
        self.task_name = None
        self.state_dim = None
        self.action_dim = None
        self.discount = 0.99
        self.num_workers = 1
        self.rollout_length = 1
        self.gradient_clip = None
        self.min_memory_size = None
        self.max_steps = 0
        self.replay_type = Config.DEFAULT_REPLAY
        self.optimization_epochs = 4
        self.mini_batch_size = 64
        self.n_step = 1
        self.target_network_update_freq = None
        self.target_network_mix = 0.001
        self.double_q = False
        self.use_gae = False
        self.gae_tau = 1.0
        self.termination_regularizer = 0
        self.categorical_v_min = None
        self.categorical_v_max = None
        self.categorical_n_atoms = 51
        self.num_quantiles = None

        # Network and optimizer property definitions
        self.network_fn = None
        self.optimizer_fn = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.shared_repr = False
        self.noisy_linear = False
        self.noisy_layer_std = 0.1

        # Replay buffer and random process property definitions
        self.replay_fn = None
        self.random_process_fn = None
        self.random_action_prob = None

        # Normalization property definitions
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()

        # Log and display property definitions
        self.tag = 'vanilla'
        self.log_interval = int(1e3)
        self.save_interval = 0
        self.eval_interval = 0
        self.eval_episodes = 10
        self.history_length = None
        self.entropy_weight = 0
        self.iteration_log_interval = 30
        self.log_level = 0

        # Miscellaneous property definitions
        self.tasks = False
        self.async_actor = True
        self.decaying_lr = False

    # Getter and setter for eval_env property
    @property
    def eval_env(self):
        return self.__eval_env

    @eval_env.setter
    def eval_env(self, env):
        self.__eval_env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.task_name = env.name

    # Add an argument to the parser
    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    # Merge configurations
    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__

        for key, value in config_dict.items():
            setattr(self, key, value)