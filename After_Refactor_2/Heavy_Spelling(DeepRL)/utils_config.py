from typing import Optional

import argparse
import torch
from .normalizer import RescaleNormalizer


class BaseConfig:
    def __init__(self, device: torch.device = torch.device('cpu'), double_q: bool = False, num_workers: int = 1,
                 gradient_clip: Optional[float] = None, entropy_weight: float = 0, use_gae: bool = False,
                 gae_tau: float = 1.0, target_network_mix: float = 0.001,
                 state_normalizer: Optional[RescaleNormalizer] = None,
                 reward_normalizer: Optional[RescaleNormalizer] = None, min_memory_size: Optional[int] = None,
                 max_steps: int = 0, iteration_log_interval: int = 30, optimization_epochs: int = 4,
                 mini_batch_size: int = 64, termination_regularizer: float = 0,
                 sgd_update_frequency: Optional[int] = None, random_action_prob: Optional[float] = None,
                 async_actor: bool = True, tasks: bool = False, replay_type: str = 'replay',
                 decaying_lr: bool = False, shared_repr: bool = False, noisy_linear: bool = False,
                 n_step: int = 1):
        self.device = device
        self.double_q = double_q
        self.num_workers = num_workers
        self.gradient_clip = gradient_clip
        self.entropy_weight = entropy_weight
        self.use_gae = use_gae
        self.gae_tau = gae_tau
        self.target_network_mix = target_network_mix
        self.state_normalizer = state_normalizer or RescaleNormalizer()
        self.reward_normalizer = reward_normalizer or RescaleNormalizer()
        self.min_memory_size = min_memory_size
        self.max_steps = max_steps
        self.iteration_log_interval = iteration_log_interval
        self.optimization_epochs = optimization_epochs
        self.mini_batch_size = mini_batch_size
        self.termination_regularizer = termination_regularizer
        self.sgd_update_frequency = sgd_update_frequency
        self.random_action_prob = random_action_prob
        self.async_actor = async_actor
        self.tasks = tasks
        self.replay_type = replay_type
        self.decaying_lr = decaying_lr
        self.shared_repr = shared_repr
        self.noisy_linear = noisy_linear
        self.n_step = n_step


class TrainingConfig(BaseConfig):
    def __init__(self, discount: float, target_network_update_freq: int, exploration_steps: int, rollout_length: int,
                 value_loss_weight: float, categorical_v_min: Optional[float] = None,
                 categorical_v_max: Optional[float] = None, categorical_n_atoms: int = 51,
                 num_quantiles: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.discount = discount
        self.target_network_update_freq = target_network_update_freq
        self.exploration_steps = exploration_steps
        self.rollout_length = rollout_length
        self.value_loss_weight = value_loss_weight
        self.categorical_v_min = categorical_v_min
        self.categorical_v_max = categorical_v_max
        self.categorical_n_atoms = categorical_n_atoms
        self.num_quantiles = num_quantiles


class ActorConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Config:
    DEFAULT_REPLAY = 'replay'
    PRIORITIZED_REPLAY = 'prioritized_replay'

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.task_fn = None
        self.optimizer_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.network_fn = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.replay_fn = None
        self.random_process_fn = None
        self.tag = 'vanilla'
        self.log_level = 0
        self.history_length = None
        self.num_quantiles = None
        self.save_interval = 0
        self.log_interval = int(1e3)
        self.eval_interval = 0
        self.eval_episodes = 10
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

        for key, value in config_dict.items():
            setattr(self, key, value)