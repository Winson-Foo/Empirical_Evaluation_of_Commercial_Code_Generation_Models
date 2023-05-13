from .normalizer import RescaleNormalizer
import argparse
import torch

class Config:
    DEVICE = torch.device('cpu')
    NOISY_LAYER_STD = 0.1
    DEFAULT_REPLAY = 'replay'
    PRIORITIZED_REPLAY = 'prioritized_replay'
    EVAL_INTERVAL = 0
    EVAL_EPISODES = 10

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
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
        self.__eval_env = None
        self.num_workers = 1
        self.rollout_length = None
        self.optimization_epochs = 4
        self.mini_batch_size = 64
        self.iteration_log_interval = 30
        self.gradient_clip = None
        self.entropy_weight = 0
        self.value_loss_weight = 1.0
        self.termination_regularizer = 0
        self.sgd_update_frequency = None
        self.random_action_prob = None
        self.log_interval = int(1e3)
        self.save_interval = 0
        self.async_actor = True
        self.tasks = False
        self.replay_type = Config.DEFAULT_REPLAY
        self.decaying_lr = False
        self.shared_repr = False
        self.noisy_linear = False
        self.n_step = 1

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
            config_dict = vars(args)
        for key, value in config_dict.items():
            setattr(self, key, value)
        
    def is_prioritized_replay(self):
        """Returns true if this config uses prioritized replay."""
        return self.replay_type == Config.PRIORITIZED_REPLAY
        
    def is_vanilla(self):
        """Returns true if this config uses vanilla DDPG."""
        return not self.is_prioritized_replay() and not self.shared_repr and not self.noisy_linear
        
    def is_parallel(self):
        """Returns true if this config uses parallel actor-learners."""
        return self.num_workers > 1
        
    def is_evaluator(self):
        """Returns true if this config is an evaluator process."""
        return self.tasks is not False and self.tasks > 1
        
    def is_target_network_enabled(self):
        """Returns true if updating target network is enabled."""
        return self.target_network_update_freq is not None and self.target_network_update_freq > 0
        
    def get_evaluation_interval(self):
        """Returns the interval at which evaluation occurs."""
        return self.eval_interval or self.log_interval
        
    def use_decay_learning_rate(self):
        """Returns true if this config uses a decaying learning rate."""
        return self.decaying_lr
        
    def use_shared_representation(self):
        """Returns true if this config uses a shared state-action representation."""
        return self.shared_repr
        
    def use_noisy_linear(self):
        """Returns true if this config uses a noisy linear layer for exploration."""
        return self.noisy_linear
        
    def get_evaluation_episodes(self):
        """Returns the number of episodes used for evaluation."""
        return self.eval_episodes if self.eval_episodes > 0 else 1