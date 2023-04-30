from .normalizer import RescaleNormalizer
import argparse
import torch


class BaseConfig:
    DEVICE = torch.device('cpu')
    NOISY_LAYER_STD = 0.1
    DEFAULT_REPLAY = 'replay'
    PRIORITIZED_REPLAY = 'prioritized_replay'


class NetworkConfig:
    def __init__(self):
        self.network_fn = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.double_q = False
        self.shared_repr = False
        self.noisy_linear = False


class ReplayConfig:
    def __init__(self):
        self.replay_fn = None
        self.min_memory_size = None
        self.rollout_length = None
        self.n_step = 1
        self.replay_type = BaseConfig.DEFAULT_REPLAY


class OptimizerConfig:
    def __init__(self):
        self.optimizer_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.gradient_clip = None
        self.optimization_epochs = 4
        self.mini_batch_size = 64
        self.entropy_weight = 0
        self.value_loss_weight = 1.0
        self.termination_regularizer = 0
        self.sgd_update_frequency = None


class NormalizerConfig:
    def __init__(self):
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()


class Config(BaseConfig, NetworkConfig, ReplayConfig, OptimizerConfig, NormalizerConfig):
    def __init__(self):
        super().__init__()

        # General
        self.task_name = None
        self.num_workers = 1
        self.tag = 'vanilla'
        self.max_steps = 0
        self.async_actor = True
        self.tasks = False
        self.decaying_lr = False
        self.random_action_prob = None

        # Environment
        self.state_dim = None
        self.action_dim = None
        self.task_fn = None
        self.eval_episodes = 10
        self.eval_interval = 0
        self.eval_env = None

        # Hyperparameters
        self.discount = None
        self.target_network_update_freq = None
        self.exploration_steps = None
        self.history_length = None
        self.gae_tau = 1.0
        self.target_network_mix = 0.001
        self.categorical_v_min = None
        self.categorical_v_max = None
        self.categorical_n_atoms = 51
        self.num_quantiles = None
        self.iteration_log_interval = 30
        self.log_interval = int(1e3)
        self.save_interval = 0

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key, value in config_dict.items():
            setattr(self, key, value)


if __name__ == '__main__':
    config = Config()
    config.add_argument('--param1', type=int, default=10)
    config.add_argument('--param2', type=float, default=0.01)
    config.merge()

    # Main program logic
    # ...