from .normalizer import RescaleNormalizer
import argparse
import torch

# Constants
NOISY_LAYER_STD = 0.1
DEFAULT_REPLAY = 'replay'
PRIORITIZED_REPLAY = 'prioritized_replay'

# Configuration options
class Config:
    def __init__(self):
        # General options
        self.task_fn = None
        self.task_name = None
        self.eval_env = None
        self.num_workers = 1
        self.tag = 'vanilla'
        self.gradient_clip = None
        self.use_gae = False
        self.gae_tau = 1.0
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
        self.min_memory_size = None
        self.max_steps = 0
        self.rollout_length = None
        self.iteration_log_interval = 30
        self.num_quantiles = None
        self.optimization_epochs = 4
        self.mini_batch_size = 64
        self.termination_regularizer = 0
        self.sgd_update_frequency = None
        self.random_action_prob = None
        self.log_interval = int(1e3)
        self.save_interval = 0
        self.eval_interval = 0
        self.eval_episodes = 10
        self.tasks = False
        self.replay_type = DEFAULT_REPLAY
        self.decaying_lr = False
        self.shared_repr = False
        self.noisy_linear = False
        self.n_step = 1

        # Actor options
        self.actor_optimizer_fn = None
        self.actor_network_fn = None

        # Critic options
        self.optimizer_fn = None
        self.critic_optimizer_fn = None
        self.network_fn = None
        self.critic_network_fn = None
        self.double_q = False
        self.value_loss_weight = 1.0
        self.categorical_v_min = None
        self.categorical_v_max = None
        self.categorical_n_atoms = 51

        # Replay options
        self.replay_fn = None
        self.default_replay = DEFAULT_REPLAY
        self.prioritized_replay = PRIORITIZED_REPLAY

        # Exploration options
        self.random_process_fn = None
        self.exploration_steps = None

        # Discount and target network options
        self.discount = None
        self.target_network_update_freq = None
        self.target_network_mix = 0.001

        # Logging and debugging options
        self.log_level = 0
        self.history_length = None

        # Device options
        self.device = torch.device('cpu')

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])