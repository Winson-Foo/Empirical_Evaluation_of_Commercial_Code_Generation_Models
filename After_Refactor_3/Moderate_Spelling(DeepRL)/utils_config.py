import argparse
import torch

from .normalizer import RescaleNormalizer


class Config:
    def __init__(self):
        self.device = torch.device('cpu')
        self.noisy_layer_std = 0.1
        self.replay_type = 'replay'
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
        self.eval_env = None
        self.log_interval = int(1e3)
        self.save_interval = 0
        self.eval_interval = 0
        self.eval_episodes = 10
        self.async_actor = True
        self.tasks = False
        self.decaying_lr = False
        self.shared_repr = False
        self.noisy_linear = False
        self.n_step = 1

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

        self.parser = argparse.ArgumentParser()
        self.add_arguments()

    def add_arguments(self):
        self.parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
        self.parser.add_argument('--noisy_layer_std', type=float, default=0.1, help='Standard deviation of noise for NoisyNet')
        self.parser.add_argument('--replay_type', type=str, default='replay', help='Replay buffer type (replay or prioritized_replay)')
        self.parser.add_argument('--history_length', type=int, default=None, help='Number of previous states to store')
        self.parser.add_argument('--double_q', action='store_true', help='Use Double Q-learning')
        self.parser.add_argument('--tag', type=str, default='vanilla', help='Tag for logging and saving')
        self.parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers for async algorithms')
        self.parser.add_argument('--gradient_clip', type=float, default=None, help='Max norm of gradients')
        self.parser.add_argument('--entropy_weight', type=float, default=0, help='Weight of entropy regularization term')
        self.parser.add_argument('--use_gae', action='store_true', help='Use Generalized Advantage Estimation')
        self.parser.add_argument('--gae_tau', type=float, default=1.0, help='Tau parameter for GAE')
        self.parser.add_argument('--target_network_mix', type=float, default=0.001, help='Mixing factor for target network updates')
        self.parser.add_argument('--categorical_v_min', type=float, default=None, help='Minimum value of categorical distribution support')
        self.parser.add_argument('--categorical_v_max', type=float, default=None, help='Maximum value of categorical distribution support')
        self.parser.add_argument('--categorical_n_atoms', type=int, default=51, help='Number of atoms for categorical distribution')
        self.parser.add_argument('--optimization_epochs', type=int, default=4, help='Number of epochs for optimization per batch')
        self.parser.add_argument('--mini_batch_size', type=int, default=64, help='Size of mini-batch for optimization')
        self.parser.add_argument('--termination_regularizer', type=float, default=0, help='Weight of termination regularization term')
        self.parser.add_argument('--sgd_update_frequency', type=int, default=None, help='Interval of SGD update for async algorithms')
        self.parser.add_argument('--random_action_prob', type=float, default=None, help='Probability of random action in epsilon-greedy policy')

    def parse_arguments(self, args=None):
        if args is None:
            args = self.parser.parse_args()
        config_dict = vars(args)
        for key, value in config_dict.items():
            setattr(self, key, value)