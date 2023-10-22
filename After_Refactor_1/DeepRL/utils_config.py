import argparse
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

from .normalizer import RescaleNormalizer


LOGGER = logging.getLogger(__name__)
DEVICE = torch.device('cpu')
NOISY_LAYER_STD = 0.1


class ReplayType:
    DEFAULT = 'replay'
    PRIORITIZED = 'prioritized_replay'


class NetworkType:
    ACTOR = 'actor'
    CRITIC = 'critic'


@dataclass
class EnvironmentConfig:
    state_dim: int
    action_dim: int
    name: str


@dataclass
class NetworkConfig:
    type: str
    network_fn: Optional[str]
    optimizer_fn: Optional[str]
    layer_sizes: List[int]
    activation: Optional[str] = 'relu'
    learning_rate: Optional[float] = 0.001
    gradient_clip: Optional[float] = None


@dataclass
class ReplayConfig:
    replay_fn: Optional[str]
    random_process_fn: Optional[str]
    discount: Optional[float] = 0.99
    batch_size: Optional[int] = 64
    max_memory_size: Optional[int] = int(1e6)
    history_length: Optional[int] = 1
    priority_alpha: Optional[float] = 0.6
    priority_beta: Optional[float] = 0.4
    priority_epsilon: Optional[float] = 1e-6
    min_memory_size: Optional[int] = None
    rollout_length: Optional[int] = None
    n_step: Optional[int] = 1


@dataclass
class LearningConfig:
    double_q: Optional[bool] = False
    entropy_weight: Optional[float] = 0.0
    use_gae: Optional[bool] = False
    gae_tau: Optional[float] = 1.0
    target_network_update_freq: Optional[int] = 1000
    target_network_mix: Optional[float] = 0.001
    sgd_update_frequency: Optional[int] = None
    value_loss_weight: Optional[float] = 1.0
    termination_regularizer: Optional[float] = 0.0


@dataclass
class TrainingConfig:
    optimization_epochs: int
    mini_batch_size: int


@dataclass
class EvaluationConfig:
    episodes: int
    interval: Optional[int] = 0
    eval_env: Optional[EnvironmentConfig] = None


@dataclass
class AgentConfig:
    network_configs: List[NetworkConfig]
    replay_config: ReplayConfig
    learning_config: LearningConfig
    training_config: TrainingConfig
    evaluation_config: EvaluationConfig
    tag: Optional[str] = 'vanilla'
    num_workers: Optional[int] = 1
    categorical_v_min: Optional[float] = None
    categorical_v_max: Optional[float] = None
    categorical_n_atoms: Optional[int] = 51
    num_quantiles: Optional[int] = None
    random_action_prob: Optional[float] = None
    shared_repr: Optional[bool] = False
    noisy_linear: Optional[bool] = False
    decaying_lr: Optional[bool] = False
    async_actor: Optional[bool] = True
    expert_task: Optional[Tuple[EnvironmentConfig, AgentConfig]] = None


class ConfigParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse(self) -> Tuple[AgentConfig, Optional[argparse.Namespace]]:
        args, remaining_args = self.parser.parse_known_args()
        config_dict = vars(args)
        config = AgentConfig(**config_dict)

        # Instantiate environment
        eval_env = EnvironmentConfig(args.state_dim, args.action_dim, args.eval_env_name)
        config.evaluation_config.eval_env = eval_env
        
        # Instantiate network configs
        network_configs = []
        network_keys = [key for key in config_dict.keys() if key.startswith('network_')]
        for network_key in network_keys:
            _, network_type, network_number, network_param = network_key.split('_')
            network_dict = {
                'type': network_type,
                'network_fn': getattr(args, f'network_{network_type}_{network_number}_fn', None),
                'optimizer_fn': getattr(args, f'{network_param}_{network_type}_{network_number}_optimizer_fn', None),
                'layer_sizes': getattr(args, f'{network_type}_{network_number}_layer_sizes', []),
                'activation': getattr(args, f'{network_type}_{network_number}_activation', 'relu'),
                'learning_rate': getattr(args, f'{network_param}_{network_type}_{network_number}_learning_rate', 0.001),
                'gradient_clip': getattr(args, f'{network_param}_{network_type}_{network_number}_gradient_clip', None),
            }
            network_configs.append(NetworkConfig(**network_dict))
        config.network_configs = network_configs

        # Instantiate replay config
        replay_fn = getattr(args, 'replay_fn', None)
        random_process_fn = getattr(args, 'random_process_fn', None)
        replay_dict = {
            'replay_fn': replay_fn or ReplayType.DEFAULT,
            'random_process_fn': random_process_fn,
            'discount': args.discount,
            'batch_size': args.batch_size,
            'max_memory_size': args.max_memory_size,
            'history_length': args.history_length,
            'priority_alpha': args.priority_alpha,
            'priority_beta': args.priority_beta,
            'priority_epsilon': args.priority_epsilon,
            'min_memory_size': getattr(args, 'min_memory_size', None),
            'rollout_length': getattr(args, 'rollout_length', None),
            'n_step': args.n_step,
        }
        config.replay_config = ReplayConfig(**replay_dict)

        # Instantiate learning config
        learning_dict = {
            'double_q': args.double_q,
            'entropy_weight': args.entropy_weight,
            'use_gae': args.use_gae,
            'gae_tau': args.gae_tau,
            'target_network_update_freq': args.target_network_update_freq,
            'target_network_mix': args.target_network_mix,
            'sgd_update_frequency': getattr(args, 'sgd_update_frequency', None),
            'value_loss_weight': args.value_loss_weight,
            'termination_regularizer': args.termination_regularizer,
        }
        config.learning_config = LearningConfig(**learning_dict)

        # Instantiate training config
        training_dict = {
            'optimization_epochs': args.optimization_epochs,
            'mini_batch_size': args.mini_batch_size,
        }
        config.training_config = TrainingConfig(**training_dict)

        # Instantiate evaluation config
        evaluation_dict = {
            'episodes': args.eval_episodes,
            'interval': args.eval_interval,
        }
        config.evaluation_config = EvaluationConfig(**evaluation_dict)

        return config, remaining_args


if __name__ == '__main__':
    config_parser = ConfigParser()
    # Define command-line arguments here
    config_parser.add_argument('--state-dim', type=int, default=4)
    config_parser.add_argument('--action-dim', type=int, default=2)
    config_parser.add_argument('--eval-env-name', type=str, default='CartPole-v0')
    config_parser.add_argument('--network-actor-1-layer-sizes', nargs='+', type=int, default=[128, 64])
    config_parser.add_argument('--network-critic-1-layer-sizes', nargs='+', type=int, default=[128, 64])
    config_parser.add_argument('--replay-fn', type=str, default='prioritized_replay', choices=[ReplayType.DEFAULT, ReplayType.PRIORITIZED])
    config_parser.add_argument('--random-process-fn', type=str, default='ou')
    config_parser.add_argument('--discount', type=float, default=0.99)
    config_parser.add_argument('--batch-size', type=int, default=64)
    config_parser.add_argument('--max-memory-size', type=int, default=int(1e6))
    config_parser.add_argument('--history-length', type=int, default=1)
    config_parser.add_argument('--learning-rate-actor-1', type=float, default=0.001)
    config_parser.add_argument('--learning-rate-critic-1', type=float, default=0.001)
    config_parser.add_argument('--double-q', action='store_true')
    config_parser.add_argument('--entropy-weight', type=float, default=0.0)
    config_parser.add_argument('--use-gae', action='store_true')
    config_parser.add_argument('--gae-tau', type=float, default=1.0)
    config_parser.add_argument('--target-network-update-freq', type=int, default=1000)
    config_parser.add_argument('--target-network-mix', type=float, default=0.001)
    config_parser.add_argument('--value-loss-weight', type=float, default=1.0)
    config_parser.add_argument('--optimization-epochs', type=int, default=4)
    config_parser.add_argument('--mini-batch-size', type=int, default=64)
    config_parser.add_argument('--eval-episodes', type=int, default=10)
    config_parser.add_argument('--eval-interval', type=int, default=0)

    config, remaining_args = config_parser.parse()

    LOGGER.debug(config)
    LOGGER.debug(remaining_args)