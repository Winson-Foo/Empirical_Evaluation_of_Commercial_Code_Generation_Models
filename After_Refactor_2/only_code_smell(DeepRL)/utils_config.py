from .normalizer import RescaleNormalizer
import argparse

class AgentConfig:
    # Environment
    task_fn = None
    __eval_env = None

    # Network
    network_fn = None
    actor_network_fn = None
    critic_network_fn = None
    shared_repr = False
    noisy_linear = False

    # Optimization
    optimizer_fn = None
    actor_optimizer_fn = None
    critic_optimizer_fn = None
    entropy_weight = 0
    gradient_clip = None
    double_q = False
    value_loss_weight = 1.0
    categorical_v_min = None
    categorical_v_max = None
    categorical_n_atoms = 51
    num_quantiles = None
    optimization_epochs = 4
    mini_batch_size = 64
    termination_regularizer = 0
    sgd_update_frequency = None

    # Replay
    replay_fn = None
    replay_type = 'replay'
    priortized_sampling = False
    n_step = 1
    max_steps = 0
    rollout_length = None
    min_memory_size = None

    # Exploration
    random_action_prob = None
    exploration_steps = None
    random_process_fn = None

    # Training
    use_gae = False
    gae_tau = 1.0
    target_network_mix = 0.001
    target_network_update_freq = None
    num_workers = 1
    iter_log_interval = 30
    decaying_lr = False

    # Normalization
    state_normalizer = RescaleNormalizer()
    reward_normalizer = RescaleNormalizer()

    # Other
    log_level = 0
    tag = 'vanilla'
    tasks = False
    async_actor = True
    eval_interval = 0
    eval_episodes = 10
    log_interval = int(1e3)
    save_interval = 0

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.discount = None

    # Properties
    @property
    def state_dim(self):
        return self.__eval_env.state_dim if self.__eval_env is not None else None

    @property
    def action_dim(self):
        return self.__eval_env.action_dim if self.__eval_env is not None else None

    @property
    def task_name(self):
        return self.__eval_env.name if self.__eval_env is not None else None

    # Methods
    def add_config_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self):
        args = self.parser.parse_args()
        for key, value in vars(args).items():
            setattr(self, key, value)