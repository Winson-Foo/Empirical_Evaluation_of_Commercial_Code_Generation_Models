## main.py

from config import Config
from network import Network
from memory import ReplayBuffer
from trainer import Trainer

config = Config()

# Set up environment
config.eval_env = MyEnv(...)
config.eval_env.seed(123)

# Set up network
config.network_fn = lambda: Network(config.state_dim, config.action_dim)
config.actor_network_fn = config.network_fn
config.critic_network_fn = config.network_fn

# Set up optimizer
config.optimizer_fn = lambda params: Adam(params, lr=5e-4)
config.actor_optimizer_fn = config.optimizer_fn
config.critic_optimizer_fn = config.optimizer_fn

# Set up replay buffer
config.replay_fn = lambda buffer_size: ReplayBuffer(buffer_size=buffer_size)

# Set up random process
config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(size=config.action_dim)

# Set up exploration steps
config.exploration_steps = 1000

# Set up target network parameter updates
config.target_network_update_freq = 500

# Run trainer
trainer = Trainer(config)
trainer.run()