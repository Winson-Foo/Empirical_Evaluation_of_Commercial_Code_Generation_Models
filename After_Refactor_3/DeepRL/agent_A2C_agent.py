from ..network import *
from ..component import *
from .BaseAgent import *


class A2CAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        """
        Take a step in the environment using the A2C algorithm.
        """
        config = self.config
        storage = self._initialize_storage()
        self.states = self.task.reset()

        # Collect experiences
        for _ in range(config.rollout_length):
            prediction = self.network(config.state_normalizer(self.states))
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
            self._record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            storage.feed(prediction)
            storage.feed({'reward': tensor(rewards).unsqueeze(-1),
                           'mask': tensor(1 - terminals).unsqueeze(-1)})
            self.states = next_states
            self.total_steps += config.num_workers

        # Compute returns and advantages
        self._compute_returns_and_advantages(storage, config)

        # Compute losses and optimize the network
        self._compute_and_optimize_losses(storage, config)

    def _initialize_storage(self):
        """
        Initialize the storage buffer for collecting experiences.
        """
        return Storage(self.config.rollout_length)

    def _record_online_return(self, info):
        """
        Record the online return for logging and monitoring.
        """
        self.record_online_return(info)

    def _compute_returns_and_advantages(self, storage, config):
        """
        Compute returns and advantages for the collected experiences.
        """
        prediction = self.network(config.state_normalizer(self.states))
        storage.feed(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()

        for i in reversed(range(config.rollout_length)):
            returns = storage.reward[i] + config.discount * storage.mask[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.reward[i] + config.discount * storage.mask[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.mask[i] + td_error
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

    def _compute_and_optimize_losses(self, storage, config):
        """
        Compute losses and optimize the network.
        """
        entries = storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])

        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        entropy_loss = entries.entropy.mean()

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()