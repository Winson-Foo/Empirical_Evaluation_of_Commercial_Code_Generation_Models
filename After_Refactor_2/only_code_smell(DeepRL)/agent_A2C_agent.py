from ..network import *
from ..component import *
from .BaseAgent import *

class Config:
    def __init__(self):
        self.rollout_length = 5
        self.num_workers = 1
        self.discount = 0.99
        self.gae_tau = 0.95
        self.use_gae = True
        self.entropy_weight = 0.01
        self.value_loss_weight = 0.5
        self.gradient_clip = 0.5
        self.task_fn = None
        self.network_fn = None
        self.optimizer_fn = None
        self.state_normalizer = None
        self.reward_normalizer = None

class A2CAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states

        self.collect_rollout(states, storage)

        prediction = self.network(config.state_normalizer(states))
        storage.feed(prediction)
        storage.placeholder()

        self.calculate_advantages(storage, config)

        entries = storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])

        policy_loss = self.calculate_policy_loss(entries)
        value_loss = self.calculate_value_loss(entries)
        entropy_loss = self.calculate_entropy_loss(entries)

        self.backpropagate(policy_loss, value_loss, entropy_loss)

    def collect_rollout(self, states, storage):
        config = self.config
        for _ in range(config.rollout_length):
            prediction = self.network(config.state_normalizer(states))
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            storage.feed(prediction)
            storage.feed({'reward': tensor(rewards).unsqueeze(-1),
                         'mask': tensor(1 - terminals).unsqueeze(-1)})

            states = next_states
            self.total_steps += config.num_workers

        self.states = states

    def calculate_advantages(self, storage, config):
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = storage.v[-1].detach()

        for i in reversed(range(config.rollout_length)):
            returns = storage.reward[i] + config.discount * storage.mask[i] * returns

            if config.use_gae:
                td_error = storage.reward[i] + config.discount * storage.mask[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.mask[i] + td_error
            else:
                advantages = returns - storage.v[i].detach()

            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

    def calculate_policy_loss(self, entries):
        return -(entries.log_pi_a * entries.advantage).mean()

    def calculate_value_loss(self, entries):
        return 0.5 * (entries.ret - entries.v).pow(2).mean()

    def calculate_entropy_loss(self, entries):
        return entries.entropy.mean()

    def backpropagate(self, policy_loss, value_loss, entropy_loss):
        self.optimizer.zero_grad()
        (policy_loss - self.config.entropy_weight * entropy_loss +
         self.config.value_loss_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()