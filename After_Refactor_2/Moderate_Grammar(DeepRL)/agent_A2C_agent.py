from ..network import *
from ..component import *
from .BaseAgent import *


class A2CAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(config.state_normalizer(states))
            next_states, rewards, terminals, info = self.task.step(prediction['action'].detach().cpu().numpy())
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            storage.feed(prediction)
            storage.feed({'reward': rewards.unsqueeze(-1),
                          'mask': (1 - terminals).unsqueeze(-1)})
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        self._update_network(storage)

    def _update_network(self, storage):
        config = self.config
        self._compute_returns(storage)

        entries = storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])
        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        entropy_loss = entries.entropy.mean()

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

    def _compute_returns(self, storage):
        config = self.config
        advantages = torch.zeros((config.num_workers, 1), device=self.device)
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


# Refactor the tensor storage and processing into a separate class   
class Storage:
    def __init__(self, num_steps):
        self.num_steps = num_steps
        self.reset()

    def reset(self):
        self.index = 0
        self.reward = torch.zeros((self.num_steps, self.num_workers, 1), device=self.device)
        self.mask = torch.zeros((self.num_steps, self.num_workers, 1), device=self.device)
        self.v = torch.zeros((self.num_steps + 1, self.num_workers, 1), device=self.device)
        self.log_pi_a = torch.zeros((self.num_steps, self.num_workers, 1), device=self.device)
        self.entropies = torch.zeros((self.num_steps, self.num_workers, 1), device=self.device)
        self.advantanges = torch.zeros((self.num_steps, self.num_workers, 1), device=self.device)
        self.returns = torch.zeros((self.num_steps, self.num_workers, 1), device=self.device)

    def feed(self, predictions):
        self.v[self.index + 1] = predictions['v']
        self.log_pi_a[self.index] = predictions['log_pi_a']
        self.entropies[self.index] = predictions['entropies']
        self.index += 1

    def feed(self, data):
        for k, v in data.items():
            setattr(self, k)[self.index] = v

    def extract(self, keys):
        return [getattr(self, k)[:self.index] for k in keys]

    def placeholder(self):
        self.v[self.index] = torch.zeros((self.num_workers, 1), device=self.device)

    def __getattr__(self, attr):
        return getattr(self.reward, attr)


# Define a separate function for the actor-critic algorithm logic
def update_a2c_network(config, storage, network, optimizer):
    advantages = torch.zeros((config.num_workers, 1), device=config.device)
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

    entries = storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])
    policy_loss = -(entries.log_pi_a * entries.advantage).mean()
    value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
    entropy_loss = entries.entropy.mean()

    optimizer.zero_grad()
    (policy_loss - config.entropy_weight * entropy_loss +
     config.value_loss_weight * value_loss).backward()
    nn.utils.clip_grad_norm_(network.parameters(), config.gradient_clip)
    optimizer.step()