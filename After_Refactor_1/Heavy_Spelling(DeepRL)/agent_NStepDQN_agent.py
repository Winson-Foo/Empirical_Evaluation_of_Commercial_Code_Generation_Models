import torch.optim as optim
import torch.nn.functional as F

from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *


class NStepDQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr)

        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)

        states = self.states
        for _ in range(config.rollout_length):
            self.network.train()
            q_values = self.network(self.config.state_normalizer(states))
            q_values = q_values['q']
            epsilon = config.random_action_prob(config.num_workers)
            actions = epsilon_greedy(epsilon, to_np(q_values))

            next_states, rewards, terminals, info = self.task.step(actions)
            rewards = config.reward_normalizer(rewards)
            self.record_online_return(info)

            storage.feed({'q': q_values,
                          'action': tensor(actions).unsqueeze(-1).long(),
                          'reward': tensor(rewards).unsqueeze(-1),
                          'mask': tensor(1 - terminals).unsqueeze(-1)})

            states = next_states
            self.total_steps += config.num_workers
            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        self.states = states

        storage.placeholder()

        with torch.no_grad():
            q_values = self.target_network(config.state_normalizer(states))['q']
            next_q_value = torch.max(q_values, dim=1, keepdim=True)[0]

        for i in reversed(range(config.rollout_length)):
            next_q_value = storage.reward[i] + config.gamma * storage.mask[i] * next_q_value
            storage.ret[i] = next_q_value

        entries = storage.extract(['q', 'action', 'ret'])
        q_values = entries.q
        actions = entries.action.squeeze(-1)
        target_values = entries.ret

        q_pred = q_values.gather(1, actions)
        loss = F.mse_loss(q_pred, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)