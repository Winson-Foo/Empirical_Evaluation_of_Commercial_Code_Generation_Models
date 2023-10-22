import torch.nn.utils as nn_utils

from ..network import Network
from ..component import Storage
from ..utils import epsilon_greedy
from .BaseAgent import BaseAgent


class NStepDQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)

        states = self.states
        for _ in range(config.rollout_length):
            q_values = self.network(self.config.state_normalizer(states))['q']

            epsilon = config.random_action_prob(config.num_workers)
            actions = epsilon_greedy(epsilon, to_np(q_values))

            next_states, rewards, terminals, info = self.task.step(actions)
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)

            storage.feed({'q_values': q_values,
                          'actions': tensor(actions).unsqueeze(-1).long(),
                          'rewards': tensor(rewards).unsqueeze(-1),
                          'masks': tensor(1 - terminals).unsqueeze(-1)})

            states = next_states

            self.total_steps += config.num_workers
            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        self.states = states

        storage.placeholder()
        ret = self.calculate_returns(storage, config, states)
        loss = self.calculate_loss(storage)

        self.optimizer.zero_grad()
        loss.backward()
        nn_utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

    def calculate_returns(self, storage, config, states):
        with torch.no_grad():
            ret = self.target_network(config.state_normalizer(states))['q'].detach()
            ret = torch.max(ret, dim=1, keepdim=True)[0]
            for i in reversed(range(config.rollout_length)):
                ret = storage.rewards[i] + config.discount * storage.masks[i] * ret
                storage.returns[i] = ret
        return ret

    def calculate_loss(self, storage):
        entries = storage.extract(['q_values', 'actions', 'returns'])
        loss = 0.5 * (entries.q_values.gather(1, entries.actions) - entries.returns).pow(2).mean()
        return loss