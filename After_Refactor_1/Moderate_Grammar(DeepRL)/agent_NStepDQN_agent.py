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
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        rollout_length = self.config.rollout_length
        config = self.config
        storage = self._collect_rollout()

        self._update_network(storage)

    def _collect_rollout(self):
        config = self.config
        storage = Storage(config.rollout_length)

        states = self.states
        for _ in range(config.rollout_length):
            q_values = self._get_q_values(states)
            actions = self._choose_actions(q_values)

            next_states, rewards, terminals, info = self.task.step(actions)
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)

            storage.feed({'q': q_values,
                          'action': tensor(actions).unsqueeze(-1).long(),
                          'reward': tensor(rewards).unsqueeze(-1),
                          'mask': tensor(1 - terminals).unsqueeze(-1)})

            states = next_states

            self.total_steps += config.num_workers

        self.states = states
        storage.placeholder()

        return storage

    def _get_q_values(self, states):
        return self.network(self.config.state_normalizer(states))['q']

    def _choose_actions(self, q_values):
        epsilon = self.config.random_action_prob(self.config.num_workers)
        return epsilon_greedy(epsilon, to_np(q_values))

    def _update_network(self, storage):
        config = self.config

        with torch.no_grad():
            target_q_values = self.target_network(config.state_normalizer(states))['q'].detach()
            target_q_values = torch.max(target_q_values, dim=1, keepdim=True)[0]

        for i in reversed(range(config.rollout_length)):
            returns = storage.reward[i] + config.discount * storage.mask[i] * target_q_values
            storage.ret[i] = returns

        entries = storage.extract(['q', 'action', 'ret'])
        loss = self._compute_loss(entries)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

    def _compute_loss(self, entries):
        return 0.5 * (entries.q.gather(1, entries.action) - entries.ret).pow(2).mean()