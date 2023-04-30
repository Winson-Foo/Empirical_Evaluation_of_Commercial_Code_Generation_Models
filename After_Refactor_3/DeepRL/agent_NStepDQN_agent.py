from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *


class NStepDQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        self._collect_rollout()
        self._update_network()

    def _collect_rollout(self):
        config = self.config
        storage = Storage(config.rollout_length)

        states = self.states
        for _ in range(config.rollout_length):
            q_values = self._get_q_values(states)
            actions = self._select_actions(q_values)

            next_states, rewards, terminals, info = self.task.step(actions)
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)

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
        self.rollout = storage

    def _get_q_values(self, states):
        q = self.network(self.config.state_normalizer(states))['q']
        return q

    def _select_actions(self, q_values):
        epsilon = self.config.random_action_prob(self.config.num_workers)
        actions = epsilon_greedy(epsilon, to_np(q_values))
        return actions

    def _update_network(self):
        config = self.config
        rollout = self.rollout

        ret = self._get_target_values()
        for i in reversed(range(config.rollout_length)):
            ret = rollout.reward[i] + config.discount * rollout.mask[i] * ret
            rollout.ret[i] = ret

        entries = rollout.extract(['q', 'action', 'ret'])
        loss = self._calculate_loss(entries)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

    def _get_target_values(self):
        states = self.states
        ret = self.target_network(self.config.state_normalizer(states))['q'].detach()
        ret = torch.max(ret, dim=1, keepdim=True)[0]
        return ret

    def _calculate_loss(self, entries):
        q_pred = entries.q.gather(1, entries.action)
        q_target = entries.ret
        loss = 0.5 * (q_pred - q_target).pow(2).mean()
        return loss