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
        rollout = self._collect_rollout()
        loss = self._compute_loss(rollout)
        self._update_network(loss)

    def _collect_rollout(self):
        config = self.config
        rollout = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            q = self.network(self.config.state_normalizer(states))['q']
            epsilon = config.random_action_prob(config.num_workers)
            actions = epsilon_greedy(epsilon, to_np(q))
            next_states, rewards, terminals, info = self.task.step(actions)
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            rollout.feed({
                'q': q,
                'action': tensor(actions).unsqueeze(-1).long(),
                'reward': tensor(rewards).unsqueeze(-1),
                'mask': tensor(1 - terminals).unsqueeze(-1),
            })
            states = next_states
            self.total_steps += config.num_workers
            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
        self.states = states
        rollout.placeholder()
        return rollout

    def _compute_loss(self, rollout):
        config = self.config
        ret = self.target_network(config.state_normalizer(self.states))['q'].detach()
        ret = torch.max(ret, dim=1, keepdim=True)[0]
        for i in reversed(range(config.rollout_length)):
            ret = rollout.reward[i] + config.discount * rollout.mask[i] * ret
            rollout.ret[i] = ret
        entries = rollout.extract(['q', 'action', 'ret'])
        loss = 0.5 * (entries.q.gather(1, entries.action) - entries.ret).pow(2).mean()
        return loss

    def _update_network(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()