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

    def train(self):
        rollout = self._rollout()
        loss = self._compute_loss(rollout)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()

    def _rollout(self):
        rollout = Storage(self.config.rollout_length)
        states = self.states
        for _ in range(self.config.rollout_length):
            q = self.network(self.config.state_normalizer(states))['q']

            epsilon = self.config.random_action_prob(self.config.num_workers)
            actions = epsilon_greedy(epsilon, to_np(q))

            next_states, rewards, terminals, info = self.task.step(actions)
            self.record_online_return(info)
            rewards = self.config.reward_normalizer(rewards)

            rollout.feed({
                'q': q,
                'action': tensor(actions).unsqueeze(-1).long(),
                'reward': tensor(rewards).unsqueeze(-1),
                'mask': tensor(1 - terminals).unsqueeze(-1)
            })

            states = next_states
            self.total_steps += self.config.num_workers

        self.states = states
        rollout.placeholder()
        return rollout

    def _compute_loss(self, rollout):
        entries = rollout.extract(['q', 'action', 'ret'])
        ret = self._compute_return(entries, rollout)
        loss = 0.5 * (entries.q.gather(1, entries.action) - ret).pow(2).mean()
        return loss

    def _compute_return(self, entries, rollout):
        ret = self.target_network(self.config.state_normalizer(self.states))['q'].detach()
        ret = torch.max(ret, dim=1, keepdim=True)[0]
        for i in reversed(range(self.config.rollout_length)):
            ret = rollout.reward[i] + self.config.discount * rollout.mask[i] * ret
            rollout.ret[i] = ret
        return ret