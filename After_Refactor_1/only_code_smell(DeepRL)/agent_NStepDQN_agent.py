from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *


# Constants
ZERO = 0
ONE = 1
NEG_ONE = -1


class NStepDQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = ZERO
        self.states = self.task.reset()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)

        states = self.states
        for _ in range(config.rollout_length):
            q = self.network(config.state_normalizer(states))['q']

            epsilon = config.random_action_prob(config.num_workers)
            actions = epsilon_greedy(epsilon, to_np(q))

            next_states, rewards, terminals, info = self.task.step(actions)
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)

            storage.feed({
                'q': q,
                'action': actions.unsqueeze(-1).long(),
                'reward': rewards.unsqueeze(-1),
                'mask': (ONE - terminals).unsqueeze(-1)
            })

            states = next_states

            self.total_steps += config.num_workers
            if self.total_steps // config.num_workers % config.target_network_update_freq == ZERO:
                self.target_network.load_state_dict(self.network.state_dict())

        self.states = states

        storage.placeholder()

        ret = self.target_network(config.state_normalizer(states))['q'].detach()
        ret = torch.max(ret, dim=1, keepdim=True)[ZERO]
        for i in reversed(range(config.rollout_length)):
            ret = storage.reward[i] + config.discount * storage.mask[i] * ret
            storage.ret[i] = ret

        entries = storage.extract(['q', 'action', 'ret'])
        loss = compute_loss(entries.q, entries.action, entries.ret)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

    def process_rollouts(self, storage, next_states):
        ret = self.target_network(self.config.state_normalizer(next_states))['q'].detach()
        ret = torch.max(ret, dim=1, keepdim=True)[ZERO]
        for i in reversed(range(self.config.rollout_length)):
            ret = storage.reward[i] + self.config.discount * storage.mask[i] * ret
            storage.ret[i] = ret

    def compute_loss(self, q, action, ret):
        return 0.5 * (q.gather(1, action) - ret).pow(2).mean()