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
        config = self.config
        storage = Storage(config.rollout_length)

        states = self.states
        for _ in range(config.rollout_length):
            q_values = self.network(self.config.state_normalizer(states))['q']
            actions = epsilon_greedy(config.random_action_prob(config.num_workers), to_np(q_values))

            next_states, rewards, terminals, info = self.task.step(actions)
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)

            storage.feed({
                'q': q_values,
                'action': tensor(actions).unsqueeze(-1).long(),
                'reward': tensor(rewards).unsqueeze(-1),
                'mask': tensor(1 - terminals).unsqueeze(-1)
            })

            states = next_states

            self.total_steps += config.num_workers
            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        self.states = states

        storage.placeholder()

        with torch.no_grad():
            q_values = self.target_network(config.state_normalizer(states))['q']
            max_q_values = torch.max(q_values, dim=1, keepdim=True)[0]

        for i in reversed(range(config.rollout_length)):
            max_q_values = storage.reward[i] + config.discount * storage.mask[i] * max_q_values
            storage.ret[i] = max_q_values

        q_values, actions, ret = storage.extract(['q', 'action', 'ret'])

        loss = (q_values.gather(1, actions) - ret).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()