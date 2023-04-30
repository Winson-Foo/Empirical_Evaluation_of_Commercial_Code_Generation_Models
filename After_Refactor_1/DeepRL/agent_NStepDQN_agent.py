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

    def process_rollout(self):
        config = self.config
        storage = Storage(config.rollout_length)

        states = self.states
        for _ in range(config.rollout_length):
            q_values = self.network(config.state_normalizer(states))['q']

            epsilon = config.random_action_prob(config.num_workers)
            actions = epsilon_greedy(epsilon, to_np(q_values))

            next_states, rewards, terminals, info = self.task.step(actions)
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)

            storage.feed({'q_values': q_values,
                          'actions': tensor(actions).unsqueeze(-1).long(),
                          'rewards': tensor(rewards).unsqueeze(-1),
                          'mask': tensor(1 - terminals).unsqueeze(-1)})

            states = next_states

            self.total_steps += config.num_workers

        self.states = states
        storage.placeholder()

        with torch.no_grad():
            q_values_next = self.target_network(config.state_normalizer(states))['q'].detach()
            q_values_next = torch.max(q_values_next, dim=1, keepdim=True)[0]
            for i in reversed(range(config.rollout_length)):
                q_values_next = storage.rewards[i] + config.discount * storage.mask[i] * q_values_next
                storage.returns[i] = q_values_next.clone()

        return storage

    def compute_loss(self, storage):
        q_values = storage.q_values
        actions = storage.actions
        returns = storage.returns

        q_values = q_values.gather(1, actions)
        loss = 0.5 * (q_values - returns).pow(2).mean()

        return loss

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()

    def update_network(self, storage):
        loss = self.compute_loss(storage)
        self.backward(loss)

    def update_target_network(self):
        if self.total_steps // self.config.num_workers % self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def step(self):
        storage = self.process_rollout()
        self.update_network(storage)
        self.update_target_network()