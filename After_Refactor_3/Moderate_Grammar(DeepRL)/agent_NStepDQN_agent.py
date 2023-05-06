from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *

class NStepDQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.task = config.task_fn()
        self.network_manager = NetworkManager(config.network_fn, self.config.state_normalizer, config.optimizer_fn, config.gradient_clip)
        self.rollout_length = config.rollout_length
        self.discount = config.discount
        self.target_network_update_freq = config.target_network_update_freq
        self.random_action_prob = config.random_action_prob
        self.reward_normalizer = config.reward_normalizer
        self.states = self.task.reset()
        self.total_steps = 0

    def step(self):
        storage = RolloutStorage(self.rollout_length)

        self.compute_q_values(self.states, storage)

        actions = self.take_actions(storage.q_values)

        next_states, rewards, terminals, info = self.task.step(actions)
        self.record_online_return(info)
        rewards = self.reward_normalizer(rewards)

        self.process_rewards(storage, rewards, terminals, next_states)

        self.total_steps += self.config.num_workers
        if self.total_steps // self.config.num_workers % self.target_network_update_freq == 0:
            self.network_manager.update_target_network()

        self.states = next_states

        self.train_network(storage)

    def compute_q_values(self, states, storage):
        with torch.no_grad():
            q_values = self.network_manager.compute_q_values(states)
            storage.feed({'q_values': q_values})

    def take_actions(self, q_values):
        epsilon = self.random_action_prob(self.config.num_workers)
        actions = epsilon_greedy(epsilon, to_np(q_values))
        return actions

    def process_rewards(self, storage, rewards, terminals, next_states):
        storage.feed({'actions': tensor(actions).unsqueeze(-1).long(),
                      'rewards': tensor(self.reward_normalizer(rewards)).unsqueeze(-1),
                      'mask': tensor(1 - terminals).unsqueeze(-1)})
        storage.placeholder()

        ret = self.network_manager.compute_q_values(next_states)
        ret = torch.max(ret, dim=self.config.action_dim, keepdim=True)[0]
        for i in reversed(range(self.rollout_length)):
            ret = storage.rewards[i] + self.discount * storage.mask[i] * ret
            storage.returns[i] = ret

    def train_network(self, storage):
        self.network_manager.train(storage.q_values, storage.actions, storage.returns)

class NetworkManager():
    def __init__(self, network_fn, state_normalizer, optimizer_fn, gradient_clip):
        self.network = network_fn()
        self.target_network = network_fn()
        self.optimizer = optimizer_fn(self.network.parameters())
        self.gradient_clip = gradient_clip
        self.state_normalizer = state_normalizer

    def compute_q_values(self, states):
        return self.network(self.state_normalizer(states))['q']

    def train(self, q_values, actions, returns):
        loss = self.compute_loss(q_values, actions, returns)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.optimizer.step()

    def compute_loss(self, q_values, actions, returns):
        q_values_for_actions = q_values.gather(1, actions)
        loss = 0.5 * (q_values_for_actions - returns).pow(2).mean()
        return loss

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())