from ..network import DQN
from ..component import Storage
from ..utils import *


class NStepDQNAgent:
    def __init__(self, config):
        self.config = config
        self.task = config.task_fn()
        self.network = DQN()
        self.target_network = DQN()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)

        states = self.states
        for _ in range(config.rollout_length):
            # Get Q values from the network and choose actions using epsilon-greedy exploration
            q_values = self.network(self.config.state_normalizer(states))['q']
            epsilon = config.random_action_prob(config.num_workers)
            actions = epsilon_greedy(epsilon, to_np(q_values))

            # Execute selected actions and preprocess the rewards
            next_states, rewards, terminals, info = self.task.step(actions)
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)

            # Store experience tuples in the storage
            storage.feed({
                'q_values': q_values,
                'actions': tensor(actions).unsqueeze(-1).long(),
                'rewards': tensor(rewards).unsqueeze(-1),
                'masks': tensor(1 - terminals).unsqueeze(-1)
            })

            states = next_states

            # Update the number of steps taken across all parallel environments
            self.total_steps += config.num_workers
            
            # Update the target network every N steps
            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        self.states = states

        # Use Bellman equation to calculate discounted returns for each experience tuple
        storage.placeholder()
        returns = self.target_network(config.state_normalizer(states))['q'].detach()
        returns = torch.max(returns, dim=1, keepdim=True)[0]
        for i in reversed(range(config.rollout_length)):
            returns = storage.rewards[i] + config.discount * storage.masks[i] * returns
            storage.returns[i] = returns

        # Calculate the loss and update the network weights
        entries = storage.extract(['q_values', 'actions', 'returns'])
        loss = 0.5 * (entries.q_values.gather(1, entries.actions) - entries.returns).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()