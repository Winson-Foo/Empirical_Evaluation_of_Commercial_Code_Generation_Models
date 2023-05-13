from typing import List
from torch import Tensor, nn
from torch.optim import Optimizer
from .BaseAgent import BaseAgent
from ..component import Storage
from ..utils import epsilon_greedy, tensor, to_np

class NStepDQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        """
        Take a step in the environment and update the agent's neural networks.
        """
        config = self.config
        storage = Storage(config.rollout_length)

        states = self.states
        for _ in range(config.rollout_length):
            # Select action based on current state
            q_values = self.network(self.config.state_normalizer(states))['q']
            epsilon = config.random_action_prob(config.num_workers)
            action = epsilon_greedy(epsilon, to_np(q_values))
            
            # Interact with the environment
            next_states, rewards, terminals, info = self.task.step(action)
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)

            # Store the experience in the replay buffer
            storage.feed({'q': q_values,
                          'action': tensor(action).unsqueeze(-1).long(),
                          'reward': tensor(rewards).unsqueeze(-1),
                          'mask': tensor(1 - terminals).unsqueeze(-1)})
            states = next_states
            self.total_steps += config.num_workers

            # Update target network periodically
            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        self.states = states
        storage.placeholder()

        # Calculate the TD target for each experience in the replay buffer
        td_targets = self._calculate_td_targets(storage.ret, storage.reward,
                                                 storage.mask, self.target_network, config)
        entries = storage.extract(['q', 'action', 'ret'])
        
        # Calculate the loss
        loss = self._calculate_loss(entries.q, entries.action, td_targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()
    
    def _calculate_td_targets(self, ret: Tensor, reward: Tensor, mask: Tensor,
                               network: nn.Module, config) -> Tensor:
        """
        Calculate the TD target for each experience using the target neural network.
        """
        with torch.no_grad():
            td_targets = ret + config.discount * mask * network(config.state_normalizer(self.states))['q']
            td_targets = td_targets.detach()
            td_targets = torch.max(td_targets, dim=1, keepdim=True)[0]
        return td_targets
    
    def _calculate_loss(self, q_values: Tensor, action: Tensor, td_targets: Tensor) -> Tensor:
        """
        Calculate the loss between the predicted Q values and the TD targets.
        """
        q_values = q_values.gather(1, action)
        loss = 0.5 * (q_values - td_targets).pow(2).mean()
        return loss