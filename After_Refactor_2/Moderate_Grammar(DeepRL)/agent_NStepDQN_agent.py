from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *


class NStepDQNAgent(BaseAgent):
    def __init__(self, config):
        """
        :param config: object that can be used to configure the
        agent. It should contain the following attrs
        config.task_fn: function that creates the task
        config.network_fn: function that returns the network
        config.optimizer_fn: function that returns the optimizer
        config.state_normalizer: function that normalizes states
        config.gradient_clip: value to clip the gradients
        config.rollout_length: length of rollout
        config.discount: discount factor gamma
        config.target_network_update_freq: frequency of target network update
        config.random_action_prob: probability of choosing a random action
        config.num_workers: number of parallel workers
        """
        super().__init__(config)
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        # Initialize variables
        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        """
        Implements the N-step DQN training step
        """
        config = self.config
        storage = Storage(config.rollout_length)

        states = self.states
        for _ in range(config.rollout_length):
            # Get q values for current state from the network
            q_values = self.network(config.state_normalizer(states))['q']

            # Choose actions using epsilon-greedy
            epsilon = config.random_action_prob(config.num_workers)
            actions = epsilon_greedy(epsilon, to_np(q_values))

            # Take a step in the environment
            next_states, rewards, terminals, info = self.task.step(actions)

            # Record online return for each worker
            self.record_online_return(info)

            # Normalize rewards
            rewards = config.reward_normalizer(rewards)

            # Store the experiences in Storage
            storage.feed({'q': q_values,
                          'action': tensor(actions).unsqueeze(-1).long(),
                          'reward': tensor(rewards).unsqueeze(-1),
                          'mask': tensor(1 - terminals).unsqueeze(-1)})

            states = next_states

            # Update total steps
            self.total_steps += config.num_workers

            # Update target network periodically
            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        self.states = states

        # Prepare placeholders in Storage for discounted returns
        storage.placeholder()

        # Calculate the target q values for next states using the target network
        target_q_values = self.target_network(config.state_normalizer(states))['q'].detach()
        target_q_values = torch.max(target_q_values, dim=1, keepdim=True)[0]

        # Calculate the discounted returns for each experience and store in Storage
        for i in reversed(range(config.rollout_length)):
            target_q_values = storage.reward[i] + config.discount * storage.mask[i] * target_q_values
            storage.ret[i] = target_q_values

        # Calculate the loss and update the network using backpropagation
        entries = storage.extract(['q', 'action', 'ret'])
        loss = 0.5 * (entries.q.gather(1, entries.action) - entries.ret).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()