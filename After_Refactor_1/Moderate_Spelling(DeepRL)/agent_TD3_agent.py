import torch
import torch.nn.functional as F
from numpy import clip, asarray
from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision


class TD3Agent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None

    def evaluate(self, state):
        """
        Evaluate the current policy given a state observation.

        Args:
            state (numpy.ndarray): The state observation.

        Returns:
            numpy.ndarray: The action to take in the current state.
        """
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return torch.Tensor.cpu(action).detach().numpy()

    def step(self):
        """
        Take a step in the environment.

        Returns:
            None
        """
        config = self.config
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = torch.Tensor.cpu(action).detach().numpy()
            action += self.random_process.sample()
        action = clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        self.replay.feed(dict(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=1-asarray(done, dtype=int),
        ))

        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.total_steps < config.warm_up:
            return

        transitions = self.replay.sample()
        states = torch.tensor(transitions.state)
        actions = torch.tensor(transitions.action)
        rewards = torch.tensor(transitions.reward).unsqueeze(-1)
        next_states = torch.tensor(transitions.next_state)
        mask = torch.tensor(transitions.mask).unsqueeze(-1)

        a_next = self.target_network(next_states)
        noise = torch.randn_like(a_next).mul(config.td3_noise)
        noise = noise.clamp(-config.td3_noise_clip, config.td3_noise_clip)

        min_a = float(self.task.action_space.low[0])
        max_a = float(self.task.action_space.high[0])
        a_next = (a_next + noise).clamp(min_a, max_a)

        q_1, q_2 = self.target_network.q(next_states, a_next)
        target = rewards + config.discount * mask * torch.min(q_1, q_2)
        target = target.detach()

        q_1, q_2 = self.network.q(states, actions)
        critic_loss = F.mse_loss(q_1, target) + F.mse_loss(q_2, target)

        self.network.zero_grad()
        critic_loss.backward()
        self.network.critic_opt.step()

        if self.total_steps % config.td3_delay == 0:
            action = self.network(states)
            policy_loss = -self.network.q(states, action)[0].mean()

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)

    def soft_update(self, target, src):
        """
        Perform a soft update of the target network weights.

        Args:
            target (nn.Module): The target network.
            src (nn.Module): The source network.

        Returns:
            None
        """
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)