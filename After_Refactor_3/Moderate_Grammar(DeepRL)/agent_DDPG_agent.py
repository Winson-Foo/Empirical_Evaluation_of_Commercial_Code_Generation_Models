from typing import Dict, Any
import numpy as np
import torch
import torch.optim as optim
import torchvision
from .BaseAgent import BaseAgent
from ..network import Network
from ..component import ReplayBuffer, RandomProcess

class DDPGAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        
        # Initialize the environment, network, and replay buffer
        self.task = config.task_fn()
        self.network = Network(config.state_dim, config.action_dim, config.hidden_dim)
        self.target_network = Network(config.state_dim, config.action_dim, config.hidden_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = ReplayBuffer(config.replay_size)
        self.random_process = RandomProcess(config.random_process_sigma, 
                                            config.random_process_theta, 
                                            config.action_dim)
        self.total_steps = 0
        self.state = None

    def soft_update(self, target, src):
        """Updates the weights of the target network using a soft update."""
        target_params = target.parameters()
        src_params = src.parameters()
        for target_param, param in zip(target_params, src_params):
            target_param.data.copy_(target_param.data * (1.0 - self.config.target_network_mix) +
                               param.data * self.config.target_network_mix)

    def eval_step(self, state):
        """Returns an action for a given state during evaluation."""
        state = self.config.state_normalizer(state)
        action = self.network(state)
        return to_np(action)

    def step(self):
        """Runs one step of the agent."""
        config = self.config
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        self.replay.feed(dict(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=1-np.asarray(done, dtype=np.int32),
        ))

        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if len(self.replay) >= config.batch_size:
            transitions = self.replay.sample(config.batch_size)
            states = torch.tensor(transitions["state"], dtype=torch.float32)
            actions = torch.tensor(transitions["action"], dtype=torch.float32)
            rewards = torch.tensor(transitions["reward"], dtype=torch.float32).unsqueeze(-1)
            next_states = torch.tensor(transitions["next_state"], dtype=torch.float32)
            mask = torch.tensor(transitions["mask"], dtype=torch.float32).unsqueeze(-1)

            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next)
            q_next = self.target_network.critic(phi_next, a_next)
            q_next = config.discount * mask * q_next
            q_next.add_(rewards)
            q_next = q_next.detach()
            phi = self.network.feature(states)
            q = self.network.critic(phi, actions)
            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            phi = self.network.feature(states)
            action = self.network.actor(phi)
            policy_loss = -self.network.critic(phi.detach(), action).mean()

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)