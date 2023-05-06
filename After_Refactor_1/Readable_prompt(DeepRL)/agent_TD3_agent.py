# Refactored Code:

# First, I will import only the necessary libraries and modules.
from torch import tensor
import torch.nn.functional as F
import numpy as np

from rl.network import Network
from rl.component import PrioritizedReplayBuffer, Normalizer
from rl.agent.BaseAgent import BaseAgent


class TD3Agent(BaseAgent):
    def __init__(self, config):
        
        super().__init__(config)
        
        self.config = config
        self.task = config.task_fn()
        self.network = Network(config.state_dim, config.action_dim)
        self.target_network = Network(config.state_dim, config.action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay_buffer = PrioritizedReplayBuffer(config.replay_buffer_size, alpha=config.prioritized_replay_alpha)
        self.state_normalizer = Normalizer(config.state_dim)
        self.reward_normalizer = Normalizer(1)

        self.total_steps = 0
        self.state = None

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def eval_step(self, state):
        self.state_normalizer.set_read_only()
        state = self.state_normalizer(tensor(state))
        action = self.network(state)
        self.state_normalizer.unset_read_only()
        return action.detach().numpy()[0]

    def step(self):
        
        if self.state is None:
            self.state_normalizer.set_state(self.task.reset())
            self.state = self.state_normalizer.normalize(self.state_normalizer.state)
            self.total_steps = 0

            self.random_process.reset_states_np()
        
        if self.total_steps < self.config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(tensor(self.state))
            action = action.detach().numpy()[0]
            action += self.random_process.sample()
            
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        
        self.record_online_return(info)
        
        reward = self.reward_normalizer.normalize(reward)
        next_state = self.state_normalizer.normalize(next_state)
        
        self.replay_buffer.add(state=self.state, action=action, reward=[reward], next_state=next_state, done=[done])
        
        if done:
            self.state_normalizer.reset()
            self.random_process.reset_states_np()
            self.state = None
        else:
            self.state = next_state
        
        self.total_steps += 1
        
        if self.total_steps >= self.config.warm_up and self.total_steps % self.config.update_frequency == 0:
            
            for _ in range(self.config.num_batches):
                experience = self.replay_buffer.sample(self.config.batch_size, beta=self.beta_schedule.value(self.total_steps))
                states = tensor(experience['state'])
                actions = tensor(experience['action'])
                rewards = tensor(experience['reward'])
                next_states = tensor(experience['next_state'])
                masks = tensor(experience['done'])
                
                with torch.no_grad():
                    next_action = self.target_network(next_states)
                    
                    noise = torch.randn_like(next_action)
                    noise = noise * self.config.td3_noise
                    noise = noise.clamp(-self.config.td3_noise_clip, self.config.td3_noise_clip)
                    
                    next_action = (next_action + noise).clamp(self.task.action_space.low[0], self.task.action_space.high[0])
                    
                    q1, q2 = self.target_network.q(next_states, next_action)
                    target_q = rewards + (1 - masks) * (self.config.discount * torch.min(q1, q2))
                
                q1, q2 = self.network.q(states, actions)
                
                q1_loss = F.mse_loss(q1, target_q)
                q2_loss = F.mse_loss(q2, target_q)
                critic_loss = q1_loss + q2_loss
                
                self.network.critic_opt.zero_grad()
                critic_loss.backward()
                self.network.critic_opt.step()
                
                if self.total_steps % self.config.policy_delay == 0:
                    
                    sampled_actions = self.network(states)
                    
                    policy_loss = -self.network.q(states, sampled_actions)[0].mean()
                    
                    self.network.actor_opt.zero_grad()
                    policy_loss.backward()
                    self.network.actor_opt.step()
                    
                    self.soft_update(self.target_network, self.network)

        return {'state': self.state}  # this line is for better readability and understandability.