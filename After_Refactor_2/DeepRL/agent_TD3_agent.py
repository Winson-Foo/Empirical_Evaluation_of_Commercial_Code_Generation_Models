from typing import Dict, Any, Union

import torch
import torch.nn.functional as F
import numpy as np

from ..network import QNetwork
from ..component import ReplayBuffer, OUProcess

class TD3Agent:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.task = config.task_fn()
        self.network = QNetwork(self.task.observation_space.shape[0], 
                                self.task.action_space.shape[0])
        self.target_network = QNetwork(self.task.observation_space.shape[0], 
                                       self.task.action_space.shape[0])
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay_buffer = ReplayBuffer(self.config.replay_size)
        self.random_process = OUProcess(self.task.action_space.shape[0])
        self.total_steps = 0
        self.current_state = None

    def _soft_update(self, target: torch.nn.Module, src: torch.nn.Module) -> None:
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.target_network_mix) +
                param.data * self.config.target_network_mix
            )

    def eval_step(self, state: np.ndarray) -> np.ndarray:
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return action.cpu().detach().numpy()

    def step(self) -> None:
        if self.current_state is None:
            self.random_process.reset_states()
            self.current_state = self.task.reset()
            self.current_state = self.config.state_normalizer(self.current_state)

        if self.total_steps < self.config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.current_state)
            action = action.cpu().detach().numpy()
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self._record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        self.replay_buffer.add_transition(
            state=self.current_state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=1-np.asarray(done, dtype=np.int32)
        )

        if done[0]:
            self.random_process.reset_states()
        self.current_state = next_state
        self.total_steps += 1

        if self.total_steps >= self.config.warm_up:
            transitions = self.replay_buffer.sample()
            states = torch.tensor(transitions.state, dtype=torch.float32)
            actions = torch.tensor(transitions.action, dtype=torch.float32)
            rewards = torch.tensor(transitions.reward, dtype=torch.float32).unsqueeze(-1)
            next_states = torch.tensor(transitions.next_state, dtype=torch.float32)
            mask = torch.tensor(transitions.mask, dtype=torch.float32).unsqueeze(-1)

            a_next = self.target_network(next_states)
            noise = torch.randn_like(a_next).mul(self.config.td3_noise)
            noise = noise.clamp(-self.config.td3_noise_clip, self.config.td3_noise_clip)

            min_a = float(self.task.action_space.low[0])
            max_a = float(self.task.action_space.high[0])
            a_next = (a_next + noise).clamp(min_a, max_a)

            q_1, q_2 = self.target_network.q(next_states, a_next)
            target = rewards + self.config.discount * mask * torch.min(q_1, q_2)
            target = target.detach()

            q_1, q_2 = self.network.q(states, actions)
            critic_loss = F.mse_loss(q_1, target) + F.mse_loss(q_2, target)

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            if self.total_steps % self.config.td3_delay:
                action = self.network(states)
                policy_loss = -self.network.q(states, action)[0].mean()

                self.network.zero_grad()
                policy_loss.backward()
                self.network.actor_opt.step()

                self._soft_update(self.target_network, self.network)

    def _record_online_return(self, info: Union[Dict[str, Any], None]) -> None:
        if info is not None and 'episode' in info:
            self.config.logger.add_scalar('online_return', info['episode']['return'])