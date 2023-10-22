from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from rl.network import ActorCriticNet
from rl.replay import ReplayBuffer
from rl.utils import soft_update, to_device, to_numpy


class DDPGAgent:
    def __init__(self, task_fn, state_normalizer, replay_fn, random_process_fn,
                 discount=0.99, actor_lr=1e-4, critic_lr=1e-3, target_network_mix=0.001,
                 buffer_size=10000, batch_size=64, warm_up=1000, max_episode_length=1000,
                 target_score=None):

        self.task_fn = task_fn
        self.state_normalizer = state_normalizer
        self.replay_fn = replay_fn
        self.random_process_fn = random_process_fn

        self.discount = discount
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.target_network_mix = target_network_mix
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.warm_up = warm_up
        self.max_episode_length = max_episode_length
        self.target_score = target_score

        self.task = None
        self.network = None
        self.target_network = None
        self.replay_buffer = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.random_process = None
        self.total_steps = 0
        self.done = False
        self.info = None
        self.rewards = []
        self.scores = []
        self.state = None

    def eval_step(self, state):
        state = self.state_normalizer(state)
        action = self.network.actor(to_device(state))
        return to_numpy(action)

    def train(self):
        self.task = self.task_fn()
        self.state_normalizer.set_read_only()
        self.network = ActorCriticNet(self.task.state_space.shape[0], self.task.action_space.shape[0]).to_device()

        self.target_network = ActorCriticNet(self.task.state_space.shape[0], self.task.action_space.shape[0]).to_device()
        self.target_network.load_state_dict(self.network.state_dict())

        self.replay_buffer = self.replay_fn(self.buffer_size)

        self.actor_optimizer = torch.optim.Adam(self.network.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.network.critic.parameters(), lr=self.critic_lr)

        self.random_process = self.random_process_fn()

        while self.total_steps < self.warm_up or not self.done:
            if self.state is None:
                self.reset()

            action = self.get_action()
            next_state, reward, self.done, self.info = self.task.step(action)
            next_state = self.state_normalizer(next_state)
            reward = self.reward_normalizer(reward)
            self.record_transition(next_state, reward, self.done)

            self.total_steps += 1
            self.state = next_state

            if len(self.rewards) > self.max_episode_length:
                self.done = True

            if len(self.replay_buffer) > self.batch_size:
                transitions = self.replay_buffer.sample(self.batch_size)
                self.update_network(transitions)

        self.state_normalizer.unset_read_only()

    def reset(self):
        self.random_process.reset_states()
        self.state = self.state_normalizer(self.task.reset())
        self.rewards.clear()
        self.done = False
        self.info = None

    def get_action(self):
        if self.total_steps < self.warm_up:
            action = self.task.action_space.sample()
        else:
            with torch.no_grad():
                action = to_numpy(self.network.actor(to_device(self.state)))
            action += self.random_process.sample()
            action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        return action

    def record_transition(self, next_state, reward, done):
        mask = 0 if done else 1
        self.replay_buffer.feed(dict(
            state=self.state,
            action=self.get_action(),
            reward=reward,
            next_state=next_state,
            mask=mask,
        ))
        self.rewards.append(reward)
        if done:
            self.scores.append(sum(self.rewards))

    def update_network(self, transitions):
        states = to_device(transitions.state)
        actions = to_device(transitions.action)
        rewards = to_device(transitions.reward).unsqueeze(-1)
        next_states = to_device(transitions.next_state)
        masks = to_device(transitions.mask).unsqueeze(-1)

        phi = self.network.feature(states)
        q = self.network.critic(phi, actions)

        phi_next = self.target_network.feature(next_states)
        a_next = self.target_network.actor(phi_next)
        q_next = self.target_network.critic(phi_next, a_next).detach()
        target_q = rewards + masks * self.discount * q_next
        critic_loss = F.mse_loss(q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        phi = self.network.feature(states)
        policy_loss = -self.network.critic(phi.detach(), self.network.actor(phi)).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.target_network, self.network, self.target_network_mix)