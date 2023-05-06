#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *

class TD3Agent(BaseAgent):
    def __init__(self, agent_config):
        BaseAgent.__init__(self, agent_config)
        self.agent_config = agent_config
        self.environment = agent_config.task_fn()
        self.actor_network = agent_config.network_fn()
        self.critic_network = agent_config.network_fn()
        self.critic_target_network = agent_config.network_fn()
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.replay_buffer = agent_config.replay_fn()
        self.random_process = agent_config.random_process_fn()
        self.total_steps = 0
        self.state = None

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.agent_config.target_network_mix) +
                               param * self.agent_config.target_network_mix)

    def evaluate_state(self, state):
        self.agent_config.state_normalizer.set_read_only()
        state = self.agent_config.state_normalizer(state)
        action = self.actor_network(state)
        self.agent_config.state_normalizer.unset_read_only()
        return to_np(action)

    def step(self):
        agent_config = self.agent_config
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.environment.reset()
            self.state = agent_config.state_normalizer(self.state)

        if self.total_steps < agent_config.warm_up:
            action = [self.environment.action_space.sample()]
        else:
            action = self.actor_network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, self.environment.action_space.low, self.environment.action_space.high)
        next_state, reward, done, info = self.environment.step(action)
        next_state = agent_config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = agent_config.reward_normalizer(reward)

        self.replay_buffer.feed(dict(
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

        if self.total_steps >= agent_config.warm_up:
            transitions = self.replay_buffer.sample()
            states = tensor(transitions.state)
            actions = tensor(transitions.action)
            rewards = tensor(transitions.reward).unsqueeze(-1)
            next_states = tensor(transitions.next_state)
            mask = tensor(transitions.mask).unsqueeze(-1)

            a_next = self.critic_target_network(next_states)
            noise = torch.randn_like(a_next).mul(agent_config.td3_noise)
            noise = noise.clamp(-agent_config.td3_noise_clip, agent_config.td3_noise_clip)

            min_action = float(self.environment.action_space.low[0])
            max_action = float(self.environment.action_space.high[0])
            a_next = (a_next + noise).clamp(min_action, max_action)

            q_1, q_2 = self.critic_target_network.q(next_states, a_next)
            target = rewards + agent_config.discount * mask * torch.min(q_1, q_2)
            target = target.detach()

            q_1, q_2 = self.critic_network.q(states, actions)
            critic_loss = F.mse_loss(q_1, target) + F.mse_loss(q_2, target)

            self.critic_network.zero_grad()
            critic_loss.backward()
            self.critic_network.critic_opt.step()

            if self.total_steps % agent_config.td3_delay:
                action = self.actor_network(states)
                policy_loss = -self.critic_network.q(states, action)[0].mean()

                self.actor_network.zero_grad()
                policy_loss.backward()
                self.actor_network.actor_opt.step()

                self.soft_update(self.critic_target_network, self.critic_network)