from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision

class TD3Agent(BaseAgent):
    def __init__(self, config: dict):
        """
        TD3Agent initialization function.
        """
        super().__init__(config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.current_state = None

    def step(self):
        """
        TD3Agent action-selection function.
        """
        config = self.config
        if self.current_state is None:
            self.random_process.reset_states()
            self.current_state = self.task.reset()
            self.current_state = config.state_normalizer(self.current_state)

        if self.total_steps < config.warm_up:
            current_action = [self.task.action_space.sample()]
        else:
            current_action = self.network(self.current_state)
            current_action = to_np(current_action)
            current_action += self.random_process.sample()
        current_action = np.clip(current_action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(current_action)
        next_state = config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = config.reward_normalizer(reward)

        self.replay.feed(dict(
            state=self.current_state,
            action=current_action,
            reward=reward,
            next_state=next_state,
            mask=1-np.asarray(done, dtype=np.int32),
        ))

        if done[0]:
            self.random_process.reset_states()
        self.current_state = next_state
        self.total_steps += 1

        if self.total_steps >= config.warm_up:
            transitions = self.replay.sample()
            states = tensor(transitions.state)
            actions = tensor(transitions.action)
            rewards = tensor(transitions.reward).unsqueeze(-1)
            next_states = tensor(transitions.next_state)
            mask = tensor(transitions.mask).unsqueeze(-1)

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

            if self.total_steps % config.td3_delay:
                current_action = self.network(states)
                policy_loss = -self.network.q(states, current_action)[0].mean()

                self.network.zero_grad()
                policy_loss.backward()
                self.network.actor_opt.step()

                self.soft_update(self.target_network, self.network)

    @staticmethod
    def soft_update(target: torch.nn.Module, src: torch.nn.Module, target_network_mix: float):
        """
        Soft update function for actor-critic networks.
        """
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - target_network_mix) +
                               param * target_network_mix)

    def eval_step(self, state):
        """
        Evaluate an action from the agent's actor network.
        """
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)