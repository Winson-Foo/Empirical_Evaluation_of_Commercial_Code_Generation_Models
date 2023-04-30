from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision


class DDPGAgent(BaseAgent):

    WARM_UP = 1000

    def __init__(self, config):
        super().__init__(config)
        self.network = self._create_network()
        self.target_network = self._create_network()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = self._create_replay_buffer()
        self.random_process = self._create_random_process()
        self.total_steps = 0
        self.state = None

    def _create_network(self):
        return self.config.network_fn()

    def _create_replay_buffer(self):
        return self.config.replay_fn()

    def _create_random_process(self):
        return self.config.random_process_fn()

    def _soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def _take_action(self):
        if self.total_steps < self.WARM_UP:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        return action

    def _update_replay_buffer(self, action, reward, next_state, done):
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

    def _update_network(self):
        if self.replay.size() >= self.WARM_UP:
            transitions = self.replay.sample()
            states = tensor(transitions.state)
            actions = tensor(transitions.action)
            rewards = tensor(transitions.reward).unsqueeze(-1)
            next_states = tensor(transitions.next_state)
            mask = tensor(transitions.mask).unsqueeze(-1)

            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next)
            q_next = self.target_network.critic(phi_next, a_next)
            q_next = self.config.discount * mask * q_next
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

            self._soft_update(self.target_network, self.network)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def step(self):
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = self.config.state_normalizer(self.state)

        action = self._take_action()
        next_state, reward, done, info = self.task.step(action)
        self._update_replay_buffer(action, reward, next_state, done)
        self.state = next_state
        self.total_steps += 1
        self._update_network()