from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision


class DDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def step(self):
        config = self.config
        if self.state is None:
            self._reset_states()
        
        self._try_network_policy()

        self._clip_action()
        
        next_state, reward, done, info = self._task_reaction()

        self._record_online_return(info)
        
        self._prepare_feed(next_state, reward, done)

        if done[0]:
            self.random_process.reset_states()

        self.state = next_state
        self.total_steps += 1

        if self.replay.size() >= config.warm_up:
            transitions = self.replay.sample()
            states, actions, rewards, next_states, mask = self._replay_trans(transitions)

            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next)
            q_next = self.target_network.critic(phi_next, a_next)
            q_next = self._calc_discount(q_next, rewards, mask)

            phi = self.network.feature(states)
            q = self.network.critic(phi, actions)
            critic_loss = self._calc_loss(q, q_next)

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            policy_loss = self._calc_policy_loss(states)

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)

    def _reset_states(self):
        self.random_process.reset_states()
        self.state = self.task.reset()
        self.state = self.config.state_normalizer(self.state)

    def _try_network_policy(self):
        if self.total_steps < self.config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
            
        self.action = np.clip(action, self.task.action_space.low, self.task.action_space.high)

    def _clip_action(self):
        self.action = np.clip(self.action, self.task.action_space.low, self.task.action_space.high)

    def _task_reaction(self):
        next_state, reward, done, info = self.task.step(self.action)
        next_state = self.config.state_normalizer(next_state)
        reward = self.config.reward_normalizer(reward)
        return next_state, reward, done, info

    def _record_online_return(self, info):
        self.record_online_return(info)

    def _prepare_feed(self, next_state, reward, done):
        self.replay.feed(dict(
            state=self.state,
            action=self.action,
            reward=reward,
            next_state=next_state,
            mask=1-np.asarray(done, dtype=np.int32),
        ))

    def _replay_trans(self, transitions):
        states = tensor(transitions.state)
        actions = tensor(transitions.action)
        rewards = tensor(transitions.reward).unsqueeze(-1)
        next_states = tensor(transitions.next_state)
        mask = tensor(transitions.mask).unsqueeze(-1)

        return states, actions, rewards, next_states, mask

    def _calc_discount(self, q_next, rewards, mask):
        q_next = self.config.discount * mask * q_next
        q_next.add_(rewards)
        q_next = q_next.detach()

        return q_next

    def _calc_loss(self, q, q_next):
        critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()

        return critic_loss

    def _calc_policy_loss(self, states):
        phi = self.network.feature(states)
        action = self.network.actor(phi)
        policy_loss = -self.network.critic(phi.detach(), action).mean()

        return policy_loss