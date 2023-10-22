from ..network import *
from ..component import *
from .BaseAgent import *


class OptionCriticAgent(BaseAgent):
    def __init__(self, agent_config):
        BaseAgent.__init__(self, agent_config)
        self.agent_config = agent_config
        self.task = agent_config.task_fn()
        self.network = agent_config.network_fn()
        self.target_network = agent_config.network_fn()
        self.optimizer = agent_config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = 0
        self.worker_index = tensor(np.arange(agent_config.num_workers)).long()

        self.states = self.agent_config.state_normalizer(self.task.reset())
        self.is_initial_states = tensor(np.ones((agent_config.num_workers))).byte()
        self.prev_options = self.is_initial_states.clone().long()

    def sample_option(self, action_prediction, epsilon, prev_option, is_initial_states):
        # Compute the probabilities of selecting each option
        q_option = action_prediction['q']
        pi_option = torch.zeros_like(q_option).add(epsilon / q_option.size(1))
        greedy_option = q_option.argmax(dim=-1, keepdim=True)
        prob = 1 - epsilon + epsilon / q_option.size(1)
        prob = torch.zeros_like(pi_option).add(prob)
        pi_option.scatter_(1, greedy_option, prob)

        # Compute the probabilities of selecting each option with termination
        mask = torch.zeros_like(q_option)
        mask[self.worker_index, prev_option] = 1
        beta = action_prediction['beta']
        pi_hat_option = (1 - beta) * mask + beta * pi_option

        # Sample options from the probability distributions
        dist = torch.distributions.Categorical(probs=pi_option)
        options = dist.sample()
        dist = torch.distributions.Categorical(probs=pi_hat_option)
        options_hat = dist.sample()

        options = torch.where(is_initial_states, options, options_hat)
        return options

    def compute_losses(self, rollout_storage, action_prediction, agent_config):
        entries = rollout_storage.extract(
            ['q', 'beta', 'log_pi', 'ret', 'advantage', 'beta_advantage', 'entropy', 'option', 'action', 'init_state', 'prev_option']
        )

        # Compute the q loss
        q_loss = (entries.q.gather(1, entries.option) - entries.ret.detach()).pow(2).mul(0.5).mean()

        # Compute the policy loss
        pi_loss = -(entries.log_pi.gather(1, entries.action) * entries.advantage.detach()) - agent_config.entropy_weight * entries.entropy
        pi_loss = pi_loss.mean()

        # Compute the termination loss
        beta_loss = (entries.beta.gather(1, entries.prev_option) * entries.beta_advantage.detach() * (1 - entries.init_state)).mean()

        return pi_loss, q_loss, beta_loss

    def update_network(self, pi_loss, q_loss, beta_loss, agent_config):
        self.optimizer.zero_grad()
        (pi_loss + q_loss + beta_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), agent_config.gradient_clip)
        self.optimizer.step()

    def step(self):
        agent_config = self.agent_config
        rollout_storage = Storage(agent_config.rollout_length, ['beta', 'option', 'beta_advantage', 'prev_option', 'init_state', 'eps'])

        # Collect rollout data
        for _ in range(agent_config.rollout_length):
            action_prediction = self.network(self.states)
            epsilon = agent_config.random_option_prob(agent_config.num_workers)
            options = self.sample_option(action_prediction, epsilon, self.prev_options, self.is_initial_states)
            action_prediction['pi'] = action_prediction['pi'][self.worker_index, options]
            action_prediction['log_pi'] = action_prediction['log_pi'][self.worker_index, options]
            dist = torch.distributions.Categorical(probs=action_prediction['pi'])
            actions = dist.sample()
            entropy = dist.entropy()

            next_states, rewards, terminals, info = self.task.step(to_np(actions))
            self.record_online_return(info)
            next_states = agent_config.state_normalizer(next_states)
            rewards = agent_config.reward_normalizer(rewards)
            rollout_storage.feed(action_prediction)
            rollout_storage.feed({'reward': tensor(rewards).unsqueeze(-1),
                                   'mask': tensor(1 - terminals).unsqueeze(-1),
                                   'option': options.unsqueeze(-1),
                                   'prev_option': self.prev_options.unsqueeze(-1),
                                   'entropy': entropy.unsqueeze(-1),
                                   'action': actions.unsqueeze(-1),
                                   'init_state': self.is_initial_states.unsqueeze(-1).float(),
                                   'eps': epsilon})

            self.is_initial_states = tensor(terminals).byte()
            self.prev_options = options
            self.states = next_states

            self.total_steps += agent_config.num_workers

            # Update the target network
            if self.total_steps // agent_config.num_workers % agent_config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        # Compute the bootstrap values
        with torch.no_grad():
            action_prediction = self.target_network(self.states)
            rollout_storage.placeholder()
            betas = action_prediction['beta'][self.worker_index, self.prev_options]
            ret = (1 - betas) * action_prediction['q'][self.worker_index, self.prev_options] + \
                  betas * torch.max(action_prediction['q'], dim=-1)[0]
            ret = ret.unsqueeze(-1)

        # Compute the advantages and target values
        for i in reversed(range(agent_config.rollout_length)):
            ret = rollout_storage.reward[i] + agent_config.discount * rollout_storage.mask[i] * ret
            adv = ret - rollout_storage.q[i].gather(1, rollout_storage.option[i])
            rollout_storage.ret[i] = ret
            rollout_storage.advantage[i] = adv

            v = rollout_storage.q[i].max(dim=-1, keepdim=True)[0] * (1 - rollout_storage.eps[i]) + rollout_storage.q[i].mean(-1).unsqueeze(-1) * \
                rollout_storage.eps[i]
            q = rollout_storage.q[i].gather(1, rollout_storage.prev_option[i])
            rollout_storage.beta_advantage[i] = q - v + agent_config.termination_regularizer

        # Compute losses and update the network
        pi_loss, q_loss, beta_loss = self.compute_losses(rollout_storage, action_prediction, agent_config)
        self.update_network(pi_loss, q_loss, beta_loss, agent_config)