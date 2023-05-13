from ..network import *
from ..component import *
from .BaseAgent import *

class OptionCriticAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())
        self.total_steps = 0
        self.worker_index = tensor(np.arange(config.num_workers)).long()
        self.states = self.config.state_normalizer(self.task.reset())
        self.is_initial_states = tensor(np.ones((config.num_workers))).byte()
        self.prev_options = self.is_initial_states.clone().long()

    def sample_option(self, prediction, epsilon, prev_options, is_initial_states):
        q_option = prediction['q']
        pi_option = self._compute_pi(q_option, epsilon)
        pi_hat_option = self._compute_pi_hat(prediction, prev_options, pi_option)

        dist = torch.distributions.Categorical(probs=pi_option)
        options = dist.sample()
        dist = torch.distributions.Categorical(probs=pi_hat_option)
        options_hat = dist.sample()
        options = torch.where(is_initial_states, options, options_hat)

        return options

    def _compute_pi(self, q_option, epsilon):
        q_size = q_option.size(1)
        pi_option = torch.zeros_like(q_option).add(epsilon / q_size)
        greedy_option = q_option.argmax(dim=-1, keepdim=True)
        prob = 1 - epsilon + epsilon / q_size
        prob = torch.zeros_like(pi_option).add(prob)
        pi_option.scatter_(1, greedy_option, prob)
        return pi_option

    def _compute_pi_hat(self, prediction, prev_options, pi_option):
        q_option = prediction['q']
        beta = prediction['beta']
        mask = torch.zeros_like(q_option)
        mask[self.worker_index, prev_options] = 1
        pi_hat_option = (1 - beta) * mask + beta * pi_option
        return pi_hat_option

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length, ['beta', 'option', 'beta_advantage', 'prev_option', 'init_state', 'eps'])
        for _ in range(config.rollout_length):
            prediction = self.network(self.states)
            epsilon = config.random_option_prob(config.num_workers)
            options = self.sample_option(prediction, epsilon, self.prev_options, self.is_initial_states)
            prediction['pi'] = prediction['pi'][self.worker_index, options]
            prediction['log_pi'] = prediction['log_pi'][self.worker_index, options]
            actions = self._sample_actions(prediction)
            entropy = self._compute_entropy(prediction)

            next_states, rewards, terminals, info = self.task.step(to_np(actions))
            self.record_online_return(info)
            next_states = config.state_normalizer(next_states)
            rewards = config.reward_normalizer(rewards)
            storage.feed(prediction)
            storage.feed({'reward': tensor(rewards).unsqueeze(-1),
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
            self.total_steps += config.num_workers

            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        entries = storage.extract(
            ['q', 'beta', 'log_pi', 'ret', 'advantage', 'beta_advantage', 'entropy', 'option', 'action', 'init_state', 'prev_option'])
        q_loss, pi_loss, beta_loss = self._compute_losses(entries, config)

        self.optimizer.zero_grad()
        (pi_loss + q_loss + beta_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

    def _sample_actions(self, prediction):
        dist = torch.distributions.Categorical(probs=prediction['pi'])
        actions = dist.sample()
        return actions

    def _compute_entropy(self, prediction):
        dist = torch.distributions.Categorical(probs=prediction['pi'])
        entropy = dist.entropy()
        return entropy

    def _compute_losses(self, entries, config):
        q_loss = (entries.q.gather(1, entries.option) - entries.ret.detach()).pow(2).mul(0.5).mean()
        pi_loss = -(entries.log_pi.gather(1, entries.action) * entries.advantage.detach()) - config.entropy_weight * entries.entropy
        pi_loss = pi_loss.mean()
        beta_loss = (entries.beta.gather(1, entries.prev_option) * entries.beta_advantage.detach() * (1 - entries.init_state)).mean()
        return q_loss, pi_loss, beta_loss