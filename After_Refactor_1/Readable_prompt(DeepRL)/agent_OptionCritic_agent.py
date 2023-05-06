## Refactored code:
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

    def sample_option(self, prediction, epsilon, prev_option, is_intial_states):
        q_option = prediction['q']
        pi_option = torch.zeros_like(q_option).add(epsilon / q_option.size(1))
        greedy_option = q_option.argmax(dim=-1, keepdim=True)
        prob = 1 - epsilon + epsilon / q_option.size(1)
        prob = torch.zeros_like(pi_option).add(prob)
        pi_option.scatter_(1, greedy_option, prob)
        mask = torch.zeros_like(q_option)
        mask[self.worker_index, prev_option] = 1
        beta = prediction['beta']
        pi_hat_option = (1 - beta) * mask + beta * pi_option
        dist = torch.distributions.Categorical(probs=pi_option)
        options = dist.sample()
        dist = torch.distributions.Categorical(probs=pi_hat_option)
        options_hat = dist.sample()
        options = torch.where(is_intial_states, options, options_hat)
        return options

    def fill_storage(self, storage, prediction, options, actions, rewards, terminals, info, entropy, epsilon):
        storage.feed({'reward': tensor(self.config.reward_normalizer(rewards)).unsqueeze(-1),
                    'mask': tensor(1 - terminals).unsqueeze(-1),
                    'option': options.unsqueeze(-1),
                    'prev_option': self.prev_options.unsqueeze(-1),
                    'entropy': entropy.unsqueeze(-1),
                    'action': actions.unsqueeze(-1),
                    'init_state': self.is_initial_states.unsqueeze(-1).float(),
                    'eps': epsilon})

    def update_states(self, next_states, terminals):
        next_states = self.config.state_normalizer(next_states)
        self.states = next_states
        self.is_initial_states = tensor(terminals).byte()
        self.prev_options = self.options

    def fill_beta_storage(self, betas, prediction, storage, ret):
        betas = prediction['beta'][self.worker_index, self.prev_options]
        ret = (1 - betas) * prediction['q'][self.worker_index, self.prev_options] + \
                    betas * torch.max(prediction['q'], dim=-1)[0]
        ret = ret.unsqueeze(-1)
        storage.ret[i] = ret

    def compute_advantages_and_beta_advantage(self, storage):
        for i in reversed(range(self.config.rollout_length)):
            ret = storage.reward[i] + self.config.discount * storage.mask[i] * ret
            adv = ret - storage.q[i].gather(1, storage.option[i])
            storage.advantage[i] = adv

            v = storage.q[i].max(dim=-1, keepdim=True)[0] * (1 - storage.eps[i]) + storage.q[i].mean(-1).unsqueeze(-1) * \
                    storage.eps[i]
            q = storage.q[i].gather(1, storage.prev_option[i])
            storage.beta_advantage[i] = q - v + self.config.termination_regularizer

    def compute_loss(self, entries):
        q_loss = (entries.q.gather(1, entries.option) - entries.ret.detach()).pow(2).mul(0.5).mean()
        pi_loss = -(entries.log_pi.gather(1,
                                          entries.action) * entries.advantage.detach()) - self.config.entropy_weight * entries.entropy
        pi_loss = pi_loss.mean()
        beta_loss = (entries.beta.gather(1, entries.prev_option) * entries.beta_advantage.detach() * (1 - entries.init_state)).mean()
        return pi_loss, q_loss, beta_loss

    def train(self):
        config = self.config
        storage = Storage(config.rollout_length, ['beta', 'option', 'beta_advantage', 'prev_option', 'init_state', 'eps'])

        for _ in range(config.rollout_length):
            prediction = self.network(self.states)
            epsilon = config.random_option_prob(config.num_workers)
            self.options = self.sample_option(prediction, epsilon, self.prev_options, self.is_initial_states)
            prediction['pi'] = prediction['pi'][self.worker_index, self.options]
            prediction['log_pi'] = prediction['log_pi'][self.worker_index, self.options]
            dist = torch.distributions.Categorical(probs=prediction['pi'])
            actions = dist.sample()
            entropy = dist.entropy()
            next_states, rewards, terminals, info = self.task.step(to_np(actions))
            self.record_online_return(info)
            self.fill_storage(storage, prediction, self.options, actions, rewards, terminals, info, entropy, epsilon)
            self.update_states(next_states, terminals)

            self.total_steps += config.num_workers
            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        with torch.no_grad():
            prediction = self.target_network(self.states)
            storage.placeholder()
            self.fill_beta_storage(prediction, betas, storage, ret)

        self.compute_advantages_and_beta_advantage(storage)
        entries = storage.extract(
            ['q', 'beta', 'log_pi', 'ret', 'advantage', 'beta_advantage', 'entropy', 'option', 'action', 'init_state', 'prev_option'])

        pi_loss, q_loss, beta_loss = self.compute_loss(entries)

        self.optimizer.zero_grad()
        (pi_loss + q_loss + beta_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()