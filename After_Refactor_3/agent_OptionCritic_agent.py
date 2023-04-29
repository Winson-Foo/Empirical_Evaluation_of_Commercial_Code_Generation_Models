from ..network import *
from ..component import *
from .BaseAgent import *


class OptionCriticAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = 0
        self.worker_index = torch.tensor(np.arange(config.num_workers)).long()

        self.states = self.config.state_normalizer(self.task.reset())
        self.is_initial_states = torch.tensor(np.ones((config.num_workers))).byte()
        self.prev_options = self.is_initial_states.clone().long()

    def sample_option(self, prediction, epsilon, prev_option, is_initial_states):
        """
        Samples options based on the given prediction, epsilon, previous options and whether the current states are initial
        :param prediction: Prediction dictionary containing q, beta, pi and log_pi
        :param epsilon: Float value representing exploration rate
        :param prev_option: Tensor of longs representing previous options
        :param is_initial_states: Tensor of bytes representing whether current states are initial
        :return: Tensor of longs representing sampled options
        """
        with torch.no_grad():
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

            options = torch.where(is_initial_states, options, options_hat)
        return options

    def step(self):
        """
        Executes one step of training
        """
        config = self.config
        rollout_length = config.rollout_length
        storage = Storage(rollout_length, ['beta', 'option', 'beta_advantage', 'prev_option', 'init_state', 'eps'])

        for _ in range(rollout_length):
            prediction = self.network(self.states)
            epsilon = config.random_option_prob(config.num_workers)
            options = self.sample_option(prediction, epsilon, self.prev_options, self.is_initial_states)
            prediction['pi'] = prediction['pi'][self.worker_index, options]
            prediction['log_pi'] = prediction['log_pi'][self.worker_index, options]
            dist = torch.distributions.Categorical(probs=prediction['pi'])
            actions = dist.sample()
            entropy = dist.entropy()

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

            self.is_initial_states = torch.tensor(terminals).byte()
            self.prev_options = options
            self.states = next_states

            self.total_steps += config.num_workers
            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        with torch.no_grad():
            prediction = self.target_network(self.states)
            storage.placeholder()
            betas = prediction['beta'][self.worker_index, self.prev_options]
            ret = (1 - betas) * prediction['q'][self.worker_index, self.prev_options] + \
                  betas * torch.max(prediction['q'], dim=-1)[0]
            ret = ret.unsqueeze(-1)

        for i in reversed(range(rollout_length)):
            ret = storage.reward[i] + config.discount * storage.mask[i] * ret
            quality = storage.q[i]
            advantages = storage.advantage[i]
            options = storage.option[i]
            prev_options = storage.prev_option[i]
            init_states = storage.init_state[i]
            epsilons = storage.eps[i]
            v = quality.max(dim=-1, keepdim=True)[0] * (1 - epsilons) + quality.mean(-1).unsqueeze(-1) * epsilons
            q = quality.gather(1, prev_options)
            storage.beta_advantage[i] = q - v + config.termination_regularizer
            storage.ret[i] = ret
            storage.advantage[i] = ret - quality.gather(1, options)

        losses = self.compute_losses(storage, config)
        self.optimizer.zero_grad()
        (losses['pi'] + losses['q'] + losses['beta']).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

    def compute_losses(self, storage, config):
        """
        Computes losses for Pi, Q, and Beta
        :param storage: Storage object containing prediction and rollout data
        :param config: Config object containing hyperparameters
        :return: Dictionary of losses
        """
        entries = storage.extract(
            ['q', 'beta', 'log_pi', 'ret', 'advantage', 'beta_advantage', 'entropy', 'option', 'action', 'init_state', 'prev_option'])

        q_loss = (entries.q.gather(1, entries.option) - entries.ret.detach()).pow(2).mul(0.5).mean()
        pi_loss = -(entries.log_pi.gather(1, entries.action) * entries.advantage.detach()) - config.entropy_weight * entries.entropy
        pi_loss = pi_loss.mean()
        beta_loss = (entries.beta.gather(1, entries.prev_option) * entries.beta_advantage.detach() * (1 - entries.init_state)).mean()

        return {'pi': pi_loss, 'q': q_loss, 'beta': beta_loss}