from typing import Dict, Tensor, List
from torch.distributions import Categorical


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

    def sample_option(self, prediction: Dict[str, Tensor], epsilon: float, prev_option: Tensor, is_initial_states: Tensor) -> Tensor:
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

    def collect_rollout(self) -> List[Dict[str, Tensor]]:
        config = self.config
        storage = Storage(config.rollout_length, ['beta', 'option', 'beta_advantage', 'prev_option', 'init_state', 'eps'])

        for _ in range(config.rollout_length):
            prediction = self.network(self.states)
            epsilon = config.random_option_prob(config.num_workers)
            options = self.sample_option(prediction, epsilon, self.prev_options, self.is_initial_states)
            prediction['pi'] = prediction['pi'][self.worker_index, options]
            prediction['log_pi'] = prediction['log_pi'][self.worker_index, options]
            dist = Categorical(probs=prediction['pi'])
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

            self.is_initial_states = tensor(terminals).byte()
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

        rollout_data = storage.extract(
            ['q', 'beta', 'log_pi', 'ret', 'advantage', 'beta_advantage', 'entropy', 'option', 'action', 'init_state', 'prev_option'])

        return rollout_data

    def update(self, rollout_data: List[Dict[str, Tensor]]) -> None:
        config = self.config

        q_loss = (rollout_data.q.gather(1, rollout_data.option) - rollout_data.ret.detach()).pow(2).mul(0.5).mean()
        pi_loss = -(rollout_data.log_pi.gather(1, rollout_data.action) * rollout_data.advantage.detach()) \
                  - config.entropy_weight * rollout_data.entropy
        pi_loss = pi_loss.mean()
        beta_loss = (rollout_data.beta.gather(1, rollout_data.prev_option) * rollout_data.beta_advantage.detach() * (
                    1 - rollout_data.init_state)).mean()

        self.optimizer.zero_grad()
        (pi_loss + q_loss + beta_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

    def step(self):
        rollout_data = self.collect_rollout()
        self.update(rollout_data)