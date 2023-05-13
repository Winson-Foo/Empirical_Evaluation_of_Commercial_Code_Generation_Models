from ..network import *
from ..component import *
from .BaseAgent import *

class OptionCriticAgent(BaseAgent):

	def __init__(self, config):
		super().__init__(config)
		self.task = config.task_fn()
		self.network = config.network_fn()
		self.target_network = config.network_fn()
		self.optimizer = config.optimizer_fn(self.network.parameters())
		self.target_network.load_state_dict(self.network.state_dict())
		self.worker_index = torch.arange(config.num_workers).long()

	def update_target_network(self):
		self.target_network.load_state_dict(self.network.state_dict())

	def sample_option(self, prediction, epsilon, prev_option, is_initial_states):
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

	def update(self):
		config = self.config
		storage = Storage(config.rollout_length, ['beta', 'option', 'beta_advantage', 'prev_option', 'init_state', 'eps'])

		for _ in range(config.rollout_length):
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

			self.is_initial_states = tensor(terminals).byte()
			self.prev_options = options
			self.states = next_states

			if (self.total_steps + config.num_workers) % config.target_network_update_freq == 0:
				self.update_target_network()

			self.total_steps += config.num_workers

		with torch.no_grad():
			prediction = self.target_network(self.states)
			storage.placeholder()
			betas = prediction['beta'][self.worker_index, self.prev_options]
			ret = (1 - betas) * prediction['q'][self.worker_index, self.prev_options] + \
				  betas * torch.max(prediction['q'], dim=-1)[0]
			ret = ret.unsqueeze(-1)

		for i in reversed(range(config.rollout_length)):
			ret = storage.reward[i] + config.discount * storage.mask[i] * ret
			adv = ret - storage.q[i].gather(1, storage.option[i])
			storage.ret[i] = ret
			storage.advantage[i] = adv

			v = storage.q[i].max(dim=-1, keepdim=True)[0] * (1 - storage.eps[i]) + storage.q[i].mean(-1).unsqueeze(-1) * \
				storage.eps[i]
			q = storage.q[i].gather(1, storage.prev_option[i])
			storage.beta_advantage[i] = q - v + config.termination_regularizer

		entries = storage.extract(
			['q', 'beta', 'log_pi', 'ret', 'advantage', 'beta_advantage', 'entropy', 'option', 'action', 'init_state', 'prev_option'])

		q_loss = (entries.q.gather(1, entries.option) - entries.ret.detach()).pow(2).mul(0.5).mean()
		pi_loss = -(entries.log_pi.gather(1,
										  entries.action) * entries.advantage.detach()) - config.entropy_weight * entries.entropy
		pi_loss = pi_loss.mean()
		beta_loss = (entries.beta.gather(1, entries.prev_option) * entries.beta_advantage.detach() * (1 - entries.init_state)).mean()

		self.optimizer.zero_grad()
		(pi_loss + q_loss + beta_loss).backward()
		nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
		self.optimizer.step()

	def learn(self):
		for i in range(self.config.max_steps // self.config.rollout_length):
			self.update()

	def act(self, states):
		# convert state to tensor if its not already one
		if isinstance(states, np.ndarray):
			states = tensor(states).unsqueeze(0)
		with torch.no_grad():
			states = self.config.state_normalizer(states)
			prediction = self.network(states)
			epsilon = self.config.random_option_prob(1)
			options = self.sample_option(prediction, epsilon, self.prev_options, self.is_initial_states)
			prediction['pi'] = prediction['pi'][0, options]
			prediction['log_pi'] = prediction['log_pi'][0, options]
			dist = torch.distributions.Categorical(probs=prediction['pi'])
			action = dist.sample().squeeze().cpu().numpy()
			self.prev_options = options
			self.is_initial_states = np.zeros((1,), dtype=np.uint8)
		return action