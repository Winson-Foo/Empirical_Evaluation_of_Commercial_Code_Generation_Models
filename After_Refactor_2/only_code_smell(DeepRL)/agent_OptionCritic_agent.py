from typing import Dict
import torch
import torch.nn as nn
import numpy as np
from torch import optim, Tensor
from ..network import Network
from ..component import Storage
from .BaseAgent import BaseAgent


class OptionCriticAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.task = config.task_fn()
        self.network: Network = config.network_fn()
        self.target_network: Network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = 0
        self.worker_index = tensor(np.arange(config.num_workers)).long()

        self.states = self.config.state_normalizer(self.task.reset())
        self.is_initial_states = tensor(np.ones((config.num_workers))).bool()
        self.prev_options = self.is_initial_states.clone().long()

    def step(self):
        storage = Storage(self.config.rollout_length, ['beta', 'option', 'beta_advantage', 'prev_option', 'init_state', 'eps'])

        for _ in range(self.config.rollout_length):
            prediction = self.network(self.states)
            epsilon = self.config.random_option_prob(self.config.num_workers)
            options = sample_option(prediction, epsilon, self.prev_options, self.is_initial_states)
            prediction = adjust_prediction(prediction, self.worker_index, options)
            actions, entropy = select_actions(prediction)
            next_states, rewards, terminals, info = self.task.step(to_np(actions))
            self.record_online_return(info)
            next_states, rewards = self.config.state_reward_normalizer(next_states, rewards)
            storage.feed(prediction=prediction, reward=rewards, mask=(1 - terminals),
                          option=options, prev_option=self.prev_options, entropy=entropy, action=actions,
                          init_state=self.is_initial_states.float(), eps=epsilon)
            self.is_initial_states = tensor(terminals).bool()
            self.prev_options = options
            self.states = self.config.state_normalizer(next_states)
            self.total_steps += self.config.num_workers
            if self.total_steps // self.config.num_workers % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        with torch.no_grad():
            prediction = self.target_network(self.states)
            storage.reset()
            betas = get_betas(prediction, self.worker_index, self.prev_options)
            ret = calculate_returns(prediction['q'], betas)
            ret = ret.unsqueeze(-1)

        for i in reversed(range(self.config.rollout_length)):
            ret = calculate_advantages(ret, storage.reward[i], storage.mask[i], storage.q[i], storage.option[i], storage.eps[i], self.config.discount)
            storage.ret[i] = ret
            storage.beta_advantage[i] = calculate_beta_advantage(storage.q[i], storage.prev_option[i], betas[i], self.config)

        entries = storage.extract(
            ['q', 'beta', 'log_pi', 'ret', 'advantage', 'beta_advantage', 'entropy', 'option', 'action', 'init_state', 'prev_option'])

        q_loss = calculate_q_loss(entries.q, entries.option, entries.ret)
        pi_loss = calculate_pi_loss(entries.log_pi, entries.action, entries.advantage, entries.entropy, self.config)
        beta_loss = calculate_beta_loss(entries.beta, entries.prev_option, entries.beta_advantage, entries.init_state, self.config)

        self.optimizer.zero_grad()
        loss = pi_loss + q_loss + beta_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()


def sample_option(prediction: Dict[str, Tensor], epsilon: float, prev_option: Tensor, is_initial_states: Tensor) -> Tensor:
    with torch.no_grad():
        q_values = prediction['q']
        options_size = q_values.size(1)
        pi_option = torch.zeros_like(q_values).add(epsilon / options_size)
        greedy_option = q_values.argmax(dim=-1, keepdim=True)
        prob = 1 - epsilon + epsilon / options_size
        prob = torch.zeros_like(pi_option).add(prob)
        pi_option.scatter_(1, greedy_option, prob)

        mask = torch.zeros_like(q_values)
        mask[range(q_values.size(0)), prev_option] = 1
        beta = prediction['beta']
        pi_hat_option = (1 - beta) * mask + beta * pi_option

        dist = torch.distributions.Categorical(probs=pi_option)
        options = dist.sample()
        dist = torch.distributions.Categorical(probs=pi_hat_option)
        options_hat = dist.sample()

        options = torch.where(is_initial_states, options, options_hat)
    return options


def adjust_prediction(prediction: Dict[str, Tensor], worker_index: Tensor, options: Tensor) -> Dict[str, Tensor]:
    prediction['pi'] = prediction['pi'][worker_index, options]
    prediction['log_pi'] = prediction['log_pi'][worker_index, options]
    prediction['q'] = prediction['q'][worker_index, options]
    prediction['beta'] = prediction['beta'][worker_index, options]
    return prediction


def select_actions(prediction: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
    dist = torch.distributions.Categorical(probs=prediction['pi'])
    actions = dist.sample()
    entropy = dist.entropy()
    return actions, entropy


def calculate_returns(q: Tensor, betas: Tensor) -> Tensor:
    max_q, _ = q.max(dim=-1)
    ret = (1 - betas) * q.gather(1, prev_option) + betas * max_q.unsqueeze(-1)
    return ret


def calculate_advantages(ret: Tensor, reward: Tensor, mask: Tensor, q: Tensor, option: Tensor,
                          eps: Tensor, discount: float) -> Tensor:
    ret = reward + discount * mask * ret
    adv = ret - q.gather(1, option)
    v = q.max(dim=-1, keepdim=True)[0] * (1 - eps) + q.mean(-1).unsqueeze(-1) * eps
    beta_adv = q.gather(1, prev_option) - v + config.termination_regularizer
    return ret, adv, beta_adv


def calculate_q_loss(q: Tensor, option: Tensor, ret: Tensor) -> Tensor:
    q_value = q.gather(1, option)
    q_loss = (q_value - ret.detach()).pow(2).mul(0.5).mean()
    return q_loss


def calculate_pi_loss(log_pi: Tensor, action: Tensor, advantage: Tensor, entropy: Tensor, config) -> Tensor:
    pi_loss = -(log_pi.gather(1, action) * advantage.detach()) - config.entropy_weight * entropy
    pi_loss = pi_loss.mean()
    return pi_loss


def calculate_beta_loss(beta: Tensor, prev_option: Tensor, beta_advantage: Tensor, init_state: Tensor, config) -> Tensor:
    beta_loss = (beta.gather(1, prev_option) * beta_advantage.detach() * (1 - init_state)).mean()
    return beta_loss


def get_betas(prediction: Dict[str, Tensor], worker_index: Tensor, prev_option: Tensor) -> Tensor:
    return prediction['beta'][worker_index, prev_option]


def tensor(data: np.ndarray) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float32)