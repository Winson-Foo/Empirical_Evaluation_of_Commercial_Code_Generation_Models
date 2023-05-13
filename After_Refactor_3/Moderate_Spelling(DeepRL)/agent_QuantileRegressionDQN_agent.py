class QuantileRegressionDQNActor(DQNActor):
    def __init__(self, config):
        super().__init__(config)

    def compute_q(self, prediction):
        q_values = prediction['quantile'].mean(-1)
        return to_np(q_values)
        

class DQNQuantileAgent(DQNAgent):
    def __init__(self, config):
        super().__init__(config)
        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = tensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles)).view(1, -1)
        
    def compute_loss(self, transitions):
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)

        quantiles_next = self.target_network(next_states)['quantile'].detach()
        a_next = torch.argmax(quantiles_next.sum(-1), dim=-1)
        quantiles_next = quantiles_next[self.batch_indices, a_next, :]

        rewards = tensor(transitions.reward).unsqueeze(-1)
        masks = tensor(transitions.mask).unsqueeze(-1)
        quantiles_next = rewards + self.config.discount ** self.config.n_step * masks * quantiles_next

        quantiles = self.network(states)['quantile']
        actions = tensor(transitions.action).long()
        quantiles = quantiles[self.batch_indices, actions, :]

        quantiles_next = quantiles_next.t().unsqueeze(-1)
        diff = quantiles_next - quantiles
        loss = huber(diff) * (self.cumulative_density - (diff.detach() < 0).float()).abs()
        return loss.sum(-1).mean(1)

    def reduce_loss(self, loss):
        return loss.mean()
    
    
class QuantileRegressionDQNAgent(DQNQuantileAgent):
    def __init__(self, config):
        super().__init__(config)
        self.actor = QuantileRegressionDQNActor(config)

class DQNQuantileAgent(DQNAgent):
    def __init__(self, config):
        super().__init__(config)
        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = tensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles)).view(1, -1)

    def get_quantile_loss(self, transitions):
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)

        quantiles_next = self.target_network(next_states)['quantile'].detach()
        best_next_actions = torch.argmax(quantiles_next.sum(-1), dim=-1)
        quantiles_next = quantiles_next[self.batch_indices, best_next_actions, :]

        rewards = tensor(transitions.reward).unsqueeze(-1)
        masks = tensor(transitions.mask).unsqueeze(-1)
        discounted_quantiles_next = rewards + self.config.discount ** self.config.n_step * masks * quantiles_next

        quantiles = self.network(states)['quantile']
        actions = tensor(transitions.action).long()
        quantiles_chosen = quantiles[self.batch_indices, actions, :]

        discounted_quantiles_next = discounted_quantiles_next.t().unsqueeze(-1)
        diff = discounted_quantiles_next - quantiles_chosen
        quantile_errors = huber(diff) * (self.cumulative_density - (diff.detach() < 0).float()).abs()
        return quantile_errors.sum(-1).mean(1)

class BaseActor:
    def __init__(self, config):
        self.network = None
        self.config = config

    def set_network(self, network):
        self.network = network
        
    def compute_q(self, prediction):
        raise NotImplementedError
        
        
class DQNActor(BaseActor):
    def __init__(self, config):
        super().__init__(config)

    def compute_q(self, prediction):
        return to_np(prediction['q_values'])


class QuantileRegressionDQNActor(BaseActor):
    def __init__(self, config):
        super().__init__(config)

    def compute_q(self, prediction):
        q_values = prediction['quantile'].mean(-1)
        return to_np(q_values)