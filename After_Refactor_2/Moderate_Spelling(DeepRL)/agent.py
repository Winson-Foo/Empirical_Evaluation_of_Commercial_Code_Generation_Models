# agent.py
class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.replay = Replay(config.replay_capacity)

    def learn(self, state, action, reward, next_state, done):
        transition = Transition(to_tensor(state),
                                action,
                                reward,
                                to_tensor(next_state),
                                done)
        self.replay.push(transition)

        if len(self.replay.buffer) >= self.config.replay_start_size:
            batch = self.replay.sample(self.config.batch_size)
            loss = self.compute_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.total_steps % self.config.target_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

            self.total_steps += 1

    def eval_step(self, state):
        state = to_tensor(state)
        q_values = to_numpy(self.network(state))
        action = epsilon_greedy(q_values, 0.0)
        return action

class DQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.network = Network(config.state_dim, config.action_dim)
        self.network_target = Network(config.state_dim, config.action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.total_steps = 0

    def compute_loss(self, batch):
        state_batch = torch.stack(batch.state)
        next_state_batch = torch.stack(batch.next_state)
        action_batch = torch.tensor(batch.action)
        reward_batch = torch.tensor(batch.reward)
        done_batch = torch.tensor(batch.done)

        q_values = self.network(state_batch)
        q_next_values = self.network_target(next_state_batch)
        max_next_q_values = q_next_values.max(1)[0]

        q_values_pred = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        q_values_target = reward_batch + self.config.discount * (1 - done_batch) * max_next_q_values
        loss = nn.functional.mse_loss(q_values_pred, q_values_target)

        return loss

class CategoricalDQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.network = CategoricalNetwork(config.state_dim,
                                           config.action_dim,
                                           config.categorical_n_atoms)
        self.network_target = CategoricalNetwork(config.state_dim,
                                                  config.action_dim,
                                                  config.categorical_n_atoms)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.total_steps = 0
        self.atoms = torch.linspace(config.categorical_v_min,
                                     config.categorical_v_max,
                                     config.categorical_n_atoms)

    def compute_loss(self, batch):
        state_batch = torch.stack(batch.state)
        next_state_batch = torch.stack(batch.next_state)
        action_batch = torch.tensor(batch.action)
        reward_batch = torch.tensor(batch.reward)
        done_batch = torch.tensor(batch.done)

        prob = self.network(state_batch)
        prob_next = self.network_target(next_state_batch)

        atoms_target = reward_batch.view(-1, 1) + self.config.discount * (1 - done_batch.view(-1, 1)) \
                        * self.atoms.view(1, -1)

        atoms_target.clamp_(self.config.categorical_v_min, self.config.categorical_v_max)

        b = (atoms_target - self.config.categorical_v_min) / self.delta_z
        l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)

        offset = to_tensor(np.arange(0, self.config.batch_size) * self.num_atoms).unsqueeze(1).expand(self.config.batch_size, self.num_atoms)

        target_prob = torch.zeros(prob_next.size())
        target_prob.view(-1).index_add_(0,
                                         (offset + action_batch * self.num_atoms + l).view(-1),
                                         (prob_next * (u.float() - b)).view(-1))
        target_prob.view(-1).index_add_(0,
                                         (offset + action_batch * self.num_atoms + u).view(-1),
                                         (prob_next * (b - l.float())).view(-1))

        loss = -torch.sum(target_prob * prob.log())

        return loss

    def eval_step(self, state):
        state = to_tensor(state)
        q_values = (self.network(state) * self.atoms).sum(-1)
        action = epsilon_greedy(q_values, 0.0)
        return action