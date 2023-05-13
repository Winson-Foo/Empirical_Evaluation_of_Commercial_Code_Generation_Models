
class BaseAgent:
    def __init__(self, config):
        self.config = config

    def start_episode(self):
        pass

    def update(self, transition):
        pass

    def end_episode(self):
        pass

class DQNActor(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.network = None

    def set_network(self, network):
        self.network = network

    def step(self, state):
        prediction = self.network(state)
        action = to_np(prediction['q_values'].argmax(-1))
        return action

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        prediction = self.network(state)
        q = (prediction['prob'] * self.config.atoms).sum(-1)
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

class DQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        config.lock = mp.Lock()
        config.atoms = np.linspace(config.categorical_v_min,
                                   config.categorical_v_max, config.categorical_n_atoms)

        self.replay = config.replay_fn()
        self.actor = DQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = range_tensor(config.batch_size)
        self.atoms = tensor(config.atoms)
        self.delta_atom = (config.categorical_v_max - config.categorical_v_min) / float(config.categorical_n_atoms - 1)

    def start_episode(self):
        self.actor.eval()

    def end_episode(self):
        self.actor.train()

    def update(self, transition):
        if len(self.replay) < self.config.replay_initial:
            return

        self.total_steps += 1

        transitions = self.replay.sample(self.config.batch_size)
        loss = self.compute_loss(transitions)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        if self.total_steps % self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def compute_loss(self, transitions):
        config = self.config
        states = config.state_normalizer(transitions.state)
        next_states = config.state_normalizer(transitions.next_state)
        with torch.no_grad():
            prob_next = self.target_network(next_states)['prob']
            q_next = (prob_next * self.atoms).sum(-1)
            if config.double_q:
                a_next = torch.argmax((self.network(next_states)['prob'] * self.atoms).sum(-1), dim=-1)
            else:
                a_next = torch.argmax(q_next, dim=-1)
            prob_next = prob_next[self.batch_indices, a_next, :]

        rewards = tensor(transitions.reward).unsqueeze(-1)
        masks = tensor(transitions.mask).unsqueeze(-1)
        atoms_target = rewards + config.discount ** config.n_step * masks * self.atoms.view(1, -1)
        atoms_target.clamp_(config.categorical_v_min, config.categorical_v_max)
        atoms_target = atoms_target.unsqueeze(1)
        target_prob = (1 - (atoms_target - self.atoms.view(1, -1, 1)).abs() / self.delta_atom).clamp(0, 1) * \
                      prob_next.unsqueeze(1)
        target_prob = target_prob.sum(-1)

        log_prob = self.network(states)['log_prob']
        actions = tensor(transitions.action).long()
        log_prob = log_prob[self.batch_indices, actions, :]
        KL = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(-1)
        return KL

    def reduce_loss(self, loss):
        return loss.mean()

class CategoricalDQNActor(DQNActor):
    def __init__(self, config):
        super().__init__(config)

    def _set_up(self):
        self.config.atoms = tensor(self.config.atoms)

    def compute_q(self, prediction):
        q_values = (prediction['prob'] * self.config.atoms).sum(-1)
        return to_np(q_values)

class CategoricalDQNAgent(DQNAgent):
    def __init__(self, config):
        super().__init__(config)

        self.actor = CategoricalDQNActor(config)
        self.actor.set_network(self.network)
        self.actor._set_up()