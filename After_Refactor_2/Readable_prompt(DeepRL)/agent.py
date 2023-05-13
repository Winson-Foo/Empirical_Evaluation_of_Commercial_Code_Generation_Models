# agent.py
class DQNAgent(BaseAgent):
    def __init__(self, config, actor):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = actor

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)
        self.total_steps = 0

    def close(self):
        close_obj(self.replay)

    def learn(self):
        transitions = self.replay.sample()
        if self.config.noisy_linear:
            self.target_network.reset_noise()
            self.network.reset_noise()
        loss = self.compute_loss(transitions)
        if isinstance(transitions, PrioritizedTransition):
            priorities = loss.abs().add(self.config.replay_eps).pow(self.config.replay_alpha)
            idxs = tensor(transitions.idx).long()
            self.replay.update_priorities(zip(to_np(idxs), to_np(priorities)))
            sampling_probs = tensor(transitions.sampling_prob)
            weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6).pow(-self.config.replay_beta())
            weights = weights / weights.max()
            loss = loss.mul(weights)

        loss = self.reduce_loss(loss)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        with self.config.lock:
            self.optimizer.step()

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return transitions

    def compute_loss(self, transitions):
        config = self.config
        states = config.state_normalizer(transitions.state)
        next_states = config.state_normalizer(transitions.next_state)
        with torch.no_grad():
            q_next = self.target_network(next_states)['q'].detach()
            if config.double_q:
                best_actions = torch.argmax(self.network(next_states)['q'], dim=-1)
                q_next = q_next.gather(1, best_actions.unsqueeze(-1)).squeeze(1)
            else:
                q_next = q_next.max(1)[0]
        masks = tensor(transitions.mask)
        rewards = tensor(transitions.reward)
        q_target = rewards + config.discount ** config.n_step * q_next * masks
        actions = tensor(transitions.action).long()
        q = self.network(states)['q']
        q = q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = q_target - q
        return loss

    def step(self):
        transitions = self.actor.step()
        self.total_steps += len(transitions)
        for states, actions, rewards, next_states, dones, info in transitions:
            self.record_online_return(info)
            self.replay.feed(dict(
                state=np.array([s[-1] if isinstance(s, LazyFrames) else s for s in states]),
                action=actions,
                reward=[self.config.reward_normalizer(r) for r in rewards],
                mask=1 - np.asarray(dones, dtype=np.int32),
            ))

        return transitions