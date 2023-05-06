
def compute_q(self, prediction: dict) -> np.ndarray:
    q_values = to_np(prediction['q'])
    return q_values

def eval_step(self, state: np.ndarray) -> np.ndarray:
    self.config.state_normalizer.set_read_only()
    state = self.config.state_normalizer(state)
    q = self.network(state)['q']
    action = to_np(q.argmax(-1))
    self.config.state_normalizer.unset_read_only()
    return action

# At the beginning of the file
NOISY_LINEAR = True
DOUBLE_Q = True
EXPLORATION_STEPS = 10000
TARGET_NETWORK_UPDATE_FREQ = 1000
SGD_UPDATE_FREQUENCY = 4
REPLAY_EPS = 1e-6
REPLAY_ALPHA = 0.5
REPLAY_BETA = 0.4

class DQNActor:
    def __init__(self, config: Dict):
        self.config = config
        self.start()

    # ...

class DQNAgent:
    def __init__(self, config: Dict):
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)
        self.total_steps = 0

def get_q_values(network: nn.Module, state: np.ndarray) -> torch.Tensor:
    return network(state)['q']

# ...
    
class DQNAgent:
    # ...
    def compute_loss(self, transitions: Transition) -> torch.Tensor:
        config = self.config
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)
        with torch.no_grad():
            q_next = get_q_values(self.target_network, next_states).detach()
            if self.config.double_q:
                best_actions = torch.argmax(get_q_values(self.network, next_states), dim=-1)
                q_next = q_next.gather(1, best_actions.unsqueeze(-1)).squeeze(1)
            else:
                q_next = q_next.max(1)[0]
        masks = tensor(transitions.mask)
        rewards = tensor(transitions.reward)
        q_target = rewards + self.config.discount ** config.n_step * q_next * masks
        actions = tensor(transitions.action).long()
        q = get_q_values(self.network, states)
        q = q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = q_target - q
        return loss
  
    def update_network(self, transitions: Transition, sampling_probs: Optional[np.ndarray] = None) -> None:
        config = self.config
        loss = self.compute_loss(transitions)
        if isinstance(transitions, PrioritizedTransition):
            priorities = loss.abs().add(config.replay_eps).pow(config.replay_alpha)
            idxs = tensor(transitions.idx).long()
            self.replay.update_priorities(zip(to_np(idxs), to_np(priorities)))
            weights = (sampling_probs * len(sampling_probs)).add(config.replay_eps).pow(-config.replay_beta)
            weights = weights / weights.max()
            loss = loss.mul(weights)

        loss = self.reduce_loss(loss)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        with config.lock:
            self.optimizer.step()

    def step(self) -> None:
        config = self.config
        transitions = self.actor.step()
        for states, actions, rewards, next_states, dones, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            self.replay.feed(dict(
                state=np.array([s[-1] if isinstance(s, LazyFrames) else s for s in states]),
                action=actions,
                reward=[config.reward_normalizer(r) for r in rewards],
                mask=1 - np.asarray(dones, dtype=np.int32),
            ))

        if self.total_steps > EXPLORATION_STEPS:
            transitions = self.replay.sample()
            if NOISY_LINEAR:
                self.target_network.reset_noise()
                self.network.reset_noise()
            self.update_network(transitions)

        if self.total_steps / SGD_UPDATE_FREQUENCY % TARGET_NETWORK_UPDATE_FREQ == 0:
            self.target_network.load_state_dict(self.network.state_dict())