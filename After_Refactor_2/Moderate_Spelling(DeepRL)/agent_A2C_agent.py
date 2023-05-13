from typing import Callable
import torch.nn.functional as F


class A2CAgent(BaseAgent):
    def __init__(
        self,
        state_shape: Tuple[int],
        action_fn: Callable[[], ActionSpace],
        network_fn: Optional[Callable[[Tuple[int], int], nn.Module]] = None,
        optimizer_fn: Optional[Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]] = None,
        rollout_length: int = 128,
        discount: float = 0.99,
        use_gae: bool = True,
        gae_tau: float = 0.95,
        entropy_weight: float = 0.01,
        value_loss_weight: float = 0.5,
        gradient_clip: float = 0.5
    ):
        super().__init__(state_shape, action_fn)
        self.network = network_fn(state_shape, self.action_space.n) if network_fn is not None else MLP(state_shape[0], self.action_space.n)
        self.optimizer = optimizer_fn(self.network.parameters()) if optimizer_fn is not None else torch.optim.Adam(self.network.parameters(), lr=1e-4)
        self.rollout_length = rollout_length
        self.discount = discount
        self.use_gae = use_gae
        self.gae_tau = gae_tau
        self.entropy_weight = entropy_weight
        self.value_loss_weight = value_loss_weight
        self.gradient_clip = gradient_clip
        self.total_steps = 0
        self.states = None
        self.reset()
    
    def reset(self):
        self.states = self.task.reset()
        self.total_steps = 0
        
    def step(self):
        storage = self._rollout()
        entries = storage.extract(['log_probs', 'values', 'returns', 'advantages', 'entropy'])
        policy_loss = -(entries.log_probs * entries.advantages).mean()
        value_loss = 0.5 * ((entries.returns - entries.values)**2).mean()
        entropy_loss = entries.entropy.mean()

        self.optimizer.zero_grad()
        (policy_loss - self.entropy_weight * entropy_loss +
         self.value_loss_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.optimizer.step()
        
    def _rollout(self):
        storage = Storage(self.rollout_length)
        states = self.states
        for i in range(self.rollout_length):
            prediction = self.network(states)
            actions = prediction['a'].squeeze(1)
            next_states, rewards, dones, infos = self.task.step(to_np(actions))
            self.record_online_return(infos)
            rewards = self.reward_normalizer(rewards)
            storage.feed(prediction)
            storage.feed({
                'r': tensor(rewards).unsqueeze(-1),
                'mask': tensor(~dones).unsqueeze(-1)
            })
            states = next_states

        self.states = states
        prediction = self.network(states)
        storage.feed(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((self.n_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(self.rollout_length)):
            returns = storage.r[i] + self.discount * storage.mask[i] * returns
            if not self.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                delta_t = storage.r[i] + self.discount * storage.mask[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * self.gae_tau * self.discount * storage.mask[i] + delta_t
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()
            
        return storage


class MLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        a = self.fc2(x)
        v = self.fc3(x)
        log_probs = F.log_softmax(a, dim=1)
        dist = torch.distributions.Categorical(logits=log_probs)
        entropy = dist.entropy().mean()
        return {'a': a, 'log_probs': log_probs, 'v': v, 'dist': dist, 'entropy': entropy}


class Storage:
    def __init__(self, capacity):
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.index = 0
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.returns = []
        self.advantages = []
        self.entropies = []

    def feed(self, prediction):
        self.states.append(prediction['s'])
        self.actions.append(prediction['a'])
        self.log_probs.append(prediction['log_probs'])
        self.values.append(prediction['v'])

    def feed(self, data):
        self.rewards.append(data['r'])
        self.masks.append(data['mask'])

    def placeholder(self):
        if self.index == self.capacity:
            return
        self.states.append(None)
        self.actions.append(None)
        self.log_probs.append(None)
        self.values.append(None)
        self.rewards.append(None)
        self.masks.append(None)

    def extract(self, fields):
        fields = [field for field in fields if getattr(self, field)[0] is not None]
        data = namedtuple('Data', fields)
        data.log_probs = torch.cat(self.log_probs)
        data.values = torch.cat(self.values)
        data.rewards = torch.cat(self.rewards)
        data.masks = torch.cat(self.masks)
        self.returns = []
        self.advantages = []
        G = 0
        for i in reversed(range(self.capacity)):
            G = self.rewards[i] + self.masks[i] * self.discount * G
            self.returns.insert(0, G)
        returns = torch.cat(self.returns)
        for i in range(self.capacity):
            if not self.use_gae:
                A = self.returns[i] - self.values[i].detach()
            else:
                delta_t = self.rewards[i] + self.masks[i] * self.discount * self.values[i + 1].detach() - self.values[i].detach()
                A = A * self.gae_tau * self.discount * self.masks[i] + delta_t
            self.advantages.append(A)
        advantages = torch.cat(self.advantages)
        data.returns = returns
        data.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        data.entropy = torch.cat(self.entropies)
        return data