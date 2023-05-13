import multiprocessing as mp

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return Transition(*zip(*samples))

    def __len__(self):
        return len(self.buffer)

class Transition:
    def __init__(self, state, action, reward, next_state, mask):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.mask = mask

class StateNormalizer:
    def __init__(self, size):
        self.size = size
        self.mean = torch.zeros(size)
        self.mean_diff = torch.zeros(size)
        self.var = torch.zeros(size)
        self.std = torch.ones(size)
        self.read_only = False

    def __call__(self, state):
        if self.read_only:
            normalized_state = (state - self.mean) / (self.std + 1e-8)
        else:
            self.mean[:] = state.mean(axis=0)
            self.mean_diff[:] = state.mean(axis=0)
            self.var[:] = (state - self.mean).pow(2).sum(axis=0) / (state.size()[0] - 1)
            self.std[:] = self.var.sqrt()
            normalized_state = (state - self.mean) / (self.std + 1e-8)
        return normalized_state

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False
