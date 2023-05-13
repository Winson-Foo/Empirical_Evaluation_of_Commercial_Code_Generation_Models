## memory.py

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def add(self, transition):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch_indices = np.random.choice(len(self.buffer), batch_size)
        state_batch = []
        action_batch = []
        next_state_batch = []
        reward_batch = []
        done_batch = []
        for i in batch_indices:
            transition = self.buffer[i]
            state_batch.append(transition.state)
            action_batch.append(transition.action)
            next_state_batch.append(transition.next_state)
            reward_batch.append(transition.reward)
            done_batch.append(transition.done)
        return (state_batch, action_batch, next_state_batch, reward_batch, done_batch)