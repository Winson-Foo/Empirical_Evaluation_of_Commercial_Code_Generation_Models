class Storage:
    def __init__(self, rollout_length):
        self.rollout_length = rollout_length
        self.reset()

    def reset(self):
        self.state = []
        self.action = []
        self.log_pi_a = []
        self.reward = []
        self.mask = []
        self.v = []
        self.advantage = []
        self.ret = []

    def feed(self, predictions):
        self.state.append(predictions['state'])
        self.action.append(predictions['action'])
        self.log_pi_a.append(predictions['log_pi_a'])
        self.v.append(predictions['v'])

    def placeholder(self):
        self.advantage.append(torch.zeros(1, 1))
        self.ret.append(torch.zeros(1, 1))

    def extract(self, fields):
        fields = [getattr(self, field) for field in fields]
        return tuple(torch.cat(field, dim=0) for field in fields)