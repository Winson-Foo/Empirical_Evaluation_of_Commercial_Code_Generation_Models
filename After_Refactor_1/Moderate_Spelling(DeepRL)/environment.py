# environment.py
class Environment:
    def __init__(self, name):
        self.name = name
        self.state_dim = None
        self.action_dim = None

    def step(self, action):
        pass

    def reset(self):
        pass