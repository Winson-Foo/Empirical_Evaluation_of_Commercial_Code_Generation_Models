from ..network import *
from ..component import *
from ..utils import *
from .base_agent import *

# Named constants
EPSILON = "epsilon"
NUM_WORKERS = "num_workers"
DISCOUNT = "discount"
TARGET_NETWORK_UPDATE_FREQ = "target_network_update_freq"
GRADIENT_CLIP = "gradient_clip"
ROLLOUT_LENGTH = "rollout_length"

class NStepDQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        config = self.config
        storage = self._collect_rollout_data()

        self.states = self.task.step(actions)

        storage.placeholder()

        # Compute the expected Q-value using the target network
        with torch.no_grad():
            ret = self.target_network(config.state_normalizer(self.states))[Q_KEY].detach()
            ret = torch.max(ret, dim=1, keepdim=True)[0]

        # Compute the expected return for each time step in the rollout
        for i in reversed(range(config.rollout_length)):
            ret = storage.reward[i] + config.discount * storage.mask[i] * ret
            storage.ret[i] = ret

        # Compute the loss and update the network
        self._optimize_network(storage)

        self.total_steps += config.num_workers

    def _collect_rollout_data(self):
        """
        Collect rollout data and store it in a Storage object.
        """

        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states

        for _ in range(config.rollout_length):
            q = self.network(config.state_normalizer(states))[Q_KEY]

            epsilon = config.random_action_prob(config.num_workers)
            actions = epsilon_greedy(epsilon, to_np(q))

            next_states, rewards, terminals, info = self.task.step(actions)
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)

            storage.feed({
                Q_KEY: q,
                ACTION_KEY: tensor(actions).unsqueeze(-1).long(),
                REWARD_KEY: tensor(rewards).unsqueeze(-1),
                MASK_KEY: tensor(1 - terminals).unsqueeze(-1)
            })

            states = next_states

        return storage

    def _optimize_network(self, storage):
        """
        Compute the loss and update the network weights.
        """

        config = self.config
        entries = storage.extract([Q_KEY, ACTION_KEY, RETURN_KEY])

        loss = 0.5 * (entries[Q_KEY].gather(1, entries[ACTION_KEY]) - entries[RETURN_KEY]).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()