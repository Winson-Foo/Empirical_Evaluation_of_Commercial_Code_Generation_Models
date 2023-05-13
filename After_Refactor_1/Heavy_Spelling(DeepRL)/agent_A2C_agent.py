from ..network import *
from ..component import *
from .BaseAgent import *


class A2CAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states

        # step through the environment for the specified number of steps
        for _ in range(config.rollout_length):
            prediction = self.network(config.state_normalizer(states))
            actions = to_np(prediction['action'])
            next_states, rewards, terminals, info = self.task.step(actions)
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)

            # store the current step's information in the storage buffer
            storage.feed(prediction)
            storage.feed({'reward': tensor(rewards).unsqueeze(-1),
                         'mask': tensor(1 - terminals).unsqueeze(-1)})

            # update the current state of each worker
            states = next_states
            self.total_steps += config.num_workers

        self.states = states

        # add the final state's prediction to the storage buffer
        prediction = self.network(config.state_normalizer(states))
        storage.feed(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()

        # compute the advantages and returns for each time step
        for i in reversed(range(config.rollout_length)):
            returns = storage.reward[i] + config.discount * storage.mask[i] * returns

            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.reward[i] + config.discount * storage.mask[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.mask[i] + td_error

            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        # extract the required entries from the storage buffer
        entries = storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])

        # compute the losses and gradients
        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        entropy_loss = entries.entropy.mean()

        self.optimizer.zero_grad()

        total_loss = policy_loss - config.entropy_weight * entropy_loss + config.value_loss_weight * value_loss
        total_loss.backward()

        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()