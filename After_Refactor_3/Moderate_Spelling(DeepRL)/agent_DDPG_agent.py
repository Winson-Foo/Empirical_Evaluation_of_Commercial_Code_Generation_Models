class DDPGAgent:
    def __init__(self, config):
        self.config = config
        self.task = config.task_fn()
        self.network = ActorCriticNetwork(config)
        self.target_network = ActorCriticNetwork(config)
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay_buffer = ReplayBuffer(config)
        self.random_process = GaussianProcess(config)

    def eval_step(self, state):
        state_normalized = self.config.state_normalizer(state)
        action = self.network.select_action(state_normalized)
        return to_np(action)

    def train_step(self):
        config = self.config

        transitions = self.replay_buffer.sample()
        states, actions, rewards, next_states, mask = transitions

        q_next = self.target_network.calculate_q_values(next_states)
        q_next = config.discount * mask * q_next
        q_next.add_(rewards)
        q_next = q_next.detach()

        critic_loss = self.network.calculate_critic_loss(states, actions, q_next)
        self.network.optimize_critic(critic_loss)

        policy_loss = self.network.calculate_policy_loss(states)
        self.network.optimize_actor(policy_loss)

        self.network.update_target_network(self.target_network)

    def step(self):
        config = self.config
        state = self.task.reset()

        for t in range(config.max_episode_steps):
            state_normalized = config.state_normalizer(state)
            action = self.network.select_action(state_normalized)
            action = to_np(action)
            action += self.random_process.sample()
            action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
            next_state, reward, done, info = self.task.step(action)
            next_state_normalized = config.state_normalizer(next_state)
            self.replay_buffer.add(state_normalized, action, reward, next_state_normalized, 1-done)

            if done:
                break

            state = next_state

            if self.replay_buffer.size() >= config.batch_size:
                self.train_step()

        self.random_process.reset_states()