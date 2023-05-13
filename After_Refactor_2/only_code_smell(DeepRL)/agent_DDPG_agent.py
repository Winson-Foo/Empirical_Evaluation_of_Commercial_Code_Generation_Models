def __init__(self, task, network, replay, random_process, config):
    self.task = task
    self.network = network
    self.target_network = network.clone()
    self.replay = replay
    self.random_process = random_process
    self.total_steps = 0
    self.state = None
    self.config = config

def eval_step(self, state):
    state = self.config.state_normalizer(state, read_only=True)
    action = self.network(state)
    return to_np(self.config.action_normalizer(action))
    
def step(self):
    if self.state is None:
        self.random_process.reset_states()
        self.state = self.task.reset()
        self.state = self.config.state_normalizer(self.state)

    if self.total_steps < self.config.warm_up:
        action = [self.task.action_space.sample()]
    else:
        action = self.network(self.state)
        action = to_np(action)
        action += self.random_process.sample()
    action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
    next_state, reward, done, info = self.task.step(action)
    next_state = self.config.state_normalizer(next_state)
    self.record_online_return(info)
    reward = self.config.reward_normalizer(reward)

    self.replay.feed(dict(
        state=self.state,
        action=action,
        reward=reward,
        next_state=next_state,
        mask=1-np.asarray(done, dtype=np.int32),
    ))

    if done[0]:
        self.random_process.reset_states()
    self.state = next_state
    self.total_steps += 1

    if self.replay.size() >= self.config.warm_up:
        transitions = self.replay.sample()
        states = tensor(transitions.state)
        actions = tensor(transitions.action)
        rewards = tensor(transitions.reward).unsqueeze(-1)
        next_states = tensor(transitions.next_state)
        mask = tensor(transitions.mask).unsqueeze(-1)

        phi_next = self.target_network.feature(next_states)
        a_next = self.target_network.actor(phi_next)
        q_next = self.target_network.critic(phi_next, a_next)
        q_next = self.config.discount * mask * q_next
        q_next.add_(rewards)
        q_next = q_next.detach()
        phi = self.network.feature(states)
        q = self.network.critic(phi, actions)
        critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()

        self.network.zero_grad()
        critic_loss.backward()
        self.network.critic_opt.step()

        phi = self.network.feature(states)
        action = self.network.actor(phi)
        policy_loss = -self.network.critic(phi.detach(), action).mean()

        self.network.zero_grad()
        policy_loss.backward()
        self.network.actor_opt.step()

        self.soft_update(self.target_network, self.network)

phi_next, a_next, q_next = self.target_network(next_states)
q_next = self.config.discount * mask * q_next.detach() + rewards
phi, q = self.network(states, actions)
critic_loss = F.mse_loss(q_next, q)
self.network.critic_opt.zero_grad()
critic_loss.backward()
self.network.critic_opt.step()

phi, action = self.network(states)
policy_loss = -self.network.critic(phi.detach(), action).mean()
self.network.actor_opt.zero_grad()
policy_loss.backward()
self.network.actor_opt.step()

def __init__(self, task, network, replay_buffer, random_process, config):
    self.task = task
    self.network = network
    self.target_network = network.clone()
    self.replay_buffer = replay_buffer
    self.random_process = random_process
    self.total_steps = 0
    self.state = None
    self.config = config

def eval_step(self, state):
    state = self.config.state_normalizer(state, read_only=True)
    action = self.network(state)
    return to_np(self.config.action_normalizer(action))

def step(self):
    if self.state is None:
        self.random_process.reset_states()
        self.state = self.task.reset()
        self.state = self.config.state_normalizer(self.state)

    if self.total_steps < self.config.warm_up:
        action = [self.task.action_space.sample()]
    else:
        action = self.network(self.state)
        action = to_np(action)
        action += self.random_process.sample()
    action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
    next_state, reward, done, info = self.task.step(action)
    next_state = self.config.state_normalizer(next_state)
    self.record_online_return(info)
    reward = self.config.reward_normalizer(reward)

    self.replay_buffer.feed(dict(
        state=self.state,
        action=action,
        reward=reward,
        next_state=next_state,
        mask=1-np.asarray(done, dtype=np.int32),
    ))

    if done[0]:
        self.random_process.reset_states()
    self.state = next_state
    self.total_steps += 1

    if self.replay_buffer.size() >= self.config.warm_up:
        transitions = self.replay_buffer.sample()
        states = tensor(transitions.state)
        actions = tensor(transitions.action)
        rewards = tensor(transitions.reward).unsqueeze(-1)
        next_states = tensor(transitions.next_state)
        masks = tensor(transitions.mask).unsqueeze(-1)

        next_phi = self.target_network.feature(next_states)
        next_a = self.target_network.actor(next_phi)
        next_q = self.target_network.critic(next_phi, next_a)
        next_q = self.config.discount_factor * masks * next_q.detach() + rewards
        next_phi_, q = self.network(states, actions)
        critic_loss = F.mse_loss(next_q, q)

        self.network.critic_opt.zero_grad()
        critic_loss.backward()
        self.network.critic_opt.step()

        phi, action = self.network(states)
        policy_loss = -self.network.critic(phi.detach(), action).mean()
        self.network.actor_opt.zero_grad()
        policy_loss.backward()
        self.network.actor_opt.step()

        self.soft_update(self.target_network, self.network)

def __init__(self, task, network, replay_buffer, random_process, config):
    self.task = task
    self.network = network
    self.target_network = network.clone()
    self.replay_buffer = replay_buffer
    self.random_process = random_process
    self.total_steps = 0
    self.state = None
    self.config = config

def eval_step(self, state):
    # set state to read-only mode and normalize
    state = self.config.state_normalizer(state, read_only=True)
    # compute action using the network and normalize
    action = self.network(state)
    return to_np(self.config.action_normalizer(action))

def step(self):
    # if starting first step
    if self.state is None:
        self.random_process.reset_states()
        self.state = self.task.reset()
        self.state = self.config.state_normalizer(self.state)

    # if still in warm-up period
    if self.total_steps < self.config.warm_up:
        action = [self.task.action_space.sample()]
    # otherwise, take action based on current state
    else:
        action = self.network(self.state)
        action = to_np(action)
        action += self.random_process.sample()
    # clip action based on action space limits
    action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
    # take step using action and observe next state, reward, and whether done
    next_state, reward, done, info = self.task.step(action)
    # normalize next state
    next_state = self.config.state_normalizer(next_state)
    # record online return based on current step info
    self.record_online_return(info)
    # normalize reward
    reward = self.config.reward_normalizer(reward)

    # add experience to replay buffer
    self.replay_buffer.feed(dict(
        state=self.state,
        action=action,
        reward=reward,
        next_state=next_state,
        mask=1-np.asarray(done, dtype=np.int32),
    ))

    # if episode is done, reset random process states
    if done[0]:
        self.random_process.reset_states()
    # update current state and total steps taken
    self.state = next_state
    self.total_steps += 1

    # if replay buffer has enough experiences
    if self.replay_buffer.size() >= self.config.warm_up:
        transitions = self.replay_buffer.sample()
        states = tensor(transitions.state)
        actions = tensor(transitions.action)
        rewards = tensor(transitions.reward).unsqueeze(-1)
        next_states = tensor(transitions.next_state)
        masks = tensor(transitions.mask).unsqueeze(-1)

        # compute next state-action value estimate
        next_phi = self.target_network.feature(next_states)
        next_a = self.target_network.actor(next_phi)
        next_q = self.target_network.critic(next_phi, next_a)
        next_q = self.config.discount_factor * masks * next_q.detach() + rewards
        # compute current state-action value estimate and critic loss
        next_phi_, q = self.network(states, actions)
        critic_loss = F.mse_loss(next_q, q)

        # update critic using critic loss
        self.network.critic_opt.zero_grad()
        critic_loss.backward()
        self.network.critic_opt.step()

        # compute policy loss and update actor using it
        phi, action = self.network(states)
        policy_loss = -self.network.critic(phi.detach(), action).mean()
        self.network.actor_opt.zero_grad()
        policy_loss.backward()
        self.network.actor_opt.step()

        # soft update target network parameters using the online network's parameters
        self.soft_update(self.target_network, self.network)