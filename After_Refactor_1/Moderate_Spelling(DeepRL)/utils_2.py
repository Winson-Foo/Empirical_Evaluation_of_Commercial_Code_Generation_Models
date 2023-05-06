# utils.py
def eval_agent(agent, env, num_episodes):
    total_reward = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
    return total_reward / num_episodes

def log_reward(reward, episode):
    pass

def save_agent(agent, save_path):
    pass