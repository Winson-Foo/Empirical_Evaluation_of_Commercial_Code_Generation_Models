# main.py
if __name__ == '__main__':
    config = Config()
    config.parse()
    env = Environment(config.task_name)
    agent = Agent(config)
    replay = ReplayMemory(config)
    network = Network(config)

    for episode in range(config.max_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            experience = (state, action, reward, next_state, done)
            replay.add(experience)
            if len(replay) >= config.batch_size:
                batch = replay.sample(config.batch_size)
                agent.train(batch)
                if network.requires_update:
                    loss = network.update(batch)
                    replay.update_priority(loss)
            state = next_state

        if episode % config.eval_interval == 0:
            reward = eval_agent(agent, env, config.eval_episodes)
            log_reward(reward, episode)

    save_agent(agent, config.save_path)