def train_agent(env, agent, episodes=1000):
    rewards_history = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        agent.decay_exploration()
        rewards_history.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(
                f"Episodio {episode + 1}/{episodes}, Recompensa Total: {total_reward:.2f}, Exploración: {agent.exploration_rate:.4f}")

    return rewards_history

# Entrenar el agente en el entorno
# env = TradingEnvironment(data)
# agent = QLearningAgent(state_size=7, action_size=3)
# train_rewards = train_agent(env, agent, episodes=1000)

# Guardar la tabla Q después del entrenamiento
# agent.save_q_table("q_table_trading.csv")
