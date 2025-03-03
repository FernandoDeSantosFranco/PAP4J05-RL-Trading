import numpy as np
import random
import pandas as pd


class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0,
                 exploration_decay=0.995, min_exploration=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration

        # Inicializar la tabla Q con ceros
        self.q_table = {}

    def get_state_key(self, state):
        """Convierte un estado en una clave para la tabla Q."""
        return tuple(state.round(2))

    def choose_action(self, state):
        """Selecciona una acción usando una política epsilon-greedy."""
        state_key = self.get_state_key(state)

        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.action_size - 1)  # Explorar

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        return np.argmax(self.q_table[state_key])  # Explotar

    def update_q_table(self, state, action, reward, next_state):
        """Actualiza la tabla Q utilizando la ecuación de Bellman."""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        best_next_action = np.argmax(self.q_table[next_state_key])
        self.q_table[state_key][action] = (1 - self.learning_rate) * self.q_table[state_key][action] + \
                                          self.learning_rate * (
                                                      reward + self.discount_factor * self.q_table[next_state_key][
                                                  best_next_action])

    def decay_exploration(self):
        """Reduce el factor de exploración gradualmente."""
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)

    def save_q_table(self, filename):
        """Guarda la tabla Q en un archivo CSV."""
        df = pd.DataFrame.from_dict(self.q_table, orient='index')
        df.to_csv(filename)

    def load_q_table(self, filename):
        """Carga la tabla Q desde un archivo CSV."""
        df = pd.read_csv(filename, index_col=0)
        self.q_table = {tuple(map(float, key.strip('()').split(','))): row.values for key, row in df.iterrows()}

#Deep Q-Learning
#Utilizar redes neuronales
#TensorFlow