import tensorflow as tf
import numpy as np


class PolicyNetwork(tf.keras.Model):
    def __init__(self, num_actions: int, hidden_units: int = 256):  
        super(PolicyNetwork, self).__init__()
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(hidden_units, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(hidden_units / num_actions, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense((hidden_units / num_actions) / num_actions, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(num_actions, activation="softmax")
        ])
    def call(self, state):
        x = self.model(state)
        return x


class PPOAgent:
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.policy_network = PolicyNetwork(num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.epsilon = 1

    def load_weights(self, checkpoint_path):
        dummy_model = PolicyNetwork(self.num_actions)
        dummy_model.load_weights(checkpoint_path)
        self.policy_network.set_weights(dummy_model.get_weights())

    def get_action(self, state):
        normalized_state = self.normalize_state(state)
        probabilities = self.policy_network.call(normalized_state).numpy()[0]

        action = np.random.choice(self.num_actions, p=probabilities)
        return action

    def normalize_state(self, state):
        normalized_state = (state - np.mean(state)) / (np.std(state) + 1e-8)
        return normalized_state

    def compute_advantages(self, rewards, values, gamma=0.8, lambbda_=0.95):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        adv = 0.0
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + gamma * adv
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            adv = delta + gamma * lambbda_ * adv
        values = values[:len(rewards)]
        advantages = returns - values
        return returns, advantages

    def update_policy(self, states, actions, old_probabilities, advantages, entropy_coeff=0.01, epsilon=None):
        epsilon = self.epsilon if epsilon is None else epsilon
        with tf.GradientTape() as tape:
            new_probabilities = self.policy_network.call(states)
            action_masks = tf.one_hot(actions, self.num_actions)
            ratio = tf.reduce_sum(action_masks * new_probabilities, axis=1) / \
                    tf.reduce_sum(action_masks * old_probabilities, axis=1)
            surrogate1 = ratio * advantages
            surrogate2 = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * advantages
            entropy = -tf.reduce_sum(new_probabilities * tf.math.log(new_probabilities + 1e-10), axis=1)
            loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2) - entropy_coeff * entropy)
        self.epsilon = epsilon * 0.5

        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
