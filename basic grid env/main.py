import numpy as np
from base_env import GridEnvironment
from model import PPOAgent
import tensorflow as tf

env = GridEnvironment(size_x=4, size_y=4, rand_goal=False, rand_start=True)

num_actions = env.action_space.n
state = env.reset()

ppo_agent = PPOAgent(num_actions)
num_episodes = 1000
max_steps_per_episode = 200

best_reward = float('-inf')
early_stop_patience = 100
early_stop_counter = 0

for episode in range(num_episodes):
    state = env.reset()
    states, actions, rewards, values, old_probabilities = [], [], [], [], []

    for step in range(max_steps_per_episode):
        if step == 0:
            first_state = state
        state = np.reshape(state, [1, -1]).astype(np.float32)
        action = ppo_agent.get_action(state)
        value = ppo_agent.policy_network(state).numpy()[0]

        states.append(state)
        actions.append(action)
        values.append(value)
        old_probabilities.append(ppo_agent.policy_network(state).numpy()[0])

        state, reward, done = env.step(action)
        rewards.append(reward)

        if done:
            break

    states = np.vstack(states)
    values = [np.array(v) for v in values]
    values = np.concatenate(values + [np.array([0.0])])
    old_probabilities = np.vstack(old_probabilities)

    returns, advantages = ppo_agent.compute_advantages(rewards, values)

    ppo_agent.update_policy(states, actions, old_probabilities, advantages)

    if episode % 10 == 0:
        print(f"Episode: {episode}, Total Reward: {np.sum(rewards)}, Episode Length: {len(rewards)}, First State: {first_state}, Last State: {state}")

    if np.sum(rewards) > best_reward:
        best_reward = np.sum(rewards)
        policy_network_weights = ppo_agent.policy_network.get_weights()
        optimizer_checkpoint = tf.train.Checkpoint(optimizer=ppo_agent.optimizer)
        ppo_agent.policy_network.save_weights(f'weights\policy_network_weights_episode_{episode}.ckpt')
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= early_stop_patience:
        print(f"Training stopped early at episode {episode} due to lack of improvement with a best reward of {best_reward}.")
        break

env.close()
