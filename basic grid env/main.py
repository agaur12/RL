import numpy as np
from base_env import GridEnvironment
#from old_model import PPOAgent
from stable_baselines3 import PPO
#import matplotlib.pyplot as plt

rewards_ = []
len_rewards = []
episodes = 0

env = GridEnvironment(size_x=11, size_y=11, rand_goal=False, rand_start=True)

num_actions = env.action_space.n
state = env.reset()

ppo_agent = PPOAgent(num_actions)
num_episodes = 10000
max_steps_per_episode = 200

best_reward = float('-inf')
early_stop_patience = 250
early_stop_counter = 0

checkpoint_path = "weights/policy_network_weights.ckpt"


"""
def graph(episodes=episodes, rewards=rewards_, len_rewards=len_rewards):
    plt.figure(1)
    plt.plot(range(episodes), rewards)
    plt.title(f"Total Reward vs Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()
    plt.figure(2)
    plt.plot(range(episodes), len_rewards)
    plt.title('Episode Length Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.show()
"""

try:
    ppo_agent.load_weights(checkpoint_path)
    print("Weights loaded successfully.")
except:
    print("No weights to load.")

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
            rewards_.append(np.sum(rewards))
            len_rewards.append(len(rewards))
            episodes += 1
            break

    states = np.vstack(states)
    values = [np.array(v) for v in values]
    values = np.concatenate(values + [np.array([0.0])])
    old_probabilities = np.vstack(old_probabilities)

    returns, advantages = ppo_agent.compute_advantages(rewards, values)

    ppo_agent.update_policy(states, actions, old_probabilities, advantages)

    if episode % 10 == 0:
        print(
            f"Episode: {episode}, Total Reward: {np.sum(rewards)}, Episode Length: {len(rewards)}, First State: {first_state}, Last State: {state}")

    if np.sum(rewards) > best_reward:
        print(
            f"Episode: {episode}, Total Reward: {np.sum(rewards)}, Episode Length: {len(rewards)}, First State: {first_state}, Last State: {state}")
        policy_network_weights = ppo_agent.policy_network.get_weights()
        optimizer_checkpoint = tf.train.Checkpoint(optimizer=ppo_agent.optimizer)
        best_reward = np.sum(rewards)
        early_stop_counter = 0
    elif np.sum(rewards) == best_reward:
        print(
            f"Episode: {episode}, Total Reward: {np.sum(rewards)}, Episode Length: {len(rewards)}, First State: {first_state}, Last State: {state}")
        early_stop_counter += 1
    else:
        None

    if early_stop_counter >= early_stop_patience:
        print(
            f"Training stopped early at episode {episode} due to lack of improvement with a best reward of {best_reward} and episode length of {len(rewards)}.")
        break

ppo_agent.policy_network.save_weights(checkpoint_path)
#print(episodes, rewards_, len_rewards)
#graph()
env.close()
