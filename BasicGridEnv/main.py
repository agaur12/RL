import numpy as np
from base_env import GridEnvironment
from stable_baselines3 import PPO, A2C
import tensorflow as tf
import os
from stable_baselines3.common.env_util import make_vec_env

rewards_ = []
len_rewards = []
episodes = 0
#models = ["PPO", "A2C"]
model_type = "PPO"
#policy = ["MlpPOLICY", "CnnPOLICY"]
policy_type = "MlpPOLICY"

env = make_vec_env(GridEnvironment, n_envs=4)

seasons = 100
num_episodes = 10000
max_steps_per_episode = 200

"""
best_reward = float('-inf')
early_stop_patience = 250
early_stop_counter = 0
"""
log_dir = "BasicGridEnv/logs"
models_dir = f"models/{model_type}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

match model_type:
    case "PPO":
        model = PPO()
    case "A2C":
        model = A2C()
