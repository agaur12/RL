import time

import numpy as np
from env import GridEnvironment
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
import tensorflow as tf
import os
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import torch
from multiprocessing import freeze_support
import keyboard

rewards_ = []
len_rewards = []
episodes = 0
models = ["PPO", "A2C"]
model_type = models[0]
policy = ["MlpPolicy", "CnnPolicy"]
policy_type = policy[0]

env = None


def make_env(size_x: int = 10, size_y: int = 10):
    return GridEnvironment(size_x, size_y, rand_goal=False, rand_start=True)


if __name__ == '__main__':
    freeze_support()
    env = SubprocVecEnv([make_env for _ in range(4)])
    #env = DummyVecEnv([make_env])
    #env = make_env()

if env is None:
    env = make_vec_env(make_env, n_envs=4)
    #env = DummyVecEnv([make_env])
    #env = make_env()

log_dir = "logs"
models_dir = f"models/{model_type}"


if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


hyperparameters = {
    'policy': policy_type,
    'env': env,
    'learning_rate': 1e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 8,
    'gamma': 0.9,
    'gae_lambda': 0.99,
    'clip_range': 0.2,
    'clip_range_vf': None,
    'normalize_advantage': True,
    'ent_coef': 0.05,
    'vf_coef': 0.7,
    'max_grad_norm': 0.3,
    'use_sde': False,
    'sde_sample_freq': -1,
    'rollout_buffer_class': None,
    'rollout_buffer_kwargs': None,
    'target_kl': None,
    'stats_window_size': 100,
    'tensorboard_log': log_dir,
    'policy_kwargs': None,
    'verbose': 2,
    'seed': None,
    'device': 'cuda',
    '_init_setup_model': True
}

try:
    match model_type:
        case "PPO":
            model = PPO(**hyperparameters)
except:
    raise Exception("No env defined")

model = model.load(f"{models_dir}/model_14", env=env)

seasons = 5
num_episodes = 10000
for season in range(seasons):
    model.learn(total_timesteps=num_episodes, reset_num_timesteps=False, tb_log_name='PPO_0', progress_bar=True)

    model.save(f"{models_dir}/model_{season}")
    
print("Model trained")

"""
env = DummyVecEnv([make_env])

model = PPO.load(f"{models_dir}/model_14", env=env)
print("Model loaded")
"""
state = env.reset()
for i in range(100000000):
    env.render()
    action, _ = model.predict(state)
    if env == make_env():
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    else:
        state, reward, done, info = env.step(action)
    time.sleep(1)
    if done.any():
        obs = env.reset()
        print("Resetting")

env.close()
