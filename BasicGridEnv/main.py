import numpy as np
from base_env import GridEnvironment
from stable_baselines3 import PPO, A2C
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
    return GridEnvironment(size_x, size_y, rand_goal=True, rand_start=True)


if __name__ == '__main__':
    freeze_support()
    n_env = 4
    env = SubprocVecEnv([make_env for _ in range(n_env)])

if env is None:
    env = make_vec_env(make_env, n_envs=4)

seasons = 5
num_episodes = 10000

learning_rate = 1e-1
n_steps = 200
n_epochs = 5
batch_size = 32
gamma = 0.8
gae_lambda = 0.95
clip_range = clip_range_vf = 0.2
normalize_advantage = True
ent_coef = 0.01
vf_coef = 0.4
max_grad_norm = 0.5
use_sde = True
sde_sample_freq = -1
rollout_buffer_class = None
target_kl = 0.1
use_rms_prop = False
rms_prop_eps = 1e-5
custom_policy_kwargs = dict(
    net_arch=[256, 64, 16, 4]
)

log_dir = "logs"
models_dir = f"models/{model_type}"



if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

try:
    match model_type:
        case "PPO":
            model = PPO(policy_type, env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
                        n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range,
                        clip_range_vf=clip_range_vf, normalize_advantage=normalize_advantage, ent_coef=ent_coef,
                        vf_coef=vf_coef, max_grad_norm=max_grad_norm, rollout_buffer_class=rollout_buffer_class,
                        target_kl=target_kl, tensorboard_log=log_dir, verbose=1, policy_kwargs=custom_policy_kwargs)
        case "A2C":
            model = A2C(policy_type, env, learning_rate=learning_rate, n_steps=n_steps, gamma=gamma,
                        gae_lambda=gae_lambda,
                        normalize_advantage=normalize_advantage, ent_coef=ent_coef, vf_coef=vf_coef,
                        max_grad_norm=max_grad_norm, use_rms_prop=use_rms_prop, rms_prop_eps=rms_prop_eps,
                        use_sde=use_sde,
                        sde_sample_freq=sde_sample_freq, tensorboard_log=log_dir, verbose=1,
                        policy_kwargs=custom_policy_kwargs)
except:
    raise Exception("No env defined")


for season in range(seasons):
    model.learn(total_timesteps=num_episodes, progress_bar=True)
    model.save(f"{models_dir}/model_{season}")


while not keyboard.is_pressed('q'):
    pass
env = None
env = DummyVecEnv([make_env])
if env is None:
    env = make_vec_env(make_env, n_envs=1)
obs = env.reset()
for i in range(100000000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    print(reward)
    if done:
        obs = env.reset()
        print("Resetting")
    if keyboard.is_pressed('q'):
        break

env.close()
