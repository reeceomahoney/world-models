import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("Pendulum", n_envs=64)

model = PPO("MlpPolicy", vec_env, verbose=1)
while True:
    model.learn(total_timesteps=1000000)
    model.save("data/ppo_pendulum")
