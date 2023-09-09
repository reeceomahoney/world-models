import numpy as np
import glob

from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def generate_gym_data(env_name, dir_name):
    envs = 1000
    eval_eps = 50
    vec_env = make_vec_env(env_name, n_envs=1)
    model = PPO.load("data/pendulum/ppo_pendulum")

    data = {"obs": [], "action": []}
    for env in range(envs):
        if env % 10 == 0:
            print(f"env: {env}")
        obs = vec_env.reset()
        obs_list, action_list = [], []
        for step in range(192):
            img = vec_env.render(mode="rgb_array")
            img = Image.fromarray(img).resize((64, 64))
            img = np.array(img)

            action, _states = model.predict(obs)
            obs_list.append(img)
            action_list.append(action)
            obs, rewards, dones, info = vec_env.step(action)

        data["obs"].append(np.array(obs_list))
        data["action"].append(np.array(action_list))

        if (env + 1) % eval_eps == 0: 
            data = {k: np.array(v) for k, v in data.items()}
            data["obs"] = np.transpose(data["obs"], (1, 0, 2, 3, 4))
            data["action"] = np.transpose(data["action"].squeeze(-1), (1, 0, 2))
            print(f"Saving data of shape: {data['obs'].shape}")

            if env == envs - 1:
                np.save(dir_name + "expert_eval.npy", data)
            else:
                np.save(dir_name + f"expert_{env}.npy", data)
            data = {"obs": [], "action": []}


if __name__ == "__main__":
    train = False
    env_name = "Pendulum"
    dir_name = "data/pendulum/"
    if train:
        generate_gym_data(env_name, dir_name)
    
    # get all expert data
    data = {"obs": [], "action": []}
    for file in glob.glob(dir_name + "batches/expert_*.npy"):
        if "eval" in file:
            continue
        data_batch = np.load(file, allow_pickle=True).item()
        data["obs"].append(data_batch["obs"])
        data["action"].append(data_batch["action"])

    data = {k: np.concatenate(v, axis=1) for k, v in data.items()}
    print(f"Saving data of shape: {data['obs'].shape}")
    np.save(dir_name + "expert.npy", data)

    # fill eval data with dummy
    data = {"obs": [], "action": []}
    np.save(dir_name + "expert_eval.npy", data)

