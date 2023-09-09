import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import dreamer


# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="raisim")
parser.add_argument("--ditto", type=str, default=True)
parser.add_argument("--agent", type=str, default=None)
parser.add_argument("--plot", type=str, default=False)
args = parser.parse_args()

# paths
home_path = Path(__file__).parents[1].absolute()
agent_path = home_path / args.agent

config, config_dict = dreamer.utils.init_config(
    Path(agent_path).parents[1] / "config.yaml", args
)

if config.env_name == "raisim":
    env_driver = dreamer.driver.RaisimDriver(config, config_dict)
    env_driver.turn_on_visualization()
else:
    env_driver = dreamer.driver.GymDriver(config, render=True)

expert_init_path = (
    home_path / "data" / config.ditto_dataset / "expert_init.npy"
)
expert_init_data = (
    torch.tensor(np.load(expert_init_path)).to(torch.float32).to(config.device)
)

env_driver.load_expert_data(expert_init_data)

agent_state_dict = torch.load(agent_path, map_location=config.device)

agent = dreamer.agent.Agent(*env_driver.env_info(), None, config)
agent.load_state_dict(agent_state_dict, strict=False)

obs, h_t, action = env_driver.reset()
timer = dreamer.utils.Timer(config.control_dt, True)

joint_angles, joint_targets = [], []
for _ in range(5):
    for step in range(500):
        timer.start()
        obs, reward, done = env_driver(action)
        h_t, action = agent(h_t, obs, deterministic=True)
        # joint_angles.append(dreamer.utils.symexp(obs[1, 3:15]).cpu().numpy())
        # joint_targets.append(action[0].cpu().numpy())
        timer.end()
        if done or step == config.eval_steps - 1:
            obs, h_t, action = env_driver.reset()

if args.plot:
    joint_angles = np.array(joint_angles)
    joint_targets = np.array(joint_targets)

    plt.figure()
    for i in range(12):
        plt.subplot(4, 3, i + 1)
        plt.plot(joint_angles[:, i], label=f"joint angle {i}")
        plt.plot(joint_targets[:, i], label=f"joint target {i}")
        plt.legend()
    plt.legend()
    plt.show()
