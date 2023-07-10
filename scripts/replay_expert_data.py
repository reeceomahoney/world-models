import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.append(str(Path(__file__).parents[1].absolute()))
import world_models.common as common

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='raisim')
parser.add_argument('--ditto', type=str, default=True)
args = parser.parse_args()

# paths
home_path = Path(__file__).parents[1].absolute()

config, config_dict = common.init_config(
    home_path / 'world_models/config.yaml', args)

if config.env_name == 'raisim':
    env_driver = common.RaisimDriver(config, config_dict)
    env_driver.turn_on_visualization()
else:
    env_driver = common.GymDriver(config, render=True)

obs_dim = 36
expert_path = home_path / 'world_models/expert_data' / config.ditto_dataset / \
    'expert.npy'
expert_data = common.load_expert_data(expert_path, obs_dim, config.device)
expert_sampler = common.ExpertSampler(config, expert_data)

expert_init_path = expert_path.parent / 'expert_init.npy'
expert_init_data = torch.tensor(np.load(expert_init_path)).to(
    torch.float32).to(config.device)
env_driver.load_expert_data(expert_init_data)

obs, h_t, action = env_driver.reset()
timer = common.Timer(config.control_dt, True)

for _ in range(20):
    sample = next(expert_sampler)
    for step in range(config.batch_length):
        timer.start()
        obs, reward, done = env_driver(sample['action'][step, 0].unsqueeze(0))
        timer.end()
        if done or step == config.eval_steps - 1:
            obs, h_t, action = env_driver.reset()
