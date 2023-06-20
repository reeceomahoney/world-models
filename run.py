import argparse
import time
from collections import OrderedDict
from pathlib import Path

import torch

import world_models.common as common
from world_models.agent import Agent


# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='raisim')
parser.add_argument('--ditto', type=str, default=False)
parser.add_argument('--agent', type=str, default=None)
args = parser.parse_args()

# paths
home_path = Path(__file__).parent.absolute()
agent_path = home_path / args.agent

config, config_dict = common.init_config(Path(agent_path).parents[1] / 'config.yaml', args)

if config.env_name == 'raisim':
    env_driver = common.RaisimDriver(config, config_dict)
    env_driver.turn_on_visualization()
else:
    env_driver = common.GymDriver(config, render=True)

agent_state_dict = torch.load(agent_path, map_location=config.device)

# backwards compatibility (remember to comment out ensemble)
old_name = lambda key: key.startswith('actor') or key.startswith('critic') or key.startswith('slow')
agent_state_dict = OrderedDict([('task_' + k, v) if old_name(k) else (k, v) for k, v in agent_state_dict.items()])

agent = Agent(*env_driver.env_info(), config)
agent.load_state_dict(agent_state_dict, strict=False)

obs, h_t, action = env_driver.reset()
timer = common.Timer(config.control_dt, True)

for _ in range(5):
    for step in range(500):
        timer.start()
        obs, reward, done = env_driver(action)
        h_t, action = agent(h_t, obs)
        timer.end()
        if done or step == config.eval_steps - 1:
            obs, h_t, action = env_driver.reset()
