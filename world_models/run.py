import argparse
import os
import time
from collections import OrderedDict

import torch

import common as common
from agent import Agent

# paths
home_path = os.path.dirname(os.path.realpath(__file__))

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='raisim')
parser.add_argument('--logdir', type=str, default=None)
parser.add_argument('--agent', type=str, default=None)
args = parser.parse_args()

log_dir = f'{home_path}/logs/{args.env}/{args.logdir}'
config, config_dict = common.init_config(log_dir + '/config.yaml', args.env)

if config.env_name == 'raisim':
    env_driver = common.RaisimDriver(config, config_dict)
    env_driver.turn_on_visualization()
else:
    env_driver = common.GymDriver(config, render=True)

agent_state_dict = torch.load(f'{log_dir}/models/agent_{args.agent}.pt', map_location=config.device)

# backwards compatibility (remember to comment out ensemble)
old_name = lambda key: key.startswith('actor') or key.startswith('critic') or key.startswith('slow')
agent_state_dict = OrderedDict([('task_' + k, v) if old_name(k) else (k, v) for k, v in agent_state_dict.items()])

agent = Agent(*env_driver.env_info(), config)
agent.load_state_dict(agent_state_dict, strict=False)

obs, h_t, action = env_driver.reset()
for _ in range(config.eval_eps):
    print("foo")
    for step in range(config.eval_steps):
        start = time.time()
        obs, reward, done = env_driver(action)
        h_t, action = agent(h_t, obs)
        delta = time.time() - start
        sim_delta = 0.04 * config.action_repeat
        if delta < sim_delta:
            time.sleep(sim_delta - delta)
        if done or step == config.eval_steps - 1:
            obs, h_t, action = env_driver.reset()
