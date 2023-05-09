import argparse
import os
import pickle

import torch

import world_models.dreamer as dreamer
import world_models.ditto as ditto

from world_models.agent import Agent
from world_models import common


# paths
home_path = os.path.dirname(os.path.realpath(__file__))
config_path = home_path + '/world_models/config.yaml'

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='raisim')
parser.add_argument('--ditto', type=str, default=False)
parser.add_argument('--agent', type=str, default=None)
parser.add_argument('--replay', type=str, default=None)
args = parser.parse_args()

# config and env
config, config_dict = common.init_config(config_path, args)
env_driver = common.get_driver(config, config_dict)
print(f'using device: {config.device}')

# agent
obs_dim, act_dim = env_driver.env_info()[:2]
agent = Agent(*env_driver.env_info(), config)
if args.agent is not None:
    agent_state_dict = torch.load(home_path + '/' + args.agent, map_location=config.device)
    agent.load_state_dict(agent_state_dict)

# replay buffer
state_replay = None
if args.replay is None:
    if args.ditto:
        replay = common.ReplayBuffer(config, {'obs': obs_dim, 'action': act_dim})
        state_replay = common.ReplayBuffer(config, {'state': config.h_dim + config.z_dim})
        expert_data = common.load_expert_data(
            home_path + '/world_models/expert_data/pos_error_cmds/expert.npy', obs_dim, config.device)
        replay.store_all(expert_data)
    else:
        replay = common.ReplayBuffer(config, {'obs': obs_dim, 'reward': 1, 'cont': 1, 'action': act_dim})
else:
    with open(home_path + '/' + args.replay, 'rb') as handle:
        replay = pickle.load(handle)
logger = common.Logger(config, agent, env_driver, replay)

if not args.ditto:
    dreamer.main(config, env_driver, agent, replay, logger)
else:
    ditto.main(config, env_driver, agent, replay, state_replay, logger)

# TODO: do p2e run w/o rewards and check speed up (w/ reward is ~183s / 1000 steps)
# TODO: do p2e run w/ velocity noise
# TODO: add vision
# TODO: plot latent space variance
# TODO: write model and world model tests
