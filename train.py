import argparse
import pickle
from pathlib import Path
from collections import OrderedDict

import torch

import world_models.dreamer as dreamer
import world_models.ditto as ditto

from world_models.agent import Agent
from world_models import common


# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='raisim')
parser.add_argument('--ditto', type=str, default=False)
parser.add_argument('--agent', type=str, default=None)
parser.add_argument('--replay', type=str, default=None)
args = parser.parse_args()

# paths
home_path = Path(__file__).parent.absolute()
config_path = home_path / 'world_models/config.yaml'
expert_path = home_path / 'world_models/expert_data/onphase/expert.npy'

# config and env
config, config_dict = common.init_config(config_path, args)
env_driver = common.get_driver(config, config_dict)
print(f'using device: {config.device}')

# agent
obs_dim, act_dim = env_driver.env_info()[:2]
agent = Agent(*env_driver.env_info(), config)
if args.agent is not None:
    agent_state_dict = torch.load(home_path / args.agent, map_location=config.device)
    # This is necessary to avoid duplicating the actor parameters
    agent_state_dict = OrderedDict([(k, v) for k, v in agent_state_dict.items() if not k.startswith('actor')])
    agent.load_state_dict(agent_state_dict, strict=False)

# replay buffer
if args.replay is None:
    if args.ditto:
        # ditto training
        replay = common.ReplayBuffer(config, {'obs': obs_dim, 'action': act_dim})
        state_replay = common.ReplayBuffer(config, {'state': config.h_dim + config.z_dim})
        expert_data = common.load_expert_data(expert_path, obs_dim, config.device)
        replay.store_all(expert_data)
        replays = (replay, state_replay)
    elif config.Plan2Explore and config.expert_replay_ratio > 0:
        # p2e training with expert data
        replay = common.ReplayBuffer(config, {'obs': obs_dim, 'cont': 1, 'action': act_dim})
        expert_replay = common.ReplayBuffer(config, {'obs': obs_dim, 'cont': 1, 'action': act_dim})
        expert_data = common.load_expert_data(expert_path, obs_dim, config.device)
        expert_replay.store_all(expert_data)
        replays = (replay, expert_replay)
    else:
        # everything else
        replays = tuple([common.ReplayBuffer(config, {'obs': obs_dim, 'reward': 1, 'cont': 1, 'action': act_dim})])
else:
    with open(home_path / args.replay, 'rb') as handle:
        replays = tuple([pickle.load(handle)])
logger = common.Logger(config, agent, env_driver, replays[0])

if not args.ditto:
    dreamer.main(config, env_driver, agent, replays, logger)
else:
    ditto.main(config, env_driver, agent, replays, logger)

# TODO: write model and world model tests
